import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from src.data import (
    RecipeRetrievalDataset,
    build_ingredient_labels,
    build_text_vocab,
    load_recipe_records,
    load_split_ids,
    make_image_transform,
)
from src.losses import symmetric_contrastive_loss
from src.metrics import retrieval_metrics
from src.model import CrossModalRecipeModel
from src.profiling import count_parameters, format_count, measure_forward_latency, profile_forward_flops, synchronize_if_needed


def parse_args():
    parser = argparse.ArgumentParser(description="Kaggle-ready improved im2recipe trainer")
    parser.add_argument("--config", default="configs/kaggle.yaml")
    parser.add_argument("--data_root")
    parser.add_argument("--image_root")
    parser.add_argument("--output_dir")
    parser.add_argument("--backbone")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--emb_dim", type=int)
    parser.add_argument("--text_dim", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--ingredient_loss_weight", type=float)
    parser.add_argument("--profile_batches", type=int)
    parser.add_argument("--max_title_len", type=int)
    parser.add_argument("--max_ing_len", type=int)
    parser.add_argument("--max_inst_len", type=int)
    parser.add_argument("--limit_train_batches", type=int)
    parser.add_argument("--limit_val_batches", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_ingredient_loss", action="store_true")
    return parser.parse_args()


def load_config(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for key, value in vars(args).items():
        if key in ("config", "no_amp") or value is None:
            continue
        cfg[key] = value
    if args.no_amp:
        cfg["amp"] = False
    if args.no_pretrained:
        cfg["pretrained"] = False
    if args.no_ingredient_loss:
        cfg["use_ingredient_loss"] = False
    return cfg


def init_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, 1, 0
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return True, rank, world_size, local_rank


def is_main(rank):
    return rank == 0


def log_main(rank, message):
    if is_main(rank):
        print(message, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


def gather_objects(obj, distributed):
    if not distributed:
        return [obj]
    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def build_loaders(cfg, distributed, rank, world_size):
    log_main(rank, f"[data] Searching Recipe1M-style files under: {cfg['data_root']}")
    records = load_recipe_records(cfg["data_root"], cfg.get("image_root"))
    log_main(rank, f"[data] Loaded {len(records)} recipes with at least one readable image path.")
    log_main(rank, "[data] Loading train/val/test split ids...")
    split_ids = {split: load_split_ids(cfg["data_root"], split) for split in ("train", "val", "test")}
    log_main(rank, "[data] Building text vocabulary from train split...")
    vocab = build_text_vocab(records, split_ids["train"], min_freq=cfg["min_freq"], max_vocab=cfg["max_vocab"])
    ingredient_labels = {}
    if cfg.get("use_ingredient_loss", True):
        log_main(rank, "[data] Building ingredient multi-label vocabulary...")
        ingredient_labels = build_ingredient_labels(records, split_ids["train"], cfg["num_ingredient_labels"])

    datasets = {
        "train": RecipeRetrievalDataset(
            records,
            split_ids["train"],
            vocab,
            ingredient_labels,
            transform=make_image_transform(True),
            train=True,
            max_title_len=cfg["max_title_len"],
            max_ing_len=cfg["max_ing_len"],
            max_inst_len=cfg["max_inst_len"],
        ),
        "val": RecipeRetrievalDataset(
            records,
            split_ids["val"],
            vocab,
            ingredient_labels,
            transform=make_image_transform(False),
            train=False,
            max_title_len=cfg["max_title_len"],
            max_ing_len=cfg["max_ing_len"],
            max_inst_len=cfg["max_inst_len"],
        ),
        "test": RecipeRetrievalDataset(
            records,
            split_ids["test"],
            vocab,
            ingredient_labels,
            transform=make_image_transform(False),
            train=False,
            max_title_len=cfg["max_title_len"],
            max_ing_len=cfg["max_ing_len"],
            max_inst_len=cfg["max_inst_len"],
        ),
    }
    train_sampler = DistributedSampler(datasets["train"], num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(datasets["val"], num_replicas=world_size, rank=rank, shuffle=False) if distributed else None
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=cfg["batch_size"],
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            sampler=val_sampler,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=torch.cuda.is_available(),
        ),
    }
    meta = {
        "vocab": vocab,
        "ingredient_labels": ingredient_labels,
        "split_sizes": {k: len(v) for k, v in datasets.items()},
    }
    return loaders, train_sampler, meta


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch, sampler=None, show_progress=True):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(loader, desc=f"train epoch {epoch}", disable=not show_progress, leave=False)
    for step, batch in enumerate(progress):
        if cfg.get("limit_train_batches") and step >= cfg["limit_train_batches"]:
            break
        batch = to_device(batch, device)
        with autocast(device_type=device.type, enabled=cfg["amp"] and torch.cuda.is_available()):
            output = model(batch)
            loss = symmetric_contrastive_loss(output["image_emb"], output["recipe_emb"], cfg["temperature"])
            if output["ingredient_logits"] is not None and batch["ingredient_targets"].numel() > 0:
                ing_loss = F.binary_cross_entropy_with_logits(output["ingredient_logits"], batch["ingredient_targets"])
                loss = loss + cfg["ingredient_loss_weight"] * ing_loss
            loss = loss / cfg["grad_accum_steps"]
        scaler.scale(loss).backward()
        if (step + 1) % cfg["grad_accum_steps"] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * cfg["grad_accum_steps"]
        steps += 1
        if show_progress:
            progress.set_postfix(loss=f"{total_loss / max(steps, 1):.4f}")
    if steps > 0 and steps % cfg["grad_accum_steps"] != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    return total_loss / max(steps, 1)


@torch.no_grad()
def evaluate(model, loader, device, cfg, distributed=False, show_progress=True):
    model.eval()
    image_embs, recipe_embs, recipe_ids, image_ids = [], [], [], []
    forward_seconds = 0.0
    forward_samples = 0
    wall_start = time.perf_counter()
    progress = tqdm(loader, desc="eval", disable=not show_progress, leave=False)
    for step, batch in enumerate(progress):
        if cfg.get("limit_val_batches") and step >= cfg["limit_val_batches"]:
            break
        batch = to_device(batch, device)
        synchronize_if_needed(device)
        forward_start = time.perf_counter()
        output = model(batch)
        synchronize_if_needed(device)
        forward_seconds += time.perf_counter() - forward_start
        forward_samples += int(batch["image"].shape[0])
        image_embs.append(output["image_emb"].detach().cpu().numpy())
        recipe_embs.append(output["recipe_emb"].detach().cpu().numpy())
        recipe_ids.extend(batch["recipe_id"])
        image_ids.extend(batch["image_id"])

    local = {
        "image_embs": np.concatenate(image_embs, axis=0) if image_embs else np.zeros((0, cfg["emb_dim"]), dtype=np.float32),
        "recipe_embs": np.concatenate(recipe_embs, axis=0) if recipe_embs else np.zeros((0, cfg["emb_dim"]), dtype=np.float32),
        "recipe_ids": recipe_ids,
        "image_ids": image_ids,
    }
    gathered = gather_objects(local, distributed)
    merged = {}
    merged["image_embs"] = np.concatenate([x["image_embs"] for x in gathered], axis=0)
    merged["recipe_embs"] = np.concatenate([x["recipe_embs"] for x in gathered], axis=0)
    merged["recipe_ids"] = sum([x["recipe_ids"] for x in gathered], [])
    merged["image_ids"] = sum([x["image_ids"] for x in gathered], [])

    if len(merged["recipe_ids"]) == 0:
        return {}, merged
    _, unique_idx = np.unique(np.asarray(merged["recipe_ids"]), return_index=True)
    unique_idx = np.sort(unique_idx)
    image_arr = merged["image_embs"][unique_idx]
    recipe_arr = merged["recipe_embs"][unique_idx]
    metrics = {}
    for prefix, query, target in (("i2r", image_arr, recipe_arr), ("r2i", recipe_arr, image_arr)):
        for key, value in retrieval_metrics(query, target).items():
            metrics[f"{prefix}_{key}"] = value
    metrics["eval_wall_seconds"] = time.perf_counter() - wall_start
    metrics["inference_forward_seconds"] = forward_seconds
    metrics["inference_samples"] = int(forward_samples)
    metrics["inference_ms_per_sample"] = forward_seconds * 1000.0 / max(forward_samples, 1)
    return metrics, merged


def save_checkpoint(path, model, optimizer, scaler, epoch, best_recall, cfg, meta):
    raw_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_recall": best_recall,
            "config": cfg,
            "vocab": meta["vocab"],
            "ingredient_labels": meta["ingredient_labels"],
        },
        path,
    )


def main():
    args = parse_args()
    cfg = load_config(args)
    distributed, rank, world_size, local_rank = init_distributed()
    seed_everything(cfg["seed"] + rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg["output_dir"])
    if is_main(rank):
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config_resolved.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        print(f"[init] distributed={distributed} world_size={world_size} device={device}", flush=True)
        print(f"[init] outputs will be written to: {output_dir}", flush=True)

    loaders, train_sampler, meta = build_loaders(cfg, distributed, rank, world_size)
    if is_main(rank):
        print("Split sizes:", meta["split_sizes"])
        print("Vocab size:", len(meta["vocab"]))
        print("Ingredient labels:", len(meta["ingredient_labels"]))
        print(f"[model] Building {cfg['backbone']} backbone. pretrained={cfg['pretrained']}", flush=True)

    model = CrossModalRecipeModel(
        vocab_size=len(meta["vocab"]),
        emb_dim=cfg["emb_dim"],
        text_dim=cfg["text_dim"],
        backbone=cfg["backbone"],
        pretrained=cfg["pretrained"],
        num_ingredient_labels=len(meta["ingredient_labels"]),
        transformer_heads=cfg["transformer_heads"],
        transformer_layers=cfg["transformer_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    log_main(rank, "[model] Model is ready.")
    raw_model = model.module if hasattr(model, "module") else model
    if is_main(rank):
        param_stats = count_parameters(raw_model)
        profile_stats = dict(param_stats)
        print(
            "[profile] Params total="
            f"{format_count(param_stats['params_total'])} "
            f"trainable={format_count(param_stats['params_trainable'])}",
            flush=True,
        )
        try:
            profile_batch = next(iter(loaders["val"]))
            profile_batch = to_device(profile_batch, device)
            flops_stats = profile_forward_flops(raw_model, profile_batch, device)
            latency_stats = measure_forward_latency(
                raw_model,
                profile_batch,
                device,
                repeats=cfg.get("profile_batches") or 10,
            )
            profile_stats.update(flops_stats)
            profile_stats.update(latency_stats)
            if "flops_per_sample" in flops_stats:
                print(
                    "[profile] FLOPs/sample="
                    f"{format_count(flops_stats['flops_per_sample'])} "
                    f"FLOPs/batch={format_count(flops_stats['flops_per_batch'])}",
                    flush=True,
                )
            else:
                print(f"[profile] FLOPs unavailable: {flops_stats.get('flops_error')}", flush=True)
            print(
                "[profile] Inference latency="
                f"{latency_stats['inference_ms_per_sample']:.3f} ms/sample "
                f"({latency_stats['inference_seconds_per_batch']:.4f} s/batch)",
                flush=True,
            )
        except Exception as exc:
            profile_stats["profile_error"] = str(exc)
            print(f"[profile] Skipped profiling: {exc}", flush=True)
        with open(output_dir / "model_profile.json", "w", encoding="utf-8") as f:
            json.dump(profile_stats, f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler("cuda", enabled=cfg["amp"] and torch.cuda.is_available())
    start_epoch = 0
    best_recall = -1.0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_recall = ckpt["best_recall"]

    history = []
    training_start = time.perf_counter()
    for epoch in range(start_epoch, cfg["epochs"]):
        log_main(rank, f"[epoch {epoch}] Training started.")
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            optimizer,
            scaler,
            device,
            cfg,
            epoch,
            train_sampler,
            show_progress=is_main(rank),
        )
        train_seconds = time.perf_counter() - train_start
        eval_start = time.perf_counter()
        val_metrics, val_outputs = evaluate(model, loaders["val"], device, cfg, distributed, show_progress=is_main(rank))
        eval_seconds = time.perf_counter() - eval_start
        epoch_seconds = time.perf_counter() - epoch_start
        recall10 = val_metrics.get("i2r_R@10", 0.0)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": epoch_seconds,
            "total_training_seconds": time.perf_counter() - training_start,
            **val_metrics,
        }
        history.append(row)
        if is_main(rank):
            print(json.dumps(row, indent=2))
            save_checkpoint(output_dir / "last.pt", model, optimizer, scaler, epoch, best_recall, cfg, meta)
            if recall10 > best_recall:
                best_recall = recall10
                save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, epoch, best_recall, cfg, meta)
                np.savez_compressed(output_dir / "val_embeddings.npz", **val_outputs)
            with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
