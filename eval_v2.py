import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data import RecipeRetrievalDataset, load_recipe_records, load_split_ids, make_image_transform
from src.metrics import retrieval_metrics
from src.model import CrossModalRecipeModel
from src.profiling import count_parameters, format_count, measure_forward_latency, profile_forward_flops, synchronize_if_needed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v2 cross-modal recipe retrieval checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root")
    parser.add_argument("--image_root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="results_v2")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--profile_batches", type=int, default=10)
    return parser.parse_args()


def to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return moved


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "This usually means training failed before saving best.pt, or --output_dir/--checkpoint points to the wrong folder.\n"
            "On Kaggle, check the files with: !find /kaggle/working -maxdepth 3 -name '*.pt' -print"
        )
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["config"]
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.image_root:
        cfg["image_root"] = args.image_root
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    records = load_recipe_records(cfg["data_root"], cfg.get("image_root"))
    split_ids = load_split_ids(cfg["data_root"], args.split)
    dataset = RecipeRetrievalDataset(
        records,
        split_ids,
        ckpt["vocab"],
        ckpt.get("ingredient_labels", {}),
        transform=make_image_transform(False),
        train=False,
        max_title_len=cfg["max_title_len"],
        max_ing_len=cfg["max_ing_len"],
        max_inst_len=cfg["max_inst_len"],
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    model = CrossModalRecipeModel(
        vocab_size=len(ckpt["vocab"]),
        emb_dim=cfg["emb_dim"],
        text_dim=cfg["text_dim"],
        backbone=cfg["backbone"],
        pretrained=False,
        num_ingredient_labels=len(ckpt.get("ingredient_labels", {})),
        transformer_heads=cfg["transformer_heads"],
        transformer_layers=cfg["transformer_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    param_stats = count_parameters(model)
    print(
        "[profile] Params total="
        f"{format_count(param_stats['params_total'])} "
        f"trainable={format_count(param_stats['params_trainable'])}",
        flush=True,
    )

    image_embs, recipe_embs, recipe_ids, image_ids = [], [], [], []
    forward_seconds = 0.0
    forward_samples = 0
    wall_start = time.perf_counter()
    for batch in tqdm(loader, desc=f"eval {args.split}", leave=False):
        batch = to_device(batch, device)
        synchronize_if_needed(device)
        forward_start = time.perf_counter()
        output = model(batch)
        synchronize_if_needed(device)
        forward_seconds += time.perf_counter() - forward_start
        forward_samples += int(batch["image"].shape[0])
        image_embs.append(output["image_emb"].cpu().numpy())
        recipe_embs.append(output["recipe_emb"].cpu().numpy())
        recipe_ids.extend(batch["recipe_id"])
        image_ids.extend(batch["image_id"])
    wall_seconds = time.perf_counter() - wall_start

    image_arr = np.concatenate(image_embs, axis=0)
    recipe_arr = np.concatenate(recipe_embs, axis=0)
    metrics = {}
    for prefix, query, target in (("i2r", image_arr, recipe_arr), ("r2i", recipe_arr, image_arr)):
        for key, value in retrieval_metrics(query, target).items():
            metrics[f"{prefix}_{key}"] = value
    metrics.update(
        {
            "eval_wall_seconds": wall_seconds,
            "inference_forward_seconds": forward_seconds,
            "inference_samples": int(forward_samples),
            "inference_ms_per_sample": forward_seconds * 1000.0 / max(forward_samples, 1),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_stats = dict(param_stats)
    try:
        profile_batch = next(iter(loader))
        profile_batch = to_device(profile_batch, device)
        profile_stats.update(profile_forward_flops(model, profile_batch, device))
        profile_stats.update(measure_forward_latency(model, profile_batch, device, repeats=args.profile_batches))
        if "flops_per_sample" in profile_stats:
            print(
                "[profile] FLOPs/sample="
                f"{format_count(profile_stats['flops_per_sample'])} "
                f"latency={profile_stats['inference_ms_per_sample']:.3f} ms/sample",
                flush=True,
            )
    except Exception as exc:
        profile_stats["profile_error"] = str(exc)
    np.savez_compressed(
        output_dir / f"{args.split}_embeddings.npz",
        image_embs=image_arr,
        recipe_embs=recipe_arr,
        recipe_ids=np.asarray(recipe_ids),
        image_ids=np.asarray(image_ids),
    )
    with open(output_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / f"{args.split}_profile.json", "w", encoding="utf-8") as f:
        json.dump(profile_stats, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
