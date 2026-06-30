import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import RecipeRetrievalDataset, load_recipe_records, load_split_ids, make_image_transform
from src.metrics import retrieval_metrics
from src.model import CrossModalRecipeModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate v2 cross-modal recipe retrieval checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root")
    parser.add_argument("--image_root")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="results_v2")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
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
    ckpt = torch.load(args.checkpoint, map_location=device)
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

    image_embs, recipe_embs, recipe_ids, image_ids = [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        output = model(batch)
        image_embs.append(output["image_emb"].cpu().numpy())
        recipe_embs.append(output["recipe_emb"].cpu().numpy())
        recipe_ids.extend(batch["recipe_id"])
        image_ids.extend(batch["image_id"])

    image_arr = np.concatenate(image_embs, axis=0)
    recipe_arr = np.concatenate(recipe_embs, axis=0)
    metrics = {}
    for prefix, query, target in (("i2r", image_arr, recipe_arr), ("r2i", recipe_arr, image_arr)):
        for key, value in retrieval_metrics(query, target).items():
            metrics[f"{prefix}_{key}"] = value

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{args.split}_embeddings.npz",
        image_embs=image_arr,
        recipe_embs=recipe_arr,
        recipe_ids=np.asarray(recipe_ids),
        image_ids=np.asarray(image_ids),
    )
    with open(output_dir / f"{args.split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
