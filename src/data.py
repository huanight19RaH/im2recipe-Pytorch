import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
INGREDIENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "bag",
    "baked",
    "bar",
    "bottle",
    "box",
    "bunch",
    "can",
    "chopped",
    "clove",
    "cloves",
    "container",
    "cooked",
    "cup",
    "cups",
    "dash",
    "diced",
    "drained",
    "dry",
    "each",
    "finely",
    "for",
    "fresh",
    "frozen",
    "g",
    "gram",
    "grams",
    "ground",
    "kg",
    "large",
    "lb",
    "lbs",
    "medium",
    "minced",
    "ml",
    "of",
    "or",
    "oz",
    "ounce",
    "ounces",
    "package",
    "packages",
    "piece",
    "pieces",
    "pinch",
    "pound",
    "pounds",
    "sliced",
    "small",
    "tablespoon",
    "tablespoons",
    "tbsp",
    "teaspoon",
    "teaspoons",
    "tsp",
    "to",
    "with",
}


def tokenize(text):
    return TOKEN_RE.findall((text or "").lower())


def ingredient_tokens(text):
    return [
        token
        for token in tokenize(text)
        if len(token) > 1 and not token.isdigit() and token not in INGREDIENT_STOPWORDS
    ]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def image_path_for(image_root, image_id):
    return Path(image_root) / image_id[0] / image_id[1] / image_id[2] / image_id[3] / image_id


def _find_file(data_root, candidates):
    root = Path(data_root)
    for name in candidates:
        path = root / name
        if path.exists():
            return path
    matches = []
    for pattern in candidates:
        matches.extend(root.glob(pattern))
    if matches:
        return matches[0]
    for pattern in candidates:
        matches.extend(root.rglob(pattern))
    if matches:
        return matches[0]
    visible = []
    if root.exists():
        visible = [str(p.relative_to(root)) for p in list(root.iterdir())[:20]]
    raise FileNotFoundError(
        "Could not find any of: "
        + ", ".join(candidates)
        + f" under {root}. Visible entries: {visible}"
    )


def load_recipe_records(data_root, image_root=None):
    data_root = Path(data_root)
    layer1_path = _find_file(data_root, ["layer1_subset.json", "layer1_subset (1).json", "layer1*.json"])
    actual_root = layer1_path.parent
    image_root = Path(image_root) if image_root else actual_root / "images" / "images"
    layer2_path = _find_file(data_root, ["layer2_subset.json", "layer2_subset (1).json", "layer2*.json"])

    layer1 = _read_json(layer1_path)
    layer2 = {entry["id"]: entry for entry in _read_json(layer2_path)}
    records = {}
    for entry in layer1:
        recipe_id = entry["id"]
        image_entries = layer2.get(recipe_id, {}).get("images", [])
        valid_images = []
        for image_entry in image_entries:
            image_id = image_entry["id"]
            path = image_path_for(image_root, image_id)
            if path.exists():
                valid_images.append({"id": image_id, "path": str(path)})
        if not valid_images:
            continue
        records[recipe_id] = {
            "id": recipe_id,
            "title": entry.get("title", ""),
            "ingredients": [x.get("text", "") for x in entry.get("ingredients", []) if x.get("text")],
            "instructions": [x.get("text", "") for x in entry.get("instructions", []) if x.get("text")],
            "partition": entry.get("partition"),
            "images": valid_images,
        }
    return records


def load_split_ids(data_root, split):
    return _read_ids(_find_file(data_root, [f"{split}_ids.txt"]))


def build_text_vocab(records, train_ids, min_freq=1, max_vocab=30000):
    counter = Counter()
    for recipe_id in train_ids:
        record = records.get(recipe_id)
        if not record:
            continue
        fields = [record["title"]] + record["ingredients"] + record["instructions"]
        for field in fields:
            counter.update(tokenize(field))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def build_ingredient_labels(records, train_ids, num_labels=100):
    counter = Counter()
    for recipe_id in train_ids:
        record = records.get(recipe_id)
        if not record:
            continue
        for ingredient in record["ingredients"]:
            counter.update(set(ingredient_tokens(ingredient)))
    labels = [token for token, _ in counter.most_common(num_labels)]
    return {token: idx for idx, token in enumerate(labels)}


def make_image_transform(train=True, image_size=224):
    if train:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class RecipeRetrievalDataset(Dataset):
    def __init__(
        self,
        records,
        ids,
        vocab,
        ingredient_label_map=None,
        transform=None,
        train=False,
        max_title_len=32,
        max_ing_len=96,
        max_inst_len=256,
    ):
        self.records = [records[recipe_id] for recipe_id in ids if recipe_id in records]
        self.vocab = vocab
        self.ingredient_label_map = ingredient_label_map or {}
        self.transform = transform
        self.train = train
        self.max_title_len = max_title_len
        self.max_ing_len = max_ing_len
        self.max_inst_len = max_inst_len

    def __len__(self):
        return len(self.records)

    def _encode(self, text, max_len):
        ids = [self.vocab.get(token, self.vocab[UNK_TOKEN]) for token in tokenize(text)[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad = max_len - len(ids)
            ids.extend([self.vocab[PAD_TOKEN]] * pad)
            mask.extend([0] * pad)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)

    def _ingredient_targets(self, ingredients):
        target = torch.zeros(len(self.ingredient_label_map), dtype=torch.float32)
        if target.numel() == 0:
            return target
        for ingredient in ingredients:
            for token in set(ingredient_tokens(ingredient)):
                idx = self.ingredient_label_map.get(token)
                if idx is not None:
                    target[idx] = 1.0
        return target

    def _load_image(self, record):
        images = list(record["images"])
        if self.train:
            random.shuffle(images)
        for image_info in images:
            try:
                image = Image.open(image_info["path"]).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, image_info["id"]
            except Exception:
                continue
        image = Image.new("RGB", (224, 224), "white")
        if self.transform:
            image = self.transform(image)
        return image, images[0]["id"] if images else ""

    def __getitem__(self, index):
        record = self.records[index]
        image, image_id = self._load_image(record)

        title_ids, title_mask = self._encode(record["title"], self.max_title_len)
        ing_ids, ing_mask = self._encode(" ".join(record["ingredients"]), self.max_ing_len)
        inst_ids, inst_mask = self._encode(" ".join(record["instructions"]), self.max_inst_len)

        return {
            "image": image,
            "title_ids": title_ids,
            "title_mask": title_mask,
            "ing_ids": ing_ids,
            "ing_mask": ing_mask,
            "inst_ids": inst_ids,
            "inst_mask": inst_mask,
            "ingredient_targets": self._ingredient_targets(record["ingredients"]),
            "recipe_id": record["id"],
            "image_id": image_id,
        }
