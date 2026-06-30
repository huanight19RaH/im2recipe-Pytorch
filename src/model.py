import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MaskedMeanPool(nn.Module):
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).float()
        summed = (x * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class TextBranch(nn.Module):
    def __init__(self, vocab_size, emb_dim=256, num_heads=4, num_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.title_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.ing_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.inst_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pool = MaskedMeanPool()
        self.fusion_score = nn.Linear(emb_dim, 1)

    def encode_field(self, ids, mask, encoder):
        x = self.embedding(ids)
        x = encoder(x, src_key_padding_mask=~mask)
        return self.pool(x, mask)

    def forward(self, title_ids, title_mask, ing_ids, ing_mask, inst_ids, inst_mask):
        title = self.encode_field(title_ids, title_mask, self.title_encoder)
        ing = self.encode_field(ing_ids, ing_mask, self.ing_encoder)
        inst = self.encode_field(inst_ids, inst_mask, self.inst_encoder)
        parts = torch.stack([title, ing, inst], dim=1)
        weights = torch.softmax(self.fusion_score(parts), dim=1)
        return (parts * weights).sum(dim=1)


def _weights_for(backbone):
    if backbone == "resnet50":
        return models.ResNet50_Weights.DEFAULT
    if backbone == "efficientnet_b0":
        return models.EfficientNet_B0_Weights.DEFAULT
    raise ValueError(f"Unknown backbone: {backbone}")


class ImageBranch(nn.Module):
    def __init__(self, backbone="resnet50", pretrained=True):
        super().__init__()
        weights = _weights_for(backbone) if pretrained else None
        if backbone == "resnet50":
            net = models.resnet50(weights=weights)
            self.features = nn.Sequential(*list(net.children())[:-1])
            self.out_dim = net.fc.in_features
        elif backbone == "efficientnet_b0":
            net = models.efficientnet_b0(weights=weights)
            self.features = nn.Sequential(net.features, net.avgpool)
            self.out_dim = net.classifier[1].in_features
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, image):
        x = self.features(image)
        return torch.flatten(x, 1)


class CrossModalRecipeModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=512,
        text_dim=256,
        backbone="resnet50",
        pretrained=True,
        num_ingredient_labels=0,
        transformer_heads=4,
        transformer_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.image_branch = ImageBranch(backbone=backbone, pretrained=pretrained)
        self.text_branch = TextBranch(
            vocab_size=vocab_size,
            emb_dim=text_dim,
            num_heads=transformer_heads,
            num_layers=transformer_layers,
            dropout=dropout,
        )
        self.image_proj = nn.Sequential(nn.Linear(self.image_branch.out_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.recipe_proj = nn.Sequential(nn.Linear(text_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))
        self.ingredient_head = nn.Linear(emb_dim, num_ingredient_labels) if num_ingredient_labels > 0 else None

    def forward(self, batch):
        image_feat = self.image_branch(batch["image"])
        recipe_feat = self.text_branch(
            batch["title_ids"],
            batch["title_mask"],
            batch["ing_ids"],
            batch["ing_mask"],
            batch["inst_ids"],
            batch["inst_mask"],
        )
        image_emb = F.normalize(self.image_proj(image_feat), dim=-1)
        recipe_emb = F.normalize(self.recipe_proj(recipe_feat), dim=-1)
        ingredient_logits = self.ingredient_head(image_emb) if self.ingredient_head is not None else None
        return {"image_emb": image_emb, "recipe_emb": recipe_emb, "ingredient_logits": ingredient_logits}
