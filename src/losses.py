import torch
import torch.nn.functional as F


def symmetric_contrastive_loss(image_emb, recipe_emb, temperature=0.07):
    logits = image_emb @ recipe_emb.t()
    logits = logits / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_i2r = F.cross_entropy(logits, labels)
    loss_r2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2r + loss_r2i) * 0.5
