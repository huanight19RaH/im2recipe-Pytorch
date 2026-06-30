import numpy as np


def retrieval_metrics(query_embs, target_embs, ks=(1, 5, 10)):
    sims = query_embs @ target_embs.T
    ranks = []
    for i in range(sims.shape[0]):
        order = np.argsort(sims[i])[::-1]
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(rank)
    ranks = np.asarray(ranks)
    metrics = {"MedR": float(np.median(ranks))}
    for k in ks:
        metrics[f"R@{k}"] = float(np.mean(ranks <= k))
    return metrics
