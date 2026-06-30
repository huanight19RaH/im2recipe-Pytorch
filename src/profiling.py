import time

import torch


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": int(total), "params_trainable": int(trainable)}


def format_count(value):
    value = float(value)
    for suffix in ("", "K", "M", "B", "T"):
        if abs(value) < 1000.0:
            return f"{value:.2f}{suffix}"
        value /= 1000.0
    return f"{value:.2f}P"


def synchronize_if_needed(device):
    if getattr(device, "type", None) == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


@torch.no_grad()
def profile_forward_flops(model, batch, device):
    was_training = model.training
    model.eval()
    try:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda" and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            model(batch)
        flops = sum(getattr(event, "flops", 0) or 0 for event in prof.key_averages())
        batch_size = int(batch["image"].shape[0])
        return {
            "flops_per_batch": int(flops),
            "flops_per_sample": int(flops / max(batch_size, 1)),
            "profile_batch_size": batch_size,
        }
    except Exception as exc:
        return {"flops_error": str(exc)}
    finally:
        model.train(was_training)


@torch.no_grad()
def measure_forward_latency(model, batch, device, warmup=2, repeats=10):
    was_training = model.training
    model.eval()
    try:
        for _ in range(max(warmup, 0)):
            model(batch)
        synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(max(repeats, 1)):
            model(batch)
        synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
        batch_size = int(batch["image"].shape[0])
        return {
            "latency_repeats": int(max(repeats, 1)),
            "latency_batch_size": batch_size,
            "inference_seconds_per_batch": elapsed / max(repeats, 1),
            "inference_ms_per_sample": elapsed * 1000.0 / max(repeats * batch_size, 1),
        }
    finally:
        model.train(was_training)
