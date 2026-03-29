"""
Quick GPU memory benchmark to find the optimal batch size for training.

Usage (run on the cloud instance):
    python scripts/benchmark_batch.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from src.models.deeplabv3plus import build_model
from src.models.losses import build_criterion

with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("No GPU found.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"GPU       : {gpu_name}")
print(f"Total VRAM: {total_vram:.1f} GB")
print()

model     = build_model(cfg).to(device)
criterion = build_criterion(cfg, device=device)
scaler    = torch.amp.GradScaler("cuda")

in_ch    = cfg["model"]["in_channels"]
img_size = cfg["training"]["img_size"]

print(f"{'Batch':>6}  {'Fwd+Bwd':>10}  {'VRAM used':>10}  {'VRAM free':>10}  Status")
print("-" * 60)

best_batch = 8
for bs in [8, 16, 24, 32, 40, 48, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        imgs  = torch.randn(bs, in_ch, img_size, img_size, device=device)
        masks = torch.randint(0, 5, (bs, img_size, img_size), device=device)

        # Forward + backward (same as real training step)
        with torch.amp.autocast("cuda"):
            logits = model(imgs)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(torch.optim.AdamW(model.parameters(), lr=1e-4))
        scaler.update()

        used = torch.cuda.max_memory_allocated() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated()) / 1024**3
        print(f"{bs:>6}  {'OK':>10}  {used:>9.1f}G  {free:>9.1f}G  ✓")
        best_batch = bs

        # Clean up
        del imgs, masks, logits, loss
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{bs:>6}  {'OOM':>10}  {'--':>10}  {'--':>10}  ✗  OUT OF MEMORY")
        break

print()
print(f"Recommended batch_size: {best_batch}")
print(f"(Largest batch that fits with headroom for val)")
print()
print("Update configs/default.yaml:")
print(f"  batch_size: {best_batch}")
