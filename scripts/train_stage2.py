#!/usr/bin/env python
"""
Train Stage 2: damage classification on building pixels only.

The model sees the full 6-channel (pre+post) image but only receives loss
signal on building pixels. Background is masked out (ignore_index=255).
This forces the model to specialise on inter-damage-level distinctions
rather than wasting capacity on the easy background vs. building separation.

Usage
-----
::

    python scripts/train_stage2.py
    python scripts/train_stage2.py --config configs/stage2.yaml
    python scripts/train_stage2.py --epochs 60 --batch_size 8
    python scripts/train_stage2.py --resume weights/stage2/latest.pth
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.optim as optim
import yaml

from src.datasets.xbd_stage2 import _S2_CLASS_NAMES, build_dataloaders_stage2
from src.models.deeplabv3plus import build_model
from src.models.losses import build_criterion
from src.training.trainer import Trainer


def main(cfg: dict) -> None:
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Stage       : 2 — damage classification (building pixels only)")
    print(f"Backbone    : {cfg['model'].get('backbone', 'mobilenet_v2')}")

    print("\nBuilding data loaders (Stage 2 — building tiles only)...")
    train_loader, val_loader = build_dataloaders_stage2(cfg)
    print(f"  Train batches : {len(train_loader)}  ({len(train_loader.dataset)} tiles)")
    print(f"  Val   batches : {len(val_loader)}  ({len(val_loader.dataset)} tiles)")

    print("\nBuilding model...")
    model = build_model(cfg).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total  ({trainable:,} trainable)")

    print("\nBuilding loss function...")
    criterion = build_criterion(cfg, dataset=None, device=device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr           = cfg["training"]["lr"],
        weight_decay = cfg["training"]["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cfg["training"]["epochs"],
        eta_min = 1e-6,
    )

    start_epoch = 0
    resume_path = cfg["training"].get("resume", "")
    if resume_path and os.path.isfile(resume_path):
        start_epoch = Trainer.load_checkpoint(resume_path, model, optimizer, scheduler, device)

    trainer = Trainer(
        model       = model,
        criterion   = criterion,
        optimizer   = optimizer,
        scheduler   = scheduler,
        device      = device,
        cfg         = cfg,
        class_names = _S2_CLASS_NAMES,
    )

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        trainer._best_miou = ckpt.get("best_miou", 0.0)

    print(f"\nStarting training  (epochs {start_epoch+1} → {cfg['training']['epochs']})\n")
    trainer.fit(train_loader, val_loader, cfg["training"]["epochs"], start_epoch)

    print(f"\nStage 2 training complete.")
    print(f"Best mIoU : {trainer._best_miou*100:.2f}%")
    print(f"Checkpoint: {cfg['training']['checkpoint_dir']}/best.pth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Stage 2 (damage classification)")
    p.add_argument("--config",     default="configs/stage2.yaml")
    p.add_argument("--epochs",     type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--lr",         type=float, default=None)
    p.add_argument("--resume",     default=None, help="Path to checkpoint")
    p.add_argument("--seed",       type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs     is not None: cfg["training"]["epochs"]     = args.epochs
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.lr         is not None: cfg["training"]["lr"]         = args.lr
    if args.resume     is not None: cfg["training"]["resume"]     = args.resume
    if args.seed       is not None: cfg["training"]["seed"]       = args.seed

    main(cfg)
