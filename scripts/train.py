#!/usr/bin/env python
"""
Train DeepLabV3+ (MobileNetV2 backbone) on the preprocessed xBD dataset.

Prerequisites
-------------
Run ``scripts/preprocess.py`` first to generate the PNG dataset.

Usage
-----
::

    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --epochs 100 --batch_size 4 --lr 5e-5
    python scripts/train.py --resume weights/latest.pth
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

from src.datasets.xbd import build_dataloaders
from src.models.deeplabv3plus import build_model
from src.models.losses import build_criterion
from src.training.trainer import Trainer


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    # ── Reproducibility ───────────────────────────────────────────────────
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"in_channels : {cfg['model']['in_channels']}  "
          f"({'pre+post' if cfg['model']['in_channels'] == 6 else 'post only'})")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nBuilding data loaders...")
    train_loader, val_loader = build_dataloaders(cfg)
    print(f"  Train batches : {len(train_loader)}  "
          f"({len(train_loader.dataset)} tiles)")
    print(f"  Val   batches : {len(val_loader)}  "
          f"({len(val_loader.dataset)} tiles)")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding model...")
    model = build_model(cfg).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total:,} total  ({trainable:,} trainable)")

    # ── Loss ──────────────────────────────────────────────────────────────
    print("\nBuilding loss function...")
    criterion = build_criterion(cfg, dataset=train_loader.dataset, device=device)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────
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

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    resume_path = cfg["training"].get("resume", "")
    if resume_path and os.path.isfile(resume_path):
        start_epoch = Trainer.load_checkpoint(resume_path, model, optimizer, scheduler, device)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model     = model,
        criterion = criterion,
        optimizer = optimizer,
        scheduler = scheduler,
        device    = device,
        cfg       = cfg,
    )

    # Sync best_miou from checkpoint if resuming
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        trainer._best_miou = ckpt.get("best_miou", 0.0)

    print(f"\nStarting training  (epochs {start_epoch+1} → {cfg['training']['epochs']})\n")
    trainer.fit(train_loader, val_loader, cfg["training"]["epochs"], start_epoch)

    print(f"\nTraining complete.")
    print(f"Best mIoU : {trainer._best_miou*100:.2f}%")
    print(f"Checkpoints : {cfg['training']['checkpoint_dir']}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepLabV3+ on xBD")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--in_channels", type=int,   default=None,
                   help="6=pre+post (default), 3=post only")
    p.add_argument("--resume",      default=None, help="Path to checkpoint")
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs      is not None: cfg["training"]["epochs"]      = args.epochs
    if args.batch_size  is not None: cfg["training"]["batch_size"]  = args.batch_size
    if args.lr          is not None: cfg["training"]["lr"]          = args.lr
    if args.in_channels is not None: cfg["model"]["in_channels"]    = args.in_channels
    if args.resume      is not None: cfg["training"]["resume"]      = args.resume
    if args.seed        is not None: cfg["training"]["seed"]        = args.seed

    main(cfg)
