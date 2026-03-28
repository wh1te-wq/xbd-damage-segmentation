"""
Training loop for xBD damage segmentation.

Features
--------
* Mixed-precision training (AMP) via ``torch.amp``.
* Optional gradient clipping (``cfg["training"]["grad_clip"]``).
* CSV log written after every epoch.
* Saves ``best.pth`` (best val mIoU) and ``latest.pth`` after every epoch.
* Supports resuming from a checkpoint.
* Clean separation: ``Trainer`` holds all state; scripts only call ``trainer.fit()``.
"""

import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.metrics import SegmentationMetrics


_CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


class Trainer:
    """
    Encapsulates the full training and evaluation logic.

    Parameters
    ----------
    model     : nn.Module
    criterion : nn.Module        — loss function (e.g. ComboLoss)
    optimizer : torch.optim.Optimizer
    scheduler : torch.optim.lr_scheduler._LRScheduler
    device    : torch.device
    cfg       : dict             — full config dictionary
    """

    def __init__(
        self,
        model:     nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device:    torch.device,
        cfg:       dict,
    ):
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.cfg       = cfg

        self.num_classes   = cfg["model"]["num_classes"]
        self.grad_clip     = float(cfg["training"].get("grad_clip", 0.0))
        self.checkpoint_dir = cfg["training"].get("checkpoint_dir", "weights")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._scaler   = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        self._best_miou = 0.0
        self._log_path  = os.path.join(self.checkpoint_dir, "train_log.csv")

    # ── Public entry point ────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int,
        start_epoch:  int = 0,
    ) -> None:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader, val_loader : DataLoader
        num_epochs   : int — total number of epochs to train.
        start_epoch  : int — epoch index to start from (>0 when resuming).
        """
        self._init_log(start_epoch)

        for epoch in range(start_epoch, num_epochs):
            t0 = time.time()
            lr = self.scheduler.get_last_lr()[0]
            print(f"\nEpoch [{epoch+1}/{num_epochs}]  lr={lr:.2e}")

            train_loss = self._train_epoch(train_loader)
            val_loss, metrics = self._eval_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            print(f"  Train loss : {train_loss:.4f}  |  Val loss : {val_loss:.4f}")
            print(str(metrics))
            print(f"  Time: {elapsed:.1f}s")

            self._log_epoch(epoch + 1, train_loss, val_loss, metrics)
            self._save_checkpoint(epoch, metrics.miou())

    # ── Train / eval ──────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(loader, desc="  train", leave=False):
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)

            self._scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self._scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self._scaler.step(self.optimizer)
            self._scaler.update()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, SegmentationMetrics]:
        self.model.eval()
        total_loss = 0.0
        metrics    = SegmentationMetrics(self.num_classes, class_names=_CLASS_NAMES)

        for imgs, masks in tqdm(loader, desc="  val  ", leave=False):
            imgs  = imgs.to(self.device)
            masks = masks.to(self.device)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                logits = self.model(imgs)
                loss   = self.criterion(logits, masks)

            total_loss += loss.item()
            metrics.update(logits.argmax(dim=1), masks)

        return total_loss / len(loader), metrics

    # ── Checkpoint handling ───────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, miou: float) -> None:
        ckpt = {
            "epoch":       epoch,
            "model":       self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "best_miou":   self._best_miou,
            "in_channels": self.cfg["model"]["in_channels"],
        }
        torch.save(ckpt, os.path.join(self.checkpoint_dir, "latest.pth"))

        if miou > self._best_miou:
            self._best_miou = miou
            torch.save(ckpt, os.path.join(self.checkpoint_dir, "best.pth"))
            print(f"  *** Best mIoU updated: {self._best_miou*100:.2f}%  — best.pth saved ***")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        model:     nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler                             = None,
        device: torch.device                  = torch.device("cpu"),
    ) -> int:
        """
        Load a checkpoint into *model* (and optionally *optimizer* / *scheduler*).

        Returns
        -------
        int — the epoch the checkpoint was saved at + 1 (next epoch to run).
        """
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if optimizer and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        next_epoch = ckpt.get("epoch", 0) + 1
        print(f"Loaded checkpoint from {checkpoint_path}  "
              f"(epoch {ckpt.get('epoch', '?')}, best mIoU={ckpt.get('best_miou', 0):.4f})")
        return next_epoch

    # ── Logging ───────────────────────────────────────────────────────────────

    def _init_log(self, start_epoch: int) -> None:
        if start_epoch == 0 or not os.path.exists(self._log_path):
            with open(self._log_path, "w", newline="") as f:
                writer = csv.writer(f)
                header = ["epoch", "train_loss", "val_loss", "pix_acc", "miou", "mean_f1"]
                header += [f"iou_{c}" for c in _CLASS_NAMES]
                writer.writerow(header)

    def _log_epoch(
        self,
        epoch:      int,
        train_loss: float,
        val_loss:   float,
        metrics:    SegmentationMetrics,
    ) -> None:
        s   = metrics.summary()
        iou = [s["iou"][c] for c in _CLASS_NAMES]
        row = [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
               f"{s['pix_acc']:.6f}", f"{s['miou']:.6f}", f"{s['mean_f1']:.6f}"]
        row += ["nan" if v != v else f"{v:.6f}" for v in iou]
        with open(self._log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
