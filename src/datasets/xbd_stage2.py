"""
Datasets for the two-stage damage segmentation pipeline.

Stage 1 — Binary building segmentation
---------------------------------------
``XBDDatasetStage1`` wraps the preprocessed xBD dataset and remaps the
5-class masks to binary: 0 = background, 1 = building (any damage level).

Stage 2 — Damage classification
---------------------------------
``XBDDatasetStage2`` trains *only* on tiles that contain at least one building
pixel and remaps the 5-class mask to 4 damage classes:

    original  →  stage-2
    ─────────────────────
    0 (bg)    →  255  (ignored by loss)
    1 (no-dmg)→  0
    2 (minor) →  1
    3 (major) →  2
    4 (destr) →  3

Ignoring background pixels forces Stage 2 to specialise entirely on building
appearance and inter-class damage distinctions — the hard problem that the
single-stage model could not solve.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.datasets.xbd import XBDDataset
from src.utils.augmentations import build_augmentor

# ─── Stage-2 remap table ─────────────────────────────────────────────────────
# Index 0-4 of this tensor maps original class IDs to stage-2 class IDs.
_S2_REMAP = torch.tensor([255, 0, 1, 2, 3], dtype=torch.long)

_S1_CLASS_NAMES = ["background", "building"]
_S2_CLASS_NAMES = ["no-damage", "minor-damage", "major-damage", "destroyed"]


# ─── Stage 1 dataset ──────────────────────────────────────────────────────────

class XBDDatasetStage1(XBDDataset):
    """
    Binary building segmentation dataset (Stage 1).

    Inherits all image loading and augmentation from :class:`XBDDataset`.
    Only the returned mask changes: any non-zero class becomes class 1.
    """

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = super().__getitem__(idx)
        return img, (mask > 0).long()

    def get_sample_weights(self, oversample_damage: float = 3.0) -> list[float]:
        """Up-weight tiles that contain buildings so they appear more often."""
        weights = []
        for stem in tqdm(self.stems, desc="  Scanning masks (stage1 weights)", leave=False):
            mask = cv2.imread(
                os.path.join(self.masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE
            )
            has_building = bool(np.any(mask > 0))
            weights.append(oversample_damage if has_building else 1.0)
        return weights


# ─── Stage 2 dataset ──────────────────────────────────────────────────────────

class XBDDatasetStage2(XBDDataset):
    """
    Damage classification dataset (Stage 2).

    Only tiles that contain at least one building pixel are kept.
    Masks are remapped: bg→255 (ignored), no-dmg→0, minor→1, major→2, destroyed→3.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Filter to building-containing tiles only
        before = len(self.stems)
        self.stems = [s for s in tqdm(self.stems,
                                      desc="  Filtering building tiles",
                                      leave=False)
                      if self._has_building(s)]
        print(f"  Stage-2 tile filter: {before} → {len(self.stems)} tiles "
              f"(kept tiles with ≥1 building pixel)")

    def _has_building(self, stem: str) -> bool:
        mask = cv2.imread(
            os.path.join(self.masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE
        )
        return mask is not None and bool(np.any(mask > 0))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, mask = super().__getitem__(idx)
        # Remap: mask values 0-4 index into _S2_REMAP
        return img, _S2_REMAP[mask.clamp(0, 4)]

    def get_sample_weights(self, oversample_damage: float = 15.0) -> list[float]:
        """
        Compute per-tile sampling weights for Stage 2.

        * minor or major pixels present → ``oversample_damage`` (default 15×)
        * destroyed only                → 4×
        * no-damage only                → 1×
        """
        weights = []
        for stem in tqdm(self.stems, desc="  Scanning masks (stage2 weights)", leave=False):
            mask = cv2.imread(
                os.path.join(self.masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE
            )
            has_minor_major = bool(np.any((mask == 2) | (mask == 3)))
            has_destroyed   = bool(np.any(mask == 4))
            if has_minor_major:
                weights.append(float(oversample_damage))
            elif has_destroyed:
                weights.append(4.0)
            else:
                weights.append(1.0)
        return weights


# ─── DataLoader factories ─────────────────────────────────────────────────────

def build_dataloaders_stage1(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build Stage-1 (binary) train/val DataLoaders."""
    aug = build_augmentor(cfg.get("augmentation", {}))

    train_ds = XBDDatasetStage1(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "train"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = aug,
        in_channels = cfg["model"]["in_channels"],
    )
    val_ds = XBDDatasetStage1(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "val"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = None,
        in_channels = cfg["model"]["in_channels"],
    )

    oversample = cfg["training"].get("oversample_damage", 3)
    weights    = train_ds.get_sample_weights(oversample_damage=oversample)
    sampler    = WeightedRandomSampler(weights, len(weights), replacement=True)

    n_bldg = sum(1 for w in weights if w > 1.0)
    n_bg   = sum(1 for w in weights if w == 1.0)
    print(f"  Tile breakdown — with-building: {n_bldg}  background-only: {n_bg}")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        sampler     = sampler,
        num_workers = cfg["training"]["workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["training"]["workers"],
        pin_memory  = True,
    )
    return train_loader, val_loader


def build_dataloaders_stage2(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build Stage-2 (damage classification) train/val DataLoaders."""
    aug = build_augmentor(cfg.get("augmentation", {}))

    train_ds = XBDDatasetStage2(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "train"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = aug,
        in_channels = cfg["model"]["in_channels"],
    )
    val_ds = XBDDatasetStage2(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "val"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = None,
        in_channels = cfg["model"]["in_channels"],
    )

    oversample = cfg["training"].get("oversample_damage", 15)
    weights    = train_ds.get_sample_weights(oversample_damage=oversample)
    sampler    = WeightedRandomSampler(weights, len(weights), replacement=True)

    n_mm   = sum(1 for w in weights if w >= oversample)
    n_dest = sum(1 for w in weights if w == 4.0)
    n_nd   = sum(1 for w in weights if w == 1.0)
    print(f"  Tile breakdown — minor/major: {n_mm}  destroyed-only: {n_dest}  no-damage: {n_nd}")
    print(f"  Oversample factor for minor/major: {oversample}×")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        sampler     = sampler,
        num_workers = cfg["training"]["workers"],
        pin_memory  = True,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg["training"]["batch_size"],
        shuffle     = False,
        num_workers = cfg["training"]["workers"],
        pin_memory  = True,
    )
    return train_loader, val_loader
