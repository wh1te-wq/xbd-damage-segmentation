"""
PyTorch Dataset and DataLoader for the preprocessed xBD damage segmentation dataset.

Expected directory layout (output of ``scripts/preprocess.py``)::

    <split_root>/
        pre/    *.png   — pre-disaster  RGB uint8 images (globally normalised)
        post/   *.png   — post-disaster RGB uint8 images (globally normalised)
        masks/  *.png   — single-channel uint8 class masks  (values 0–4)

Input tensor
------------
* ``in_channels=6`` (default): pre + post concatenated as [R_pre G_pre B_pre
  R_post G_post B_post] — recommended for change-detection semantics.
* ``in_channels=3``: post-disaster image only.

Normalisation
-------------
ImageNet mean/std is applied to each 3-channel block after converting to float.
The images were already globally histogram-stretched to uint8 during preprocessing,
so ImageNet normalisation is a reasonable final step.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

from src.utils.augmentations import Augmentor, build_augmentor

# ImageNet normalisation — applied per 3-channel block
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

NUM_CLASSES = 5


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XBDDataset(Dataset):
    """
    xBD semantic segmentation dataset.

    Parameters
    ----------
    split_root : str
        Directory containing ``pre/``, ``post/``, and ``masks/`` sub-directories.
    img_size : int
        Both spatial dimensions are resized to this value.
    augmentor : Augmentor or None
        Augmentor instance applied during training.  Pass ``None`` for val/test.
    in_channels : int
        6 = pre+post (recommended) | 3 = post only.
    """

    def __init__(
        self,
        split_root: str,
        img_size: int = 512,
        augmentor: Augmentor | None = None,
        in_channels: int = 6,
    ):
        self.pre_dir   = os.path.join(split_root, "pre")
        self.post_dir  = os.path.join(split_root, "post")
        self.masks_dir = os.path.join(split_root, "masks")
        self.img_size  = img_size
        self.augmentor = augmentor
        self.in_channels = in_channels

        self.stems = sorted(
            Path(f).stem
            for f in os.listdir(self.post_dir)
            if f.endswith(".png")
        )
        if not self.stems:
            raise RuntimeError(f"No PNG files found in {self.post_dir}")

        # Normalisation: duplicate ImageNet stats for 6-channel input
        mean = _IMAGENET_MEAN * (2 if in_channels == 6 else 1)
        std  = _IMAGENET_STD  * (2 if in_channels == 6 else 1)
        self._normalize = transforms.Normalize(mean=mean, std=std)

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]

        post = self._load_rgb(os.path.join(self.post_dir,  stem + ".png"))
        mask = cv2.imread(os.path.join(self.masks_dir, stem + ".png"), cv2.IMREAD_GRAYSCALE)
        pre  = self._load_rgb(os.path.join(self.pre_dir, stem + ".png")) \
               if self.in_channels == 6 else None

        # Resize
        if post.shape[0] != self.img_size or post.shape[1] != self.img_size:
            sz = (self.img_size, self.img_size)
            post = cv2.resize(post, sz, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, sz, interpolation=cv2.INTER_NEAREST)
            if pre is not None:
                pre = cv2.resize(pre, sz, interpolation=cv2.INTER_LINEAR)

        # Augmentation (training only)
        if self.augmentor is not None and pre is not None:
            pre, post, mask = self.augmentor(pre, post, mask)
        elif self.augmentor is not None:
            # 3-channel mode: augment post + mask only (pass dummy pre)
            _dummy = np.zeros_like(post)
            _dummy, post, mask = self.augmentor(_dummy, post, mask)

        # Build image tensor
        if pre is not None:
            combined   = np.concatenate([pre, post], axis=2)   # HxWx6
            img_tensor = torch.from_numpy(
                combined.transpose(2, 0, 1).astype(np.float32) / 255.0
            )
        else:
            img_tensor = torch.from_numpy(
                post.transpose(2, 0, 1).astype(np.float32) / 255.0
            )

        img_tensor  = self._normalize(img_tensor)
        mask_tensor = torch.from_numpy(mask).long().clamp(0, NUM_CLASSES - 1)

        return img_tensor, mask_tensor

    # ── Damage-aware sampling ─────────────────────────────────────────────────

    def get_sample_weights(self, oversample_damage: float = 10.0) -> list[float]:
        """
        Compute per-tile sampling weights so that tiles containing minor-damage
        (class 2) or major-damage (class 3) are oversampled during training.

        Weight scheme
        -------------
        - Contains minor or major damage  → ``oversample_damage``  (default 10×)
        - Contains only destroyed         → 3×
        - No damage at all (bg only)      → 1×
        - Has no-damage buildings only    → 1×

        Parameters
        ----------
        oversample_damage : float
            Sampling multiplier for tiles with minor/major damage pixels.

        Returns
        -------
        list[float] of length ``len(self)``.
        """
        weights = []
        for stem in tqdm(self.stems, desc="  Scanning masks for damage weights", leave=False):
            mask = cv2.imread(
                os.path.join(self.masks_dir, stem + ".png"),
                cv2.IMREAD_GRAYSCALE,
            )
            has_minor_major = bool(np.any((mask == 2) | (mask == 3)))
            has_destroyed   = bool(np.any(mask == 4))

            if has_minor_major:
                weights.append(float(oversample_damage))
            elif has_destroyed:
                weights.append(3.0)
            else:
                weights.append(1.0)

        return weights

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _load_rgb(path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # HxWx3 uint8


# ─── DataLoader factory ───────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation :class:`DataLoader` objects from a config dict.

    Parameters
    ----------
    cfg : dict
        Full config dictionary (e.g. loaded from ``configs/default.yaml``).

    Returns
    -------
    train_loader, val_loader
    """
    aug = build_augmentor(cfg.get("augmentation", {}))

    train_ds = XBDDataset(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "train"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = aug,
        in_channels = cfg["model"]["in_channels"],
    )
    val_ds = XBDDataset(
        split_root  = os.path.join(cfg["data"]["dataset_dir"], "val"),
        img_size    = cfg["training"]["img_size"],
        augmentor   = None,
        in_channels = cfg["model"]["in_channels"],
    )

    # Damage-aware WeightedRandomSampler: oversample tiles with minor/major damage
    oversample = cfg["training"].get("oversample_damage", 10)
    sample_weights = train_ds.get_sample_weights(oversample_damage=oversample)
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )
    n_minor_major = sum(1 for w in sample_weights if w >= oversample)
    n_destroyed   = sum(1 for w in sample_weights if w == 3.0)
    n_easy        = sum(1 for w in sample_weights if w == 1.0)
    print(f"  Tile breakdown — minor/major: {n_minor_major}  "
          f"destroyed-only: {n_destroyed}  no-damage/bg: {n_easy}")
    print(f"  Oversample factor for minor/major tiles: {oversample}×")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg["training"]["batch_size"],
        sampler     = sampler,          # replaces shuffle=True
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
