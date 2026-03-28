"""
Loss functions for xBD damage segmentation.

Combo loss (CE + Dice + Focal) inspired by xview2_1st_place_solution (losses.py).

Why combine multiple losses?
-----------------------------
* **CrossEntropyLoss** (with class weights): standard pixel-wise classification;
  class weights address the severe background/building imbalance.
* **Soft DiceLoss**: directly optimises the overlap metric (IoU-like), useful
  when pixel counts per class are highly unbalanced.
* **FocalLoss**: down-weights easy negatives so the model focuses on hard
  boundary pixels and small damaged regions.

Class weights
-------------
Computed as inverse-frequency from a random subset of training masks.
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# ─── Individual losses ────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.

    Computes the mean Dice coefficient across all classes (including background)
    and returns ``1 - mean_dice``.

    Parameters
    ----------
    num_classes : int
    smooth      : float — Laplace smoothing to avoid division by zero.
    ignore_index: int   — class index to exclude from the loss.
    """

    def __init__(self, num_classes: int, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.num_classes  = num_classes
        self.smooth       = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits:  (N, C, H, W)
        # targets: (N, H, W)  int64
        valid = targets != self.ignore_index
        t     = targets.clone()
        t[~valid] = 0   # temporary; excluded from sum via mask

        probs  = F.softmax(logits, dim=1)           # (N, C, H, W)
        t_1hot = F.one_hot(t, self.num_classes)     # (N, H, W, C)
        t_1hot = t_1hot.permute(0, 3, 1, 2).float() # (N, C, H, W)

        # Mask out ignored pixels
        vm = valid.unsqueeze(1).float()
        probs  = probs  * vm
        t_1hot = t_1hot * vm

        inter = (probs * t_1hot).sum(dim=(0, 2, 3))          # (C,)
        union = probs.sum(dim=(0, 2, 3)) + t_1hot.sum(dim=(0, 2, 3))   # (C,)
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """
    Multi-class focal loss.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Parameters
    ----------
    gamma        : float — focusing parameter (default 2.0).
    ignore_index : int   — passed to F.cross_entropy.
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.gamma        = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, reduction="none")
        pt   = torch.exp(-ce)
        loss = (1.0 - pt) ** self.gamma * ce
        # Mask out ignored pixels
        mask = targets != self.ignore_index
        return loss[mask].mean() if mask.any() else loss.mean()


# ─── Combo loss ───────────────────────────────────────────────────────────────

class ComboLoss(nn.Module):
    """
    Weighted combination of CrossEntropy + Dice + Focal losses.

    ``total = ce_weight * CE + dice_weight * Dice + focal_weight * Focal``

    Parameters
    ----------
    num_classes   : int
    loss_cfg      : dict — ``cfg["loss"]`` sub-dict from the config file.
    class_weights : torch.Tensor or None — passed to CrossEntropyLoss.
    """

    def __init__(
        self,
        num_classes: int,
        loss_cfg: dict,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce    = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
        self.dice  = DiceLoss(num_classes, ignore_index=255)
        self.focal = FocalLoss(gamma=loss_cfg.get("focal_gamma", 2.0), ignore_index=255)

        self.ce_w    = float(loss_cfg.get("ce_weight",    1.0))
        self.dice_w  = float(loss_cfg.get("dice_weight",  0.5))
        self.focal_w = float(loss_cfg.get("focal_weight", 0.5))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.ce_w    * self.ce(logits, targets)    +
            self.dice_w  * self.dice(logits, targets)  +
            self.focal_w * self.focal(logits, targets)
        )


# ─── Class weight estimation ──────────────────────────────────────────────────

def compute_class_weights(
    dataset,
    num_classes: int,
    max_samples: int = 500,
) -> torch.Tensor:
    """
    Estimate inverse-frequency class weights from a random subset of masks.

    Sampling a subset (rather than the full dataset) is sufficient for a good
    estimate and avoids loading tens of thousands of images.

    Parameters
    ----------
    dataset     : torch.utils.data.Dataset
        Must return ``(img_tensor, mask_tensor)`` pairs.
    num_classes : int
    max_samples : int

    Returns
    -------
    torch.Tensor of shape ``(num_classes,)`` with float32 weights.
    """
    counts  = np.zeros(num_classes, dtype=np.float64)
    indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))

    for i in tqdm(indices, desc="  Estimating class weights", leave=False):
        _, mask = dataset[i]
        for c in range(num_classes):
            counts[c] += (mask == c).sum().item()

    counts  = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.sum() * num_classes   # normalise so sum == num_classes

    return torch.tensor(weights, dtype=torch.float32)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_criterion(
    cfg: dict,
    dataset=None,
    device: torch.device = torch.device("cpu"),
) -> nn.Module:
    """
    Build the loss function from a config dict.

    Parameters
    ----------
    cfg     : dict  — full config dictionary.
    dataset : XBDDataset or None
        Training dataset used for class weight estimation.
        If ``None`` or ``cfg["training"]["use_class_weights"]`` is False,
        no class weighting is applied.
    device  : torch.device

    Returns
    -------
    ComboLoss instance.
    """
    class_weights = None
    if dataset is not None and cfg["training"].get("use_class_weights", True):
        class_weights = compute_class_weights(
            dataset, cfg["model"]["num_classes"]
        ).to(device)

    return ComboLoss(
        num_classes   = cfg["model"]["num_classes"],
        loss_cfg      = cfg.get("loss", {}),
        class_weights = class_weights,
    )
