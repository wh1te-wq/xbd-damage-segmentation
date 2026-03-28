"""
Segmentation evaluation metrics.

Accumulates a confusion matrix over batches then computes:
    - Per-class IoU  (Intersection over Union / Jaccard index)
    - Mean IoU (mIoU) — NaN classes (never predicted and never present) are excluded
    - Overall pixel accuracy
    - Per-class F1 score (Dice coefficient)
    - Mean F1

All metric computations are done in NumPy after moving tensors to CPU so the
GPU memory is not held during evaluation.
"""

import numpy as np
import torch


class SegmentationMetrics:
    """
    Online confusion-matrix accumulator.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes.
    class_names : list[str] or None
        Optional class labels for pretty-printing.
    """

    def __init__(
        self,
        num_classes: int,
        class_names: list[str] | None = None,
    ):
        self.num_classes  = num_classes
        self.class_names  = class_names or [str(i) for i in range(num_classes)]
        self.conf_mat     = np.zeros((num_classes, num_classes), dtype=np.int64)

    # ── Accumulation ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Zero the confusion matrix."""
        self.conf_mat[:] = 0

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """
        Update the confusion matrix with one batch.

        Parameters
        ----------
        pred   : torch.Tensor  (N, H, W)  int64 — argmax predictions.
        target : torch.Tensor  (N, H, W)  int64 — ground-truth labels.
        """
        pred   = pred.cpu().numpy().ravel()
        target = target.cpu().numpy().ravel()
        valid  = (target >= 0) & (target < self.num_classes)
        idx    = target[valid] * self.num_classes + pred[valid]
        self.conf_mat += (
            np.bincount(idx, minlength=self.num_classes ** 2)
            .reshape(self.num_classes, self.num_classes)
        )

    # ── Per-class metrics ─────────────────────────────────────────────────────

    def iou_per_class(self) -> np.ndarray:
        """
        Return per-class IoU as a float32 array of shape ``(num_classes,)``.
        Classes that never appear in predictions *or* targets get ``NaN``.
        """
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        denom = tp + fp + fn
        return np.where(denom > 0, tp / (denom + 1e-7), np.nan)

    def f1_per_class(self) -> np.ndarray:
        """
        Return per-class F1 (Dice) as a float32 array of shape ``(num_classes,)``.
        """
        tp = np.diag(self.conf_mat)
        fp = self.conf_mat.sum(axis=0) - tp
        fn = self.conf_mat.sum(axis=1) - tp
        denom = 2 * tp + fp + fn
        return np.where(denom > 0, 2 * tp / (denom + 1e-7), np.nan)

    # ── Aggregate metrics ─────────────────────────────────────────────────────

    def miou(self) -> float:
        """Mean IoU, NaN classes excluded."""
        return float(np.nanmean(self.iou_per_class()))

    def mean_f1(self) -> float:
        """Mean F1 / Dice, NaN classes excluded."""
        return float(np.nanmean(self.f1_per_class()))

    def pixel_accuracy(self) -> float:
        """Fraction of correctly classified pixels."""
        correct = np.diag(self.conf_mat).sum()
        total   = self.conf_mat.sum()
        return float(correct / (total + 1e-7))

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return all aggregate metrics as a plain dict."""
        iou = self.iou_per_class()
        f1  = self.f1_per_class()
        return {
            "miou":      self.miou(),
            "mean_f1":   self.mean_f1(),
            "pix_acc":   self.pixel_accuracy(),
            "iou":       {self.class_names[i]: float(iou[i]) for i in range(self.num_classes)},
            "f1":        {self.class_names[i]: float(f1[i])  for i in range(self.num_classes)},
        }

    def __str__(self) -> str:
        s   = self.summary()
        iou = s["iou"]
        f1  = s["f1"]
        lines = [
            f"  Pixel acc : {s['pix_acc']*100:.2f}%",
            f"  mIoU      : {s['miou']*100:.2f}%",
            f"  Mean F1   : {s['mean_f1']*100:.2f}%",
        ]
        for name in self.class_names:
            v_iou = iou[name]
            v_f1  = f1[name]
            if not np.isnan(v_iou):
                lines.append(f"    {name:15s}  IoU={v_iou*100:.1f}%  F1={v_f1*100:.1f}%")
        return "\n".join(lines)
