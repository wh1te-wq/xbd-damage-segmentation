#!/usr/bin/env python
"""
Run inference with a trained DeepLabV3+ checkpoint on new images.

Two modes
---------
**Single pair**  (``--pre`` + ``--post``)::

    python scripts/predict.py --pre pre.png --post post.png --out pred.png

**Directory**  (``--input_dir``)::

    python scripts/predict.py --input_dir D:/dataset/test --out_dir D:/predictions

    The directory must contain ``pre/`` and ``post/`` sub-directories.
    Predictions are written to ``<out_dir>/masks/``.

**Evaluation** (``--input_dir`` + ``--evaluate``)::

    python scripts/predict.py --input_dir D:/dataset/test --evaluate

    Also computes metrics against ``<input_dir>/masks/`` ground-truth masks.

Post-processing (inspired by xview2_1st_place_solution)
--------------------------------------------------------
Optionally applies tiered confidence thresholds per damage class, as tuned
during the xView2 competition.  Controlled by ``cfg["inference"]["damage_thresholds"]``.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
import yaml
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm

from src.models.deeplabv3plus import build_model
from src.training.metrics import SegmentationMetrics
from src.training.trainer import Trainer

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]
_CLASS_NAMES   = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_CLASSES    = 5

# Colour palette for visualisation (BGR)
_PALETTE = {
    0: (50,  50,  50),   # background — dark grey
    1: (0,  200,   0),   # no-damage  — green
    2: (0,  200, 255),   # minor      — yellow-ish (in BGR: orange)
    3: (0, 100, 255),    # major      — orange-red
    4: (0,   0, 200),    # destroyed  — red
}


# ─── Inference helpers ────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, cfg: dict, device: torch.device) -> torch.nn.Module:
    """Load a model from a checkpoint file."""
    model = build_model(cfg).to(device)
    Trainer.load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    return model


def _to_tensor(img_rgb: np.ndarray, in_channels_half: int = 3) -> torch.Tensor:
    """HxWx3 uint8 → normalised float tensor (1, 3, H, W)."""
    mean = _IMAGENET_MEAN
    std  = _IMAGENET_STD
    t = torch.from_numpy(img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return transforms.Normalize(mean=mean, std=std)(t).unsqueeze(0)


def predict_pair(
    model:      torch.nn.Module,
    pre_path:   str,
    post_path:  str,
    img_size:   int,
    in_channels: int,
    device:     torch.device,
    thresholds: list[float] | None = None,
) -> np.ndarray:
    """
    Run inference on a single pre/post image pair.

    Returns
    -------
    np.ndarray  (H, W) uint8 — class mask at the original resolution.
    """
    def _load(p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(p)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    post = _load(post_path)
    orig_h, orig_w = post.shape[:2]
    post_r = cv2.resize(post, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    if in_channels == 6:
        pre   = _load(pre_path)
        pre_r = cv2.resize(pre, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        combined = np.concatenate([pre_r, post_r], axis=2)   # HxWx6
        mean = _IMAGENET_MEAN * 2
        std  = _IMAGENET_STD  * 2
        t = torch.from_numpy(combined.transpose(2, 0, 1).astype(np.float32) / 255.0)
        tensor = transforms.Normalize(mean=mean, std=std)(t).unsqueeze(0).to(device)
    else:
        tensor = _to_tensor(post_r).to(device)

    with torch.no_grad():
        logits = model(tensor)                      # (1, C, H, W)
        probs  = torch.softmax(logits, dim=1)[0]    # (C, H, W)

    # Tiered threshold post-processing (xview2_1st_place technique)
    if thresholds is not None:
        pred = torch.zeros(img_size, img_size, dtype=torch.long)
        for cls_idx, thresh in enumerate(thresholds, start=1):
            mask = probs[cls_idx] >= thresh
            pred[mask] = cls_idx
    else:
        pred = probs.argmax(dim=0)

    pred_np = pred.cpu().numpy().astype(np.uint8)
    return cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def colorise(mask: np.ndarray) -> np.ndarray:
    """Map a class mask to a BGR colour image for visualisation."""
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, colour in _PALETTE.items():
        canvas[mask == cls_id] = colour
    return canvas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict, args: argparse.Namespace) -> None:
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint  = args.checkpoint or cfg["inference"]["checkpoint"]
    img_size    = cfg["inference"]["img_size"]
    in_channels = cfg["model"]["in_channels"]
    thresholds  = cfg["inference"].get("damage_thresholds") if args.use_thresholds else None

    print(f"Device    : {device}")
    print(f"Checkpoint: {checkpoint}")

    model = load_model(checkpoint, cfg, device)

    # ── Single-pair mode ──────────────────────────────────────────────────
    if args.pre and args.post:
        pred = predict_pair(model, args.pre, args.post, img_size, in_channels, device, thresholds)
        out  = args.out or "prediction.png"
        cv2.imwrite(out, pred)
        cv2.imwrite(out.replace(".png", "_colour.png"), colorise(pred))
        print(f"Saved: {out}")
        return

    # ── Directory mode ────────────────────────────────────────────────────
    input_dir = args.input_dir or cfg["inference"].get("input_dir", "")
    out_dir   = args.out_dir  or cfg["inference"].get("out_dir", "predictions")

    pre_dir  = os.path.join(input_dir, "pre")
    post_dir = os.path.join(input_dir, "post")
    mask_dir = os.path.join(input_dir, "masks")   # for evaluation

    pred_dir   = os.path.join(out_dir, "masks")
    colour_dir = os.path.join(out_dir, "colour")
    os.makedirs(pred_dir,   exist_ok=True)
    os.makedirs(colour_dir, exist_ok=True)

    stems = sorted(Path(f).stem for f in os.listdir(post_dir) if f.endswith(".png"))
    print(f"Found {len(stems)} tiles in {input_dir}")

    metrics = SegmentationMetrics(NUM_CLASSES, _CLASS_NAMES) if args.evaluate else None

    for stem in tqdm(stems, desc="Predicting"):
        pre_path  = os.path.join(pre_dir,  stem + ".png")
        post_path = os.path.join(post_dir, stem + ".png")

        pred = predict_pair(model, pre_path, post_path, img_size, in_channels, device, thresholds)
        cv2.imwrite(os.path.join(pred_dir,   stem + ".png"), pred)
        cv2.imwrite(os.path.join(colour_dir, stem + ".png"), colorise(pred))

        if metrics:
            gt_path = os.path.join(mask_dir, stem + ".png")
            if os.path.exists(gt_path):
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                metrics.update(
                    torch.from_numpy(pred).long(),
                    torch.from_numpy(gt).long(),
                )

    print(f"\nPredictions saved to: {out_dir}")

    if metrics:
        print("\n=== Evaluation Results ===")
        print(str(metrics))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xBD inference")
    p.add_argument("--config",         default="configs/default.yaml")
    p.add_argument("--checkpoint",     default=None, help="Override cfg checkpoint path")
    # Single-pair mode
    p.add_argument("--pre",            default=None, help="Pre-disaster PNG path")
    p.add_argument("--post",           default=None, help="Post-disaster PNG path")
    p.add_argument("--out",            default=None, help="Output PNG path (single mode)")
    # Directory mode
    p.add_argument("--input_dir",      default=None, help="Directory with pre/ post/ sub-dirs")
    p.add_argument("--out_dir",        default=None, help="Output directory for masks")
    p.add_argument("--evaluate",       action="store_true",
                   help="Compute metrics against ground-truth masks in input_dir/masks/")
    p.add_argument("--use_thresholds", action="store_true",
                   help="Apply tiered per-class confidence thresholds (xview2_1st_place)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args)
