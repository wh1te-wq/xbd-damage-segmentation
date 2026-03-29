#!/usr/bin/env python
"""
Two-stage inference for xBD damage segmentation.

Pipeline
--------
1. **Stage 1** (ResNet-50 binary model) — decides *where* buildings are.
2. **Stage 2** (ResNet-50 damage model) — decides *how damaged* each building pixel is.
3. **Merge**: pixels predicted as background by Stage 1 become class 0;
   all other pixels take their Stage-2 damage label (shifted +1 to recover
   the original 5-class encoding).

Final output classes
--------------------
    0 — background
    1 — no-damage
    2 — minor-damage
    3 — major-damage
    4 — destroyed

Usage
-----
**Single pair**::

    python scripts/predict_twostage.py --pre pre.png --post post.png --out pred.png

**Directory**::

    python scripts/predict_twostage.py --input_dir D:/dataset/test --out_dir D:/predictions

**Evaluate** (requires ``<input_dir>/masks/`` ground-truth)::

    python scripts/predict_twostage.py --input_dir D:/dataset/test --evaluate
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

# Final 5-class names (original encoding)
_FINAL_CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]
NUM_FINAL_CLASSES  = 5

# Colour palette for visualisation (BGR)
_PALETTE = {
    0: (50,  50,  50),   # background — dark grey
    1: (0,  200,   0),   # no-damage  — green
    2: (0,  200, 255),   # minor      — orange (BGR)
    3: (0, 100, 255),    # major      — orange-red
    4: (0,   0, 200),    # destroyed  — red
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _make_tensor(pre: np.ndarray, post: np.ndarray, in_channels: int,
                 device: torch.device) -> torch.Tensor:
    """Build a normalised (1, C, H, W) tensor from RGB uint8 arrays."""
    if in_channels == 6:
        combined = np.concatenate([pre, post], axis=2)   # HxWx6
        mean = _IMAGENET_MEAN * 2
        std  = _IMAGENET_STD  * 2
    else:
        combined = post
        mean = _IMAGENET_MEAN
        std  = _IMAGENET_STD
    t = torch.from_numpy(combined.transpose(2, 0, 1).astype(np.float32) / 255.0)
    return transforms.Normalize(mean=mean, std=std)(t).unsqueeze(0).to(device)


def load_model_from_cfg(cfg_path: str, device: torch.device) -> torch.nn.Module:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg).to(device)
    ckpt_path = cfg["inference"]["checkpoint"]
    Trainer.load_checkpoint(ckpt_path, model, device=device)
    model.eval()
    return model, cfg


def colorise(mask: np.ndarray) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, colour in _PALETTE.items():
        canvas[mask == cls_id] = colour
    return canvas


# ─── Core inference ───────────────────────────────────────────────────────────

@torch.no_grad()
def predict_pair(
    model_s1:   torch.nn.Module,
    model_s2:   torch.nn.Module,
    pre_path:   str,
    post_path:  str,
    img_size:   int,
    in_channels: int,
    device:     torch.device,
) -> np.ndarray:
    """
    Run two-stage inference on a single pre/post pair.

    Returns
    -------
    np.ndarray (H, W) uint8 — 5-class mask at original resolution.
    """
    post = _load_rgb(post_path)
    orig_h, orig_w = post.shape[:2]
    post_r = cv2.resize(post, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    pre_r = None
    if in_channels == 6:
        pre   = _load_rgb(pre_path)
        pre_r = cv2.resize(pre, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    else:
        pre_r = post_r  # placeholder; only post will be used

    tensor = _make_tensor(pre_r, post_r, in_channels, device)

    # Stage 1: building mask (0=background, 1=building)
    logits_s1  = model_s1(tensor)                          # (1, 2, H, W)
    bldg_mask  = logits_s1.argmax(dim=1)[0]                # (H, W) — 0 or 1

    # Stage 2: damage prediction (0=no-dmg, 1=minor, 2=major, 3=destroyed)
    logits_s2  = model_s2(tensor)                          # (1, 4, H, W)
    damage_pred = logits_s2.argmax(dim=1)[0]               # (H, W) — 0-3

    # Merge: background stays 0; building pixels → damage label + 1
    final = torch.where(bldg_mask == 0, torch.zeros_like(damage_pred), damage_pred + 1)
    final_np = final.cpu().numpy().astype(np.uint8)

    return cv2.resize(final_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading Stage 1 model...")
    model_s1, cfg_s1 = load_model_from_cfg(args.config_s1, device)
    print(f"  {args.config_s1}  →  {cfg_s1['inference']['checkpoint']}")

    print("Loading Stage 2 model...")
    model_s2, cfg_s2 = load_model_from_cfg(args.config_s2, device)
    print(f"  {args.config_s2}  →  {cfg_s2['inference']['checkpoint']}")

    img_size    = cfg_s1["inference"]["img_size"]
    in_channels = cfg_s1["model"]["in_channels"]

    # ── Single-pair mode ──────────────────────────────────────────────────
    if args.pre and args.post:
        pred = predict_pair(model_s1, model_s2, args.pre, args.post,
                            img_size, in_channels, device)
        out  = args.out or "prediction_twostage.png"
        cv2.imwrite(out, pred)
        cv2.imwrite(out.replace(".png", "_colour.png"), colorise(pred))
        print(f"Saved: {out}")
        return

    # ── Directory mode ────────────────────────────────────────────────────
    input_dir = args.input_dir
    out_dir   = args.out_dir or "predictions_twostage"
    pre_dir   = os.path.join(input_dir, "pre")
    post_dir  = os.path.join(input_dir, "post")
    mask_dir  = os.path.join(input_dir, "masks")

    pred_dir   = os.path.join(out_dir, "masks")
    colour_dir = os.path.join(out_dir, "colour")
    os.makedirs(pred_dir,   exist_ok=True)
    os.makedirs(colour_dir, exist_ok=True)

    stems = sorted(Path(f).stem for f in os.listdir(post_dir) if f.endswith(".png"))
    print(f"\nFound {len(stems)} tiles in {input_dir}")

    metrics = SegmentationMetrics(NUM_FINAL_CLASSES, _FINAL_CLASS_NAMES) \
              if args.evaluate else None

    for stem in tqdm(stems, desc="Predicting (two-stage)"):
        pred = predict_pair(
            model_s1, model_s2,
            os.path.join(pre_dir,  stem + ".png"),
            os.path.join(post_dir, stem + ".png"),
            img_size, in_channels, device,
        )
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
        print("\n=== Two-Stage Evaluation Results ===")
        print(str(metrics))


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage xBD inference")
    p.add_argument("--config_s1",  default="configs/stage1.yaml",
                   help="Stage-1 config (binary building segmentation)")
    p.add_argument("--config_s2",  default="configs/stage2.yaml",
                   help="Stage-2 config (damage classification)")
    # Single-pair mode
    p.add_argument("--pre",        default=None, help="Pre-disaster PNG")
    p.add_argument("--post",       default=None, help="Post-disaster PNG")
    p.add_argument("--out",        default=None, help="Output PNG (single mode)")
    # Directory mode
    p.add_argument("--input_dir",  default=None, help="Directory with pre/ post/ sub-dirs")
    p.add_argument("--out_dir",    default=None, help="Output directory")
    p.add_argument("--evaluate",   action="store_true",
                   help="Compute metrics against ground-truth masks")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
