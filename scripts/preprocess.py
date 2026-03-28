#!/usr/bin/env python
"""
Preprocess xBD GeoTIFF dataset into PNG images + segmentation masks.

What this script does
---------------------
1. Collect all tile pairs from ``tier1`` and ``tier3`` (training splits).
2. Compute **global** normalization statistics separately for pre- and
   post-disaster images (sampling ``norm_sample_size`` images per phase).
3. Split training tiles into train / val at the **disaster-event level** to
   prevent spatial-correlation data leakage.
4. Convert GeoTIFFs → uint8 PNGs using the global stats.
5. Rasterise label JSON polygons → single-channel class masks.
6. Process ``test`` and ``hold`` splits into separate directories.

Output layout
-------------
::

    <dataset_dir>/
        norm_stats.json
        train/  pre/ post/ masks/
        val/    pre/ post/ masks/
        test/   pre/ post/ masks/
        hold/   pre/ post/ masks/

Usage
-----
::

    python scripts/preprocess.py
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --data_root D:/data/xbd --dataset_dir D:/data/processed
"""

import argparse
import json
import os
import sys

# Allow importing from project root regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import yaml
from tqdm import tqdm

from src.preprocessing.masks import build_mask_from_label
from src.preprocessing.normalize import apply_global_norm, compute_and_save_stats, load_stats
from src.preprocessing.splits import collect_eval_pairs, collect_pairs, event_level_split


# ─── Processing helpers ───────────────────────────────────────────────────────

def process_split(
    pairs:       list[tuple[str, str]],
    data_root:   str,
    out_dir:     str,
    stats:       dict,
    split_label: str = "",
) -> int:
    """
    Save pre/post PNG images and class-mask PNGs for a list of tile pairs.

    Parameters
    ----------
    pairs      : list of (source_split_name, base_id)
    data_root  : raw xBD GeoTIFF root directory
    out_dir    : destination directory (will contain pre/, post/, masks/)
    stats      : normalization statistics dict with "pre" and "post" keys
    split_label: label shown in the progress bar

    Returns
    -------
    int — number of successfully saved tiles
    """
    pre_dir   = os.path.join(out_dir, "pre")
    post_dir  = os.path.join(out_dir, "post")
    masks_dir = os.path.join(out_dir, "masks")
    for d in (pre_dir, post_dir, masks_dir):
        os.makedirs(d, exist_ok=True)

    saved = 0
    for src_split, base in tqdm(pairs, desc=f"  {split_label}", leave=False):
        pre_tif  = os.path.join(data_root, src_split, "images", base + "_pre_disaster.tif")
        post_tif = os.path.join(data_root, src_split, "images", base + "_post_disaster.tif")
        lbl_json = os.path.join(data_root, src_split, "labels", base + "_post_disaster.json")

        if not all(os.path.exists(p) for p in (pre_tif, post_tif, lbl_json)):
            continue

        # Images — use phase-specific normalisation stats
        pre_img  = apply_global_norm(pre_tif,  stats["pre"])
        post_img = apply_global_norm(post_tif, stats["post"])

        cv2.imwrite(os.path.join(pre_dir,  base + ".png"), cv2.cvtColor(pre_img,  cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(post_dir, base + ".png"), cv2.cvtColor(post_img, cv2.COLOR_RGB2BGR))

        # Mask
        import rasterio
        with rasterio.open(post_tif) as src:
            h, w = src.height, src.width
        mask = build_mask_from_label(lbl_json, height=h, width=w)
        cv2.imwrite(os.path.join(masks_dir, base + ".png"), mask)

        saved += 1

    return saved


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(cfg: dict) -> None:
    data_root   = cfg["data"]["data_root"]
    dataset_dir = cfg["data"]["dataset_dir"]
    tiers       = cfg["data"].get("tiers", ["tier1", "tier3"])
    val_ratio   = cfg["data"].get("val_ratio", 0.1)
    n_sample    = cfg["data"].get("norm_sample_size", 500)
    seed        = cfg["training"].get("seed", 42)

    os.makedirs(dataset_dir, exist_ok=True)

    # ── 1. Collect training tile pairs ────────────────────────────────────
    all_pairs = collect_pairs(data_root, tiers)
    print(f"Total training tile pairs: {len(all_pairs)}")

    # ── 2. Global normalization stats (pre and post separately) ───────────
    stats_path = os.path.join(dataset_dir, "norm_stats.json")
    if os.path.exists(stats_path):
        print(f"Loading existing norm stats: {stats_path}")
        stats = load_stats(stats_path)
    else:
        print(f"Computing norm stats (sampling {n_sample} images per phase)...")
        stats = compute_and_save_stats(all_pairs, data_root, stats_path, n_sample=n_sample)
        print(f"  pre  lo={[round(x,1) for x in stats['pre']['lo']]}  "
              f"hi={[round(x,1) for x in stats['pre']['hi']]}")
        print(f"  post lo={[round(x,1) for x in stats['post']['lo']]}  "
              f"hi={[round(x,1) for x in stats['post']['hi']]}")

    # ── 3. Event-level train / val split ──────────────────────────────────
    train_pairs, val_pairs, val_events = event_level_split(all_pairs, val_ratio, seed)
    print(f"\nEvent-level split:")
    print(f"  Train tiles : {len(train_pairs)}")
    print(f"  Val   tiles : {len(val_pairs)}  (events: {val_events})")

    # ── 4. Process train ──────────────────────────────────────────────────
    print("\n=== TRAIN ===")
    n = process_split(train_pairs, data_root, os.path.join(dataset_dir, "train"), stats, "train")
    print(f"  Saved {n} tiles.")

    # ── 5. Process val ────────────────────────────────────────────────────
    print("\n=== VAL ===")
    n = process_split(val_pairs, data_root, os.path.join(dataset_dir, "val"), stats, "val")
    print(f"  Saved {n} tiles.")

    # ── 6. Process test and hold (kept separate) ──────────────────────────
    for split_name in ("test", "hold"):
        pairs = collect_eval_pairs(data_root, split_name)
        if not pairs:
            continue
        print(f"\n=== {split_name.upper()} ({len(pairs)} pairs) ===")
        n = process_split(pairs, data_root, os.path.join(dataset_dir, split_name), stats, split_name)
        print(f"  Saved {n} tiles.")

    print(f"\nPreprocessing complete. Dataset at: {dataset_dir}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="xBD preprocessing pipeline")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--data_root",   default=None, help="Override cfg.data.data_root")
    p.add_argument("--dataset_dir", default=None, help="Override cfg.data.dataset_dir")
    p.add_argument("--val_ratio",   type=float, default=None)
    p.add_argument("--seed",        type=int,   default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Command-line overrides
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.dataset_dir:
        cfg["data"]["dataset_dir"] = args.dataset_dir
    if args.val_ratio is not None:
        cfg["data"]["val_ratio"] = args.val_ratio
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    main(cfg)
