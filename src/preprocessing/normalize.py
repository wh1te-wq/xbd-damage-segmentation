"""
Global normalization statistics for xBD satellite imagery.

Motivation
----------
Per-image percentile stretching produces inconsistent brightness across samples,
which makes the model's job harder. Instead we compute *global* 2nd / 98th
percentile values from a representative sample of the training set and apply the
same stretch to every image.

Pre-disaster and post-disaster images are treated separately because they are
acquired on different dates and may have different lighting / sensor conditions.
The resulting statistics are saved to ``norm_stats.json`` in the dataset root so
that the same values can be reproduced at inference time.
"""

import json
import os
import random

import numpy as np
import rasterio
from tqdm import tqdm


# ─── Statistics computation ───────────────────────────────────────────────────

def compute_global_stats(
    tif_paths: list[str],
    n_sample: int = 500,
    desc: str = "norm stats",
) -> dict:
    """
    Sample up to *n_sample* GeoTIFFs and compute per-channel 2nd / 98th
    percentile across all sampled pixels.

    Parameters
    ----------
    tif_paths : list[str]
        Paths to GeoTIFF files (all from the same phase, i.e. all pre or all post).
    n_sample : int
        Maximum number of files to sample.
    desc : str
        Label shown in the tqdm progress bar.

    Returns
    -------
    dict
        ``{"lo": [r, g, b], "hi": [r, g, b]}`` — one float per channel.
    """
    sampled = random.sample(tif_paths, min(n_sample, len(tif_paths)))
    per_channel: list[list[np.ndarray]] = [[], [], []]

    for path in tqdm(sampled, desc=f"  Computing {desc}"):
        with rasterio.open(path) as src:
            img = src.read().astype(np.float32)   # (bands, H, W)
        for c in range(min(3, img.shape[0])):
            flat = img[c].ravel()
            # Subsample each image to bound memory usage
            step = max(1, len(flat) // 4096)
            per_channel[c].append(flat[::step])

    lo, hi = [], []
    for c in range(3):
        all_vals = np.concatenate(per_channel[c])
        lo.append(float(np.percentile(all_vals, 2)))
        hi.append(float(np.percentile(all_vals, 98)))

    return {"lo": lo, "hi": hi}


def compute_and_save_stats(
    train_pairs: list[tuple[str, str]],
    data_root: str,
    out_path: str,
    n_sample: int = 500,
) -> dict:
    """
    Compute separate normalization statistics for pre- and post-disaster images
    from the training set, then save them to *out_path* as JSON.

    Parameters
    ----------
    train_pairs : list of (split_name, base_id)
        Training tile identifiers, e.g. ``[("tier1", "guatemala-volcano_00000000"), ...]``.
    data_root : str
        Root directory of raw xBD GeoTIFFs.
    out_path : str
        Destination JSON file (e.g. ``dataset/norm_stats.json``).
    n_sample : int
        Images sampled **per phase** (total sampled = 2 × n_sample).

    Returns
    -------
    dict
        ``{"pre": {lo, hi}, "post": {lo, hi}}``
    """
    half = n_sample // 2

    pre_tifs = [
        os.path.join(data_root, sp, "images", base + "_pre_disaster.tif")
        for sp, base in train_pairs
    ]
    post_tifs = [
        os.path.join(data_root, sp, "images", base + "_post_disaster.tif")
        for sp, base in train_pairs
    ]

    pre_stats  = compute_global_stats(pre_tifs,  n_sample=half, desc="pre  phase")
    post_stats = compute_global_stats(post_tifs, n_sample=half, desc="post phase")

    stats = {"pre": pre_stats, "post": post_stats}

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    return stats


# ─── Application ──────────────────────────────────────────────────────────────

def apply_global_norm(tif_path: str, phase_stats: dict) -> np.ndarray:
    """
    Read a GeoTIFF and linearly stretch each channel to ``[0, 255]`` using
    pre-computed per-phase statistics.

    Parameters
    ----------
    tif_path : str
        Path to the GeoTIFF.
    phase_stats : dict
        Statistics for this specific phase: ``{"lo": [r,g,b], "hi": [r,g,b]}``.
        Pass ``norm_stats["pre"]`` for pre-disaster images and
        ``norm_stats["post"]`` for post-disaster images.

    Returns
    -------
    np.ndarray
        ``uint8`` array of shape ``(H, W, 3)``.
    """
    with rasterio.open(tif_path) as src:
        img = src.read().astype(np.float32)   # (bands, H, W)

    n_bands = img.shape[0]
    result  = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.uint8)

    for c in range(3):
        band   = img[c] if c < n_bands else np.zeros_like(img[0])
        lo, hi = phase_stats["lo"][c], phase_stats["hi"][c]
        if hi > lo:
            stretched = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
        else:
            stretched = np.zeros_like(band)
        result[:, :, c] = stretched.astype(np.uint8)

    return result   # HxWx3


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def load_stats(stats_path: str) -> dict:
    """Load normalization statistics from a JSON file."""
    with open(stats_path) as f:
        return json.load(f)
