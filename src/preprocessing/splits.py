"""
Dataset splitting utilities for xBD.

Key design: **event-level split** (inspired by competition best practices).

A naive tile-level shuffle leaks information because tiles from the same disaster
event (e.g. ``guatemala-volcano``) cover adjacent geographic areas, share the
same damage patterns, and have very similar visual statistics.  Splitting at the
*event* level guarantees that every tile of a given event lands in exactly one
partition, eliminating spatial correlation leakage.

Event name extraction
---------------------
xBD filenames follow the pattern::

    <disaster-name>_<8-digit-tile-index>_<phase>_disaster.<ext>

e.g. ``guatemala-volcano_00000000_post_disaster.tif``

The base ID (without phase suffix) is ``guatemala-volcano_00000000``.
The event name is obtained by ``base.rsplit("_", 1)[0]`` → ``"guatemala-volcano"``.
"""

import glob
import os
import random
from collections import defaultdict
from pathlib import Path


# ─── Collection ───────────────────────────────────────────────────────────────

def collect_pairs(
    data_root: str,
    tier_splits: list[str] = ("tier1", "tier3"),
) -> list[tuple[str, str]]:
    """
    Collect all training tile pairs from the given tier splits.

    Parameters
    ----------
    data_root : str
        Root directory of raw xBD GeoTIFFs.
    tier_splits : list[str]
        Names of the tier sub-directories to include (default: tier1 + tier3).

    Returns
    -------
    list of (split_name, base_id)
        e.g. ``[("tier1", "guatemala-volcano_00000000"), ...]``
    """
    pairs: list[tuple[str, str]] = []

    for split in tier_splits:
        img_dir = os.path.join(data_root, split, "images")
        if not os.path.isdir(img_dir):
            print(f"[splits] Warning: {img_dir} not found, skipping.")
            continue
        for tif in sorted(glob.glob(os.path.join(img_dir, "*_post_disaster.tif"))):
            base = Path(tif).stem.replace("_post_disaster", "")
            pairs.append((split, base))

    return pairs


def collect_eval_pairs(
    data_root: str,
    split_name: str,
) -> list[tuple[str, str]]:
    """
    Collect all tile pairs from a fixed evaluation split (``test`` or ``hold``).

    Parameters
    ----------
    data_root : str
        Root directory of raw xBD GeoTIFFs.
    split_name : str
        Sub-directory name: ``"test"`` or ``"hold"``.

    Returns
    -------
    list of (split_name, base_id)
    """
    img_dir = os.path.join(data_root, split_name, "images")
    if not os.path.isdir(img_dir):
        print(f"[splits] Warning: {img_dir} not found.")
        return []
    pairs = []
    for tif in sorted(glob.glob(os.path.join(img_dir, "*_post_disaster.tif"))):
        base = Path(tif).stem.replace("_post_disaster", "")
        pairs.append((split_name, base))
    return pairs


# ─── Event-level split ────────────────────────────────────────────────────────

def event_level_split(
    pairs: list[tuple[str, str]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Split *pairs* into train / val by grouping at the disaster-event level.

    Every tile belonging to the same event is assigned to the same partition,
    preventing spatial-correlation data leakage between train and val sets.

    Parameters
    ----------
    pairs : list of (split_name, base_id)
        All available training tile pairs.
    val_ratio : float
        Approximate fraction of tiles to allocate to validation.
        The actual ratio may differ slightly because we split whole events.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_pairs : list of (split_name, base_id)
    val_pairs   : list of (split_name, base_id)
    val_events  : list[str]  — disaster event names assigned to validation
    """
    rng = random.Random(seed)

    # Group tiles by disaster event name
    event_to_pairs: dict[str, list] = defaultdict(list)
    for split, base in pairs:
        event = base.rsplit("_", 1)[0]   # "guatemala-volcano_00000000" → "guatemala-volcano"
        event_to_pairs[event].append((split, base))

    events = sorted(event_to_pairs.keys())
    rng.shuffle(events)

    # Greedily assign events to val until we reach the target tile count
    val_tile_target = max(1, int(len(pairs) * val_ratio))
    val_events: list[str] = []
    train_events: list[str] = []
    val_count = 0

    for ev in events:
        if val_count < val_tile_target:
            val_events.append(ev)
            val_count += len(event_to_pairs[ev])
        else:
            train_events.append(ev)

    train_pairs = [p for ev in train_events for p in event_to_pairs[ev]]
    val_pairs   = [p for ev in val_events   for p in event_to_pairs[ev]]

    return train_pairs, val_pairs, val_events
