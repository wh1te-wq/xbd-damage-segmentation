"""
Polygon-to-raster mask generation for the xBD dataset.

Approach adapted from:
  - xView2_baseline  : border-aware polygon masking with Shapely + OpenCV fillPoly
  - xview2_1st_place : damage-encoded pixel values (1-4), dual pre/post masks

Key design decisions
--------------------
* class_id == 0 (un-classified / empty subtype) objects are SKIPPED with
  `continue`; the mask is zero-initialised so drawing class 0 is a no-op.
* Coordinates are ROUNDED before int32 cast to avoid systematic 0.5-px
  boundary drift on small buildings.
* Interior holes in polygons are filled back to 0 after exterior fill.
* MultiPolygon geometries: every constituent polygon is rasterised.
* Class conflict: per-pixel max-class wins (highest damage takes precedence).
"""

import json

import cv2
import numpy as np
from shapely import wkt as shapely_wkt
from shapely.geometry import MultiPolygon, Polygon

# ─── Class mapping ────────────────────────────────────────────────────────────

NUM_CLASSES = 5

SUBTYPE_TO_CLASS: dict[str, int] = {
    "no-damage":     1,
    "minor-damage":  2,
    "major-damage":  3,
    "destroyed":     4,
    "un-classified": 0,   # → skipped
    "":              0,   # → skipped
}

CLASS_NAMES = ["background", "no-damage", "minor-damage", "major-damage", "destroyed"]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _draw_polygon(mask: np.ndarray, poly: Polygon, class_id: int) -> None:
    """
    Rasterise a single Shapely Polygon onto *mask* with max-class priority.

    Steps
    -----
    1. Fill exterior ring into a temporary mask.
    2. Erase interior holes (background wins inside holes).
    3. Merge into global mask: new class wins only if it is higher than existing.
    """
    temp = np.zeros_like(mask)

    # Round before int cast → avoids systematic 0.5-px drift on small buildings
    ext_coords = np.round(np.array(poly.exterior.coords)).astype(np.int32)
    cv2.fillPoly(temp, [ext_coords], class_id)

    for interior in poly.interiors:
        int_coords = np.round(np.array(interior.coords)).astype(np.int32)
        cv2.fillPoly(temp, [int_coords], 0)

    # Per-pixel max: highest damage class always wins
    np.maximum(mask, temp, out=mask)


# ─── Public API ───────────────────────────────────────────────────────────────

def build_mask_from_label(
    label_path: str,
    height: int = 1024,
    width: int = 1024,
) -> np.ndarray:
    """
    Parse an xBD label JSON and return a ``(height, width)`` uint8 class mask.

    Uses pixel-space ``xy`` coordinates (not geographic ``lng_lat`` coordinates).

    Parameters
    ----------
    label_path : str
        Path to a ``*_post_disaster.json`` label file.
    height, width : int
        Spatial dimensions of the target mask (must match the paired image).

    Returns
    -------
    np.ndarray
        uint8 array of shape ``(height, width)``, values in ``[0, NUM_CLASSES-1]``.
    """
    with open(label_path) as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    for feat in data["features"]["xy"]:
        subtype  = feat["properties"].get("subtype", "")
        class_id = SUBTYPE_TO_CLASS.get(subtype, 0)

        # Skip background-class objects (mask already zero-initialised)
        if class_id == 0:
            continue

        try:
            geom = shapely_wkt.loads(feat["wkt"])
        except Exception:
            continue

        if geom.is_empty:
            continue

        if geom.geom_type == "Polygon":
            polys = [geom]
        elif geom.geom_type == "MultiPolygon":
            # Draw every constituent polygon (not just the first)
            polys = list(geom.geoms)
        else:
            continue

        for poly in polys:
            _draw_polygon(mask, poly, class_id)

    return mask
