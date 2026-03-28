"""
Augmentation pipeline for xBD pre/post disaster image pairs.

Design principles (from xview2_1st_place_solution)
---------------------------------------------------
* Geometric transforms (flip, rotate, scale-crop) are applied **consistently**
  to the pre-image, post-image, and mask so spatial alignment is preserved.
* Color transforms (brightness, contrast, HSV jitter) are applied
  **independently** to pre and post images — the two acquisitions differ in
  lighting conditions, so identical color augmentation is unrealistic.
* Noise is applied independently to each image.

All transforms operate on raw ``uint8`` NumPy arrays (HxWx3) and return
arrays of the same type/shape.
"""

import random

import cv2
import numpy as np


class Augmentor:
    """
    Stochastic augmentor for (pre, post, mask) triplets.

    Parameters
    ----------
    p_hflip          : probability of random horizontal flip.
    p_vflip          : probability of random vertical flip.
    p_rotate90       : probability of random 90°-multiple rotation.
    p_color          : probability of applying colour jitter per image.
    p_noise          : probability of adding Gaussian noise per image.
    brightness_limit : max absolute brightness shift (fraction of 255).
    contrast_limit   : max multiplicative contrast shift (fraction).
    hue_shift_limit  : max hue shift in degrees.
    sat_shift_limit  : max saturation shift (0–255 scale).
    noise_std        : standard deviation of Gaussian noise (fraction of 255).
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_rotate90: float = 0.5,
        p_color: float = 0.5,
        p_noise: float = 0.3,
        brightness_limit: float = 0.2,
        contrast_limit: float = 0.2,
        hue_shift_limit: int = 10,
        sat_shift_limit: int = 20,
        noise_std: float = 0.03,
    ):
        self.p_hflip          = p_hflip
        self.p_vflip          = p_vflip
        self.p_rotate90       = p_rotate90
        self.p_color          = p_color
        self.p_noise          = p_noise
        self.brightness_limit = brightness_limit
        self.contrast_limit   = contrast_limit
        self.hue_shift_limit  = int(hue_shift_limit)
        self.sat_shift_limit  = int(sat_shift_limit)
        self.noise_std        = noise_std

    # ── Public call ───────────────────────────────────────────────────────────

    def __call__(
        self,
        pre:  np.ndarray,
        post: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply augmentations to a (pre, post, mask) triplet.

        Parameters
        ----------
        pre, post : np.ndarray  HxWx3 uint8
        mask      : np.ndarray  HxW   uint8

        Returns
        -------
        (pre, post, mask) — same shapes and dtype, contiguous arrays.
        """
        # 1. Geometric — same random transform applied to all three
        pre, post, mask = self._geometric(pre, post, mask)

        # 2. Colour — independent random transform per image
        if random.random() < self.p_color:
            pre  = self._color_jitter(pre)
        if random.random() < self.p_color:
            post = self._color_jitter(post)

        # 3. Noise — independent per image
        if random.random() < self.p_noise:
            pre  = self._gaussian_noise(pre)
        if random.random() < self.p_noise:
            post = self._gaussian_noise(post)

        return (
            np.ascontiguousarray(pre),
            np.ascontiguousarray(post),
            np.ascontiguousarray(mask),
        )

    # ── Geometric transforms ──────────────────────────────────────────────────

    def _geometric(
        self,
        pre:  np.ndarray,
        post: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply identical geometric transforms to all three arrays."""
        # Horizontal flip
        if random.random() < self.p_hflip:
            pre  = cv2.flip(pre,  1)
            post = cv2.flip(post, 1)
            mask = cv2.flip(mask, 1)

        # Vertical flip
        if random.random() < self.p_vflip:
            pre  = cv2.flip(pre,  0)
            post = cv2.flip(post, 0)
            mask = cv2.flip(mask, 0)

        # 90°-multiple rotation
        if random.random() < self.p_rotate90:
            k = random.choice([1, 2, 3])
            pre  = np.rot90(pre,  k)
            post = np.rot90(post, k)
            mask = np.rot90(mask, k)

        return pre, post, mask

    # ── Colour transforms ─────────────────────────────────────────────────────

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """
        Apply random brightness, contrast, saturation, and hue jitter.
        Operations are shuffled each call (inspired by xView2_baseline transforms.py).
        """
        ops = [self._brightness, self._contrast, self._hsv_shift]
        random.shuffle(ops)
        for op in ops:
            img = op(img)
        return img

    def _brightness(self, img: np.ndarray) -> np.ndarray:
        delta = int(random.uniform(-self.brightness_limit, self.brightness_limit) * 255)
        img   = img.astype(np.int16)
        img   = np.clip(img + delta, 0, 255).astype(np.uint8)
        return img

    def _contrast(self, img: np.ndarray) -> np.ndarray:
        factor = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
        img    = img.astype(np.float32)
        img    = np.clip(img * factor, 0, 255).astype(np.uint8)
        return img

    def _hsv_shift(self, img: np.ndarray) -> np.ndarray:
        """Shift hue and saturation in HSV space."""
        hsv         = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
        hue_shift   = random.randint(-self.hue_shift_limit, self.hue_shift_limit)
        sat_shift   = random.randint(-self.sat_shift_limit, self.sat_shift_limit)
        hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + sat_shift, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # ── Noise ─────────────────────────────────────────────────────────────────

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        """Add per-pixel Gaussian noise."""
        noise = np.random.normal(0, self.noise_std * 255, img.shape).astype(np.int16)
        return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_augmentor(aug_cfg: dict) -> Augmentor:
    """
    Build an :class:`Augmentor` from a config dict (``cfg["augmentation"]``).
    """
    return Augmentor(
        p_hflip          = aug_cfg.get("p_hflip",          0.5),
        p_vflip          = aug_cfg.get("p_vflip",          0.5),
        p_rotate90       = aug_cfg.get("p_rotate90",       0.5),
        p_color          = aug_cfg.get("p_color",          0.5),
        p_noise          = aug_cfg.get("p_noise",          0.3),
        brightness_limit = aug_cfg.get("brightness_limit", 0.2),
        contrast_limit   = aug_cfg.get("contrast_limit",   0.2),
        hue_shift_limit  = aug_cfg.get("hue_shift_limit",  10),
        sat_shift_limit  = aug_cfg.get("sat_shift_limit",  20),
        noise_std        = aug_cfg.get("noise_std",        0.03),
    )
