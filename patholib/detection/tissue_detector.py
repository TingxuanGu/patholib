"""
tissue_detector.py - Tissue region detection for whole-slide pathology images.

Separates tissue from glass/background using thresholding on HSV saturation
or grayscale intensity, with optional morphological cleaning.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage as ndi


def detect_tissue(
    image: np.ndarray,
    method: str = "otsu",
    min_area: int = 500,
    closing_radius: int = 5,
    saturation_threshold: Optional[int] = None,
) -> np.ndarray:
    """Detect tissue regions and return a binary mask (tissue=True).

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    method : str
        Thresholding strategy. One of "otsu" (HSV saturation Otsu),
        "grayscale" (inverted grayscale Otsu), or "saturation" (fixed).
    min_area : int
        Connected components smaller than this (pixels) are removed.
    closing_radius : int
        Disk radius for morphological closing. 0 to skip.
    saturation_threshold : int or None
        Fixed saturation threshold [0, 255]. Required for method="saturation".

    Returns
    -------
    mask : np.ndarray
        Boolean array (H, W), True = tissue.
    """
    _validate_image(image)
    if method == "otsu":
        mask = _threshold_otsu_saturation(image)
    elif method == "grayscale":
        mask = _threshold_otsu_grayscale(image)
    elif method == "saturation":
        if saturation_threshold is None:
            raise ValueError("saturation_threshold required when method=saturation")
        mask = _threshold_fixed_saturation(image, saturation_threshold)
    else:
        raise ValueError(f"Unknown method: {method}")
    if closing_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * closing_radius + 1, 2 * closing_radius + 1))
        mask_u8 = mask.astype(np.uint8) * 255
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
        mask = mask_u8 > 0
    mask = ndi.binary_fill_holes(mask)
    if min_area > 0:
        mask = _remove_small_objects(mask, min_area)
    return mask


def compute_tissue_area(mask: np.ndarray, mpp: Optional[float] = None) -> float:
    """Compute tissue area. Returns um^2 if mpp given, else pixel count."""
    pixel_count = float(np.count_nonzero(mask))
    if mpp is not None:
        return pixel_count * (mpp ** 2)
    return pixel_count


def get_tissue_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) bounding box of tissue. (0,0,0,0) if empty."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    row_min, row_max = np.where(rows)[0][[0, -1]]
    col_min, col_max = np.where(cols)[0][[0, -1]]
    x = int(col_min)
    y = int(row_min)
    w = int(col_max - col_min + 1)
    h = int(row_max - row_min + 1)
    return (x, y, w, h)


# --- Private helpers ---


def _validate_image(image: np.ndarray) -> None:
    """Raise ValueError for non-conforming input images."""
    if not (image.ndim == 3 and image.shape[2] == 3):
        raise ValueError(f"Expected 3-channel RGB image, got shape {image.shape}.")
    if not (image.dtype == np.uint8):
        warnings.warn(f"Image dtype is {image.dtype}; converting to uint8.", stacklevel=3)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure image is uint8, handling both [0,1] and [0,255] float ranges."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        vmax = float(np.nanmax(image)) if image.size else 0.0
        if vmax <= 1.0:
            return (np.clip(image, 0, 1) * 255).astype(np.uint8)
        return np.clip(image, 0, 255).astype(np.uint8)
    return image.astype(np.uint8)


def _threshold_otsu_saturation(image: np.ndarray) -> np.ndarray:
    """Otsu threshold on the HSV saturation channel."""
    image = _to_uint8(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _threshold_otsu_grayscale(image: np.ndarray) -> np.ndarray:
    """Otsu threshold on inverted grayscale (tissue darker than bg)."""
    image = _to_uint8(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray_inv = 255 - gray
    _, binary = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary > 0


def _threshold_fixed_saturation(image: np.ndarray, threshold: int) -> np.ndarray:
    """Fixed threshold on the HSV saturation channel."""
    image = _to_uint8(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return hsv[:, :, 1] > threshold


def _remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area pixels."""
    labelled, num_features = ndi.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labelled.ravel())
    too_small = component_sizes < min_area
    too_small[0] = False
    mask_out = mask.copy()
    mask_out[too_small[labelled]] = False
    return mask_out
