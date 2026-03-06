"""
cell_detector_cv.py - Classical computer vision cell/nucleus detection.

Uses adaptive thresholding + watershed segmentation for nucleus detection
in H&E stained pathology images.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology, feature, segmentation


DEFAULT_PARAMS: Dict[str, Any] = {
    "min_area": 30,
    "max_area": 800,
    "threshold_method": "adaptive",
    "adaptive_block_size": 51,
    "adaptive_c": 10,
    "dist_transform_threshold": 0.4,
    "min_circularity": 0.3,
}


def detect_nuclei_watershed(
    image: np.ndarray,
    hematoxylin_channel: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Detect nuclei using adaptive thresholding + watershed segmentation.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    hematoxylin_channel : np.ndarray or None
        Pre-extracted hematoxylin channel (grayscale uint8). If None,
        the image is converted to grayscale.
    params : dict or None
        Detection parameters. Missing keys fall back to DEFAULT_PARAMS.

    Returns
    -------
    labels : np.ndarray
        Labeled image where each nucleus has a unique integer label.
    centroids : list of (y, x)
        Centroid coordinates for each detected nucleus.
    properties : list of dict
        Per-nucleus properties: area, circularity, bbox, mean_intensity.
    """
    cfg = dict(DEFAULT_PARAMS)
    if params is not None:
        cfg.update(params)

    # Step 1: prepare grayscale input
    if hematoxylin_channel is not None:
        gray = hematoxylin_channel.copy()
        if gray.dtype == np.float64 or gray.dtype == np.float32:
            vmax = float(np.nanmax(gray)) if gray.size else 0.0
            if vmax <= 1.0:
                gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
            else:
                gray = np.clip(gray, 0, 255).astype(np.uint8)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    intensity_image = gray.copy()

    # Step 2: Gaussian blur
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)

    # Step 3: threshold
    binary = _apply_threshold(blurred, cfg)

    # Step 4: morphological opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Step 5: distance transform
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Step 6: find local maxima as watershed seeds
    dist_thresh = cfg["dist_transform_threshold"]
    dist_max = dist.max()
    if dist_max > 0:
        sure_fg = (dist > dist_thresh * dist_max).astype(np.uint8)
    else:
        sure_fg = np.zeros_like(binary)

    # Label the sure foreground markers
    markers, n_markers = ndi.label(sure_fg)

    # Background region: dilated binary minus sure foreground
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_bg = cv2.dilate(binary, kernel_dilate, iterations=3)

    # Unknown region
    unknown = sure_bg - sure_fg

    # Increment markers so background is 1, unknown is 0
    markers = markers + 1
    markers[unknown == 1] = 0

    # Step 7: watershed segmentation
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.astype(np.int32)
    cv2.watershed(img_bgr, markers_ws)

    # Build clean label image: background=0, boundaries=-1 in watershed output
    labels = markers_ws.copy()
    labels[labels <= 1] = 0  # background and boundary
    labels = labels - 1  # shift so first object is label 1
    labels[labels < 0] = 0

    # Step 8-9: filter by area/circularity and compute properties
    labels, centroids, properties = _extract_and_filter_properties(
        labels, intensity_image, cfg
    )

    return labels, centroids, properties


# --- Private helpers ---


def _apply_threshold(gray: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    """Apply thresholding to a grayscale image and return a binary mask."""
    method = cfg["threshold_method"]

    if method == "adaptive":
        block_size = cfg["adaptive_block_size"]
        c_val = cfg["adaptive_c"]
        if block_size % 2 == 0:
            block_size += 1
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, block_size, c_val
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif isinstance(method, (int, float)):
        _, binary = cv2.threshold(gray, int(method), 255, cv2.THRESH_BINARY_INV)
    else:
        raise ValueError(f"Unknown threshold_method: {method}")

    return binary


def _compute_circularity(area: float, perimeter: float) -> float:
    """Compute circularity = 4 * pi * area / perimeter^2."""
    if perimeter == 0:
        return 0.0
    return (4.0 * np.pi * area) / (perimeter ** 2)


def _extract_and_filter_properties(
    labels: np.ndarray,
    intensity_image: np.ndarray,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Extract region properties, filter by area and circularity, relabel."""
    min_area = cfg["min_area"]
    max_area = cfg["max_area"]
    min_circ = cfg["min_circularity"]

    regions = measure.regionprops(labels, intensity_image=intensity_image)

    filtered_centroids: List[Tuple[float, float]] = []
    filtered_props: List[Dict[str, Any]] = []
    keep_labels = set()

    for region in regions:
        area = region.area
        if area < min_area or area > max_area:
            continue

        perimeter = region.perimeter
        circularity = _compute_circularity(float(area), float(perimeter))
        if circularity < min_circ:
            continue

        keep_labels.add(region.label)
        cy, cx = region.centroid
        filtered_centroids.append((float(cy), float(cx)))

        rmin, cmin, rmax, cmax = region.bbox
        prop_dict = {
            "label": region.label,
            "area": int(area),
            "circularity": round(circularity, 4),
            "bbox": (int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)),
            "mean_intensity": round(float(region.mean_intensity), 2),
            "centroid": (float(cy), float(cx)),
            "perimeter": round(float(perimeter), 2),
        }
        filtered_props.append(prop_dict)

    # Build a clean label image with only kept nuclei
    clean_labels = np.zeros_like(labels)
    new_id = 1
    label_map = {}
    for old_label in sorted(keep_labels):
        label_map[old_label] = new_id
        clean_labels[labels == old_label] = new_id
        new_id += 1

    # Update labels in property dicts
    for prop in filtered_props:
        prop["label"] = label_map[prop["label"]]

    return clean_labels, filtered_centroids, filtered_props
