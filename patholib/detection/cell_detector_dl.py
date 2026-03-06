"""
cell_detector_dl.py - Deep learning cell detection wrapper for Cellpose.

Provides nucleus and cytoplasm detection using Cellpose models with
graceful import handling and a consistent return format matching
cell_detector_cv.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --- Graceful cellpose import ---

_CELLPOSE_AVAILABLE = False
_CELLPOSE_IMPORT_ERROR: Optional[str] = None

try:
    from cellpose import models as _cp_models
    _CELLPOSE_AVAILABLE = True
except ImportError as exc:
    _CELLPOSE_IMPORT_ERROR = (
        f"Cellpose is not installed. Install with: pip install cellpose  " 
        f"(original error: {exc})"
    )
    _cp_models = None  # type: ignore[assignment]

try:
    from skimage import measure as _sk_measure
except ImportError:
    _sk_measure = None  # type: ignore[assignment]


DEFAULT_PARAMS: Dict[str, Any] = {
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "min_size": 15,
}


def is_cellpose_available() -> bool:
    """Return True if cellpose is installed and importable."""
    return _CELLPOSE_AVAILABLE


def _require_cellpose() -> None:
    """Raise RuntimeError if cellpose is not available."""
    if not _CELLPOSE_AVAILABLE:
        raise RuntimeError(_CELLPOSE_IMPORT_ERROR)


def _require_skimage() -> None:
    """Raise RuntimeError if scikit-image is not available."""
    if _sk_measure is None:
        raise RuntimeError("scikit-image is required. Install with: pip install scikit-image")


def detect_nuclei_cellpose(
    image: np.ndarray,
    model_type: str = "nuclei",
    gpu: bool = True,
    diameter: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Detect nuclei using Cellpose.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    model_type : str
        Cellpose model type. Default "nuclei".
    gpu : bool
        Whether to use GPU acceleration.
    diameter : float or None
        Expected nucleus diameter in pixels. None for auto-estimation.
    params : dict or None
        Additional cellpose parameters (flow_threshold, cellprob_threshold, min_size).

    Returns
    -------
    labels : np.ndarray
    centroids : list of (y, x)
    properties : list of dict
    """
    return _run_cellpose(image, model_type=model_type, gpu=gpu,
                         diameter=diameter, params=params, channels=[0, 0])


def detect_cells_cellpose(
    image: np.ndarray,
    model_type: str = "cyto3",
    gpu: bool = True,
    diameter: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Detect cells (cytoplasm) using Cellpose.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    model_type : str
        Cellpose model type. Default "cyto3".
    gpu : bool
        Whether to use GPU acceleration.
    diameter : float or None
        Expected cell diameter in pixels. None for auto-estimation.
    params : dict or None
        Additional cellpose parameters.

    Returns
    -------
    labels : np.ndarray
    centroids : list of (y, x)
    properties : list of dict
    """
    # For cytoplasm models, channel 0 = cytoplasm (grayscale/red), channel 1 = nuclei (green/blue)
    # When using RGB input: [0, 0] tells cellpose to use grayscale
    # For cyto models with color: [2, 3] means cytoplasm=green, nucleus=blue
    return _run_cellpose(image, model_type=model_type, gpu=gpu,
                         diameter=diameter, params=params, channels=[0, 0])


def _run_cellpose(
    image: np.ndarray,
    model_type: str,
    gpu: bool,
    diameter: Optional[float],
    params: Optional[Dict[str, Any]],
    channels: List[int],
) -> Tuple[np.ndarray, List[Tuple[float, float]], List[Dict[str, Any]]]:
    """Core cellpose inference and post-processing."""
    _require_cellpose()
    _require_skimage()

    cfg = dict(DEFAULT_PARAMS)
    if params is not None:
        cfg.update(params)

    # Initialize model
    logger.info("Loading Cellpose model: %s (gpu=%s)", model_type, gpu)
    model = _cp_models.Cellpose(model_type=model_type, gpu=gpu)

    # Run inference
    masks, flows, styles, diams = model.eval(
        image,
        diameter=diameter,
        channels=channels,
        flow_threshold=cfg["flow_threshold"],
        cellprob_threshold=cfg["cellprob_threshold"],
        min_size=cfg["min_size"],
    )

    labels = masks.astype(np.int32)
    logger.info("Cellpose detected %d objects (estimated diameter=%s)", labels.max(), diams)

    # Extract properties using skimage
    intensity_image = None
    if image.ndim == 3:
        import cv2
        intensity_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        intensity_image = image

    regions = _sk_measure.regionprops(labels, intensity_image=intensity_image)

    centroids: List[Tuple[float, float]] = []
    properties: List[Dict[str, Any]] = []

    for region in regions:
        cy, cx = region.centroid
        centroids.append((float(cy), float(cx)))

        area = region.area
        perimeter = region.perimeter
        circ = 0.0
        if perimeter > 0:
            circ = (4.0 * np.pi * area) / (perimeter ** 2)

        rmin, cmin, rmax, cmax = region.bbox
        prop_dict = {
            "label": region.label,
            "area": int(area),
            "circularity": round(circ, 4),
            "bbox": (int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)),
            "mean_intensity": round(float(region.mean_intensity), 2),
            "centroid": (float(cy), float(cx)),
            "perimeter": round(float(perimeter), 2),
        }
        properties.append(prop_dict)

    return labels, centroids, properties
