"""
patholib.stain.stain_normalizer
================================
Stain normalization for H&E and IHC images.

Implements:
  - Macenko method (Macenko et al., 2009)
  - Reinhard method (Reinhard et al., 2001)

Both methods align the colour distribution of a source image to a target image,
reducing inter-slide staining variability.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def normalize_stain(
    source: np.ndarray,
    target: Optional[np.ndarray] = None,
    method: str = 'reinhard',
) -> np.ndarray:
    """
    Normalize staining of source image to match target.

    Parameters
    ----------
    source : np.ndarray
        Source RGB image (uint8).
    target : np.ndarray, optional
        Target RGB image (uint8). If None, uses built-in reference statistics.
    method : str
        'reinhard' or 'macenko'.

    Returns
    -------
    np.ndarray
        Normalized RGB image (uint8).
    """
    if method.lower() == 'reinhard':
        return _reinhard_normalize(source, target)
    elif method.lower() == 'macenko':
        return _macenko_normalize(source, target)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# Default reference statistics (Lab colour space, from a well-stained H&E slide)
_DEFAULT_LAB_STATS = {
    'mean': np.array([71.0, 12.0, -5.0]),
    'std': np.array([16.0, 8.0, 6.0]),
}


def _reinhard_normalize(
    source: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Reinhard colour transfer in Lab colour space.
    """
    from skimage.color import rgb2lab, lab2rgb

    src_lab = rgb2lab(source)

    if target is not None:
        tgt_lab = rgb2lab(target)
        tgt_mean = tgt_lab.reshape(-1, 3).mean(axis=0)
        tgt_std = tgt_lab.reshape(-1, 3).std(axis=0) + 1e-6
    else:
        tgt_mean = _DEFAULT_LAB_STATS['mean']
        tgt_std = _DEFAULT_LAB_STATS['std']

    src_flat = src_lab.reshape(-1, 3)
    src_mean = src_flat.mean(axis=0)
    src_std = src_flat.std(axis=0) + 1e-6

    # Transfer: scale and shift each channel
    normalized = (src_flat - src_mean) * (tgt_std / src_std) + tgt_mean
    normalized = normalized.reshape(src_lab.shape)

    # Convert back to RGB
    rgb_out = lab2rgb(normalized)
    rgb_out = np.clip(rgb_out * 255, 0, 255).astype(np.uint8)

    logger.info("Reinhard normalization applied.")
    return rgb_out


def _macenko_normalize(
    source: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Macenko stain normalization via SVD of OD space.
    Simplified implementation for the MVP pipeline.
    """
    def _get_stain_vectors(img):
        od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1) / 256.0)
        od_thresh = od[od.sum(axis=1) > 0.15]
        if od_thresh.shape[0] < 10:
            return np.eye(3)[:2]
        _, _, Vt = np.linalg.svd(od_thresh, full_matrices=False)
        plane = Vt[:2, :]
        return plane

    def _project(img, vectors):
        od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1) / 256.0)
        coords = od @ vectors.T
        return coords

    src_vecs = _get_stain_vectors(source)
    src_coords = _project(source, src_vecs)

    if target is not None:
        tgt_vecs = _get_stain_vectors(target)
        tgt_coords = _project(target, tgt_vecs)
    else:
        # Use source vectors but standardize intensity
        tgt_vecs = src_vecs
        tgt_coords = src_coords

    # Normalize concentrations
    src_99 = np.percentile(src_coords, 99, axis=0).clip(min=1e-6)
    tgt_99 = np.percentile(tgt_coords, 99, axis=0).clip(min=1e-6)

    norm_coords = src_coords * (tgt_99 / src_99)

    # Reconstruct OD
    od_norm = norm_coords @ tgt_vecs[:2]
    rgb_norm = np.exp(-od_norm) * 256.0 - 1
    rgb_norm = np.clip(rgb_norm, 0, 255).astype(np.uint8)
    rgb_norm = rgb_norm.reshape(source.shape)

    logger.info("Macenko normalization applied.")
    return rgb_norm
