"""
patholib.stain.color_deconv
============================
Colour deconvolution for H&E and IHC stains.

Implements the Ruifrok & Johnston method (2001) for separating
Hematoxylin, DAB, Eosin, and other common stain combinations.

Reference:
  Ruifrok AC, Johnston DA. Quantification of histochemical staining by color
  deconvolution. Anal Quant Cytol Histol. 2001;23(4):291-9.
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default stain matrices (rows = stain vectors, normalised to unit length)
# H-DAB (Hematoxylin + DAB, standard IHC)
HDAB_MATRIX = np.array([
    [0.650, 0.704, 0.286],  # Hematoxylin
    [0.268, 0.570, 0.776],  # DAB
    [0.707, 0.424, 0.566],  # Residual
], dtype=np.float64)

# H-E (Hematoxylin + Eosin, standard H&E)
HE_MATRIX = np.array([
    [0.644, 0.717, 0.267],  # Hematoxylin
    [0.093, 0.954, 0.283],  # Eosin
    [0.630, 0.350, 0.690],  # Residual (background/other)
], dtype=np.float64)


def separate_stains(
    rgb: np.ndarray,
    stain_matrix: Optional[np.ndarray] = None,
    stain_type: str = 'hdab',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separate stains from an RGB image using colour deconvolution.

    Parameters
    ----------
    rgb : np.ndarray
        Input RGB image, uint8, shape (H, W, 3).
    stain_matrix : np.ndarray, optional
        3x3 stain matrix. If None, uses preset based on stain_type.
    stain_type : str
        'hdab' for H-DAB (IHC) or 'he' for H&E. Ignored if stain_matrix is provided.

    Returns
    -------
    ch1, ch2, ch3 : np.ndarray
        Optical density images for each stain channel.
        For 'hdab': (hematoxylin, dab, residual).
        For 'he': (hematoxylin, eosin, residual).
        Each is float64, range roughly [0, ~3], higher = more stain.
    """
    if stain_matrix is None:
        if stain_type.lower() in ('hdab', 'h-dab', 'h_dab', 'ihc'):
            stain_matrix = HDAB_MATRIX.copy()
        elif stain_type.lower() in ('he', 'h-e', 'h_e', 'hne'):
            stain_matrix = HE_MATRIX.copy()
        else:
            raise ValueError(f"Unknown stain_type: {stain_type}. Use 'hdab' or 'he'.")

    # Normalise rows to unit length
    norms = np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    M = stain_matrix / norms

    # Invert the stain matrix
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        logger.warning("Stain matrix is singular; using pseudo-inverse.")
        M_inv = np.linalg.pinv(M)

    # Convert RGB to optical density (OD)
    img = rgb.astype(np.float64)
    img = np.clip(img, 1, 255)  # avoid log(0)
    od = -np.log(img / 255.0)

    # Deconvolve: shape (H, W, 3) @ (3, 3).T -> (H, W, 3) stain channels
    h, w = od.shape[:2]
    od_flat = od.reshape(-1, 3)
    stains = od_flat @ M_inv.T
    stains = stains.reshape(h, w, 3)

    ch1 = stains[:, :, 0]
    ch2 = stains[:, :, 1]
    ch3 = stains[:, :, 2]

    logger.info(
        "Color deconvolution (%s): ch1 range=[%.3f, %.3f], ch2 range=[%.3f, %.3f]",
        stain_type, ch1.min(), ch1.max(), ch2.min(), ch2.max()
    )

    return ch1, ch2, ch3


def od_to_rgb(od_channel: np.ndarray, stain_vector: np.ndarray) -> np.ndarray:
    """
    Reconstruct an RGB image from a single OD channel and its stain vector.

    Useful for visualization (e.g., showing only the DAB channel as pseudo-colour).

    Parameters
    ----------
    od_channel : np.ndarray
        2D optical density image.
    stain_vector : np.ndarray
        Length-3 stain vector (will be normalised).

    Returns
    -------
    np.ndarray
        uint8 RGB image showing just this stain.
    """
    sv = stain_vector / (np.linalg.norm(stain_vector) + 1e-12)
    od3 = np.outer(od_channel.ravel(), sv).reshape(*od_channel.shape, 3)
    rgb = np.exp(-od3) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)
