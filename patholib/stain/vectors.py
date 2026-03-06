"""
patholib.stain.vectors
======================
Preset stain vectors for common IHC and H&E staining protocols.
"""

import numpy as np

STAIN_VECTORS = {
    'hdab': {
        'hematoxylin': np.array([0.650, 0.704, 0.286]),
        'dab':         np.array([0.268, 0.570, 0.776]),
        'residual':    np.array([0.707, 0.424, 0.566]),
    },
    'he': {
        'hematoxylin': np.array([0.644, 0.717, 0.267]),
        'eosin':       np.array([0.093, 0.954, 0.283]),
        'residual':    np.array([0.630, 0.350, 0.690]),
    },
    'hdab_alt': {
        'hematoxylin': np.array([0.651, 0.701, 0.290]),
        'dab':         np.array([0.269, 0.568, 0.778]),
        'residual':    np.array([0.710, 0.426, 0.560]),
    },
}


def get_matrix(stain_type: str) -> np.ndarray:
    """Return 3x3 stain matrix for the given stain type."""
    key = stain_type.lower().replace('-', '').replace('_', '')
    if key not in STAIN_VECTORS:
        raise ValueError(f"Unknown stain type: {stain_type}. Available: {list(STAIN_VECTORS.keys())}")
    d = STAIN_VECTORS[key]
    return np.array(list(d.values()), dtype=np.float64)
