"""
Positive cell percentage calculation.

Simple percentage scoring: (positive cells / total cells) * 100.
Also provides labeling index calculation (alias for Ki-67).
"""

import numpy as np
from typing import Optional, List


def compute_percentage(positive: int, total: int) -> float:
    """Return positive cell percentage (0.0-100.0).

    Parameters
    ----------
    positive : int
        Number of positively stained cells.
    total : int
        Total number of detected cells.

    Returns
    -------
    float
        Percentage of positive cells, or 0.0 if total == 0.
    """
    if total <= 0:
        return 0.0
    return float(positive) / float(total) * 100.0


def compute_labeling_index(positive: int, total: int) -> float:
    """Compute labeling index as percentage (0.0-100.0).

    Parameters
    ----------
    positive : int
        Number of Ki-67-positive nuclei.
    total : int
        Total number of detected nuclei.

    Returns
    -------
    float
        Labeling index in [0, 100].
    """
    return compute_percentage(positive, total)


def compute_percentage_from_array(
    intensities: np.ndarray,
    threshold: float,
) -> float:
    """Compute positive percentage by thresholding an intensity array.

    Parameters
    ----------
    intensities : np.ndarray
        Per-cell DAB intensity values (0-1 normalised).
    threshold : float
        Minimum intensity to count as positive.

    Returns
    -------
    float
        Percentage of cells exceeding the threshold.
    """
    arr = np.asarray(intensities, dtype=float)
    if arr.size == 0:
        return 0.0
    positive = int((arr >= threshold).sum())
    return compute_percentage(positive, arr.size)


def compute_tps(positive_tumor_cells: int, total_tumor_cells: int) -> float:
    """Compute Tumor Proportion Score (TPS) used for PD-L1 reporting.

    Parameters
    ----------
    positive_tumor_cells : int
        Tumor cells with membrane PD-L1 staining (>= 1+).
    total_tumor_cells : int
        Total viable tumor cells counted.

    Returns
    -------
    float
        TPS percentage (0-100).
    """
    return compute_percentage(positive_tumor_cells, total_tumor_cells)
