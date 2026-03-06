"""
Positive cell percentage calculation.

Simple percentage scoring: (positive cells / total cells) * 100.
Also provides labeling index calculation (alias for Ki-67).
"""

import numpy as np
from typing import Optional


def compute_percentage(positive_count: int, total_count: int) -> float:
    """Compute positive cell percentage.

    Parameters
    ----------
    positive_count : int
        Number of positive cells.
    total_count : int
        Total number of cells.

    Returns
    -------
    percentage : float
        Positive percentage (0-100).
    """
    if total_count <= 0:
        return 0.0
    pct = positive_count / total_count * 100
    return min(max(pct, 0.0), 100.0)


def compute_labeling_index(positive_count: int, total_count: int) -> float:
    """Compute Ki-67 labeling index (alias for percentage).

    Parameters
    ----------
    positive_count : int
        Number of Ki-67-positive cells.
    total_count : int
        Total number of cells.

    Returns
    -------
    li : float
        Labeling index (0-100).
    """
    return compute_percentage(positive_count, total_count)


def compute_percentage_from_array(
    values: np.ndarray,
    threshold: float = 0.10,
) -> float:
    """Compute positive percentage from an array of values.

    Parameters
    ----------
    values : numpy.ndarray
        Array of measurement values (e.g., OD values).
    threshold : float
        Threshold for positivity.

    Returns
    -------
    percentage : float
        Positive percentage (0-100).
    """
    values = np.asarray(values)
    if len(values) == 0:
        return 0.0

    positive = int(np.sum(values >= threshold))
    return compute_percentage(positive, len(values))
