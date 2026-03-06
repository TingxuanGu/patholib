"""
H-score calculation for immunohistochemistry.

H-score = 1 * (% weak positive) + 2 * (% moderate positive) + 3 * (% strong positive)
Range: 0 to 300.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_h_score(
    weak_count: int,
    moderate_count: int,
    strong_count: int,
    total_cells: int,
) -> float:
    """Compute H-score from cell counts per grade.

    Parameters
    ----------
    weak_count : int
        Number of weakly positive (1+) cells.
    moderate_count : int
        Number of moderately positive (2+) cells.
    strong_count : int
        Number of strongly positive (3+) cells.
    total_cells : int
        Total number of cells evaluated.

    Returns
    -------
    h_score : float
        H-score value (0-300).
    """
    if total_cells <= 0:
        return 0.0

    pct_weak = weak_count / total_cells * 100
    pct_moderate = moderate_count / total_cells * 100
    pct_strong = strong_count / total_cells * 100

    h_score = 1 * pct_weak + 2 * pct_moderate + 3 * pct_strong

    return min(h_score, 300.0)


def h_score_from_od_values(
    od_values: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, int]]:
    """Compute H-score from an array of optical density values.

    Parameters
    ----------
    od_values : numpy.ndarray
        Array of mean OD values, one per cell.
    thresholds : dict or None
        OD thresholds. Default: weak=0.10, moderate=0.25, strong=0.45.

    Returns
    -------
    h_score : float
        H-score value (0-300).
    counts : dict
        Cell counts per grade.
    """
    if thresholds is None:
        thresholds = {'weak': 0.10, 'moderate': 0.25, 'strong': 0.45}
    # Validate thresholds
    assert thresholds["weak"] < thresholds["moderate"] < thresholds["strong"], \
        "Thresholds must be in ascending order: weak < moderate < strong"

    od_values = np.asarray(od_values, dtype=np.float64)
    total = len(od_values)

    if total == 0:
        return 0.0, {'negative': 0, 'weak': 0, 'moderate': 0, 'strong': 0}

    negative = int(np.sum(od_values < thresholds["weak"]))
    weak = int(np.sum((od_values >= thresholds["weak"]) & (od_values < thresholds["moderate"])))
    moderate = int(np.sum((od_values >= thresholds["moderate"]) & (od_values < thresholds["strong"])))
    strong = int(np.sum(od_values >= thresholds["strong"]))

    counts = {'negative': negative, 'weak': weak, 'moderate': moderate, 'strong': strong}

    h_score = compute_h_score(weak, moderate, strong, total)

    return h_score, counts
