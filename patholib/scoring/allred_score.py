"""
Allred score calculation for ER/PR immunohistochemistry.

Allred Score = Proportion Score (PS, 0-5) + Intensity Score (IS, 0-3)
Range: 0 or 2-8 (score of 1 is not possible in practice).

Reference:
    Allred DC, et al. Prognostic and predictive factors in breast cancer
    by immunohistochemical analysis. Mod Pathol. 1998;11(2):155-168.
"""

from typing import Tuple


# Proportion Score thresholds (percentage of positive cells)
PROPORTION_THRESHOLDS = [
    (0, 0),     # 0%        -> PS 0
    (1, 1),     # >0 to 1%  -> PS 1
    (10, 2),    # >1 to 10% -> PS 2
    (33, 3),    # >10 to 33%-> PS 3
    (66, 4),    # >33 to 66%-> PS 4
    (100, 5),   # >66%      -> PS 5
]

# Intensity Score thresholds (normalised DAB stain intensity 0-1)
INTENSITY_THRESHOLDS = [
    (0.10, 1),   # weak
    (0.25, 2),   # moderate
    (0.45, 3),   # strong
]


def proportion_score(positive_percentage: float) -> int:
    """Map positive percentage to Allred Proportion Score (0-5)."""
    p = float(positive_percentage)
    if p <= 0:
        return 0
    if p < 1:
        return 1
    if p < 10:
        return 2
    if p < 33:
        return 3
    if p < 67:
        return 4
    return 5


def intensity_score(
    mean_intensity: float,
    weak: float = 0.10,
    moderate: float = 0.25,
    strong: float = 0.45,
) -> int:
    """Map representative stain intensity to Allred Intensity Score (0-3).

    Parameters
    ----------
    mean_intensity : float
        Representative DAB optical density value, normalised 0-1.
    weak, moderate, strong : float
        Lower boundaries for each intensity grade.
    """
    v = float(mean_intensity)
    if v >= strong:
        return 3
    if v >= moderate:
        return 2
    if v >= weak:
        return 1
    return 0


def compute_allred(
    positive_percentage: float,
    representative_intensity: float,
    weak: float = 0.10,
    moderate: float = 0.25,
    strong: float = 0.45,
) -> Tuple[int, int, int]:
    """Compute Allred score.

    Parameters
    ----------
    positive_percentage : float
        Percentage of positive cells (0-100).
    representative_intensity : float
        Representative DAB intensity of positive cells (0-1).
    weak, moderate, strong : float
        Intensity grade thresholds.

    Returns
    -------
    (allred_total, proportion_score, intensity_score) : Tuple[int, int, int]
        allred_total is PS + IS, clamped so that 1 maps to 0 per convention.
    """
    ps = proportion_score(positive_percentage)
    ins = intensity_score(representative_intensity, weak, moderate, strong)
    total = ps + ins
    # Allred convention: combined score of 1 does not exist; treat as 0
    if total == 1:
        total = 0
    return total, ps, ins
