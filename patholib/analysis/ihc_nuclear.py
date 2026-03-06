"""
patholib.analysis.ihc_nuclear
==============================
Nuclear IHC analysis (Ki-67, p53, ER, PR, etc.).

Workflow:
  1. H-DAB colour deconvolution
  2. Nuclear detection (Cellpose or watershed)
  3. Per-cell DAB intensity measurement
  4. Grade classification (neg/weak/moderate/strong)
  5. Scoring (H-score, percentage, Allred for ER/PR)
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def analyze_nuclear_ihc(image, params):
    """
    Full nuclear IHC analysis pipeline.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image.
    params : dict
        Analysis parameters (thresholds, detection method, marker, etc.).

    Returns
    -------
    dict
        Analysis results including cell data, scores, and overlay.
    """
    from patholib.stain.color_deconv import separate_stains

    # 1. Colour deconvolution
    hematoxylin, dab, _ = separate_stains(image, stain_type='hdab')

    # 2. Nuclear detection
    labels = _detect_nuclei(hematoxylin, image, params)

    # 3. Measure and classify
    from skimage import measure
    weak_t = params.get("weak_threshold", 0.10)
    mod_t = params.get("moderate_threshold", 0.25)
    strong_t = params.get("strong_threshold", 0.45)
    min_area = params.get("min_area", 30)
    max_area = params.get("max_area", 800)

    cell_data = []
    gc = {0: 0, 1: 0, 2: 0, 3: 0}

    for rp in measure.regionprops(labels, intensity_image=dab):
        a = rp.area
        if a < min_area or a > max_area:
            continue
        md = float(rp.mean_intensity)
        grade = 3 if md >= strong_t else (2 if md >= mod_t else (1 if md >= weak_t else 0))
        gc[grade] += 1
        cell_data.append({
            "centroid": (float(rp.centroid[0]), float(rp.centroid[1])),
            "area": int(a),
            "label": int(rp.label),
            "grade": grade,
            "intensity_mean": md,
            "cell_type": "positive" if grade > 0 else "negative",
        })

    # 4. Scoring
    total = len(cell_data)
    pos = sum(1 for c in cell_data if c["grade"] > 0)
    neg = total - pos
    hs = ((gc[1] + 2 * gc[2] + 3 * gc[3]) / total * 100) if total > 0 else 0.0
    pp = (pos / total * 100) if total > 0 else 0.0

    result = {
        "total_cells": total,
        "positive_cells": pos,
        "negative_cells": neg,
        "h_score": float(round(hs, 1)),
        "positive_percentage": float(round(pp, 1)),
        "grade_counts": gc,
        "grade_percentages": {g: round(n / total * 100, 1) if total > 0 else 0.0
                              for g, n in gc.items()},
        "stain_type": "nuclear",
        "marker": params.get("marker", "Unknown"),
        "cell_data": cell_data,
        "labels": labels,
    }

    # Allred for ER/PR
    marker_upper = params.get("marker", "").upper()
    allred_markers = {"ER", "PR", "ESR1", "PGR", "ESTROGEN", "PROGESTERONE"}
    if any(m in marker_upper for m in allred_markers):
        pos_intensities = [c["intensity_mean"] for c in cell_data if c["grade"] > 0]
        rep = float(np.median(pos_intensities)) if pos_intensities else 0.0
        try:
            from patholib.scoring.allred_score import compute_allred
            allred, ps, ins = compute_allred(pp, rep)
            result["allred_score"] = allred
        except ImportError:
            result["allred_score"] = "N/A (scoring module unavailable)"
    else:
        result["allred_score"] = "N/A (not ER/PR marker)"

    # Overlay
    try:
        from patholib.viz.overlay import create_detection_overlay, blend_overlay
        ov = create_detection_overlay(image, labels, cell_data, overlay_type="ihc")
        result["overlay"] = blend_overlay(image, ov, alpha=0.5)
    except Exception:
        result["overlay"] = image.copy()

    logger.info("Nuclear IHC analysis complete: %d cells, H-score=%.1f, pos%%=%.1f%%",
                total, hs, pp)
    return result


def _detect_nuclei(hematoxylin, rgb, params):
    """Try Cellpose first, fall back to watershed."""
    method = params.get("detection_method", "cellpose")
    fail_fast = params.get("fail_fast", False)
    use_gpu = params.get("use_gpu", False)

    if method == "cellpose":
        try:
            from patholib.detection.cell_detector_dl import detect_nuclei_cellpose
            labels, _, _ = detect_nuclei_cellpose(rgb, model_type="nuclei", gpu=use_gpu)
            return labels
        except Exception as e:
            if fail_fast:
                raise RuntimeError(f"Cellpose detection failed: {e}") from e
            warnings.warn(f"Cellpose unavailable ({e}); falling back to watershed.")

    # Watershed fallback
    from patholib.detection.cell_detector_cv import detect_nuclei_watershed
    labels, _, _ = detect_nuclei_watershed(rgb, hematoxylin_channel=hematoxylin, params=params)
    return labels
