"""
patholib.analysis.ihc_membrane
================================
Membrane IHC analysis (HER2, CD markers, etc.).

Measures DAB intensity in a ring-shaped region around each nucleus
to approximate membrane staining.
"""

import logging
import warnings

import numpy as np

logger = logging.getLogger(__name__)


def analyze_membrane_ihc(image, params):
    """
    Membrane IHC analysis pipeline.

    Parameters
    ----------
    image : np.ndarray
        RGB uint8 image.
    params : dict
        Must include ring_width, thresholds, min/max_area.

    Returns
    -------
    dict
        Analysis results.
    """
    from patholib.stain.color_deconv import separate_stains
    from skimage import measure, morphology

    hematoxylin, dab, _ = separate_stains(image, stain_type='hdab')

    # Nuclear detection
    labels = _detect_nuclei(hematoxylin, image, params)

    ring_width = params.get("ring_width", 4)
    weak_t = params.get("weak_threshold", 0.10)
    mod_t = params.get("moderate_threshold", 0.25)
    strong_t = params.get("strong_threshold", 0.45)
    min_area = params.get("min_area", 30)
    max_area = params.get("max_area", 800)

    # Create dilated ring mask for membrane measurement
    dilated = morphology.dilation(labels > 0, morphology.disk(ring_width))
    membrane_mask = dilated & ~(labels > 0)

    cell_data = []
    gc = {0: 0, 1: 0, 2: 0, 3: 0}

    for rp in measure.regionprops(labels):
        a = rp.area
        if a < min_area or a > max_area:
            continue

        # Get ring region for this cell
        cell_mask = labels == rp.label
        cell_dilated = morphology.dilation(cell_mask, morphology.disk(ring_width))
        ring = cell_dilated & ~cell_mask
        ring_pixels = dab[ring]

        if ring_pixels.size == 0:
            md = 0.0
        else:
            md = float(np.mean(ring_pixels))

        grade = 3 if md >= strong_t else (2 if md >= mod_t else (1 if md >= weak_t else 0))
        gc[grade] += 1
        cell_data.append({
            "centroid": (float(rp.centroid[0]), float(rp.centroid[1])),
            "area": int(a),
            "label": int(rp.label),
            "grade": grade,
            "intensity_mean": md,
            "membrane_intensity": md,
            "cell_type": "positive" if grade > 0 else "negative",
        })

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
        "stain_type": "membrane",
        "marker": params.get("marker", "Unknown"),
        "allred_score": "N/A (membrane marker)",
        "cell_data": cell_data,
        "labels": labels,
    }

    try:
        from patholib.viz.overlay import create_detection_overlay, blend_overlay
        ov = create_detection_overlay(image, labels, cell_data, overlay_type="ihc")
        result["overlay"] = blend_overlay(image, ov, alpha=0.5)
    except Exception:
        result["overlay"] = image.copy()

    logger.info("Membrane IHC analysis complete: %d cells, H-score=%.1f", total, hs)
    return result


def _detect_nuclei(hematoxylin, rgb, params):
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

    from patholib.detection.cell_detector_cv import detect_nuclei_watershed
    labels, _, _ = detect_nuclei_watershed(rgb, hematoxylin_channel=hematoxylin, params=params)
    return labels
