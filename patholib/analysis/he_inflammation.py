"""
patholib.analysis.he_inflammation
=================================
Inflammation scoring from H&E stained histopathology images.

Provides quantitative assessment of inflammatory infiltrate density
in H&E-stained tissue sections through automated nucleus detection,
morphological classification, and grid-based scoring.
"""

import numpy as np
from skimage import measure, morphology, filters
from scipy import ndimage
import warnings


DEFAULT_PARAMS = {
    "detection_method": "cellpose",
    "inflammatory_max_area": 80,
    "inflammatory_min_circularity": 0.7,
    "grid_size_um": 200,
    "grid_size_px": 400,
    "mpp": None,
    "mild_threshold": 50,
    "moderate_threshold": 200,
    "severe_threshold": 500,
    "min_area": 15,
    "max_area": 500,
}


def analyze_inflammation(image, params=None):
    """
    Analyze inflammation in an H&E stained image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image array with shape (H, W, 3) and dtype uint8.
    params : dict, optional
        Analysis parameters. Missing keys filled from DEFAULT_PARAMS.

    Returns
    -------
    dict
        total_nuclei, inflammatory_cells, parenchymal_cells,
        inflammatory_density, inflammation_score (0-3),
        grid_scores, grid_densities, cell_data, labels.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    rgb = _validate_rgb(image)
    hematoxylin = _he_color_deconvolution(rgb)
    labels = _detect_nuclei(rgb, hematoxylin, p)
    cell_data = _extract_morphology(labels, p)
    cell_data, n_inflammatory, n_parenchymal = _classify_cells(cell_data, p)
    tissue_mask, tissue_area_px = _tissue_mask(rgb)
    density = _inflammatory_density(n_inflammatory, tissue_area_px, p["mpp"])
    grid_scores, grid_densities = _grid_scoring(rgb.shape, cell_data, p, tissue_mask)
    score = _score_from_density(density, p)
    return {
        "total_nuclei": len(cell_data),
        "inflammatory_cells": n_inflammatory,
        "parenchymal_cells": n_parenchymal,
        "inflammatory_density": float(density),
        "inflammation_score": int(score),
        "grid_scores": grid_scores,
        "grid_densities": grid_densities,
        "cell_data": cell_data,
        "labels": labels,
    }


def _validate_rgb(image):
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected RGB (H,W,3); got {image.shape}.")
    rgb = image[:, :, :3]
    if np.issubdtype(rgb.dtype, np.floating):
        vmax = float(np.nanmax(rgb)) if rgb.size else 0.0
        if vmax <= 1.0:
            rgb = np.clip(rgb, 0.0, 1.0) * 255.0
        else:
            rgb = np.clip(rgb, 0.0, 255.0)
        return rgb.astype(np.uint8)
    if np.issubdtype(rgb.dtype, np.integer):
        return np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb.astype(np.uint8)


def _he_color_deconvolution(rgb):
    try:
        from patholib.stain.color_deconv import separate_stains
        h, _e, _residual = separate_stains(rgb, stain_type='he')
        return h
    except (ImportError, ModuleNotFoundError):
        pass
    img = np.clip(rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    od = -np.log(img)
    he = np.array([[0.6500286, 0.7041306, 0.2860126],
                   [0.0728011, 0.9904030, 0.1181305]])
    he = he / np.linalg.norm(he, axis=1, keepdims=True)
    third = np.cross(he[0], he[1])
    third /= (np.linalg.norm(third) + 1e-12)
    M = np.vstack([he, third])
    deconv = od.reshape(-1, 3) @ np.linalg.inv(M)
    return np.clip(deconv[:, 0].reshape(rgb.shape[:2]), 0, None)


def _detect_nuclei(rgb, hematoxylin, p):
    if p["detection_method"] == "cellpose":
        try:
            from patholib.detection.cell_detector_dl import detect_nuclei_cellpose
            labels, _, _ = detect_nuclei_cellpose(
                rgb,
                model_type="nuclei",
                gpu=bool(p.get("use_gpu", False)),
                params=p,
            )
            return labels
        except Exception as exc:
            if p.get("fail_fast"):
                raise RuntimeError(f"Cellpose detection failed: {exc}") from exc
            warnings.warn(f"Cellpose unavailable ({exc}); falling back to watershed.")
    return _detect_watershed(hematoxylin, p)


def _detect_watershed(hematoxylin, p):
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as skwatershed
    h_norm = hematoxylin / (hematoxylin.max() + 1e-8)
    fg = h_norm[h_norm > 0.01]
    thresh = filters.threshold_otsu(fg) if fg.size > 100 else 0.2
    binary = h_norm > thresh
    binary = morphology.remove_small_objects(binary, min_size=p["min_area"])
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=5, labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    markers = ndimage.label(morphology.dilation(markers > 0, morphology.disk(2)))[0]
    return skwatershed(-dist, markers, mask=binary).astype(np.int32)


def _extract_morphology(labels, p):
    cells = []
    for rp in measure.regionprops(labels):
        a = rp.area
        if a < p["min_area"] or a > p["max_area"]:
            continue
        peri = rp.perimeter
        circ = min((4 * np.pi * a) / (peri ** 2), 1.0) if peri > 0 else 0.0
        cells.append({
            "centroid": (float(rp.centroid[0]), float(rp.centroid[1])),
            "area": int(a),
            "circularity": float(circ),
            "eccentricity": float(rp.eccentricity),
            "solidity": float(rp.solidity),
            "label": int(rp.label),
            "cell_type": None,
        })
    return cells


def _classify_cells(cell_data, p):
    n_i = n_p = 0
    for c in cell_data:
        if c["area"] <= p["inflammatory_max_area"] and c["circularity"] >= p["inflammatory_min_circularity"]:
            c["cell_type"] = "inflammatory"
            n_i += 1
        else:
            c["cell_type"] = "parenchymal"
            n_p += 1
    return cell_data, n_i, n_p


def _tissue_mask(rgb):
    gray = np.mean(rgb.astype(np.float64), axis=2) / 255.0
    mask = gray < 0.85
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    return mask, int(mask.sum())


def _inflammatory_density(n, tpx, mpp):
    if tpx == 0:
        return 0.0
    if mpp is not None and mpp > 0:
        mm2 = tpx * mpp ** 2 / 1e6
        return float(n / mm2) if mm2 > 0 else 0.0
    kpx = tpx / 1000.0
    return float(n / kpx) if kpx > 0 else 0.0


def _grid_scoring(shape, cell_data, p, tissue_mask):
    h, w = shape[:2]
    mpp = p.get("mpp")
    if mpp and mpp > 0:
        gs = max(1, int(p["grid_size_um"] / mpp))
    else:
        gs = max(1, int(p.get("grid_size_px", p["grid_size_um"])))
    nr = max(1, int(np.ceil(h / gs)))
    nc = max(1, int(np.ceil(w / gs)))
    counts = np.zeros((nr, nc), dtype=np.float64)
    for c in cell_data:
        if c["cell_type"] != "inflammatory":
            continue
        cy, cx = c["centroid"]
        counts[min(int(cy // gs), nr - 1), min(int(cx // gs), nc - 1)] += 1
    densities = np.zeros((nr, nc), dtype=np.float64)
    scores = np.zeros((nr, nc), dtype=np.int32)
    for r in range(nr):
        for c in range(nc):
            r0, r1 = r * gs, min(r * gs + gs, h)
            c0, c1 = c * gs, min(c * gs + gs, w)
            tpx = int(tissue_mask[r0:r1, c0:c1].sum())
            if tpx == 0:
                continue
            d = _inflammatory_density(counts[r, c], tpx, mpp)
            densities[r, c] = d
            scores[r, c] = _score_from_density(d, p)
    return scores, densities


def _score_from_density(density, p):
    if density < p["mild_threshold"]:
        return 0
    if density < p["moderate_threshold"]:
        return 1
    if density < p["severe_threshold"]:
        return 2
    return 3
