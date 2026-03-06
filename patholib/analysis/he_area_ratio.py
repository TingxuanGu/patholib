"""
patholib.analysis.he_area_ratio
================================
Tumor/necrosis area ratio measurement from H&E stained images.

Segments tissue regions (normal, tumor, necrosis, stroma) and computes
area statistics including tumor ratio and necrosis-to-tumor ratio.
"""

import numpy as np
from skimage import morphology, filters, color, measure
from scipy import ndimage


DEFAULT_PARAMS = {
    "method": "threshold",
    "classifier_path": None,
    "mpp": None,
    "necrosis_eosin_threshold": 0.15,
    "necrosis_nuclear_density_threshold": 0.05,
    "patch_size": 64,
    "stride": 32,
}

CLASS_BACKGROUND = 0
CLASS_NORMAL = 1
CLASS_TUMOR = 2
CLASS_NECROSIS = 3
CLASS_STROMA = 4

CLASS_NAMES = {
    CLASS_BACKGROUND: "background",
    CLASS_NORMAL: "normal",
    CLASS_TUMOR: "tumor",
    CLASS_NECROSIS: "necrosis",
    CLASS_STROMA: "stroma",
}


def analyze_area_ratio(image, params=None):
    """
    Analyze tissue area ratios in an H&E stained image.

    Parameters
    ----------
    image : numpy.ndarray
        RGB image (H, W, 3), uint8.
    params : dict, optional
        Override any key in DEFAULT_PARAMS.

    Returns
    -------
    dict
        tissue_area_px, tissue_area_um2, regions dict, tumor_ratio,
        necrosis_ratio, segmentation_mask.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    rgb = _validate_rgb(image)

    tissue_mask = _detect_tissue(rgb)
    tissue_area_px = int(tissue_mask.sum())
    tissue_area_um2 = _px_to_um2(tissue_area_px, p["mpp"])

    if p["method"] == "classifier" and p["classifier_path"] is not None:
        seg_mask = _classify_regions_model(rgb, tissue_mask, p)
    else:
        seg_mask = _classify_regions_threshold(rgb, tissue_mask, p)

    regions = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        if cls_id == CLASS_BACKGROUND:
            continue
        area_px = int((seg_mask == cls_id).sum())
        area_um2 = _px_to_um2(area_px, p["mpp"])
        pct = (area_px / tissue_area_px * 100.0) if tissue_area_px > 0 else 0.0
        regions[cls_name] = {
            "area_px": area_px,
            "area_um2": area_um2,
            "percentage": float(round(pct, 2)),
        }

    tumor_px = regions["tumor"]["area_px"]
    tumor_ratio = (tumor_px / tissue_area_px * 100.0) if tissue_area_px > 0 else 0.0
    necrosis_px = regions["necrosis"]["area_px"]
    necrosis_ratio = (necrosis_px / tumor_px * 100.0) if tumor_px > 0 else 0.0

    return {
        "tissue_area_px": tissue_area_px,
        "tissue_area_um2": tissue_area_um2,
        "regions": regions,
        "tumor_ratio": float(round(tumor_ratio, 2)),
        "necrosis_ratio": float(round(necrosis_ratio, 2)),
        "segmentation_mask": seg_mask,
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


def _px_to_um2(area_px, mpp):
    if mpp is not None and mpp > 0:
        return float(area_px * mpp ** 2)
    return None


def _detect_tissue(rgb):
    gray = np.mean(rgb.astype(np.float64), axis=2) / 255.0
    mask = gray < 0.85
    sat = color.rgb2hsv(rgb)[:, :, 1]
    mask = mask | (sat > 0.05)
    mask = morphology.remove_small_objects(mask, min_size=1000)
    mask = morphology.remove_small_holes(mask, area_threshold=1000)
    mask = morphology.binary_closing(mask, morphology.disk(5))
    return mask


def _he_deconvolution(rgb):
    img = np.clip(rgb.astype(np.float64) / 255.0, 1e-6, 1.0)
    od = -np.log(img)
    he = np.array([[0.6500286, 0.7041306, 0.2860126],
                   [0.0728011, 0.9904030, 0.1181305]])
    he = he / np.linalg.norm(he, axis=1, keepdims=True)
    third = np.cross(he[0], he[1])
    third /= (np.linalg.norm(third) + 1e-12)
    M = np.vstack([he, third])
    deconv = od.reshape(-1, 3) @ np.linalg.inv(M)
    deconv = deconv.reshape(rgb.shape[0], rgb.shape[1], 3)
    hematoxylin = np.clip(deconv[:, :, 0], 0, None)
    eosin = np.clip(deconv[:, :, 1], 0, None)
    return hematoxylin, eosin


def _compute_nuclear_density_map(hematoxylin, patch_size, stride):
    h, w = hematoxylin.shape
    fg = hematoxylin[hematoxylin > 0.01]
    thresh = filters.threshold_otsu(fg) if fg.size > 100 else 0.2
    nuc_mask = hematoxylin > thresh
    density_map = np.zeros_like(hematoxylin, dtype=np.float64)
    count_map = np.zeros_like(hematoxylin, dtype=np.float64)
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = nuc_mask[y:y + patch_size, x:x + patch_size]
            d = patch.mean()
            density_map[y:y + patch_size, x:x + patch_size] += d
            count_map[y:y + patch_size, x:x + patch_size] += 1.0
    count_map[count_map == 0] = 1.0
    return density_map / count_map


def _classify_regions_threshold(rgb, tissue_mask, p):
    h, w = rgb.shape[:2]
    seg = np.zeros((h, w), dtype=np.int32)
    hematoxylin, eosin = _he_deconvolution(rgb)

    nuc_density = _compute_nuclear_density_map(
        hematoxylin, p["patch_size"], p["stride"]
    )

    necrosis_mask = (
        (eosin > p["necrosis_eosin_threshold"])
        & (nuc_density < p["necrosis_nuclear_density_threshold"])
        & tissue_mask
    )
    necrosis_mask = morphology.remove_small_objects(necrosis_mask, min_size=200)
    necrosis_mask = morphology.binary_closing(necrosis_mask, morphology.disk(3))

    stroma_mask = (
        (eosin > p["necrosis_eosin_threshold"] * 0.5)
        & (eosin <= p["necrosis_eosin_threshold"])
        & (nuc_density < 0.15)
        & tissue_mask
        & ~necrosis_mask
    )
    stroma_mask = morphology.remove_small_objects(stroma_mask, min_size=200)

    high_cell_thresh = np.percentile(nuc_density[tissue_mask], 65) if tissue_mask.any() else 0.2
    tumor_mask = (
        (nuc_density > high_cell_thresh)
        & tissue_mask
        & ~necrosis_mask
        & ~stroma_mask
    )
    tumor_mask = morphology.remove_small_objects(tumor_mask, min_size=200)
    tumor_mask = morphology.binary_closing(tumor_mask, morphology.disk(3))

    normal_mask = tissue_mask & ~necrosis_mask & ~stroma_mask & ~tumor_mask

    seg[necrosis_mask] = CLASS_NECROSIS
    seg[stroma_mask] = CLASS_STROMA
    seg[tumor_mask] = CLASS_TUMOR
    seg[normal_mask] = CLASS_NORMAL
    return seg


def _classify_regions_model(rgb, tissue_mask, p):
    import warnings
    warnings.warn(
        "Classifier method is temporarily disabled (MVP patch): "
        "TextureClassifier interface not yet validated. "
        "Falling back to threshold method.",
        UserWarning,
        stacklevel=2,
    )
    return _classify_regions_threshold(rgb, tissue_mask, p)
