#!/usr/bin/env python3
"""
analyze_ihc.py - CLI entry point for IHC image analysis.

Usage:
    python analyze_ihc.py --input slide.tif --stain-type nuclear --marker Ki67
"""

import argparse
import os
import sys
import time
import numpy as np


def build_parser():
    p = argparse.ArgumentParser(description="Quantitative IHC image analysis")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--stain-type", required=True,
                   choices=["nuclear", "membrane", "cytoplasmic"])
    p.add_argument("--marker", default="Unknown")
    p.add_argument("--detection-method", default="cellpose",
                   choices=["cellpose", "watershed"])
    p.add_argument("--stain-vector", default="auto", choices=["auto", "custom"])
    p.add_argument("--scoring", default="h-score,percentage")
    p.add_argument("--weak-threshold", type=float, default=0.10)
    p.add_argument("--moderate-threshold", type=float, default=0.25)
    p.add_argument("--strong-threshold", type=float, default=0.45)
    p.add_argument("--ring-width", type=int, default=4)
    p.add_argument("--min-area", type=int, default=30)
    p.add_argument("--max-area", type=int, default=800)
    p.add_argument("--normalize-stain", action="store_true")
    p.add_argument("--stain-reference", default=None)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--save-overlay", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    p.add_argument("--tile-size", type=int, default=256)
    p.add_argument("--magnification", type=float, default=20.0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)
    return p


def load_image(path):
    try:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    except ImportError:
        pass
    from skimage.io import imread
    return imread(path)[:, :, :3]


def apply_stain_normalization(image, reference_path=None):
    try:
        from patholib.stain.stain_normalizer import normalize_stain
        target = load_image(reference_path) if reference_path else None
        return normalize_stain(image, target=target, method='macenko')
    except (ImportError, ModuleNotFoundError):
        import warnings
        warnings.warn("Stain normalization not available; skipping.")
        return image


def build_params(args):
    return {
        "detection_method": args.detection_method,
        "stain_vector": args.stain_vector,
        "scoring_methods": [s.strip() for s in args.scoring.split(",")],
        "weak_threshold": args.weak_threshold,
        "moderate_threshold": args.moderate_threshold,
        "strong_threshold": args.strong_threshold,
        "ring_width": args.ring_width,
        "min_area": args.min_area,
        "max_area": args.max_area,
        "stain_type": args.stain_type,
        "marker": args.marker,
        "tile_size": args.tile_size,
        "magnification": args.magnification,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }


def run_analysis(image, params):
    st = params["stain_type"]
    try:
        if st == "nuclear":
            from patholib.analysis.ihc_nuclear import analyze_nuclear_ihc
            return analyze_nuclear_ihc(image, params)
        elif st == "membrane":
            from patholib.analysis.ihc_membrane import analyze_membrane_ihc
            return analyze_membrane_ihc(image, params)
        elif st == "cytoplasmic":
            from patholib.analysis.ihc_cytoplasmic import analyze_cytoplasmic_ihc
            return analyze_cytoplasmic_ihc(image, params)
    except (ImportError, ModuleNotFoundError):
        pass
    return _generic_ihc_analysis(image, params)


def _generic_ihc_analysis(image, params):
    """Fallback IHC analysis using H-DAB deconvolution and thresholding."""
    from skimage import measure, morphology, filters
    from scipy import ndimage
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    img_f = np.clip(image.astype(np.float64) / 255.0, 1e-6, 1.0)
    od = -np.log(img_f)
    hdab = np.array([[0.6500286, 0.7041306, 0.2860126],
                     [0.2688158, 0.5700040, 0.7780420]])
    hdab = hdab / np.linalg.norm(hdab, axis=1, keepdims=True)
    third = np.cross(hdab[0], hdab[1])
    third /= (np.linalg.norm(third) + 1e-12)
    M = np.vstack([hdab, third])
    deconv = od.reshape(-1, 3) @ np.linalg.inv(M)
    deconv = deconv.reshape(image.shape[0], image.shape[1], 3)
    hematoxylin = np.clip(deconv[:, :, 0], 0, None)
    dab = np.clip(deconv[:, :, 1], 0, None)

    h_norm = hematoxylin / (hematoxylin.max() + 1e-8)
    fg = h_norm[h_norm > 0.01]
    thresh = filters.threshold_otsu(fg) if fg.size > 100 else 0.2
    binary = h_norm > thresh
    binary = morphology.remove_small_objects(binary, min_size=params["min_area"])
    binary = morphology.remove_small_holes(binary, area_threshold=50)
    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=5, labels=binary, exclude_border=False)
    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i
    markers = ndimage.label(morphology.dilation(markers > 0, morphology.disk(2)))[0]
    labels = watershed(-dist, markers, mask=binary)

    weak_t = params["weak_threshold"]
    mod_t = params["moderate_threshold"]
    strong_t = params["strong_threshold"]
    cell_data = []
    gc = {0: 0, 1: 0, 2: 0, 3: 0}

    for rp in measure.regionprops(labels, intensity_image=dab):
        a = rp.area
        if a < params["min_area"] or a > params["max_area"]:
            continue
        md = rp.mean_intensity
        grade = 3 if md >= strong_t else (2 if md >= mod_t else (1 if md >= weak_t else 0))
        gc[grade] += 1
        cell_data.append({
            "centroid": (float(rp.centroid[0]), float(rp.centroid[1])),
            "area": int(a), "label": int(rp.label),
            "grade": grade, "intensity_mean": float(md),
            "cell_type": "positive" if grade > 0 else "negative",
        })

    total = len(cell_data)
    pos = sum(1 for c in cell_data if c["grade"] > 0)
    neg = total - pos
    hs = ((gc[1] + 2 * gc[2] + 3 * gc[3]) / total * 100) if total > 0 else 0.0
    pp = (pos / total * 100) if total > 0 else 0.0

    positive_intensities = [
        c["intensity_mean"] for c in cell_data if c["grade"] > 0
    ]
    representative_intensity = (
        float(np.median(positive_intensities)) if positive_intensities else 0.0
    )

    try:
        from patholib.scoring.allred_score import compute_allred
        allred, _, _ = compute_allred(pp, representative_intensity)
    except Exception:
        if pp <= 0:
            ps_val = 0
        elif pp < 1:
            ps_val = 1
        elif pp < 10:
            ps_val = 2
        elif pp < 33:
            ps_val = 3
        elif pp < 67:
            ps_val = 4
        else:
            ps_val = 5
        iscore = (
            3 if representative_intensity >= strong_t
            else (2 if representative_intensity >= mod_t else (1 if representative_intensity >= weak_t else 0))
        )
        allred = ps_val + iscore
        if allred == 1:
            allred = 0

    try:
        from patholib.viz.overlay import create_detection_overlay, blend_overlay
        ov = create_detection_overlay(image, labels, cell_data, overlay_type="ihc")
        blended = blend_overlay(image, ov, alpha=0.5)
    except Exception:
        blended = image.copy()

    return {
        "total_cells": total, "positive_cells": pos, "negative_cells": neg,
        "h_score": float(round(hs, 1)), "positive_percentage": float(round(pp, 1)),
        "allred_score": int(allred), "grade_counts": gc,
        "grade_percentages": {g: round(n / total * 100, 1) if total > 0 else 0.0
                              for g, n in gc.items()},
        "stain_type": params["stain_type"], "marker": params["marker"],
        "cell_data": cell_data, "labels": labels, "overlay": blended,
    }


def print_summary(results, elapsed):
    print("")
    print("=" * 60)
    print("IHC ANALYSIS RESULTS")
    print("=" * 60)
    print("  Marker:              " + str(results.get("marker", "Unknown")))
    print("  Stain type:          " + str(results.get("stain_type", "Unknown")))
    print("  Total cells:         " + str(results.get("total_cells", 0)))
    print("  Positive cells:      " + str(results.get("positive_cells", 0)))
    print("  Negative cells:      " + str(results.get("negative_cells", 0)))
    pp = results.get("positive_percentage", 0)
    print("  Positive pct:        " + str(round(pp, 1)) + "%")
    print("  H-Score:             " + str(round(results.get("h_score", 0), 1)))
    print("  Allred Score:        " + str(results.get("allred_score", 0)))
    gc = results.get("grade_counts", {})
    gp = results.get("grade_percentages", {})
    print("  Grade distribution:")
    names = ["Negative", "Weak", "Moderate", "Strong"]
    for g in range(4):
        print("    " + names[g] + ": " + str(gc.get(g, 0)) + " (" + str(gp.get(g, 0)) + "%)")
    print("  Elapsed: " + str(round(elapsed, 2)) + " seconds")
    print("=" * 60)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        print("Error: input not found: " + args.input, file=sys.stderr)
        sys.exit(1)
    print("Loading image: " + args.input)
    image = load_image(args.input)
    print("  Image shape: " + str(image.shape))
    if args.normalize_stain:
        print("Applying stain normalization...")
        image = apply_stain_normalization(image, args.stain_reference)
    params = build_params(args)
    print("Running " + args.stain_type + " IHC analysis for marker: " + args.marker)
    t0 = time.time()
    results = run_analysis(image, params)
    elapsed = time.time() - t0
    print_summary(results, elapsed)
    try:
        from patholib.viz.report import generate_ihc_report
    except (ImportError, ModuleNotFoundError):
        from report import generate_ihc_report
    rp = generate_ihc_report(results, args.input, params, args.output_dir,
                             save_overlay=args.save_overlay, save_csv=args.save_csv)
    print("Report saved to: " + rp)


if __name__ == "__main__":
    main()
