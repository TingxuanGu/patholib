#!/usr/bin/env python3
"""
analyze_he.py - CLI entry point for H&E image analysis.

Usage:
    python analyze_he.py --input slide.tif --mode inflammation
    python analyze_he.py --input slide.png --mode area-ratio
    python analyze_he.py --input slide.tif --mode both --save-heatmap --save-overlay
"""

import argparse
import os
import sys
import time
import numpy as np


def build_parser():
    p = argparse.ArgumentParser(description="Quantitative H&E image analysis")
    p.add_argument("--input", required=True, help="Input image path")
    p.add_argument("--mode", required=True,
                   choices=["inflammation", "area-ratio", "both"],
                   help="Analysis mode")
    p.add_argument("--detection-method", default="cellpose",
                   choices=["cellpose", "watershed"])
    p.add_argument("--mpp", type=float, default=None,
                   help="Microns per pixel (auto-detect if None)")
    p.add_argument("--inflammatory-max-area", type=int, default=80)
    p.add_argument("--inflammatory-min-circularity", type=float, default=0.7)
    p.add_argument("--grid-size", type=int, default=200,
                   help="Grid cell size in micrometers")
    p.add_argument("--classifier-path", default=None,
                   help="Path to trained region classifier model")
    p.add_argument("--normalize-stain", action="store_true")
    p.add_argument("--stain-reference", default=None)
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--save-overlay", action="store_true")
    p.add_argument("--save-heatmap", action="store_true")
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


def build_inflammation_params(args):
    return {
        "detection_method": args.detection_method,
        "inflammatory_max_area": args.inflammatory_max_area,
        "inflammatory_min_circularity": args.inflammatory_min_circularity,
        "grid_size_um": args.grid_size,
        "mpp": args.mpp,
    }


def build_area_ratio_params(args):
    params = {"mpp": args.mpp}
    if args.classifier_path:
        params["method"] = "classifier"
        params["classifier_path"] = args.classifier_path
    else:
        params["method"] = "threshold"
    return params


def run_inflammation(image, params):
    from patholib.analysis.he_inflammation import analyze_inflammation
    return analyze_inflammation(image, params)


def run_area_ratio(image, params):
    from patholib.analysis.he_area_ratio import analyze_area_ratio
    return analyze_area_ratio(image, params)


def make_overlays(image, results, mode, args):
    """Generate overlay and heatmap images if requested."""
    combined = dict(results)

    if args.save_overlay:
        try:
            from patholib.viz.overlay import (create_detection_overlay,
                                              create_segmentation_overlay,
                                              blend_overlay)
            if mode in ("inflammation", "both") and "labels" in results:
                cell_data = results.get("cell_data", [])
                det_ov = create_detection_overlay(image, results["labels"],
                                                  cell_data, overlay_type="he")
                blended = blend_overlay(image, det_ov, alpha=0.5)
                combined["overlay"] = blended

            if mode in ("area-ratio", "both") and "segmentation_mask" in results:
                seg_ov = create_segmentation_overlay(image,
                                                     results["segmentation_mask"],
                                                     alpha=0.4)
                seg_blended = blend_overlay(image, seg_ov, alpha=0.5)
                combined["segmentation_overlay"] = seg_blended
        except Exception as exc:
            import warnings
            warnings.warn("Could not create overlay: " + str(exc))

    if args.save_heatmap and mode in ("inflammation", "both"):
        try:
            from patholib.viz.heatmap import (create_density_heatmap,
                                              overlay_heatmap)
            cell_data = results.get("cell_data", [])
            inflam_coords = [(c["centroid"][0], c["centroid"][1])
                             for c in cell_data if c.get("cell_type") == "inflammatory"]
            if inflam_coords:
                hm = create_density_heatmap(image.shape, inflam_coords, sigma=50)
                hm_img = overlay_heatmap(image, hm, alpha=0.5, cmap="jet")
                combined["heatmap"] = hm_img
        except Exception as exc:
            import warnings
            warnings.warn("Could not create heatmap: " + str(exc))

    return combined


def print_inflammation_summary(results):
    print("  --- Inflammation ---")
    print("  Total nuclei:        " + str(results.get("total_nuclei", 0)))
    print("  Inflammatory cells:  " + str(results.get("inflammatory_cells", 0)))
    print("  Parenchymal cells:   " + str(results.get("parenchymal_cells", 0)))
    density = results.get("inflammatory_density", 0)
    print("  Inflammatory density: " + str(round(density, 2)))
    score = results.get("inflammation_score", 0)
    score_labels = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}
    print("  Inflammation score:  " + str(score) + " (" + score_labels.get(score, "Unknown") + ")")


def print_area_ratio_summary(results):
    print("  --- Area Ratio ---")
    print("  Tissue area (px):    " + str(results.get("tissue_area_px", 0)))
    um2 = results.get("tissue_area_um2")
    if um2 is not None:
        print("  Tissue area (um2):   " + str(round(um2, 1)))
    print("  Tumor ratio:         " + str(results.get("tumor_ratio", 0)) + "%")
    print("  Necrosis ratio:      " + str(results.get("necrosis_ratio", 0)) + "%")
    regions = results.get("regions", {})
    print("  Regions:")
    for name, info in regions.items():
        pct = info.get("percentage", 0)
        print("    " + name + ": " + str(info.get("area_px", 0)) + " px (" + str(pct) + "%)")


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

    all_results = {}
    mode = args.mode
    t0 = time.time()

    if mode in ("inflammation", "both"):
        print("Running inflammation analysis...")
        i_params = build_inflammation_params(args)
        i_results = run_inflammation(image, i_params)
        if mode == "both":
            all_results["inflammation"] = i_results
        else:
            all_results = i_results

    if mode in ("area-ratio", "both"):
        print("Running area ratio analysis...")
        a_params = build_area_ratio_params(args)
        a_results = run_area_ratio(image, a_params)
        if mode == "both":
            all_results["area_ratio"] = a_results
        else:
            all_results = a_results

    elapsed = time.time() - t0

    # Generate overlays/heatmaps
    flat_results = {}
    if mode == "both":
        for k, v in all_results.get("inflammation", {}).items():
            flat_results[k] = v
        for k, v in all_results.get("area_ratio", {}).items():
            flat_results[k] = v
        flat_results["inflammation"] = all_results.get("inflammation", {})
        flat_results["area_ratio"] = all_results.get("area_ratio", {})
    else:
        flat_results = all_results

    flat_results = make_overlays(image, flat_results, mode, args)

    # Print summary
    print("")
    print("=" * 60)
    print("H&E ANALYSIS RESULTS")
    print("=" * 60)
    if mode in ("inflammation", "both"):
        i_res = all_results.get("inflammation", all_results)
        print_inflammation_summary(i_res)
    if mode in ("area-ratio", "both"):
        a_res = all_results.get("area_ratio", all_results)
        print_area_ratio_summary(a_res)
    print("  Elapsed: " + str(round(elapsed, 2)) + " seconds")
    print("=" * 60)

    # Generate report
    try:
        from patholib.viz.report import generate_he_report
    except (ImportError, ModuleNotFoundError):
        from report import generate_he_report
    report_results = flat_results
    report_params = {}
    if mode in ("inflammation", "both"):
        report_params.update(build_inflammation_params(args))
    if mode in ("area-ratio", "both"):
        report_params.update(build_area_ratio_params(args))

    rp = generate_he_report(report_results, args.input, report_params,
                            args.output_dir, mode,
                            save_overlay=args.save_overlay,
                            save_heatmap=args.save_heatmap,
                            save_csv=args.save_csv)
    print("Report saved to: " + rp)


if __name__ == "__main__":
    main()
