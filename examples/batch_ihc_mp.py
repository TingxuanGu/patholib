#!/usr/bin/env python3
"""
Batch GPX4 IHC analysis with MULTIPROCESSING for 64-core CPU.

Parallelizes tile processing within each sample using multiprocessing.Pool.
"""

import argparse
import os
import re
import json
import csv
import sys
import gc
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import warnings
warnings.filterwarnings('ignore')

EXAMPLES_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from PIL import Image
    from skimage import measure, morphology
    from scipy import ndimage
    IMPORT_ERROR = None
except ImportError as exc:
    Image = None
    measure = morphology = ndimage = None
    IMPORT_ERROR = exc

from patholib.io.image_loader import load_image, get_wsi_info
from patholib.stain.color_deconv import separate_stains

# Processing parameters
TISSUE_LEVEL = 6
ANALYSIS_LEVEL = 2
TILE_SIZE = 2048
TILE_OVERLAP = 0

# Cell detection parameters
MIN_AREA = 5
MAX_AREA = 200

# Cytoplasmic measurement
CYTO_EXPANSION = 3

# DAB intensity grading thresholds
WEAK_THRESHOLD = 0.10
MODERATE_THRESHOLD = 0.25
STRONG_THRESHOLD = 0.45

# Marker
MARKER = "GPX4"

# Multiprocessing
N_WORKERS = min(60, cpu_count())  # Use up to 60 cores

# Group ordering
GROUP_ORDER = ["con", "4NQO", "4NQO+Low-Se", "4NQO+Low-Se+L-MSC", "4NQO+Low-Se+Se-Met"]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Batch GPX4 IHC analysis for .mrxs whole-slide images with multiprocessing."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input .mrxs slides",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for reports, overlays, and CSVs (default: <input-dir>/results)",
    )
    return parser


def _resolve_io_dirs(args):
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")
    output_dir = (
        os.path.abspath(args.output_dir)
        if args.output_dir
        else os.path.join(input_dir, "results")
    )
    return input_dir, output_dir


def _require_dependencies():
    if IMPORT_ERROR is not None:
        raise RuntimeError(
            "examples/batch_ihc_mp.py requires optional imaging dependencies. "
            "Install Pillow, scipy, and scikit-image before running it."
        ) from IMPORT_ERROR


def parse_group(filename):
    """Extract group name from filename."""
    m = re.match(r'^(.+)-(\d+)\.mrxs$', filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]


def detect_tissue_region(wsi_path, level=6):
    """Detect tissue bounding box at low resolution."""
    _require_dependencies()
    thumb = load_image(wsi_path, level=level)
    gray = np.mean(thumb.astype(float), axis=2) / 255.0

    tissue = (gray > 0.15) & (gray < 0.90)
    tissue = ndimage.binary_fill_holes(tissue)
    tissue = morphology.remove_small_objects(tissue, min_size=100)

    rows = np.any(tissue, axis=1)
    cols = np.any(tissue, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (rmin, rmax, cmin, cmax)


def detect_nuclei_watershed_ihc(hematoxylin_od, rgb_tile):
    """Detect nuclei from hematoxylin OD using watershed."""
    _require_dependencies()
    from skimage.filters import threshold_otsu
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed as skwatershed

    h, w = hematoxylin_od.shape

    hem = np.clip(hematoxylin_od, 0, None)
    hmax = hem.max()
    if hmax <= 0:
        return np.zeros((h, w), dtype=np.int32)
    hem_norm = hem / hmax

    fg_mask = hem_norm > 0.01
    fg_values = hem_norm[fg_mask]
    if fg_values.size < 50:
        return np.zeros((h, w), dtype=np.int32)

    try:
        thresh = threshold_otsu(fg_values)
    except ValueError:
        return np.zeros((h, w), dtype=np.int32)

    binary = (hem_norm > thresh) & fg_mask
    binary = morphology.remove_small_objects(binary, min_size=MIN_AREA)
    binary = morphology.remove_small_holes(binary, area_threshold=50)

    if binary.sum() < MIN_AREA:
        return np.zeros((h, w), dtype=np.int32)

    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=3, labels=binary, exclude_border=False)

    if len(coords) == 0:
        return np.zeros((h, w), dtype=np.int32)

    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    labels = skwatershed(-dist, markers, mask=binary).astype(np.int32)
    return labels


def measure_cytoplasmic_dab(labels, dab_channel):
    """Measure DAB intensity in cytoplasmic ring."""
    _require_dependencies()
    cell_data = []
    grade_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    for rp in measure.regionprops(labels):
        area = rp.area
        if area < MIN_AREA or area > MAX_AREA:
            continue

        cell_mask = labels == rp.label
        cell_dilated = morphology.dilation(cell_mask, morphology.disk(CYTO_EXPANSION))
        cyto_mask = cell_dilated & ~cell_mask
        cyto_pixels = dab_channel[cyto_mask]

        if cyto_pixels.size == 0:
            mean_dab = 0.0
        else:
            mean_dab = float(np.mean(cyto_pixels))

        if mean_dab >= STRONG_THRESHOLD:
            grade = 3
        elif mean_dab >= MODERATE_THRESHOLD:
            grade = 2
        elif mean_dab >= WEAK_THRESHOLD:
            grade = 1
        else:
            grade = 0

        grade_counts[grade] += 1

        cy, cx = rp.centroid
        cell_data.append({
            "centroid": (float(cy), float(cx)),
            "area": int(area),
            "grade": grade,
            "cytoplasmic_intensity": round(mean_dab, 4),
            "cell_type": "positive" if grade > 0 else "negative",
        })

    return cell_data, grade_counts


def process_single_tile(args):
    """Worker function to process one tile (for multiprocessing).

    Returns lightweight data only (no large arrays through IPC).
    """
    _require_dependencies()
    wsi_path, tile_x_l0, tile_y_l0, tx, ty, stride, ds_analysis = args

    try:
        tile = load_image(wsi_path, level=ANALYSIS_LEVEL,
                          region=(tile_x_l0, tile_y_l0, TILE_SIZE, TILE_SIZE))
    except Exception:
        return None

    # Check tissue content
    gray = np.mean(tile.astype(float), axis=2)
    tissue_px = int(np.sum((gray > 30) & (gray < 230)))
    if tissue_px < 100:
        return None

    # H-DAB color deconvolution
    hematoxylin, dab, _ = separate_stains(tile, stain_type='hdab')

    # Detect nuclei
    labels = detect_nuclei_watershed_ihc(hematoxylin, tile)

    # Measure cytoplasmic DAB
    cells, grade_counts = measure_cytoplasmic_dab(labels, dab)

    # Adjust coordinates to global (analysis-level)
    tile_y_al = int(ty * stride)
    tile_x_al = int(tx * stride)
    for cell in cells:
        cy, cx = cell["centroid"]
        cell["centroid"] = (tile_y_al + cy, tile_x_al + cx)

    # Return lightweight data only - no tile/labels arrays
    return {
        "cells": cells,
        "grade_counts": grade_counts,
        "tissue_px": tissue_px,
    }


def analyze_wsi_tiled_mp(wsi_path, info):
    """Analyze WSI using multiprocessing for tile processing."""
    _require_dependencies()
    tissue_bbox = detect_tissue_region(wsi_path, level=TISSUE_LEVEL)
    if tissue_bbox is None:
        print("  No tissue detected")
        return None

    rmin, rmax, cmin, cmax = tissue_bbox
    print(f"  Tissue bbox at level {TISSUE_LEVEL}: rows [{rmin},{rmax}], cols [{cmin},{cmax}]")

    ds_tissue = info['level_downsamples'][TISSUE_LEVEL]
    ds_analysis = info['level_downsamples'][ANALYSIS_LEVEL]

    y0_l0 = int(rmin * ds_tissue)
    y1_l0 = int((rmax + 1) * ds_tissue)
    x0_l0 = int(cmin * ds_tissue)
    x1_l0 = int((cmax + 1) * ds_tissue)

    tissue_h = int((y1_l0 - y0_l0) / ds_analysis)
    tissue_w = int((x1_l0 - x0_l0) / ds_analysis)
    print(f"  Tissue region at level {ANALYSIS_LEVEL}: {tissue_w}x{tissue_h} pixels")

    stride = TILE_SIZE - TILE_OVERLAP
    n_tiles_y = max(1, (tissue_h + stride - 1) // stride)
    n_tiles_x = max(1, (tissue_w + stride - 1) // stride)
    total_tiles = n_tiles_y * n_tiles_x
    print(f"  Processing {n_tiles_y}x{n_tiles_x} = {total_tiles} tiles with {N_WORKERS} workers...")

    # Prepare tile arguments
    tile_args = []
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tile_x_l0 = x0_l0 + int(tx * stride * ds_analysis)
            tile_y_l0 = y0_l0 + int(ty * stride * ds_analysis)
            tile_args.append((wsi_path, tile_x_l0, tile_y_l0, tx, ty, stride, ds_analysis))

    # Process tiles in parallel
    all_cells = []
    total_tissue_area = 0
    grade_counts_total = {0: 0, 1: 0, 2: 0, 3: 0}

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(process_single_tile, tile_args)

    print(f"  Aggregating results from {len(results)} tiles...")
    for result in results:
        if result is None:
            continue

        all_cells.extend(result["cells"])
        for g in range(4):
            grade_counts_total[g] += result["grade_counts"][g]
        total_tissue_area += result["tissue_px"]

    # Create overlay from low-res WSI + cell markers
    print(f"  Creating overlay (loading level {ANALYSIS_LEVEL + 2} background)...")
    try:
        # Load a lower-res version for overlay background
        overlay_level = min(ANALYSIS_LEVEL + 2, info['level_count'] - 1)
        ds_overlay = info['level_downsamples'][overlay_level]
        bg = load_image(wsi_path, level=overlay_level,
                        region=(x0_l0, y0_l0,
                                int((x1_l0 - x0_l0) / ds_overlay),
                                int((y1_l0 - y0_l0) / ds_overlay)))
        # Scale factor from analysis-level coords to overlay coords
        scale = ds_analysis / ds_overlay

        overlay_thumb = bg.copy()
        colors = {0: (180, 180, 180), 1: (0, 200, 0), 2: (255, 200, 0), 3: (255, 0, 0)}
        for cell in all_cells:
            cy, cx = cell["centroid"]
            oy = int(cy * scale)
            ox = int(cx * scale)
            grade = cell["grade"]
            color = colors.get(grade, (180, 180, 180))

            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    yy, xx = oy + dy, ox + dx
                    if 0 <= yy < overlay_thumb.shape[0] and 0 <= xx < overlay_thumb.shape[1]:
                        overlay_thumb[yy, xx] = color
    except Exception as e:
        print(f"  Warning: overlay creation failed ({e}), using blank")
        overlay_thumb = np.zeros((100, 100, 3), dtype=np.uint8)

    print(f"  Detected {len(all_cells)} cells total")

    # Compute metrics
    total = len(all_cells)
    pos = sum(1 for c in all_cells if c["grade"] > 0)
    neg = total - pos
    h_score = ((grade_counts_total[1] + 2 * grade_counts_total[2] +
                3 * grade_counts_total[3]) / total * 100) if total > 0 else 0.0
    pos_pct = (pos / total * 100) if total > 0 else 0.0

    mpp_base = float(info.get('mpp_x') or 0.271)
    mpp = mpp_base * ds_analysis
    tissue_area_mm2 = (total_tissue_area * mpp * mpp) / 1e6

    grade_pcts = {g: round(n / total * 100, 1) if total > 0 else 0.0
                  for g, n in grade_counts_total.items()}

    return {
        "total_cells": total,
        "positive_cells": pos,
        "negative_cells": neg,
        "h_score": round(h_score, 1),
        "positive_percentage": round(pos_pct, 1),
        "grade_counts": grade_counts_total,
        "grade_percentages": grade_pcts,
        "tissue_area_mm2": round(tissue_area_mm2, 4),
        "stain_type": "cytoplasmic",
        "marker": MARKER,
        "cell_data": all_cells,
        "overlay": overlay_thumb,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_dir, output_dir = _resolve_io_dirs(args)
    _require_dependencies()

    os.makedirs(output_dir, exist_ok=True)

    mrxs_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.mrxs'))
    print(f"Found {len(mrxs_files)} .mrxs files")
    print(f"Using {N_WORKERS} worker processes\n")

    all_results = {}
    group_results = defaultdict(list)

    for i, fname in enumerate(mrxs_files, 1):
        path = os.path.join(input_dir, fname)
        name = os.path.splitext(fname)[0]
        group = parse_group(fname)

        print(f"\n{'='*60}")
        print(f"[{i}/{len(mrxs_files)}] {fname}  (group: {group})")
        print(f"{'='*60}")

        try:
            info = get_wsi_info(path)
            print(f"  WSI: {info['level_count']} levels, base {info['dimensions'][0]}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        try:
            results = analyze_wsi_tiled_mp(path, info)
            if results is None:
                continue
        except Exception as e:
            print(f"  ERROR in analysis: {e}")
            import traceback
            traceback.print_exc()
            continue

        total = results["total_cells"]
        pos = results["positive_cells"]
        neg = results["negative_cells"]
        hs = results["h_score"]
        pp = results["positive_percentage"]
        gc_map = results["grade_counts"]

        print(f"  Total cells:          {total}")
        print(f"  Positive cells:       {pos}")
        print(f"  Negative cells:       {neg}")
        print(f"  H-score:              {hs}")
        print(f"  Positive %:           {pp}")
        print(f"  Grade distribution:   0:{gc_map[0]}  1:{gc_map[1]}  2:{gc_map[2]}  3:{gc_map[3]}")

        slide_result = {
            "group": group,
            "total_cells": total,
            "positive_cells": pos,
            "negative_cells": neg,
            "h_score": hs,
            "positive_percentage": pp,
            "grade_counts": {str(k): v for k, v in gc_map.items()},
            "grade_percentages": results["grade_percentages"],
            "tissue_area_mm2": results["tissue_area_mm2"],
        }
        all_results[name] = slide_result
        group_results[group].append(slide_result)

        # Save per-slide JSON
        report = {k: v for k, v in slide_result.items()}
        report["stain_type"] = "cytoplasmic"
        report["marker"] = MARKER
        report_path = os.path.join(output_dir, f"{name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save cell data CSV
        cell_data = results.get("cell_data", [])
        if cell_data:
            csv_path = os.path.join(output_dir, f"{name}_cells.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "centroid", "area", "grade", "cytoplasmic_intensity", "cell_type"
                ])
                writer.writeheader()
                writer.writerows(cell_data)

        # Save overlay
        overlay = results.get("overlay")
        if overlay is not None:
            overlay_path = os.path.join(output_dir, f"{name}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)

        del results
        gc.collect()

    # Per-slide comparison
    print(f"\n{'='*70}")
    print("PER-SLIDE RESULTS")
    print(f"{'='*70}")
    print(f"{'Slide':<30} {'Group':<25} {'Cells':>7} {'Pos':>5} {'Pos%':>7} {'H-score':>8}")
    print("-" * 85)
    for name, r in all_results.items():
        print(f"{name:<30} {r['group']:<25} {r['total_cells']:>7} "
              f"{r['positive_cells']:>5} {r['positive_percentage']:>6.1f}% "
              f"{r['h_score']:>7.1f}")

    # Group summary
    print(f"\n{'='*70}")
    print("GROUP COMPARISON (mean ± SD)")
    print(f"{'='*70}")
    print(f"{'Group':<25} {'n':>3} {'H-score':>16} {'Pos%':>16} {'Cells':>14}")
    print("-" * 80)

    group_summary = {}
    all_groups = list(dict.fromkeys(GROUP_ORDER + list(group_results.keys())))

    for grp in all_groups:
        if grp not in group_results:
            continue
        recs = group_results[grp]
        n = len(recs)
        h_scores = [r["h_score"] for r in recs]
        pos_pcts = [r["positive_percentage"] for r in recs]
        cells = [r["total_cells"] for r in recs]

        summary = {
            "n": n,
            "h_score_mean": round(float(np.mean(h_scores)), 1),
            "h_score_sd": round(float(np.std(h_scores, ddof=1)) if n > 1 else 0, 1),
            "positive_percentage_mean": round(float(np.mean(pos_pcts)), 1),
            "positive_percentage_sd": round(float(np.std(pos_pcts, ddof=1)) if n > 1 else 0, 1),
            "total_cells_mean": round(float(np.mean(cells)), 1),
            "total_cells_sd": round(float(np.std(cells, ddof=1)) if n > 1 else 0, 1),
        }

        for g in range(4):
            vals = [r["grade_counts"].get(str(g), 0) for r in recs]
            summary[f"grade_{g}_mean"] = round(float(np.mean(vals)), 1)
            summary[f"grade_{g}_sd"] = round(float(np.std(vals, ddof=1)) if n > 1 else 0, 1)

        group_summary[grp] = summary

        hs_str = f"{summary['h_score_mean']:.1f}±{summary['h_score_sd']:.1f}"
        pp_str = f"{summary['positive_percentage_mean']:.1f}±{summary['positive_percentage_sd']:.1f}"
        tc_str = f"{summary['total_cells_mean']:.0f}±{summary['total_cells_sd']:.0f}"
        print(f"{grp:<25} {n:>3} {hs_str:>16} {pp_str:>16} {tc_str:>14}")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "per_slide": all_results,
            "per_group": group_summary,
            "parameters": {
                "analysis_level": ANALYSIS_LEVEL,
                "tile_size": TILE_SIZE,
                "min_area": MIN_AREA,
                "max_area": MAX_AREA,
                "cyto_expansion": CYTO_EXPANSION,
                "weak_threshold": WEAK_THRESHOLD,
                "moderate_threshold": MODERATE_THRESHOLD,
                "strong_threshold": STRONG_THRESHOLD,
                "marker": MARKER,
                "detection_method": "watershed",
                "n_workers": N_WORKERS,
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
