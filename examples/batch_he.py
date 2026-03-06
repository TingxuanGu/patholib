#!/usr/bin/env python3
"""
Batch H&E inflammation analysis for WYJ HE whole slide images (.mrxs).

Uses tile-based processing at level 2 (4x downsample, ~1.08 um/px) with
adaptive thresholding for nucleus detection.
"""

import os
import re
import json
import csv
import sys
from collections import defaultdict

import numpy as np
from PIL import Image
from skimage import filters, morphology, measure
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as skwatershed
from skimage.filters import gaussian
from scipy import ndimage
import warnings
warnings.filterwarnings('ignore')

from patholib.io.image_loader import load_image, get_wsi_info


INPUT_DIR = "/home/bio/桌面/Tingxuan Gu/analysis/WYJ HE-IHC results/WYJ HE"
OUTPUT_DIR = os.path.join(INPUT_DIR, "results")

# Processing parameters
TISSUE_LEVEL = 6  # For tissue detection
ANALYSIS_LEVEL = 2  # For cell detection (4x downsample, ~1.08 um/px)
TILE_SIZE = 2048  # Tile size at analysis level
TILE_OVERLAP = 0  # No overlap to avoid double-counting

# Cell detection parameters (tuned for level 2)
MIN_AREA = 8
MAX_AREA = 200
INFLAMMATORY_MAX_AREA = 40
INFLAMMATORY_MIN_CIRCULARITY = 0.7  # Stricter: lymphocytes are very round
INFLAMMATORY_H_OD_THRESH = 0.075  # Min hematoxylin OD for inflammatory cells (higher = darker staining)

# Inflammation scoring thresholds (cells per mm²)
MILD_THRESHOLD = 50
MODERATE_THRESHOLD = 200
SEVERE_THRESHOLD = 500

SCORE_LABELS = {0: "None", 1: "Mild", 2: "Moderate", 3: "Severe"}


def parse_group(filename):
    """Extract group name from filename like '4NQO+Low-Se+L-MSC-7.mrxs'."""
    m = re.match(r'^(.+)-(\d+)\.mrxs$', filename)
    if m:
        return m.group(1)
    return os.path.splitext(filename)[0]


def detect_tissue_region(wsi_path, level=6):
    """Detect tissue bounding box at low resolution."""
    thumb = load_image(wsi_path, level=level)
    gray = np.mean(thumb.astype(float), axis=2) / 255.0
    
    # Tissue mask: exclude black background and white glass
    tissue = (gray > 0.15) & (gray < 0.90)
    tissue = ndimage.binary_fill_holes(tissue)
    tissue = morphology.remove_small_objects(tissue, min_size=100)
    
    # Find bounding box
    rows = np.any(tissue, axis=1)
    cols = np.any(tissue, axis=0)
    if not rows.any() or not cols.any():
        return None
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    return (rmin, rmax, cmin, cmax)


def detect_nuclei_adaptive(img):
    """Detect nuclei using adaptive thresholding + watershed."""
    gray = np.mean(img.astype(float), axis=2)

    # Tissue mask
    tissue_mask = (gray > 30) & (gray < 230)
    if tissue_mask.sum() < 100:
        return np.zeros(img.shape[:2], dtype=np.int32), []

    # Hematoxylin channel via color deconvolution (higher OD = darker staining)
    hed = rgb2hed(img)
    hematoxylin_od = hed[:, :, 0]

    # Adaptive thresholding on inverted grayscale (keep gray for segmentation)
    inv = 255 - gray
    inv_smooth = gaussian(inv, sigma=1)
    local_thresh = filters.threshold_local(inv_smooth, 51, method='gaussian', offset=-10)
    binary = (inv_smooth > local_thresh) & tissue_mask

    # Clean up
    binary = morphology.remove_small_objects(binary, min_size=MIN_AREA)
    binary = morphology.remove_small_holes(binary, area_threshold=20)

    if binary.sum() == 0:
        return np.zeros(img.shape[:2], dtype=np.int32), []

    # Watershed segmentation
    dist = ndimage.distance_transform_edt(binary)
    coords = peak_local_max(dist, min_distance=3, labels=binary, exclude_border=False)

    markers = np.zeros(binary.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, 1):
        markers[r, c] = i

    labels = skwatershed(-dist, markers, mask=binary).astype(np.int32)

    # Extract cell data with hematoxylin OD measurement
    cell_data = []
    for rp in measure.regionprops(labels, intensity_image=hematoxylin_od):
        area = rp.area
        if area < MIN_AREA or area > MAX_AREA:
            continue

        peri = rp.perimeter
        circ = min((4 * np.pi * area) / (peri ** 2), 1.0) if peri > 0 else 0
        mean_int = rp.mean_intensity  # Hematoxylin OD: higher = darker

        # Classify cell type: inflammatory = small + round + high hematoxylin OD
        if (area <= INFLAMMATORY_MAX_AREA and
            circ >= INFLAMMATORY_MIN_CIRCULARITY and
            mean_int > INFLAMMATORY_H_OD_THRESH):
            cell_type = "inflammatory"
        else:
            cell_type = "parenchymal"

        cy, cx = rp.centroid
        cell_data.append({
            "centroid": (cy, cx),
            "area": area,
            "circularity": circ,
            "hematoxylin_od": mean_int,
            "cell_type": cell_type,
        })

    return labels, cell_data


def analyze_wsi_tiled(wsi_path, info):
    """Analyze WSI using tile-based processing."""
    # Detect tissue region at low resolution
    tissue_bbox = detect_tissue_region(wsi_path, level=TISSUE_LEVEL)
    if tissue_bbox is None:
        print("  No tissue detected")
        return None
    
    rmin, rmax, cmin, cmax = tissue_bbox
    print(f"  Tissue bbox at level {TISSUE_LEVEL}: rows [{rmin},{rmax}], cols [{cmin},{cmax}]")
    
    # Convert to level-0 coordinates and analysis-level tile dimensions
    ds_tissue = info['level_downsamples'][TISSUE_LEVEL]
    ds_analysis = info['level_downsamples'][ANALYSIS_LEVEL]

    # Level-0 coordinates of tissue bounding box
    y0_l0 = int(rmin * ds_tissue)
    y1_l0 = int((rmax + 1) * ds_tissue)
    x0_l0 = int(cmin * ds_tissue)
    x1_l0 = int((cmax + 1) * ds_tissue)

    # Tissue size in analysis-level pixels
    tissue_h = int((y1_l0 - y0_l0) / ds_analysis)
    tissue_w = int((x1_l0 - x0_l0) / ds_analysis)
    print(f"  Tissue region at level {ANALYSIS_LEVEL}: {tissue_w}x{tissue_h} pixels")

    # Generate tiles
    all_cells = []
    total_tissue_area = 0

    stride = TILE_SIZE - TILE_OVERLAP
    n_tiles_y = max(1, (tissue_h + stride - 1) // stride)
    n_tiles_x = max(1, (tissue_w + stride - 1) // stride)
    print(f"  Processing {n_tiles_y}x{n_tiles_x} = {n_tiles_y*n_tiles_x} tiles...")

    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            # Level-0 coordinates for read_region location
            tile_x_l0 = x0_l0 + int(tx * stride * ds_analysis)
            tile_y_l0 = y0_l0 + int(ty * stride * ds_analysis)

            # Load tile (region: x,y in level-0 coords; w,h at analysis level)
            try:
                tile = load_image(wsi_path, level=ANALYSIS_LEVEL,
                                region=(tile_x_l0, tile_y_l0, TILE_SIZE, TILE_SIZE))
            except:
                continue

            # Detect nuclei
            labels, cells = detect_nuclei_adaptive(tile)

            # Adjust coordinates to global (analysis-level)
            tile_y_al = int(ty * stride)
            tile_x_al = int(tx * stride)
            for cell in cells:
                cy, cx = cell["centroid"]
                cell["centroid"] = (tile_y_al + cy, tile_x_al + cx)
            
            all_cells.extend(cells)
            
            # Count tissue area
            gray = np.mean(tile.astype(float), axis=2)
            tissue_px = np.sum((gray > 30) & (gray < 230))
            total_tissue_area += tissue_px
    
    print(f"  Detected {len(all_cells)} cells total")
    
    # Compute metrics
    n_inflammatory = sum(1 for c in all_cells if c["cell_type"] == "inflammatory")
    n_parenchymal = sum(1 for c in all_cells if c["cell_type"] == "parenchymal")
    
    # Convert tissue area to mm²
    mpp_base = float(info.get('mpp_x') or 0.271)
    ds_analysis = info['level_downsamples'][ANALYSIS_LEVEL]
    mpp = mpp_base * ds_analysis
    tissue_area_mm2 = (total_tissue_area * mpp * mpp) / 1e6
    
    inflammatory_density = n_inflammatory / tissue_area_mm2 if tissue_area_mm2 > 0 else 0
    
    # Inflammation score
    if inflammatory_density < MILD_THRESHOLD:
        score = 0
    elif inflammatory_density < MODERATE_THRESHOLD:
        score = 1
    elif inflammatory_density < SEVERE_THRESHOLD:
        score = 2
    else:
        score = 3
    
    return {
        "total_nuclei": len(all_cells),
        "inflammatory_cells": n_inflammatory,
        "parenchymal_cells": n_parenchymal,
        "tissue_area_mm2": tissue_area_mm2,
        "inflammatory_density": inflammatory_density,
        "inflammation_score": score,
        "cell_data": all_cells,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Discover all .mrxs files
    mrxs_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith('.mrxs'))
    print(f"Found {len(mrxs_files)} .mrxs files\n")
    
    all_results = {}
    group_results = defaultdict(list)
    
    for i, fname in enumerate(mrxs_files, 1):
        path = os.path.join(INPUT_DIR, fname)
        name = os.path.splitext(fname)[0]
        group = parse_group(fname)
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(mrxs_files)}] {fname}  (group: {group})")
        print(f"{'='*60}")
        
        # Get WSI info
        try:
            info = get_wsi_info(path)
            print(f"  WSI: {info['level_count']} levels, base {info['dimensions'][0]}")
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue
        
        # Analyze
        try:
            results = analyze_wsi_tiled(path, info)
            if results is None:
                continue
        except Exception as e:
            print(f"  ERROR in analysis: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        total = results["total_nuclei"]
        n_inf = results["inflammatory_cells"]
        n_par = results["parenchymal_cells"]
        density = results["inflammatory_density"]
        score = results["inflammation_score"]
        
        print(f"  Total nuclei:         {total}")
        print(f"  Inflammatory cells:   {n_inf}")
        print(f"  Parenchymal cells:    {n_par}")
        print(f"  Tissue area:          {results['tissue_area_mm2']:.2f} mm²")
        print(f"  Inflammatory density: {density:.2f} cells/mm²")
        print(f"  Inflammation score:   {score} ({SCORE_LABELS.get(score, '?')})")
        
        slide_result = {
            "group": group,
            "total_nuclei": total,
            "inflammatory_cells": n_inf,
            "parenchymal_cells": n_par,
            "tissue_area_mm2": round(results["tissue_area_mm2"], 4),
            "inflammatory_density": round(density, 4),
            "inflammation_score": score,
            "inflammation_label": SCORE_LABELS.get(score, "?"),
        }
        all_results[name] = slide_result
        group_results[group].append(slide_result)
        
        # Save per-slide JSON report
        report = {k: v for k, v in results.items() if k != "cell_data"}
        report["group"] = group
        report_path = os.path.join(OUTPUT_DIR, f"{name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save cell data CSV
        cell_data = results.get("cell_data", [])
        if cell_data:
            csv_path = os.path.join(OUTPUT_DIR, f"{name}_cells.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["centroid", "area", "circularity", "hematoxylin_od", "cell_type"])
                writer.writeheader()
                writer.writerows(cell_data)
            print(f"  Saved: {csv_path}")
        
        # Free memory
        import gc
        gc.collect()
    
    # --- Per-slide comparison ---
    print(f"\n{'='*70}")
    print("PER-SLIDE RESULTS")
    print(f"{'='*70}")
    print(f"{'Slide':<30} {'Group':<25} {'Nuclei':>7} {'Inflam':>7} {'Density':>8} {'Score':>6}")
    print("-" * 85)
    for name, r in all_results.items():
        print(f"{name:<30} {r['group']:<25} {r['total_nuclei']:>7} "
              f"{r['inflammatory_cells']:>7} {r['inflammatory_density']:>8.2f} "
              f"{r['inflammation_score']:>3} ({r['inflammation_label']})")
    
    # --- Group summary with mean±SD ---
    print(f"\n{'='*70}")
    print("GROUP COMPARISON (mean ± SD)")
    print(f"{'='*70}")
    print(f"{'Group':<25} {'n':>3} {'Nuclei':>14} {'Inflam':>14} {'Density':>16} {'Score':>12}")
    print("-" * 90)
    
    group_summary = {}
    group_order = ["con", "4NQO", "4NQO+Low-Se", "4NQO+Low-Se+L-MSC", "4NQO+Low-Se+Se-Met"]
    all_groups = list(dict.fromkeys(group_order + list(group_results.keys())))
    
    for grp in all_groups:
        if grp not in group_results:
            continue
        recs = group_results[grp]
        n = len(recs)
        nuclei = [r["total_nuclei"] for r in recs]
        inflam = [r["inflammatory_cells"] for r in recs]
        dens = [r["inflammatory_density"] for r in recs]
        scores = [r["inflammation_score"] for r in recs]
        
        summary = {
            "n": n,
            "total_nuclei_mean": round(float(np.mean(nuclei)), 1),
            "total_nuclei_sd": round(float(np.std(nuclei, ddof=1)) if n > 1 else 0, 1),
            "inflammatory_cells_mean": round(float(np.mean(inflam)), 1),
            "inflammatory_cells_sd": round(float(np.std(inflam, ddof=1)) if n > 1 else 0, 1),
            "inflammatory_density_mean": round(float(np.mean(dens)), 4),
            "inflammatory_density_sd": round(float(np.std(dens, ddof=1)) if n > 1 else 0, 4),
            "inflammation_score_mean": round(float(np.mean(scores)), 2),
            "inflammation_score_sd": round(float(np.std(scores, ddof=1)) if n > 1 else 0, 2),
            "slides": [r for r in recs],
        }
        group_summary[grp] = summary
        
        nuc_str = f"{summary['total_nuclei_mean']:.0f}±{summary['total_nuclei_sd']:.0f}"
        inf_str = f"{summary['inflammatory_cells_mean']:.0f}±{summary['inflammatory_cells_sd']:.0f}"
        den_str = f"{summary['inflammatory_density_mean']:.2f}±{summary['inflammatory_density_sd']:.2f}"
        sco_str = f"{summary['inflammation_score_mean']:.1f}±{summary['inflammation_score_sd']:.1f}"
        print(f"{grp:<25} {n:>3} {nuc_str:>14} {inf_str:>14} {den_str:>16} {sco_str:>12}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "per_slide": all_results,
            "per_group": {k: {kk: vv for kk, vv in v.items() if kk != "slides"}
                          for k, v in group_summary.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
