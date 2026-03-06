#!/usr/bin/env python3
"""Quick test: run recalibrated H&E inflammation analysis on con-1 and 4NQO-2."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from batch_he_wyjhe_v2 import (
    analyze_wsi_tiled, SCORE_LABELS,
    INFLAMMATORY_MIN_CIRCULARITY, INFLAMMATORY_INTENSITY_THRESH
)
from patholib.io.image_loader import get_wsi_info
import numpy as np

INPUT_DIR = "/home/bio/桌面/Tingxuan Gu/analysis/WYJ HE-IHC results/WYJ HE"
TEST_SLIDES = ["con-1.mrxs", "4NQO-2.mrxs"]

print(f"Recalibration parameters:")
print(f"  INFLAMMATORY_MIN_CIRCULARITY = {INFLAMMATORY_MIN_CIRCULARITY}")
print(f"  INFLAMMATORY_INTENSITY_THRESH = {INFLAMMATORY_INTENSITY_THRESH}")
print()

for fname in TEST_SLIDES:
    path = os.path.join(INPUT_DIR, fname)
    name = os.path.splitext(fname)[0]

    print(f"{'='*60}")
    print(f"Analyzing: {fname}")
    print(f"{'='*60}")

    info = get_wsi_info(path)
    print(f"  WSI: {info['level_count']} levels, base {info['dimensions'][0]}")

    results = analyze_wsi_tiled(path, info)
    if results is None:
        print("  No tissue detected!")
        continue

    total = results["total_nuclei"]
    n_inf = results["inflammatory_cells"]
    n_par = results["parenchymal_cells"]
    density = results["inflammatory_density"]
    score = results["inflammation_score"]
    pct = (n_inf / total * 100) if total > 0 else 0

    # Intensity statistics for inflammatory cells
    inf_cells = [c for c in results["cell_data"] if c["cell_type"] == "inflammatory"]
    par_cells = [c for c in results["cell_data"] if c["cell_type"] == "parenchymal"]

    print(f"\n  --- Results ---")
    print(f"  Total nuclei:            {total}")
    print(f"  Inflammatory cells:      {n_inf} ({pct:.1f}%)")
    print(f"  Parenchymal cells:       {n_par}")
    print(f"  Tissue area:             {results['tissue_area_mm2']:.2f} mm²")
    print(f"  Inflammatory density:    {density:.2f} cells/mm²")
    print(f"  Inflammation score:      {score} ({SCORE_LABELS.get(score, '?')})")

    if inf_cells:
        inf_intensities = [c["mean_intensity"] for c in inf_cells]
        print(f"\n  Inflammatory cell intensity: mean={np.mean(inf_intensities):.1f}, "
              f"median={np.median(inf_intensities):.1f}, "
              f"range=[{np.min(inf_intensities):.1f}, {np.max(inf_intensities):.1f}]")

    if par_cells:
        par_intensities = [c["mean_intensity"] for c in par_cells]
        print(f"  Parenchymal cell intensity:  mean={np.mean(par_intensities):.1f}, "
              f"median={np.median(par_intensities):.1f}, "
              f"range=[{np.min(par_intensities):.1f}, {np.max(par_intensities):.1f}]")
    print()

print("="*60)
print("COMPARISON")
print("="*60)
print("Expected: con-1 should be Score 0-1 (None/Mild baseline)")
print("Expected: 4NQO-2 should have higher density/score than con-1")
