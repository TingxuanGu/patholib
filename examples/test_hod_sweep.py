#!/usr/bin/env python3
"""Sweep Hematoxylin OD thresholds on con-1 and 4NQO-2 to find optimal cutoff."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from batch_he_wyjhe_v2 import (
    analyze_wsi_tiled, SCORE_LABELS,
    INFLAMMATORY_MAX_AREA, INFLAMMATORY_MIN_CIRCULARITY,
    MILD_THRESHOLD, MODERATE_THRESHOLD, SEVERE_THRESHOLD,
)
from patholib.io.image_loader import get_wsi_info
import numpy as np

INPUT_DIR = "/home/bio/桌面/Tingxuan Gu/analysis/WYJ HE-IHC results/WYJ HE"
TEST_SLIDES = ["con-1.mrxs", "4NQO-2.mrxs"]

# Run analysis once per slide to collect all cell data
slide_data = {}
for fname in TEST_SLIDES:
    path = os.path.join(INPUT_DIR, fname)
    name = os.path.splitext(fname)[0]
    print(f"Loading {fname}...")
    info = get_wsi_info(path)
    results = analyze_wsi_tiled(path, info)
    slide_data[name] = {
        "cells": results["cell_data"],
        "tissue_area_mm2": results["tissue_area_mm2"],
    }
    print(f"  {len(results['cell_data'])} cells, {results['tissue_area_mm2']:.2f} mm²\n")

# --- Raw H-OD distributions for all nuclei ---
print("=" * 70)
print("RAW HEMATOXYLIN OD DISTRIBUTIONS (all detected nuclei)")
print("=" * 70)
for name in ["con-1", "4NQO-2"]:
    cells = slide_data[name]["cells"]
    all_od = [c["hematoxylin_od"] for c in cells]
    # Small+round subset (morphologically inflammatory candidates)
    cand_od = [c["hematoxylin_od"] for c in cells
               if c["area"] <= INFLAMMATORY_MAX_AREA and c["circularity"] >= INFLAMMATORY_MIN_CIRCULARITY]
    # Large/irregular (parenchymal by morphology)
    other_od = [c["hematoxylin_od"] for c in cells
                if not (c["area"] <= INFLAMMATORY_MAX_AREA and c["circularity"] >= INFLAMMATORY_MIN_CIRCULARITY)]

    print(f"\n  {name}:")
    print(f"    All nuclei ({len(all_od)}):              "
          f"mean={np.mean(all_od):.4f}, median={np.median(all_od):.4f}, "
          f"std={np.std(all_od):.4f}, range=[{np.min(all_od):.4f}, {np.max(all_od):.4f}]")
    print(f"    Small+round candidates ({len(cand_od)}): "
          f"mean={np.mean(cand_od):.4f}, median={np.median(cand_od):.4f}, "
          f"std={np.std(cand_od):.4f}, range=[{np.min(cand_od):.4f}, {np.max(cand_od):.4f}]")
    print(f"    Other nuclei ({len(other_od)}):          "
          f"mean={np.mean(other_od):.4f}, median={np.median(other_od):.4f}, "
          f"std={np.std(other_od):.4f}, range=[{np.min(other_od):.4f}, {np.max(other_od):.4f}]")

    # Percentile breakdown for candidates
    pcts = [10, 25, 50, 75, 90, 95, 99]
    vals = np.percentile(cand_od, pcts)
    pct_str = ", ".join(f"P{p}={v:.4f}" for p, v in zip(pcts, vals))
    print(f"    Candidate percentiles: {pct_str}")

# --- Threshold sweep ---
print(f"\n{'='*70}")
print("THRESHOLD SWEEP: Hematoxylin OD (higher = darker staining)")
print(f"{'='*70}")

thresholds = [0.050, 0.075, 0.100, 0.125, 0.150, 0.175, 0.200, 0.225, 0.250, 0.275, 0.300]

header = f"{'H-OD':>6} |"
for name in ["con-1", "4NQO-2"]:
    header += f" {'inf%':>7} {'density':>10} {'score':>12} |"
print(header)
print("-" * 80)

for thresh in thresholds:
    row = f"{thresh:>6.3f} |"
    for name in ["con-1", "4NQO-2"]:
        d = slide_data[name]
        cells = d["cells"]
        area = d["tissue_area_mm2"]

        # Re-classify at this threshold (H-OD: higher = darker, so > thresh)
        n_inf = sum(1 for c in cells
                    if c["area"] <= INFLAMMATORY_MAX_AREA
                    and c["circularity"] >= INFLAMMATORY_MIN_CIRCULARITY
                    and c["hematoxylin_od"] > thresh)
        total = len(cells)
        pct = (n_inf / total * 100) if total > 0 else 0
        density = n_inf / area if area > 0 else 0

        if density < MILD_THRESHOLD:
            score = 0
        elif density < MODERATE_THRESHOLD:
            score = 1
        elif density < MODERATE_THRESHOLD:
            score = 1
        elif density < SEVERE_THRESHOLD:
            score = 2
        else:
            score = 3

        row += f" {pct:>6.1f}% {density:>9.1f} {score:>2}({SCORE_LABELS[score]:>8}) |"
    print(row)

print()
print(f"Scoring: None<{MILD_THRESHOLD}, Mild<{MODERATE_THRESHOLD}, Moderate<{SEVERE_THRESHOLD}, Severe>={SEVERE_THRESHOLD} cells/mm²")
print("Target: con-1 = Score 0-1 (baseline), 4NQO-2 density > con-1")
