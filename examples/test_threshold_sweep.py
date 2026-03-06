#!/usr/bin/env python3
"""Sweep intensity thresholds on already-detected cells to find optimal cutoff."""

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

# Run analysis once (uses current threshold but we'll re-classify post-hoc)
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

# Sweep thresholds
thresholds = [100, 110, 120, 125, 130, 135, 140, 145, 150, 160]

print(f"{'Thresh':>6} | {'con-1 inf%':>10} {'density':>10} {'score':>6} | {'4NQO-2 inf%':>11} {'density':>10} {'score':>6}")
print("-" * 80)

for thresh in thresholds:
    row = f"{thresh:>6} |"
    for name in ["con-1", "4NQO-2"]:
        d = slide_data[name]
        cells = d["cells"]
        area = d["tissue_area_mm2"]

        # Re-classify at this threshold
        n_inf = sum(1 for c in cells
                    if c["area"] <= INFLAMMATORY_MAX_AREA
                    and c["circularity"] >= INFLAMMATORY_MIN_CIRCULARITY
                    and c["mean_intensity"] < thresh)
        total = len(cells)
        pct = (n_inf / total * 100) if total > 0 else 0
        density = n_inf / area if area > 0 else 0

        if density < MILD_THRESHOLD:
            score = 0
        elif density < MODERATE_THRESHOLD:
            score = 1
        elif density < SEVERE_THRESHOLD:
            score = 2
        else:
            score = 3

        row += f" {pct:>9.1f}% {density:>9.1f} {score:>3}({SCORE_LABELS[score]:>8}) |"
    print(row)

print()
print("Scoring thresholds: None<50, Mild<200, Moderate<500, Severe>=500 cells/mm²")
print("Target: con-1 = Score 0-1, 4NQO-2 clearly higher than con-1")
