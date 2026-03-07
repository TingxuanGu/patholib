#!/usr/bin/env python3
"""Sweep intensity thresholds on already-detected cells to find optimal cutoff."""

import argparse
import os
import sys

import numpy as np

EXAMPLES_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(EXAMPLES_DIR)
sys.path.insert(0, EXAMPLES_DIR)
sys.path.insert(0, REPO_ROOT)

from patholib.io.image_loader import get_wsi_info

TEST_SLIDES = ["con-1.mrxs", "4NQO-2.mrxs"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the test .mrxs slides",
    )
    return parser


def main(argv=None):
    from batch_he import (
        analyze_wsi_tiled,
        SCORE_LABELS,
        INFLAMMATORY_MAX_AREA,
        INFLAMMATORY_MIN_CIRCULARITY,
        MILD_THRESHOLD,
        MODERATE_THRESHOLD,
        SEVERE_THRESHOLD,
    )

    args = build_parser().parse_args(argv)
    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        raise SystemExit(f"Input directory not found: {input_dir}")

    slide_data = {}
    for fname in TEST_SLIDES:
        path = os.path.join(input_dir, fname)
        name = os.path.splitext(fname)[0]
        print(f"Loading {fname}...")
        info = get_wsi_info(path)
        results = analyze_wsi_tiled(path, info)
        slide_data[name] = {
            "cells": results["cell_data"],
            "tissue_area_mm2": results["tissue_area_mm2"],
        }
        print(f"  {len(results['cell_data'])} cells, {results['tissue_area_mm2']:.2f} mm²\n")

    thresholds = [100, 110, 120, 125, 130, 135, 140, 145, 150, 160]

    print(f"{'Thresh':>6} | {'con-1 inf%':>10} {'density':>10} {'score':>6} | {'4NQO-2 inf%':>11} {'density':>10} {'score':>6}")
    print("-" * 80)

    for thresh in thresholds:
        row = f"{thresh:>6} |"
        for name in ["con-1", "4NQO-2"]:
            data = slide_data[name]
            cells = data["cells"]
            area = data["tissue_area_mm2"]

            n_inf = sum(
                1
                for cell in cells
                if cell["area"] <= INFLAMMATORY_MAX_AREA
                and cell["circularity"] >= INFLAMMATORY_MIN_CIRCULARITY
                and cell["hematoxylin_od"] > thresh / 255.0
            )
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


if __name__ == "__main__":
    main()
