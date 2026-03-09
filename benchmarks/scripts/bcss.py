#!/usr/bin/env python3
"""Run and evaluate the BCSS phase-1 benchmark."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import bcss


def build_parser():
    parser = argparse.ArgumentParser(description="BCSS benchmark utilities for patholib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run patholib area-ratio analysis on BCSS images")
    run_parser.add_argument("--images-dir", required=True, help="Directory containing BCSS images")
    run_parser.add_argument("--output-dir", required=True, help="Directory for reports and segmentation masks")
    run_parser.add_argument("--mpp", type=float, default=None)
    run_parser.add_argument("--normalize-stain", action="store_true")
    run_parser.add_argument("--stain-reference", default=None)
    run_parser.add_argument("--overwrite", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate patholib outputs on BCSS masks")
    eval_parser.add_argument("--masks-dir", required=True, help="Directory containing BCSS ground-truth masks")
    eval_parser.add_argument("--predictions-dir", required=True, help="Directory containing *_he_segmentation.npy")
    eval_parser.add_argument("--label-map-json", default=None, help="Optional JSON mapping from raw BCSS labels to patholib classes")
    eval_parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Skip masks without prediction arrays instead of failing",
    )
    eval_parser.add_argument("--output-json", default=None, help="Path to aggregated JSON summary")
    eval_parser.add_argument("--output-csv", default=None, help="Path to per-image CSV summary")
    return parser


def _default_eval_paths(predictions_dir):
    return (
        os.path.join(predictions_dir, "bcss_eval_summary.json"),
        os.path.join(predictions_dir, "bcss_per_image.csv"),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        params = bcss.build_default_area_ratio_params(mpp=args.mpp)
        summary = bcss.run_bcss_images(
            images_dir=args.images_dir,
            output_dir=args.output_dir,
            params=params,
            normalize_stain=args.normalize_stain,
            stain_reference=args.stain_reference,
            overwrite=args.overwrite,
        )
        print(
            f"Completed BCSS run: {summary['images_completed']} processed, "
            f"{summary['images_skipped']} skipped"
        )
        return

    output_json, output_csv = _default_eval_paths(args.predictions_dir)
    if args.output_json:
        output_json = args.output_json
    if args.output_csv:
        output_csv = args.output_csv

    summary, rows = bcss.evaluate_bcss_predictions(
        masks_dir=args.masks_dir,
        predictions_dir=args.predictions_dir,
        label_map_path=args.label_map_json,
        require_predictions=not args.allow_missing_predictions,
    )
    bcss.write_summary_json(summary, output_json)
    bcss.write_per_image_csv(rows, output_csv)

    print("BCSS evaluation complete")
    print(f"  Images evaluated: {summary['images_evaluated']}")
    print(f"  Tumor Dice:      {summary['per_class']['tumor']['dice']:.4f}")
    print(f"  Stroma Dice:     {summary['per_class']['stroma']['dice']:.4f}")
    print(f"  Necrosis Dice:   {summary['per_class']['necrosis']['dice']:.4f}")
    print(f"  Tumor MAE:       {summary['tumor_ratio_mae']:.4f}")
    print(f"  Necrosis MAE:    {summary['necrosis_ratio_mae']:.4f}")
    print(f"  Summary:         {output_json}")
    print(f"  Per-image:       {output_csv}")


if __name__ == "__main__":
    main()
