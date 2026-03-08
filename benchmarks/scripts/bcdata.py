#!/usr/bin/env python3
"""Run and evaluate the BCData phase-1 benchmark."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import bcdata


def build_parser():
    parser = argparse.ArgumentParser(description="BCData benchmark utilities for patholib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run patholib on a BCData split")
    run_parser.add_argument("--dataset-root", required=True, help="Path to the BCData root directory")
    run_parser.add_argument("--split", default="test", choices=bcdata.BCDATA_SPLITS)
    run_parser.add_argument("--output-dir", required=True, help="Directory for patholib reports and CSVs")
    run_parser.add_argument("--detection-method", default="watershed", choices=["watershed", "cellpose"])
    run_parser.add_argument("--marker", default="Ki67")
    run_parser.add_argument("--weak-threshold", type=float, default=0.10)
    run_parser.add_argument("--moderate-threshold", type=float, default=0.25)
    run_parser.add_argument("--strong-threshold", type=float, default=0.45)
    run_parser.add_argument("--min-area", type=int, default=30)
    run_parser.add_argument("--max-area", type=int, default=800)
    run_parser.add_argument("--normalize-stain", action="store_true")
    run_parser.add_argument("--stain-reference", default=None)
    run_parser.add_argument("--save-overlay", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--use-gpu", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate patholib outputs on a BCData split")
    eval_parser.add_argument("--dataset-root", required=True, help="Path to the BCData root directory")
    eval_parser.add_argument("--split", default="test", choices=bcdata.BCDATA_SPLITS)
    eval_parser.add_argument("--predictions-dir", required=True, help="Directory containing *_ihc_cells.csv files")
    eval_parser.add_argument("--match-radius", type=float, default=6.0, help="Point-match radius in pixels")
    eval_parser.add_argument(
        "--coord-order",
        default="xy",
        choices=["xy", "yx"],
        help="Interpretation of BCData coordinates in each annotation row",
    )
    eval_parser.add_argument(
        "--allow-missing-predictions",
        action="store_true",
        help="Skip images without prediction CSVs instead of failing",
    )
    eval_parser.add_argument("--output-json", default=None, help="Path to aggregated JSON summary")
    eval_parser.add_argument("--output-csv", default=None, help="Path to per-image CSV summary")
    return parser


def _default_eval_paths(predictions_dir, split):
    return (
        os.path.join(predictions_dir, f"bcdata_{split}_eval_summary.json"),
        os.path.join(predictions_dir, f"bcdata_{split}_per_image.csv"),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        params = bcdata.build_default_ihc_params(
            detection_method=args.detection_method,
            marker=args.marker,
            fail_fast=args.fail_fast,
            use_gpu=args.use_gpu,
            weak_threshold=args.weak_threshold,
            moderate_threshold=args.moderate_threshold,
            strong_threshold=args.strong_threshold,
            min_area=args.min_area,
            max_area=args.max_area,
        )
        summary = bcdata.run_bcdata_split(
            dataset_root=args.dataset_root,
            split=args.split,
            output_dir=args.output_dir,
            params=params,
            normalize_stain=args.normalize_stain,
            stain_reference=args.stain_reference,
            save_overlay=args.save_overlay,
            overwrite=args.overwrite,
        )
        print(
            f"Completed BCData run for split={args.split}: "
            f"{summary['images_completed']} processed, {summary['images_skipped']} skipped"
        )
        return

    output_json, output_csv = _default_eval_paths(args.predictions_dir, args.split)
    if args.output_json:
        output_json = args.output_json
    if args.output_csv:
        output_csv = args.output_csv

    summary, per_image_rows = bcdata.evaluate_bcdata_split(
        dataset_root=args.dataset_root,
        split=args.split,
        predictions_dir=args.predictions_dir,
        radius_px=args.match_radius,
        coord_order=args.coord_order,
        require_predictions=not args.allow_missing_predictions,
    )
    bcdata.write_summary_json(summary, output_json)
    bcdata.write_per_image_csv(per_image_rows, output_csv)

    print(f"BCData evaluation complete for split={args.split}")
    print(f"  Images evaluated: {summary['images_evaluated']}")
    print(f"  Positive F1: {summary['positive']['f1']:.4f}")
    print(f"  Negative F1: {summary['negative']['f1']:.4f}")
    print(f"  Mean F1:     {summary['mean_f1']:.4f}")
    print(f"  Ki67 MAE:    {summary['positive_percentage_mae']:.4f}")
    print(f"  Ki67 RMSE:   {summary['positive_percentage_rmse']:.4f}")
    print(f"  Summary:     {output_json}")
    print(f"  Per-image:   {output_csv}")


if __name__ == "__main__":
    main()
