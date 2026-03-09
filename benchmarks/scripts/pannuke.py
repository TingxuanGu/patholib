#!/usr/bin/env python3
"""Run and evaluate the PanNuke phase-1 benchmark."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import pannuke


def build_parser():
    parser = argparse.ArgumentParser(description="PanNuke benchmark utilities for patholib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run patholib on a PanNuke image array")
    run_parser.add_argument("--images-npy", required=True, help="Path to PanNuke images.npy")
    run_parser.add_argument("--output-dir", required=True, help="Directory for prediction arrays and summaries")
    run_parser.add_argument("--detection-method", default="watershed", choices=["watershed", "cellpose"])
    run_parser.add_argument("--inflammatory-max-area", type=int, default=80)
    run_parser.add_argument("--inflammatory-min-circularity", type=float, default=0.7)
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--use-gpu", action="store_true")
    run_parser.add_argument("--start-index", type=int, default=0)
    run_parser.add_argument("--limit", type=int, default=None)

    eval_parser = subparsers.add_parser("eval", help="Evaluate patholib outputs on PanNuke masks")
    eval_parser.add_argument("--masks-npy", required=True, help="Path to PanNuke masks.npy")
    eval_parser.add_argument("--predictions-dir", required=True, help="Directory containing PanNuke prediction arrays")
    eval_parser.add_argument(
        "--inflammatory-channel",
        type=int,
        default=pannuke.DEFAULT_INFLAMMATORY_CHANNEL,
        help="Index of the inflammatory channel in the PanNuke mask stack",
    )
    eval_parser.add_argument("--output-json", default=None, help="Path to aggregated JSON summary")
    eval_parser.add_argument("--output-csv", default=None, help="Path to per-patch CSV summary")
    return parser


def _default_eval_paths(predictions_dir):
    return (
        os.path.join(predictions_dir, "pannuke_eval_summary.json"),
        os.path.join(predictions_dir, "pannuke_per_patch.csv"),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        params = pannuke.build_default_inflammation_params(
            detection_method=args.detection_method,
            inflammatory_max_area=args.inflammatory_max_area,
            inflammatory_min_circularity=args.inflammatory_min_circularity,
            fail_fast=args.fail_fast,
            use_gpu=args.use_gpu,
        )
        summary = pannuke.run_pannuke_images(
            images_npy_path=args.images_npy,
            output_dir=args.output_dir,
            params=params,
            start_index=args.start_index,
            limit=args.limit,
        )
        print(
            f"Completed PanNuke run: start={summary['start_index']}, "
            f"count={summary['images_evaluated']}"
        )
        return

    output_json, output_csv = _default_eval_paths(args.predictions_dir)
    if args.output_json:
        output_json = args.output_json
    if args.output_csv:
        output_csv = args.output_csv

    summary, rows = pannuke.evaluate_pannuke_predictions(
        masks_npy_path=args.masks_npy,
        predictions_dir=args.predictions_dir,
        inflammatory_channel=args.inflammatory_channel,
    )
    pannuke.write_summary_json(summary, output_json)
    pannuke.write_per_image_csv(rows, output_csv)

    print("PanNuke evaluation complete")
    print(f"  Patches evaluated:  {summary['images_evaluated']}")
    print(f"  Binary Dice:        {summary['binary_nuclei_dice']:.4f}")
    print(f"  AJI:                {summary['aji']:.4f}")
    print(f"  PQ:                 {summary['pq']:.4f}")
    print(f"  Inflammatory F1:    {summary['inflammatory_f1']:.4f}")
    print(f"  Summary:            {output_json}")
    print(f"  Per-patch:          {output_csv}")


if __name__ == "__main__":
    main()
