#!/usr/bin/env python3
"""Run and evaluate the HER2-IHC-40x phase-1 benchmark."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import her2_ihc_40x


def build_parser():
    parser = argparse.ArgumentParser(description="HER2-IHC-40x benchmark utilities for patholib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run patholib on a HER2-IHC-40x split")
    run_parser.add_argument("--dataset-root", required=True, help="Path to extracted HER2-IHC-40x data")
    run_parser.add_argument("--split", default="test", choices=her2_ihc_40x.HER2_SPLITS)
    run_parser.add_argument("--output-dir", required=True, help="Directory for patholib reports and CSVs")
    run_parser.add_argument("--detection-method", default="watershed", choices=["watershed", "cellpose"])
    run_parser.add_argument("--weak-threshold", type=float, default=0.10)
    run_parser.add_argument("--moderate-threshold", type=float, default=0.25)
    run_parser.add_argument("--strong-threshold", type=float, default=0.45)
    run_parser.add_argument("--ring-width", type=int, default=4)
    run_parser.add_argument("--min-area", type=int, default=30)
    run_parser.add_argument("--max-area", type=int, default=800)
    run_parser.add_argument("--normalize-stain", action="store_true")
    run_parser.add_argument("--stain-reference", default=None)
    run_parser.add_argument("--save-overlay", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument("--use-gpu", action="store_true")
    run_parser.add_argument("--overwrite", action="store_true")

    eval_parser = subparsers.add_parser("eval", help="Evaluate patholib outputs on HER2-IHC-40x")
    eval_parser.add_argument("--dataset-root", required=True, help="Path to extracted HER2-IHC-40x data")
    eval_parser.add_argument("--split", default="test", choices=her2_ihc_40x.HER2_SPLITS)
    eval_parser.add_argument("--reports-dir", required=True, help="Directory containing *_ihc_report.json files")
    eval_parser.add_argument("--zero-cutoff", type=float, default=1.0)
    eval_parser.add_argument("--positive-cutoff", type=float, default=10.0)
    eval_parser.add_argument("--strong-grade-cutoff", type=float, default=2.5)
    eval_parser.add_argument("--strong-fraction-cutoff", type=float, default=0.30)
    eval_parser.add_argument(
        "--allow-missing-reports",
        action="store_true",
        help="Skip images without *_ihc_report.json instead of failing",
    )
    eval_parser.add_argument("--output-json", default=None, help="Path to aggregated JSON summary")
    eval_parser.add_argument("--output-csv", default=None, help="Path to per-image CSV summary")
    return parser


def _default_eval_paths(reports_dir, split):
    return (
        os.path.join(reports_dir, f"her2_ihc_40x_{split}_eval_summary.json"),
        os.path.join(reports_dir, f"her2_ihc_40x_{split}_per_image.csv"),
    )


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        params = her2_ihc_40x.build_default_membrane_params(
            detection_method=args.detection_method,
            fail_fast=args.fail_fast,
            use_gpu=args.use_gpu,
            weak_threshold=args.weak_threshold,
            moderate_threshold=args.moderate_threshold,
            strong_threshold=args.strong_threshold,
            ring_width=args.ring_width,
            min_area=args.min_area,
            max_area=args.max_area,
        )
        summary = her2_ihc_40x.run_her2_split(
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
            f"Completed HER2-IHC-40x run for split={args.split}: "
            f"{summary['images_completed']} processed, {summary['images_skipped']} skipped"
        )
        return

    output_json, output_csv = _default_eval_paths(args.reports_dir, args.split)
    if args.output_json:
        output_json = args.output_json
    if args.output_csv:
        output_csv = args.output_csv

    summary, rows = her2_ihc_40x.evaluate_her2_split(
        dataset_root=args.dataset_root,
        split=args.split,
        reports_dir=args.reports_dir,
        zero_cutoff=args.zero_cutoff,
        positive_cutoff=args.positive_cutoff,
        strong_grade_cutoff=args.strong_grade_cutoff,
        strong_fraction_cutoff=args.strong_fraction_cutoff,
        require_reports=not args.allow_missing_reports,
    )
    her2_ihc_40x.write_summary_json(summary, output_json)
    her2_ihc_40x.write_per_image_csv(rows, output_csv)

    print(f"HER2-IHC-40x evaluation complete for split={args.split}")
    print(f"  Images evaluated: {summary['images_evaluated']}")
    print(f"  Accuracy:         {summary['accuracy']:.4f}")
    print(f"  Macro F1:         {summary['macro_f1']:.4f}")
    print(f"  QWK:              {summary['quadratic_weighted_kappa']:.4f}")
    print(f"  Summary:          {output_json}")
    print(f"  Per-image:        {output_csv}")


if __name__ == "__main__":
    main()
