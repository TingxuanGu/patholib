#!/usr/bin/env python3
"""Aggregate BCData and HER2-IHC-40x phase-1 benchmark summaries."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import phase1_summary


def build_parser():
    parser = argparse.ArgumentParser(description="Aggregate phase-1 benchmark eval summaries")
    parser.add_argument(
        "--eval-json",
        action="append",
        required=True,
        help="Path to one evaluation summary JSON. Repeat for multiple methods/datasets.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for aggregated CSV and markdown outputs",
    )
    parser.add_argument(
        "--commit",
        default="",
        help="Optional git commit hash to record in the aggregated rows",
    )
    parser.add_argument(
        "--run-date",
        default=None,
        help="Optional ISO date override. Defaults to each eval summary file mtime.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    long_rows, wide_rows = phase1_summary.aggregate_eval_summaries(
        eval_summary_paths=args.eval_json,
        commit=args.commit,
        run_date=args.run_date,
    )

    long_csv_path = os.path.join(args.output_dir, "phase1_metrics_long.csv")
    wide_csv_path = os.path.join(args.output_dir, "phase1_summary.csv")
    markdown_path = os.path.join(args.output_dir, "phase1_summary.md")

    phase1_summary.write_csv(long_rows, long_csv_path, phase1_summary.LONG_FIELDNAMES)
    phase1_summary.write_csv(wide_rows, wide_csv_path, phase1_summary.WIDE_FIELDNAMES)
    phase1_summary.write_markdown(
        phase1_summary.render_markdown_summary(wide_rows),
        markdown_path,
    )

    print("Phase 1 benchmark aggregation complete")
    print(f"  Eval summaries: {len(args.eval_json)}")
    print(f"  Long CSV:       {long_csv_path}")
    print(f"  Wide CSV:       {wide_csv_path}")
    print(f"  Markdown:       {markdown_path}")


if __name__ == "__main__":
    main()
