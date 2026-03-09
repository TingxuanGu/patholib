#!/usr/bin/env python3
"""Run or smoke-test the full phase-1 benchmark workflow."""

import argparse
import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
BENCHMARKS_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BENCHMARKS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from benchmarks import phase1_orchestration


def build_parser():
    parser = argparse.ArgumentParser(description="Orchestrate the phase-1 benchmark workflow")
    parser.add_argument("--output-dir", required=True, help="Root directory for benchmark outputs")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=phase1_orchestration.PHASE1_DATASETS,
        default=list(phase1_orchestration.PHASE1_DATASETS),
        help="Subset of phase-1 datasets to run",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["watershed", "cellpose"],
        default=list(phase1_orchestration.DEFAULT_METHODS),
        help="Detection methods for datasets that support them",
    )
    parser.add_argument("--commit", default="", help="Optional git commit hash recorded in the aggregate tables")
    parser.add_argument("--run-date", default=None, help="Optional ISO timestamp override")
    parser.add_argument("--smoke", action="store_true", help="Run a synthetic smoke pass without real benchmark data")
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--normalize-stain", action="store_true")
    parser.add_argument("--stain-reference", default=None)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--bcdata-root", default=None)
    parser.add_argument("--bcdata-split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--bcdata-match-radius", type=float, default=6.0)
    parser.add_argument("--bcdata-coord-order", default="xy", choices=["xy", "yx"])

    parser.add_argument("--her2-root", default=None)
    parser.add_argument("--her2-split", default="test", choices=["train", "test"])

    parser.add_argument("--bcss-images-dir", default=None)
    parser.add_argument("--bcss-masks-dir", default=None)
    parser.add_argument("--bcss-label-map-json", default=None)
    parser.add_argument("--bcss-mpp", type=float, default=None)

    parser.add_argument("--pannuke-images-npy", default=None)
    parser.add_argument("--pannuke-masks-npy", default=None)
    parser.add_argument("--pannuke-start-index", type=int, default=0)
    parser.add_argument("--pannuke-limit", type=int, default=None)
    parser.add_argument("--pannuke-inflammatory-channel", type=int, default=1)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    result = phase1_orchestration.run_phase1(vars(args))

    print("Phase-1 benchmark workflow complete")
    print(f"  Smoke mode:   {result['smoke']}")
    print(f"  Datasets:     {', '.join(result['datasets'])}")
    print(f"  Methods:      {', '.join(result['methods'])}")
    print(f"  Eval JSONs:   {len(result['eval_json_paths'])}")
    print(f"  Output root:  {result['output_dir']}")
    print(f"  Summary dir:  {result['summary_dir']}")


if __name__ == "__main__":
    main()
