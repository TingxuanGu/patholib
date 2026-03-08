"""Unified phase-1 benchmark summary helpers."""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from typing import Iterable


DATASET_SPECS = {
    "BCData": {
        "task": "ihc_nuclear",
        "scale": "official",
        "metrics": (
            ("images_evaluated", "images_evaluated"),
            ("positive_f1", ("positive", "f1")),
            ("negative_f1", ("negative", "f1")),
            ("mean_f1", "mean_f1"),
            ("positive_percentage_mae", "positive_percentage_mae"),
            ("positive_percentage_rmse", "positive_percentage_rmse"),
            ("positive_percentage_pearson_r", "positive_percentage_pearson_r"),
        ),
    },
    "HER2-IHC-40x": {
        "task": "ihc_membrane",
        "scale": "40x",
        "metrics": (
            ("images_evaluated", "images_evaluated"),
            ("accuracy", "accuracy"),
            ("macro_f1", "macro_f1"),
            ("quadratic_weighted_kappa", "quadratic_weighted_kappa"),
        ),
    },
}

LONG_FIELDNAMES = [
    "phase",
    "dataset",
    "split",
    "task",
    "method_family",
    "method_name",
    "detection_backend",
    "normalization",
    "mpp_or_scale",
    "metric_name",
    "metric_value",
    "commit",
    "run_date",
    "notes",
]

WIDE_FIELDNAMES = [
    "phase",
    "dataset",
    "split",
    "task",
    "method_family",
    "method_name",
    "detection_backend",
    "normalization",
    "mpp_or_scale",
    "images_evaluated",
    "positive_f1",
    "negative_f1",
    "mean_f1",
    "positive_percentage_mae",
    "positive_percentage_rmse",
    "positive_percentage_pearson_r",
    "accuracy",
    "macro_f1",
    "quadratic_weighted_kappa",
    "commit",
    "run_date",
    "notes",
]


def load_json(path: str) -> dict:
    """Read a JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def iso_date_from_path(path: str) -> str:
    """Return UTC-ish local ISO timestamp from file mtime."""
    ts = os.path.getmtime(path)
    return dt.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _nested_value(payload: dict, path_spec):
    if isinstance(path_spec, tuple):
        value = payload
        for key in path_spec:
            value = value.get(key, {})
        return value if value != {} else None
    return payload.get(path_spec)


def detect_dataset(summary: dict) -> str:
    """Validate and return the dataset name from an eval summary."""
    dataset = summary.get("dataset")
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unsupported benchmark dataset: {dataset}")
    return dataset


def infer_run_summary_path(eval_summary_path: str, summary: dict) -> str | None:
    """Guess the corresponding run-summary JSON in the same directory."""
    dataset = detect_dataset(summary)
    split = summary.get("split", "")
    directory = os.path.dirname(eval_summary_path)
    if dataset == "BCData":
        candidate = os.path.join(directory, f"bcdata_{split}_run_summary.json")
    elif dataset == "HER2-IHC-40x":
        candidate = os.path.join(directory, f"her2_ihc_40x_{split}_run_summary.json")
    else:
        return None
    return candidate if os.path.isfile(candidate) else None


def infer_method_name(eval_summary_path: str, run_summary: dict | None) -> str:
    """Infer a readable method name."""
    if run_summary:
        params = run_summary.get("parameters", {})
        backend = params.get("detection_method")
        if backend:
            return str(backend)
    return os.path.basename(os.path.dirname(eval_summary_path)) or "unknown"


def infer_detection_backend(run_summary: dict | None, method_name: str) -> str:
    """Infer detection backend for summary rows."""
    if run_summary:
        params = run_summary.get("parameters", {})
        backend = params.get("detection_method")
        if backend:
            return str(backend)
    return method_name


def infer_normalization(run_summary: dict | None) -> str:
    """Infer stain-normalization status."""
    if run_summary is None:
        return "unknown"
    return "on" if bool(run_summary.get("normalize_stain")) else "off"


def build_notes(summary: dict) -> str:
    """Build compact dataset-specific note text."""
    dataset = detect_dataset(summary)
    if dataset == "BCData":
        return (
            f"match_radius_px={summary.get('match_radius_px', '')}; "
            f"coord_order={summary.get('annotation_coord_order', '')}"
        )
    heuristic = summary.get("heuristic", {})
    return (
        f"zero_cutoff={heuristic.get('zero_cutoff', '')}; "
        f"positive_cutoff={heuristic.get('positive_cutoff', '')}; "
        f"strong_grade_cutoff={heuristic.get('strong_grade_cutoff', '')}; "
        f"strong_fraction_cutoff={heuristic.get('strong_fraction_cutoff', '')}"
    )


def build_common_metadata(
    eval_summary_path: str,
    eval_summary: dict,
    run_summary: dict | None = None,
    commit: str = "",
    run_date: str | None = None,
    notes: str | None = None,
) -> dict:
    """Build shared metadata for long/wide result rows."""
    dataset = detect_dataset(eval_summary)
    spec = DATASET_SPECS[dataset]
    method_name = infer_method_name(eval_summary_path, run_summary)
    return {
        "phase": "phase1",
        "dataset": dataset,
        "split": str(eval_summary.get("split", "")),
        "task": spec["task"],
        "method_family": "patholib",
        "method_name": method_name,
        "detection_backend": infer_detection_backend(run_summary, method_name),
        "normalization": infer_normalization(run_summary),
        "mpp_or_scale": spec["scale"],
        "commit": commit,
        "run_date": run_date or iso_date_from_path(eval_summary_path),
        "notes": notes if notes is not None else build_notes(eval_summary),
    }


def long_rows_from_summary(
    eval_summary_path: str,
    eval_summary: dict,
    run_summary: dict | None = None,
    commit: str = "",
    run_date: str | None = None,
    notes: str | None = None,
) -> list[dict]:
    """Expand one eval summary into long-format metric rows."""
    dataset = detect_dataset(eval_summary)
    metadata = build_common_metadata(
        eval_summary_path=eval_summary_path,
        eval_summary=eval_summary,
        run_summary=run_summary,
        commit=commit,
        run_date=run_date,
        notes=notes,
    )
    rows = []
    for metric_name, source in DATASET_SPECS[dataset]["metrics"]:
        rows.append(
            {
                **metadata,
                "metric_name": metric_name,
                "metric_value": _nested_value(eval_summary, source),
            }
        )
    return rows


def wide_row_from_summary(
    eval_summary_path: str,
    eval_summary: dict,
    run_summary: dict | None = None,
    commit: str = "",
    run_date: str | None = None,
    notes: str | None = None,
) -> dict:
    """Convert one eval summary into a wide-format unified result row."""
    dataset = detect_dataset(eval_summary)
    row = {name: "" for name in WIDE_FIELDNAMES}
    row.update(
        build_common_metadata(
            eval_summary_path=eval_summary_path,
            eval_summary=eval_summary,
            run_summary=run_summary,
            commit=commit,
            run_date=run_date,
            notes=notes,
        )
    )
    for metric_name, source in DATASET_SPECS[dataset]["metrics"]:
        row[metric_name] = _nested_value(eval_summary, source)
    return row


def aggregate_eval_summaries(
    eval_summary_paths: Iterable[str],
    commit: str = "",
    run_date: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Aggregate multiple benchmark eval summaries into long and wide rows."""
    long_rows = []
    wide_rows = []
    for path in eval_summary_paths:
        eval_summary = load_json(path)
        run_summary_path = infer_run_summary_path(path, eval_summary)
        run_summary = load_json(run_summary_path) if run_summary_path else None
        long_rows.extend(
            long_rows_from_summary(
                eval_summary_path=path,
                eval_summary=eval_summary,
                run_summary=run_summary,
                commit=commit,
                run_date=run_date,
            )
        )
        wide_rows.append(
            wide_row_from_summary(
                eval_summary_path=path,
                eval_summary=eval_summary,
                run_summary=run_summary,
                commit=commit,
                run_date=run_date,
            )
        )
    return long_rows, wide_rows


def write_csv(rows: Iterable[dict], output_path: str, fieldnames: list[str]) -> None:
    """Write rows to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _format_metric(value) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_markdown_summary(wide_rows: Iterable[dict]) -> str:
    """Render a compact markdown summary table."""
    wide_rows = list(wide_rows)
    lines = [
        "# Phase 1 Benchmark Summary",
        "",
        "| Dataset | Split | Method | Backend | Norm | Images | Positive F1 | Mean F1 | Ki67 MAE | Accuracy | Macro F1 | QWK |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in wide_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["split"]),
                    str(row["method_name"]),
                    str(row["detection_backend"]),
                    str(row["normalization"]),
                    _format_metric(row["images_evaluated"]),
                    _format_metric(row["positive_f1"]),
                    _format_metric(row["mean_f1"]),
                    _format_metric(row["positive_percentage_mae"]),
                    _format_metric(row["accuracy"]),
                    _format_metric(row["macro_f1"]),
                    _format_metric(row["quadratic_weighted_kappa"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(markdown_text: str, output_path: str) -> None:
    """Write markdown summary output."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown_text)
