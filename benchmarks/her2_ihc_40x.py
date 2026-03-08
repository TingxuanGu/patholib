"""HER2-IHC-40x benchmark helpers for patholib phase-1 evaluation."""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable

import numpy as np


HER2_CLASSES = ("0", "1+", "2+", "3+")
HER2_SPLITS = ("train", "test")


def normalize_her2_label(value: str) -> str | None:
    """Normalize dataset or prediction label strings to HER2 classes."""
    token = str(value).strip().lower().replace(" ", "")
    mapping = {
        "0": "0",
        "score0": "0",
        "class0": "0",
        "1": "1+",
        "1+": "1+",
        "1plus": "1+",
        "score1": "1+",
        "class1": "1+",
        "2": "2+",
        "2+": "2+",
        "2plus": "2+",
        "score2": "2+",
        "class2": "2+",
        "3": "3+",
        "3+": "3+",
        "3plus": "3+",
        "score3": "3+",
        "class3": "3+",
    }
    return mapping.get(token)


def resolve_split_dir(dataset_root: str, split: str) -> str:
    """Resolve the HER2-IHC-40x split directory from common extracted layouts."""
    if split not in HER2_SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    split_title = split.title()
    candidates = [
        os.path.join(dataset_root, "Patches", split_title),
        os.path.join(dataset_root, split_title),
        os.path.join(dataset_root, "patches", split_title),
        os.path.join(dataset_root, "patches", split),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a HER2-IHC-40x split directory. "
        f"Checked: {candidates}"
    )


def infer_label_from_path(path: str) -> str | None:
    """Infer the HER2 class from a file or directory path."""
    current = os.path.abspath(path)
    while True:
        label = normalize_her2_label(os.path.basename(current))
        if label is not None:
            return label
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def iter_her2_images(dataset_root: str, split: str) -> list[dict]:
    """Return labeled HER2-IHC-40x image paths for one split."""
    split_dir = resolve_split_dir(dataset_root, split)
    records = []
    for root, _, files in os.walk(split_dir):
        label = infer_label_from_path(root)
        if label is None:
            continue
        for name in sorted(files):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                records.append(
                    {
                        "image_path": os.path.join(root, name),
                        "label": label,
                    }
                )
    return records


def build_default_membrane_params(
    detection_method: str = "watershed",
    marker: str = "HER2",
    fail_fast: bool = False,
    use_gpu: bool = False,
    weak_threshold: float = 0.10,
    moderate_threshold: float = 0.25,
    strong_threshold: float = 0.45,
    ring_width: int = 4,
    min_area: int = 30,
    max_area: int = 800,
) -> dict:
    """Build a reusable patholib membrane-IHC parameter set for HER2 runs."""
    return {
        "detection_method": detection_method,
        "weak_threshold": weak_threshold,
        "moderate_threshold": moderate_threshold,
        "strong_threshold": strong_threshold,
        "ring_width": ring_width,
        "min_area": min_area,
        "max_area": max_area,
        "stain_type": "membrane",
        "marker": marker,
        "fail_fast": fail_fast,
        "use_gpu": use_gpu,
    }


def normalize_grade_counts(raw_counts: dict | None) -> dict[int, int]:
    """Normalize JSON grade-count keys to integer grades."""
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    if not raw_counts:
        return counts
    for key, value in raw_counts.items():
        counts[int(key)] = int(value)
    return counts


def predict_her2_grade(
    summary: dict,
    zero_cutoff: float = 1.0,
    positive_cutoff: float = 10.0,
    strong_grade_cutoff: float = 2.5,
    strong_fraction_cutoff: float = 0.30,
) -> str:
    """Map patholib membrane summary outputs to a HER2 patch label.

    This is a benchmark heuristic, not a clinically validated HER2 scoring rule.
    """
    positive_pct = float(summary.get("positive_percentage", 0.0))
    grade_counts = normalize_grade_counts(summary.get("grade_counts"))
    weak = grade_counts[1]
    moderate = grade_counts[2]
    strong = grade_counts[3]
    positive = weak + moderate + strong

    if positive <= 0 or positive_pct < zero_cutoff:
        return "0"
    if positive_pct <= positive_cutoff:
        return "1+"

    weighted_grade = (weak + 2 * moderate + 3 * strong) / positive if positive > 0 else 0.0
    strong_fraction = strong / positive if positive > 0 else 0.0
    if strong > 0 and (weighted_grade >= strong_grade_cutoff or strong_fraction >= strong_fraction_cutoff):
        return "3+"
    if moderate > 0 or strong > 0:
        return "2+"
    return "1+"


def read_report_summary(report_path: str) -> dict:
    """Read a patholib IHC report and return its summary block."""
    with open(report_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("summary", {})


def confusion_matrix(rows: Iterable[dict]) -> np.ndarray:
    """Return a 4x4 confusion matrix ordered by HER2_CLASSES."""
    label_to_index = {label: idx for idx, label in enumerate(HER2_CLASSES)}
    matrix = np.zeros((len(HER2_CLASSES), len(HER2_CLASSES)), dtype=np.int64)
    for row in rows:
        truth = label_to_index[row["ground_truth"]]
        pred = label_to_index[row["predicted_label"]]
        matrix[truth, pred] += 1
    return matrix


def accuracy(rows: Iterable[dict]) -> float:
    """Compute patch-level accuracy."""
    rows = list(rows)
    if not rows:
        return 0.0
    correct = sum(1 for row in rows if row["ground_truth"] == row["predicted_label"])
    return float(correct / len(rows))


def macro_f1(rows: Iterable[dict]) -> float:
    """Compute macro F1 across HER2 classes."""
    matrix = confusion_matrix(rows)
    scores = []
    for idx in range(len(HER2_CLASSES)):
        tp = int(matrix[idx, idx])
        fp = int(np.sum(matrix[:, idx]) - tp)
        fn = int(np.sum(matrix[idx, :]) - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        scores.append(f1)
    return float(np.mean(scores)) if scores else 0.0


def quadratic_weighted_kappa(rows: Iterable[dict]) -> float:
    """Compute quadratic weighted kappa across HER2 classes."""
    rows = list(rows)
    if not rows:
        return 0.0

    observed = confusion_matrix(rows).astype(np.float64)
    total = float(np.sum(observed))
    if total <= 0:
        return 0.0

    hist_true = np.sum(observed, axis=1)
    hist_pred = np.sum(observed, axis=0)
    expected = np.outer(hist_true, hist_pred) / total

    n = len(HER2_CLASSES)
    weights = np.zeros((n, n), dtype=np.float64)
    denom = float((n - 1) ** 2)
    for i in range(n):
        for j in range(n):
            weights[i, j] = ((i - j) ** 2) / denom

    observed_score = float(np.sum(weights * observed) / total)
    expected_total = float(np.sum(expected))
    if expected_total <= 0:
        return 0.0
    expected_score = float(np.sum(weights * expected) / expected_total)
    if expected_score <= 0:
        return 0.0
    return float(1.0 - observed_score / expected_score)


def class_metrics(rows: Iterable[dict]) -> dict:
    """Return per-class precision/recall/F1 metrics."""
    matrix = confusion_matrix(rows)
    metrics = {}
    for idx, label in enumerate(HER2_CLASSES):
        tp = int(matrix[idx, idx])
        fp = int(np.sum(matrix[:, idx]) - tp)
        fn = int(np.sum(matrix[idx, :]) - tp)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
        metrics[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(np.sum(matrix[idx, :])),
        }
    return metrics


def run_her2_split(
    dataset_root: str,
    split: str,
    output_dir: str,
    params: dict,
    normalize_stain: bool = False,
    stain_reference: str | None = None,
    save_overlay: bool = False,
    overwrite: bool = False,
) -> dict:
    """Run patholib membrane analysis on a HER2-IHC-40x split."""
    import analyze_ihc
    from patholib.viz.report import generate_ihc_report

    os.makedirs(output_dir, exist_ok=True)
    image_records = iter_her2_images(dataset_root, split)
    completed = 0
    skipped = 0
    summaries = []

    for record in image_records:
        image_path = record["image_path"]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        report_path = os.path.join(output_dir, f"{stem}_ihc_report.json")
        if os.path.exists(report_path) and not overwrite:
            skipped += 1
            continue

        image = analyze_ihc.load_image(image_path)
        if normalize_stain:
            image = analyze_ihc.apply_stain_normalization(image, stain_reference)
        results = analyze_ihc.run_analysis(image, params)
        generate_ihc_report(
            results,
            image_path,
            params,
            output_dir,
            save_overlay=save_overlay,
            save_csv=True,
        )
        summaries.append(
            {
                "image_stem": stem,
                "ground_truth": record["label"],
                "positive_percentage": results.get("positive_percentage", 0.0),
                "h_score": results.get("h_score", 0.0),
                "grade_counts": results.get("grade_counts", {}),
            }
        )
        completed += 1

    summary = {
        "dataset": "HER2-IHC-40x",
        "split": split,
        "images_total": len(image_records),
        "images_completed": completed,
        "images_skipped": skipped,
        "parameters": params,
        "normalize_stain": normalize_stain,
        "summaries": summaries,
    }
    summary_path = os.path.join(output_dir, f"her2_ihc_40x_{split}_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate_her2_split(
    dataset_root: str,
    split: str,
    reports_dir: str,
    zero_cutoff: float = 1.0,
    positive_cutoff: float = 10.0,
    strong_grade_cutoff: float = 2.5,
    strong_fraction_cutoff: float = 0.30,
    require_reports: bool = True,
) -> tuple[dict, list[dict]]:
    """Evaluate patholib HER2-IHC-40x reports against patch labels."""
    rows = []
    missing_reports = []
    for record in iter_her2_images(dataset_root, split):
        image_path = record["image_path"]
        stem = os.path.splitext(os.path.basename(image_path))[0]
        report_path = os.path.join(reports_dir, f"{stem}_ihc_report.json")
        if not os.path.isfile(report_path):
            missing_reports.append(stem)
            if require_reports:
                raise FileNotFoundError(f"Report not found: {report_path}")
            continue

        summary = read_report_summary(report_path)
        predicted = predict_her2_grade(
            summary,
            zero_cutoff=zero_cutoff,
            positive_cutoff=positive_cutoff,
            strong_grade_cutoff=strong_grade_cutoff,
            strong_fraction_cutoff=strong_fraction_cutoff,
        )
        rows.append(
            {
                "image_stem": stem,
                "ground_truth": record["label"],
                "predicted_label": predicted,
                "positive_percentage": float(summary.get("positive_percentage", 0.0)),
                "h_score": float(summary.get("h_score", 0.0)),
            }
        )

    matrix = confusion_matrix(rows)
    summary = {
        "dataset": "HER2-IHC-40x",
        "split": split,
        "images_evaluated": len(rows),
        "missing_reports": missing_reports,
        "heuristic": {
            "zero_cutoff": float(zero_cutoff),
            "positive_cutoff": float(positive_cutoff),
            "strong_grade_cutoff": float(strong_grade_cutoff),
            "strong_fraction_cutoff": float(strong_fraction_cutoff),
        },
        "accuracy": accuracy(rows),
        "macro_f1": macro_f1(rows),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(rows),
        "class_metrics": class_metrics(rows),
        "confusion_matrix": matrix.tolist(),
        "confusion_labels": list(HER2_CLASSES),
    }
    return summary, rows


def write_per_image_csv(rows: Iterable[dict], output_path: str) -> None:
    """Write per-image HER2 benchmark predictions to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "image_stem",
        "ground_truth",
        "predicted_label",
        "positive_percentage",
        "h_score",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict, output_path: str) -> None:
    """Write HER2 benchmark summary JSON."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
