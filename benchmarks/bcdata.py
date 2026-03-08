"""BCData benchmark helpers for patholib phase-1 evaluation."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Iterable

import numpy as np


BCDATA_SPLITS = ("train", "validation", "test")


def iter_bcdata_images(dataset_root: str, split: str) -> list[str]:
    """Return sorted image paths for a BCData split."""
    image_dir = os.path.join(dataset_root, "images", split)
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"BCData image directory not found: {image_dir}")

    image_paths = []
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
            image_paths.append(os.path.join(image_dir, name))
    return image_paths


def get_annotation_paths(dataset_root: str, split: str, image_stem: str) -> tuple[str, str]:
    """Return positive/negative annotation paths for one BCData patch."""
    base_dir = os.path.join(dataset_root, "annotations", split)
    positive_path = os.path.join(base_dir, "positive", f"{image_stem}.h5")
    negative_path = os.path.join(base_dir, "negative", f"{image_stem}.h5")
    return positive_path, negative_path


def load_bcdata_coordinates(annotation_path: str, coord_order: str = "xy") -> list[tuple[float, float]]:
    """Load BCData point annotations from an h5 file."""
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to load BCData annotations. Install it before running benchmark evaluation."
        ) from exc

    if coord_order not in {"xy", "yx"}:
        raise ValueError(f"Unsupported coord_order: {coord_order}")

    with h5py.File(annotation_path, "r") as handle:
        if "coordinates" not in handle:
            raise KeyError(f"Missing 'coordinates' dataset in: {annotation_path}")
        coords = np.asarray(handle["coordinates"])

    if coords.size == 0:
        return []
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Unexpected coordinate shape in {annotation_path}: {coords.shape}")

    if coord_order == "xy":
        x_vals = coords[:, 0]
        y_vals = coords[:, 1]
    else:
        y_vals = coords[:, 0]
        x_vals = coords[:, 1]
    return [(float(x), float(y)) for x, y in zip(x_vals, y_vals)]


def load_prediction_cells(csv_path: str) -> list[dict]:
    """Load patholib per-cell CSV output and normalize class labels."""
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            label = infer_prediction_label(row)
            if label is None:
                continue
            rows.append(
                {
                    "x": float(row["centroid_x"]),
                    "y": float(row["centroid_y"]),
                    "label": label,
                }
            )
    return rows


def infer_prediction_label(row: dict) -> str | None:
    """Infer BCData-compatible positive/negative label from a patholib CSV row."""
    cell_type = str(row.get("cell_type", "")).strip().lower()
    if cell_type in {"positive", "negative"}:
        return cell_type

    grade = row.get("grade")
    if grade is None or grade == "":
        return None
    try:
        return "positive" if int(float(grade)) > 0 else "negative"
    except ValueError:
        return None


def split_cells_by_label(cells: Iterable[dict]) -> dict[str, list[tuple[float, float]]]:
    """Split cell dictionaries into positive/negative coordinate lists."""
    groups = {"positive": [], "negative": []}
    for cell in cells:
        label = cell["label"]
        if label not in groups:
            continue
        groups[label].append((float(cell["x"]), float(cell["y"])))
    return groups


def greedy_match_points(
    gt_points: Iterable[tuple[float, float]],
    pred_points: Iterable[tuple[float, float]],
    radius_px: float,
) -> tuple[int, int, int]:
    """Greedy one-to-one point matching within a radius."""
    gt_points = list(gt_points)
    pred_points = list(pred_points)
    if radius_px <= 0:
        raise ValueError("radius_px must be > 0")

    if not gt_points:
        return 0, len(pred_points), 0
    if not pred_points:
        return 0, 0, len(gt_points)

    radius_sq = radius_px * radius_px
    candidates = []
    for gt_idx, (gx, gy) in enumerate(gt_points):
        for pred_idx, (px, py) in enumerate(pred_points):
            dist_sq = (gx - px) ** 2 + (gy - py) ** 2
            if dist_sq <= radius_sq:
                candidates.append((dist_sq, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0])
    matched_gt = set()
    matched_pred = set()
    for _, gt_idx, pred_idx in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)

    tp = len(matched_gt)
    fp = len(pred_points) - tp
    fn = len(gt_points) - tp
    return tp, fp, fn


def precision_recall_f1(tp: int, fp: int, fn: int) -> dict:
    """Compute precision, recall, and F1 from counts."""
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def positive_percentage(positive_count: int, negative_count: int) -> float:
    """Return Ki-67 positive percentage in the 0-100 scale."""
    total = positive_count + negative_count
    if total <= 0:
        return 0.0
    return float(positive_count / total * 100.0)


def evaluate_patch(
    gt_positive: Iterable[tuple[float, float]],
    gt_negative: Iterable[tuple[float, float]],
    pred_positive: Iterable[tuple[float, float]],
    pred_negative: Iterable[tuple[float, float]],
    radius_px: float,
) -> dict:
    """Evaluate one BCData patch."""
    gt_positive = list(gt_positive)
    gt_negative = list(gt_negative)
    pred_positive = list(pred_positive)
    pred_negative = list(pred_negative)

    pos_tp, pos_fp, pos_fn = greedy_match_points(gt_positive, pred_positive, radius_px)
    neg_tp, neg_fp, neg_fn = greedy_match_points(gt_negative, pred_negative, radius_px)

    gt_pp = positive_percentage(len(gt_positive), len(gt_negative))
    pred_pp = positive_percentage(len(pred_positive), len(pred_negative))
    abs_error = abs(pred_pp - gt_pp)

    positive_metrics = precision_recall_f1(pos_tp, pos_fp, pos_fn)
    negative_metrics = precision_recall_f1(neg_tp, neg_fp, neg_fn)

    return {
        "gt_positive": len(gt_positive),
        "gt_negative": len(gt_negative),
        "pred_positive": len(pred_positive),
        "pred_negative": len(pred_negative),
        "positive": positive_metrics,
        "negative": negative_metrics,
        "mean_f1": float((positive_metrics["f1"] + negative_metrics["f1"]) / 2.0),
        "gt_positive_percentage": float(gt_pp),
        "pred_positive_percentage": float(pred_pp),
        "positive_percentage_abs_error": float(abs_error),
    }


def evaluate_prediction_file(
    positive_annotation_path: str,
    negative_annotation_path: str,
    prediction_csv_path: str,
    radius_px: float,
    coord_order: str = "xy",
) -> dict:
    """Evaluate one prediction CSV against BCData annotations."""
    gt_positive = load_bcdata_coordinates(positive_annotation_path, coord_order=coord_order)
    gt_negative = load_bcdata_coordinates(negative_annotation_path, coord_order=coord_order)
    prediction_cells = load_prediction_cells(prediction_csv_path)
    grouped = split_cells_by_label(prediction_cells)
    return evaluate_patch(
        gt_positive=gt_positive,
        gt_negative=gt_negative,
        pred_positive=grouped["positive"],
        pred_negative=grouped["negative"],
        radius_px=radius_px,
    )


def aggregate_patch_results(rows: Iterable[dict]) -> dict:
    """Aggregate per-image BCData metrics into a split summary."""
    rows = list(rows)
    pos_tp = sum(row["positive"]["tp"] for row in rows)
    pos_fp = sum(row["positive"]["fp"] for row in rows)
    pos_fn = sum(row["positive"]["fn"] for row in rows)
    neg_tp = sum(row["negative"]["tp"] for row in rows)
    neg_fp = sum(row["negative"]["fp"] for row in rows)
    neg_fn = sum(row["negative"]["fn"] for row in rows)

    positive_metrics = precision_recall_f1(pos_tp, pos_fp, pos_fn)
    negative_metrics = precision_recall_f1(neg_tp, neg_fp, neg_fn)

    abs_errors = [row["positive_percentage_abs_error"] for row in rows]
    gt_pps = [row["gt_positive_percentage"] for row in rows]
    pred_pps = [row["pred_positive_percentage"] for row in rows]
    sq_errors = [(pred - gt) ** 2 for pred, gt in zip(pred_pps, gt_pps)]

    if len(rows) >= 2 and np.std(gt_pps) > 0 and np.std(pred_pps) > 0:
        pearson_r = float(np.corrcoef(gt_pps, pred_pps)[0, 1])
    else:
        pearson_r = 0.0

    return {
        "images_evaluated": len(rows),
        "positive": positive_metrics,
        "negative": negative_metrics,
        "mean_f1": float((positive_metrics["f1"] + negative_metrics["f1"]) / 2.0),
        "positive_percentage_mae": float(np.mean(abs_errors)) if abs_errors else 0.0,
        "positive_percentage_rmse": float(math.sqrt(np.mean(sq_errors))) if sq_errors else 0.0,
        "positive_percentage_pearson_r": pearson_r,
    }


def build_default_ihc_params(
    detection_method: str = "watershed",
    marker: str = "Ki67",
    fail_fast: bool = False,
    use_gpu: bool = False,
    weak_threshold: float = 0.10,
    moderate_threshold: float = 0.25,
    strong_threshold: float = 0.45,
    min_area: int = 30,
    max_area: int = 800,
) -> dict:
    """Build a reusable patholib nuclear-IHC parameter set for BCData runs."""
    return {
        "detection_method": detection_method,
        "weak_threshold": weak_threshold,
        "moderate_threshold": moderate_threshold,
        "strong_threshold": strong_threshold,
        "ring_width": 4,
        "min_area": min_area,
        "max_area": max_area,
        "stain_type": "nuclear",
        "marker": marker,
        "fail_fast": fail_fast,
        "use_gpu": use_gpu,
    }


def run_bcdata_split(
    dataset_root: str,
    split: str,
    output_dir: str,
    params: dict,
    normalize_stain: bool = False,
    stain_reference: str | None = None,
    save_overlay: bool = False,
    overwrite: bool = False,
) -> dict:
    """Run patholib on every image in a BCData split and save reports."""
    import analyze_ihc
    from patholib.viz.report import generate_ihc_report

    os.makedirs(output_dir, exist_ok=True)
    image_paths = iter_bcdata_images(dataset_root, split)
    completed = 0
    skipped = 0
    summaries = []

    for image_path in image_paths:
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
                "total_cells": results.get("total_cells", 0),
                "positive_cells": results.get("positive_cells", 0),
                "negative_cells": results.get("negative_cells", 0),
                "positive_percentage": results.get("positive_percentage", 0.0),
            }
        )
        completed += 1

    summary_path = os.path.join(output_dir, f"bcdata_{split}_run_summary.json")
    summary = {
        "dataset": "BCData",
        "split": split,
        "images_total": len(image_paths),
        "images_completed": completed,
        "images_skipped": skipped,
        "parameters": params,
        "normalize_stain": normalize_stain,
        "summaries": summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate_bcdata_split(
    dataset_root: str,
    split: str,
    predictions_dir: str,
    radius_px: float = 6.0,
    coord_order: str = "xy",
    require_predictions: bool = True,
) -> tuple[dict, list[dict]]:
    """Evaluate one BCData split against patholib CSV predictions."""
    per_image_rows = []
    missing_predictions = []

    for image_path in iter_bcdata_images(dataset_root, split):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        pos_path, neg_path = get_annotation_paths(dataset_root, split, stem)
        pred_csv_path = os.path.join(predictions_dir, f"{stem}_ihc_cells.csv")

        if not os.path.isfile(pred_csv_path):
            missing_predictions.append(stem)
            if require_predictions:
                raise FileNotFoundError(f"Prediction CSV not found: {pred_csv_path}")
            continue
        if not os.path.isfile(pos_path):
            raise FileNotFoundError(f"Positive annotation not found: {pos_path}")
        if not os.path.isfile(neg_path):
            raise FileNotFoundError(f"Negative annotation not found: {neg_path}")

        metrics = evaluate_prediction_file(
            positive_annotation_path=pos_path,
            negative_annotation_path=neg_path,
            prediction_csv_path=pred_csv_path,
            radius_px=radius_px,
            coord_order=coord_order,
        )
        metrics["image_stem"] = stem
        per_image_rows.append(metrics)

    summary = aggregate_patch_results(per_image_rows)
    summary.update(
        {
            "dataset": "BCData",
            "split": split,
            "match_radius_px": float(radius_px),
            "annotation_coord_order": coord_order,
            "missing_predictions": missing_predictions,
        }
    )
    return summary, per_image_rows


def write_per_image_csv(rows: Iterable[dict], output_path: str) -> None:
    """Write per-image BCData metrics to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "image_stem",
        "gt_positive",
        "gt_negative",
        "pred_positive",
        "pred_negative",
        "positive_tp",
        "positive_fp",
        "positive_fn",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "negative_tp",
        "negative_fp",
        "negative_fn",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "mean_f1",
        "gt_positive_percentage",
        "pred_positive_percentage",
        "positive_percentage_abs_error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_stem": row["image_stem"],
                    "gt_positive": row["gt_positive"],
                    "gt_negative": row["gt_negative"],
                    "pred_positive": row["pred_positive"],
                    "pred_negative": row["pred_negative"],
                    "positive_tp": row["positive"]["tp"],
                    "positive_fp": row["positive"]["fp"],
                    "positive_fn": row["positive"]["fn"],
                    "positive_precision": row["positive"]["precision"],
                    "positive_recall": row["positive"]["recall"],
                    "positive_f1": row["positive"]["f1"],
                    "negative_tp": row["negative"]["tp"],
                    "negative_fp": row["negative"]["fp"],
                    "negative_fn": row["negative"]["fn"],
                    "negative_precision": row["negative"]["precision"],
                    "negative_recall": row["negative"]["recall"],
                    "negative_f1": row["negative"]["f1"],
                    "mean_f1": row["mean_f1"],
                    "gt_positive_percentage": row["gt_positive_percentage"],
                    "pred_positive_percentage": row["pred_positive_percentage"],
                    "positive_percentage_abs_error": row["positive_percentage_abs_error"],
                }
            )


def write_summary_json(summary: dict, output_path: str) -> None:
    """Write BCData benchmark summary JSON."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
