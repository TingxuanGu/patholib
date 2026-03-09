"""BCSS benchmark helpers for patholib phase-1 evaluation."""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable

import numpy as np

CLASS_ID_BY_NAME = {
    "background": 0,
    "normal": 1,
    "tumor": 2,
    "necrosis": 3,
    "stroma": 4,
}

IGNORED_LABEL = -1
EVAL_CLASSES = ("tumor", "stroma", "necrosis")


def iter_image_paths(images_dir: str) -> list[str]:
    """Return sorted image paths under an input directory."""
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    paths = []
    for root, _, files in os.walk(images_dir):
        for name in sorted(files):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")):
                paths.append(os.path.join(root, name))
    return paths


def iter_mask_paths(masks_dir: str) -> list[str]:
    """Return sorted ground-truth mask paths under a directory."""
    if not os.path.isdir(masks_dir):
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")
    paths = []
    for root, _, files in os.walk(masks_dir):
        for name in sorted(files):
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")):
                paths.append(os.path.join(root, name))
    return paths


def stem(path: str) -> str:
    """Return basename without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def load_array_image(path: str) -> np.ndarray:
    """Load image or mask arrays from common formats."""
    if path.lower().endswith(".npy"):
        return np.load(path)

    try:
        from PIL import Image

        return np.array(Image.open(path))
    except ImportError:
        from skimage.io import imread

        return imread(path)


def save_prediction_mask(mask: np.ndarray, output_path: str) -> None:
    """Save a predicted segmentation mask as .npy."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.save(output_path, mask.astype(np.int32))


def load_label_map(path: str) -> dict:
    """Load a BCSS label-map JSON file."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _parse_target_label(value) -> int:
    token = str(value).strip().lower()
    if token == "ignore":
        return IGNORED_LABEL
    if token in CLASS_ID_BY_NAME:
        return CLASS_ID_BY_NAME[token]
    return int(value)


def normalize_ground_truth_mask(mask: np.ndarray, label_map: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a raw BCSS mask to patholib class ids and a valid-pixel mask."""
    mask = np.asarray(mask)
    if label_map is None:
        if mask.ndim == 3:
            raise ValueError("RGB masks require a label_map to collapse colors into patholib classes.")
        normalized = mask.astype(np.int32)
        valid_mask = normalized != IGNORED_LABEL
        return normalized, valid_mask

    mapping = label_map.get("mapping", {})
    map_type = label_map.get("type", "int")
    normalized = np.full(mask.shape[:2], IGNORED_LABEL, dtype=np.int32)

    if map_type == "int":
        if mask.ndim != 2:
            raise ValueError("Integer label maps require a 2D mask.")
        for raw_value, target in mapping.items():
            normalized[mask == int(raw_value)] = _parse_target_label(target)
    elif map_type == "rgb":
        if mask.ndim != 3 or mask.shape[2] < 3:
            raise ValueError("RGB label maps require a color mask.")
        rgb = mask[:, :, :3]
        for raw_value, target in mapping.items():
            color = tuple(int(part) for part in raw_value.split(","))
            hit = np.all(rgb == np.array(color, dtype=rgb.dtype), axis=2)
            normalized[hit] = _parse_target_label(target)
    else:
        raise ValueError(f"Unsupported label-map type: {map_type}")

    valid_mask = normalized != IGNORED_LABEL
    return normalized, valid_mask


def _intersection_counts(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray, class_id: int) -> dict:
    pred = (pred_mask == class_id) & valid_mask
    gt = (gt_mask == class_id) & valid_mask
    intersection = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    dice = (2.0 * intersection / (pred_area + gt_area)) if pred_area + gt_area > 0 else 0.0
    iou = (intersection / union) if union > 0 else 0.0
    return {
        "intersection": intersection,
        "pred_area": pred_area,
        "gt_area": gt_area,
        "union": union,
        "dice": float(dice),
        "iou": float(iou),
    }


def _tissue_area(mask: np.ndarray, valid_mask: np.ndarray) -> int:
    return int(np.logical_and(mask != CLASS_ID_BY_NAME["background"], valid_mask).sum())


def _tumor_ratio(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    tissue_area = _tissue_area(mask, valid_mask)
    tumor_area = int(np.logical_and(mask == CLASS_ID_BY_NAME["tumor"], valid_mask).sum())
    return float(tumor_area / tissue_area * 100.0) if tissue_area > 0 else 0.0


def _necrosis_ratio(mask: np.ndarray, valid_mask: np.ndarray) -> float:
    tumor_area = int(np.logical_and(mask == CLASS_ID_BY_NAME["tumor"], valid_mask).sum())
    necrosis_area = int(np.logical_and(mask == CLASS_ID_BY_NAME["necrosis"], valid_mask).sum())
    return float(necrosis_area / tumor_area * 100.0) if tumor_area > 0 else 0.0


def evaluate_mask_pair(pred_mask: np.ndarray, gt_mask: np.ndarray, valid_mask: np.ndarray) -> dict:
    """Evaluate one predicted segmentation mask against BCSS ground truth."""
    per_class = {}
    for class_name in EVAL_CLASSES:
        class_id = CLASS_ID_BY_NAME[class_name]
        per_class[class_name] = _intersection_counts(pred_mask, gt_mask, valid_mask, class_id)

    gt_tumor_ratio = _tumor_ratio(gt_mask, valid_mask)
    pred_tumor_ratio = _tumor_ratio(pred_mask, valid_mask)
    gt_necrosis_ratio = _necrosis_ratio(gt_mask, valid_mask)
    pred_necrosis_ratio = _necrosis_ratio(pred_mask, valid_mask)

    return {
        "per_class": per_class,
        "gt_tumor_ratio": float(gt_tumor_ratio),
        "pred_tumor_ratio": float(pred_tumor_ratio),
        "gt_necrosis_ratio": float(gt_necrosis_ratio),
        "pred_necrosis_ratio": float(pred_necrosis_ratio),
        "tumor_ratio_abs_error": float(abs(pred_tumor_ratio - gt_tumor_ratio)),
        "necrosis_ratio_abs_error": float(abs(pred_necrosis_ratio - gt_necrosis_ratio)),
    }


def aggregate_mask_results(rows: Iterable[dict]) -> dict:
    """Aggregate per-image BCSS metrics across a dataset."""
    rows = list(rows)
    summary = {"images_evaluated": len(rows), "per_class": {}}

    for class_name in EVAL_CLASSES:
        intersection = sum(row["per_class"][class_name]["intersection"] for row in rows)
        pred_area = sum(row["per_class"][class_name]["pred_area"] for row in rows)
        gt_area = sum(row["per_class"][class_name]["gt_area"] for row in rows)
        union = sum(row["per_class"][class_name]["union"] for row in rows)
        dice = (2.0 * intersection / (pred_area + gt_area)) if pred_area + gt_area > 0 else 0.0
        iou = (intersection / union) if union > 0 else 0.0
        summary["per_class"][class_name] = {
            "dice": float(dice),
            "iou": float(iou),
        }

    summary["tumor_ratio_mae"] = (
        float(np.mean([row["tumor_ratio_abs_error"] for row in rows])) if rows else 0.0
    )
    summary["necrosis_ratio_mae"] = (
        float(np.mean([row["necrosis_ratio_abs_error"] for row in rows])) if rows else 0.0
    )
    return summary


def build_default_area_ratio_params(mpp: float | None = None) -> dict:
    """Build a reusable patholib area-ratio parameter set."""
    return {
        "mpp": mpp,
        "method": "threshold",
    }


def run_bcss_images(
    images_dir: str,
    output_dir: str,
    params: dict,
    normalize_stain: bool = False,
    stain_reference: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Run patholib H&E area-ratio analysis on a directory of BCSS images."""
    import analyze_he
    from patholib.viz.report import generate_he_report

    os.makedirs(output_dir, exist_ok=True)
    image_paths = iter_image_paths(images_dir)
    completed = 0
    skipped = 0
    summaries = []

    for image_path in image_paths:
        image_stem = stem(image_path)
        report_path = os.path.join(output_dir, f"{image_stem}_he_report.json")
        mask_path = os.path.join(output_dir, f"{image_stem}_he_segmentation.npy")
        if os.path.exists(report_path) and os.path.exists(mask_path) and not overwrite:
            skipped += 1
            continue

        image = analyze_he.load_image(image_path)
        if normalize_stain:
            image = analyze_he.apply_stain_normalization(image, stain_reference)
        results = analyze_he.run_area_ratio(image, params)
        generate_he_report(
            results,
            image_path,
            params,
            output_dir,
            mode="area-ratio",
            save_overlay=False,
            save_heatmap=False,
            save_csv=False,
        )
        save_prediction_mask(results["segmentation_mask"], mask_path)
        summaries.append(
            {
                "image_stem": image_stem,
                "tumor_ratio": results.get("tumor_ratio", 0.0),
                "necrosis_ratio": results.get("necrosis_ratio", 0.0),
            }
        )
        completed += 1

    summary = {
        "dataset": "BCSS",
        "images_total": len(image_paths),
        "images_completed": completed,
        "images_skipped": skipped,
        "parameters": params,
        "normalize_stain": normalize_stain,
        "summaries": summaries,
    }
    summary_path = os.path.join(output_dir, "bcss_run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate_bcss_predictions(
    masks_dir: str,
    predictions_dir: str,
    label_map_path: str | None = None,
    require_predictions: bool = True,
) -> tuple[dict, list[dict]]:
    """Evaluate predicted BCSS segmentation masks against ground truth."""
    label_map = load_label_map(label_map_path) if label_map_path else None
    rows = []
    missing_predictions = []

    for mask_path in iter_mask_paths(masks_dir):
        image_stem = stem(mask_path)
        pred_path = os.path.join(predictions_dir, f"{image_stem}_he_segmentation.npy")
        if not os.path.isfile(pred_path):
            missing_predictions.append(image_stem)
            if require_predictions:
                raise FileNotFoundError(f"Prediction mask not found: {pred_path}")
            continue

        gt_raw = load_array_image(mask_path)
        gt_mask, valid_mask = normalize_ground_truth_mask(gt_raw, label_map=label_map)
        pred_mask = np.asarray(np.load(pred_path), dtype=np.int32)
        if pred_mask.shape != gt_mask.shape:
            raise ValueError(
                f"Shape mismatch for {image_stem}: prediction {pred_mask.shape}, ground truth {gt_mask.shape}"
            )
        metrics = evaluate_mask_pair(pred_mask, gt_mask, valid_mask)
        metrics["image_stem"] = image_stem
        rows.append(metrics)

    summary = aggregate_mask_results(rows)
    summary.update(
        {
            "dataset": "BCSS",
            "missing_predictions": missing_predictions,
            "label_map": label_map_path,
        }
    )
    return summary, rows


def write_per_image_csv(rows: Iterable[dict], output_path: str) -> None:
    """Write per-image BCSS metrics to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "image_stem",
        "tumor_dice",
        "tumor_iou",
        "stroma_dice",
        "stroma_iou",
        "necrosis_dice",
        "necrosis_iou",
        "gt_tumor_ratio",
        "pred_tumor_ratio",
        "tumor_ratio_abs_error",
        "gt_necrosis_ratio",
        "pred_necrosis_ratio",
        "necrosis_ratio_abs_error",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_stem": row["image_stem"],
                    "tumor_dice": row["per_class"]["tumor"]["dice"],
                    "tumor_iou": row["per_class"]["tumor"]["iou"],
                    "stroma_dice": row["per_class"]["stroma"]["dice"],
                    "stroma_iou": row["per_class"]["stroma"]["iou"],
                    "necrosis_dice": row["per_class"]["necrosis"]["dice"],
                    "necrosis_iou": row["per_class"]["necrosis"]["iou"],
                    "gt_tumor_ratio": row["gt_tumor_ratio"],
                    "pred_tumor_ratio": row["pred_tumor_ratio"],
                    "tumor_ratio_abs_error": row["tumor_ratio_abs_error"],
                    "gt_necrosis_ratio": row["gt_necrosis_ratio"],
                    "pred_necrosis_ratio": row["pred_necrosis_ratio"],
                    "necrosis_ratio_abs_error": row["necrosis_ratio_abs_error"],
                }
            )


def write_summary_json(summary: dict, output_path: str) -> None:
    """Write a BCSS benchmark summary JSON."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
