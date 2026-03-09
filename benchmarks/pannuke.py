"""PanNuke benchmark helpers for patholib phase-1 evaluation."""

from __future__ import annotations

import csv
import json
import os
from typing import Iterable

import numpy as np


DEFAULT_INFLAMMATORY_CHANNEL = 1


def load_npy_array(path: str) -> np.ndarray:
    """Load a numpy array from disk."""
    return np.load(path)


def relabel_instances(instance_map: np.ndarray) -> np.ndarray:
    """Relabel a 2D instance map to contiguous ids."""
    instance_map = np.asarray(instance_map)
    relabeled = np.zeros(instance_map.shape, dtype=np.int32)
    next_id = 1
    for raw_id in np.unique(instance_map):
        if raw_id <= 0:
            continue
        relabeled[instance_map == raw_id] = next_id
        next_id += 1
    return relabeled


def merge_instance_channels(mask_stack: np.ndarray) -> np.ndarray:
    """Merge a channel-wise instance stack into one 2D instance map."""
    mask_stack = np.asarray(mask_stack)
    if mask_stack.ndim != 3:
        raise ValueError(f"Expected (H, W, C) mask stack, got {mask_stack.shape}")
    merged = np.zeros(mask_stack.shape[:2], dtype=np.int32)
    next_id = 1
    for channel in range(mask_stack.shape[2]):
        channel_map = mask_stack[:, :, channel]
        for raw_id in np.unique(channel_map):
            if raw_id <= 0:
                continue
            merged[channel_map == raw_id] = next_id
            next_id += 1
    return merged


def extract_inflammatory_instances(mask_stack: np.ndarray, channel: int = DEFAULT_INFLAMMATORY_CHANNEL) -> np.ndarray:
    """Extract a relabeled inflammatory-channel instance map from PanNuke masks."""
    mask_stack = np.asarray(mask_stack)
    if mask_stack.ndim != 3:
        raise ValueError(f"Expected (H, W, C) mask stack, got {mask_stack.shape}")
    if channel < 0 or channel >= mask_stack.shape[2]:
        raise ValueError(f"Inflammatory channel {channel} out of range for {mask_stack.shape}")
    return relabel_instances(mask_stack[:, :, channel])


def build_default_inflammation_params(
    detection_method: str = "watershed",
    inflammatory_max_area: int = 80,
    inflammatory_min_circularity: float = 0.7,
    fail_fast: bool = False,
    use_gpu: bool = False,
) -> dict:
    """Build a reusable patholib inflammation parameter set for PanNuke runs."""
    return {
        "detection_method": detection_method,
        "inflammatory_max_area": inflammatory_max_area,
        "inflammatory_min_circularity": inflammatory_min_circularity,
        "fail_fast": fail_fast,
        "use_gpu": use_gpu,
    }


def _compute_overlap_info(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_mask = np.asarray(gt_mask, dtype=np.int32)
    pred_mask = np.asarray(pred_mask, dtype=np.int32)
    gt_ids = [int(x) for x in np.unique(gt_mask) if x > 0]
    pred_ids = [int(x) for x in np.unique(pred_mask) if x > 0]

    gt_areas = {gt_id: int(np.sum(gt_mask == gt_id)) for gt_id in gt_ids}
    pred_areas = {pred_id: int(np.sum(pred_mask == pred_id)) for pred_id in pred_ids}

    overlaps = {}
    if gt_ids and pred_ids:
        factor = int(pred_mask.max()) + 1
        codes = gt_mask.astype(np.int64) * factor + pred_mask.astype(np.int64)
        valid = np.logical_and(gt_mask > 0, pred_mask > 0)
        unique_codes, counts = np.unique(codes[valid], return_counts=True)
        for code, count in zip(unique_codes, counts):
            gt_id = int(code // factor)
            pred_id = int(code % factor)
            overlaps[(gt_id, pred_id)] = int(count)

    return gt_ids, pred_ids, gt_areas, pred_areas, overlaps


def binary_dice_score(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute binary Dice over foreground nuclei."""
    gt_fg = np.asarray(gt_mask) > 0
    pred_fg = np.asarray(pred_mask) > 0
    intersection = int(np.logical_and(gt_fg, pred_fg).sum())
    total = int(gt_fg.sum() + pred_fg.sum())
    return float(2.0 * intersection / total) if total > 0 else 0.0


def _iou_candidates(gt_mask: np.ndarray, pred_mask: np.ndarray):
    gt_ids, pred_ids, gt_areas, pred_areas, overlaps = _compute_overlap_info(gt_mask, pred_mask)
    candidates = []
    for (gt_id, pred_id), intersection in overlaps.items():
        union = gt_areas[gt_id] + pred_areas[pred_id] - intersection
        iou = float(intersection / union) if union > 0 else 0.0
        candidates.append((iou, gt_id, pred_id, intersection, union))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return gt_ids, pred_ids, gt_areas, pred_areas, candidates


def match_instances(gt_mask: np.ndarray, pred_mask: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """Greedily match instances by IoU threshold."""
    gt_ids, pred_ids, _, _, candidates = _iou_candidates(gt_mask, pred_mask)
    matched_gt = set()
    matched_pred = set()
    matched_ious = []

    for iou, gt_id, pred_id, _, _ in candidates:
        if iou < iou_threshold:
            continue
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    f1 = (2.0 * tp / (2.0 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    sq = (float(np.mean(matched_ious)) if matched_ious else 0.0)
    rq = (tp / (tp + 0.5 * fp + 0.5 * fn)) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = sq * rq
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "f1": float(f1),
        "sq": float(sq),
        "rq": float(rq),
        "pq": float(pq),
    }


def aggregated_jaccard_index(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    """Compute AJI on two instance masks."""
    gt_ids, pred_ids, gt_areas, pred_areas, candidates = _iou_candidates(gt_mask, pred_mask)
    matched_gt = set()
    matched_pred = set()
    intersection_sum = 0
    union_sum = 0

    for iou, gt_id, pred_id, intersection, union in candidates:
        if iou <= 0:
            continue
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        intersection_sum += intersection
        union_sum += union

    for gt_id in gt_ids:
        if gt_id not in matched_gt:
            union_sum += gt_areas[gt_id]
    for pred_id in pred_ids:
        if pred_id not in matched_pred:
            union_sum += pred_areas[pred_id]

    return float(intersection_sum / union_sum) if union_sum > 0 else 0.0


def evaluate_patch(
    gt_mask_stack: np.ndarray,
    pred_instances: np.ndarray,
    pred_inflammatory_instances: np.ndarray,
    inflammatory_channel: int = DEFAULT_INFLAMMATORY_CHANNEL,
) -> dict:
    """Evaluate one PanNuke patch."""
    gt_all = merge_instance_channels(gt_mask_stack)
    gt_inflammatory = extract_inflammatory_instances(gt_mask_stack, channel=inflammatory_channel)
    pred_all = relabel_instances(pred_instances)
    pred_inflammatory = relabel_instances(pred_inflammatory_instances)

    all_match = match_instances(gt_all, pred_all, iou_threshold=0.5)
    inflammatory_match = match_instances(gt_inflammatory, pred_inflammatory, iou_threshold=0.5)

    return {
        "binary_nuclei_dice": binary_dice_score(gt_all, pred_all),
        "aji": aggregated_jaccard_index(gt_all, pred_all),
        "pq": all_match["pq"],
        "all_nuclei_f1": all_match["f1"],
        "inflammatory_precision": float(
            inflammatory_match["tp"] / (inflammatory_match["tp"] + inflammatory_match["fp"])
        )
        if (inflammatory_match["tp"] + inflammatory_match["fp"]) > 0
        else 0.0,
        "inflammatory_recall": float(
            inflammatory_match["tp"] / (inflammatory_match["tp"] + inflammatory_match["fn"])
        )
        if (inflammatory_match["tp"] + inflammatory_match["fn"]) > 0
        else 0.0,
        "inflammatory_f1": inflammatory_match["f1"],
    }


def aggregate_patch_results(rows: Iterable[dict]) -> dict:
    """Aggregate PanNuke per-patch metrics across a dataset slice."""
    rows = list(rows)
    keys = (
        "binary_nuclei_dice",
        "aji",
        "pq",
        "all_nuclei_f1",
        "inflammatory_precision",
        "inflammatory_recall",
        "inflammatory_f1",
    )
    summary = {"dataset": "PanNuke", "images_evaluated": len(rows)}
    for key in keys:
        summary[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    return summary


def run_pannuke_images(
    images_npy_path: str,
    output_dir: str,
    params: dict,
    start_index: int = 0,
    limit: int | None = None,
) -> dict:
    """Run patholib inflammation analysis on a PanNuke image array."""
    import analyze_he

    images = np.asarray(np.load(images_npy_path))
    if images.ndim != 4 or images.shape[-1] < 3:
        raise ValueError(f"Expected image array shaped (N, H, W, 3), got {images.shape}")

    end_index = images.shape[0] if limit is None else min(images.shape[0], start_index + limit)
    subset = images[start_index:end_index]
    pred_instances = np.zeros(subset.shape[:3], dtype=np.int32)
    pred_inflammatory = np.zeros(subset.shape[:3], dtype=np.int32)

    os.makedirs(output_dir, exist_ok=True)
    summaries = []
    for offset, image in enumerate(subset):
        results = analyze_he.run_inflammation(image[:, :, :3], params)
        labels = np.asarray(results["labels"], dtype=np.int32)
        if labels.shape != subset.shape[1:3]:
            raise ValueError(
                f"Prediction label shape {labels.shape} does not match image shape {subset.shape[1:3]}"
            )
        inflammatory_labels = np.zeros_like(labels, dtype=np.int32)
        inflammatory_ids = {
            int(cell["label"])
            for cell in results.get("cell_data", [])
            if cell.get("cell_type") == "inflammatory"
        }
        for label_id in inflammatory_ids:
            inflammatory_labels[labels == label_id] = label_id

        pred_instances[offset] = labels
        pred_inflammatory[offset] = relabel_instances(inflammatory_labels)
        summaries.append(
            {
                "patch_index": int(start_index + offset),
                "total_nuclei": int(results.get("total_nuclei", 0)),
                "inflammatory_cells": int(results.get("inflammatory_cells", 0)),
            }
        )

    np.save(os.path.join(output_dir, "pannuke_pred_instances.npy"), pred_instances)
    np.save(os.path.join(output_dir, "pannuke_pred_inflammatory_instances.npy"), pred_inflammatory)

    summary = {
        "dataset": "PanNuke",
        "images_total": int(images.shape[0]),
        "start_index": int(start_index),
        "images_evaluated": int(subset.shape[0]),
        "parameters": params,
        "summaries": summaries,
    }
    with open(os.path.join(output_dir, "pannuke_run_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def evaluate_pannuke_predictions(
    masks_npy_path: str,
    predictions_dir: str,
    inflammatory_channel: int = DEFAULT_INFLAMMATORY_CHANNEL,
) -> tuple[dict, list[dict]]:
    """Evaluate PanNuke predictions saved by run_pannuke_images."""
    gt_masks = np.asarray(np.load(masks_npy_path))
    pred_instances = np.asarray(np.load(os.path.join(predictions_dir, "pannuke_pred_instances.npy")))
    pred_inflammatory = np.asarray(
        np.load(os.path.join(predictions_dir, "pannuke_pred_inflammatory_instances.npy"))
    )
    run_summary_path = os.path.join(predictions_dir, "pannuke_run_summary.json")
    run_summary = {}
    if os.path.isfile(run_summary_path):
        with open(run_summary_path, encoding="utf-8") as handle:
            run_summary = json.load(handle)

    start_index = int(run_summary.get("start_index", 0))
    images_evaluated = pred_instances.shape[0]
    gt_subset = gt_masks[start_index:start_index + images_evaluated]
    if gt_subset.shape[0] != images_evaluated:
        raise ValueError("Ground-truth mask array is shorter than the prediction slice.")

    rows = []
    for idx in range(images_evaluated):
        metrics = evaluate_patch(
            gt_mask_stack=gt_subset[idx],
            pred_instances=pred_instances[idx],
            pred_inflammatory_instances=pred_inflammatory[idx],
            inflammatory_channel=inflammatory_channel,
        )
        metrics["patch_index"] = int(start_index + idx)
        rows.append(metrics)

    summary = aggregate_patch_results(rows)
    summary.update(
        {
            "split": "",
            "start_index": start_index,
            "inflammatory_channel": int(inflammatory_channel),
        }
    )
    return summary, rows


def write_per_image_csv(rows: Iterable[dict], output_path: str) -> None:
    """Write per-patch PanNuke metrics to CSV."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = [
        "patch_index",
        "binary_nuclei_dice",
        "aji",
        "pq",
        "all_nuclei_f1",
        "inflammatory_precision",
        "inflammatory_recall",
        "inflammatory_f1",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict, output_path: str) -> None:
    """Write a PanNuke benchmark summary JSON."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
