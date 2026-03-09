"""Phase-1 benchmark orchestration helpers."""

from __future__ import annotations

import datetime as dt
import json
import os

from benchmarks import bcdata, bcss, her2_ihc_40x, pannuke, phase1_summary


PHASE1_DATASETS = ("BCData", "HER2-IHC-40x", "BCSS", "PanNuke")
DEFAULT_METHODS = ("watershed",)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(path: str, payload: dict) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _method_dir(output_dir: str, dataset_name: str, method_name: str) -> str:
    return os.path.join(output_dir, dataset_name, method_name)


def _run_bcdata_real(args: dict, output_dir: str, method_name: str) -> str:
    params = bcdata.build_default_ihc_params(
        detection_method=method_name,
        use_gpu=bool(args.get("use_gpu", False)),
        fail_fast=bool(args.get("fail_fast", False)),
    )
    bcdata.run_bcdata_split(
        dataset_root=args["bcdata_root"],
        split=args.get("bcdata_split", "test"),
        output_dir=output_dir,
        params=params,
        normalize_stain=bool(args.get("normalize_stain", False)),
        stain_reference=args.get("stain_reference"),
        save_overlay=False,
        overwrite=bool(args.get("overwrite", False)),
    )
    summary, rows = bcdata.evaluate_bcdata_split(
        dataset_root=args["bcdata_root"],
        split=args.get("bcdata_split", "test"),
        predictions_dir=output_dir,
        radius_px=float(args.get("bcdata_match_radius", 6.0)),
        coord_order=args.get("bcdata_coord_order", "xy"),
    )
    eval_json = os.path.join(output_dir, f"bcdata_{args.get('bcdata_split', 'test')}_eval_summary.json")
    per_image_csv = os.path.join(output_dir, f"bcdata_{args.get('bcdata_split', 'test')}_per_image.csv")
    bcdata.write_summary_json(summary, eval_json)
    bcdata.write_per_image_csv(rows, per_image_csv)
    return eval_json


def _run_her2_real(args: dict, output_dir: str, method_name: str) -> str:
    params = her2_ihc_40x.build_default_membrane_params(
        detection_method=method_name,
        use_gpu=bool(args.get("use_gpu", False)),
        fail_fast=bool(args.get("fail_fast", False)),
    )
    her2_ihc_40x.run_her2_split(
        dataset_root=args["her2_root"],
        split=args.get("her2_split", "test"),
        output_dir=output_dir,
        params=params,
        normalize_stain=bool(args.get("normalize_stain", False)),
        stain_reference=args.get("stain_reference"),
        save_overlay=False,
        overwrite=bool(args.get("overwrite", False)),
    )
    summary, rows = her2_ihc_40x.evaluate_her2_split(
        dataset_root=args["her2_root"],
        split=args.get("her2_split", "test"),
        reports_dir=output_dir,
    )
    eval_json = os.path.join(output_dir, f"her2_ihc_40x_{args.get('her2_split', 'test')}_eval_summary.json")
    per_image_csv = os.path.join(output_dir, f"her2_ihc_40x_{args.get('her2_split', 'test')}_per_image.csv")
    her2_ihc_40x.write_summary_json(summary, eval_json)
    her2_ihc_40x.write_per_image_csv(rows, per_image_csv)
    return eval_json


def _run_bcss_real(args: dict, output_dir: str) -> str:
    params = bcss.build_default_area_ratio_params(mpp=args.get("bcss_mpp"))
    bcss.run_bcss_images(
        images_dir=args["bcss_images_dir"],
        output_dir=output_dir,
        params=params,
        normalize_stain=bool(args.get("normalize_stain", False)),
        stain_reference=args.get("stain_reference"),
        overwrite=bool(args.get("overwrite", False)),
    )
    summary, rows = bcss.evaluate_bcss_predictions(
        masks_dir=args["bcss_masks_dir"],
        predictions_dir=output_dir,
        label_map_path=args.get("bcss_label_map_json"),
    )
    eval_json = os.path.join(output_dir, "bcss_eval_summary.json")
    per_image_csv = os.path.join(output_dir, "bcss_per_image.csv")
    bcss.write_summary_json(summary, eval_json)
    bcss.write_per_image_csv(rows, per_image_csv)
    return eval_json


def _run_pannuke_real(args: dict, output_dir: str, method_name: str) -> str:
    params = pannuke.build_default_inflammation_params(
        detection_method=method_name,
        use_gpu=bool(args.get("use_gpu", False)),
        fail_fast=bool(args.get("fail_fast", False)),
    )
    pannuke.run_pannuke_images(
        images_npy_path=args["pannuke_images_npy"],
        output_dir=output_dir,
        params=params,
        start_index=int(args.get("pannuke_start_index", 0)),
        limit=args.get("pannuke_limit"),
    )
    summary, rows = pannuke.evaluate_pannuke_predictions(
        masks_npy_path=args["pannuke_masks_npy"],
        predictions_dir=output_dir,
        inflammatory_channel=int(args.get("pannuke_inflammatory_channel", pannuke.DEFAULT_INFLAMMATORY_CHANNEL)),
    )
    eval_json = os.path.join(output_dir, "pannuke_eval_summary.json")
    per_patch_csv = os.path.join(output_dir, "pannuke_per_patch.csv")
    pannuke.write_summary_json(summary, eval_json)
    pannuke.write_per_image_csv(rows, per_patch_csv)
    return eval_json


def _smoke_metrics(dataset_name: str, method_name: str) -> dict:
    offset = 0.04 if method_name == "cellpose" else 0.0
    if dataset_name == "BCData":
        return {
            "dataset": "BCData",
            "split": "test",
            "images_evaluated": 3,
            "positive": {"f1": 0.78 + offset},
            "negative": {"f1": 0.74 + offset},
            "mean_f1": 0.76 + offset,
            "positive_percentage_mae": 5.0 - offset,
            "positive_percentage_rmse": 6.2 - offset,
            "positive_percentage_pearson_r": 0.81 + offset,
            "match_radius_px": 6.0,
            "annotation_coord_order": "xy",
        }
    if dataset_name == "HER2-IHC-40x":
        return {
            "dataset": "HER2-IHC-40x",
            "split": "test",
            "images_evaluated": 3,
            "accuracy": 0.67 + offset,
            "macro_f1": 0.61 + offset,
            "quadratic_weighted_kappa": 0.72 + offset,
            "heuristic": {
                "zero_cutoff": 1.0,
                "positive_cutoff": 10.0,
                "strong_grade_cutoff": 2.5,
                "strong_fraction_cutoff": 0.30,
            },
        }
    if dataset_name == "BCSS":
        return {
            "dataset": "BCSS",
            "images_evaluated": 2,
            "per_class": {
                "tumor": {"dice": 0.70, "iou": 0.55},
                "stroma": {"dice": 0.62, "iou": 0.47},
                "necrosis": {"dice": 0.50, "iou": 0.35},
            },
            "tumor_ratio_mae": 9.5,
            "necrosis_ratio_mae": 12.5,
            "label_map": "smoke",
        }
    return {
        "dataset": "PanNuke",
        "split": "test",
        "images_evaluated": 4,
        "binary_nuclei_dice": 0.73 + offset,
        "aji": 0.56 + offset,
        "pq": 0.49 + offset,
        "all_nuclei_f1": 0.68 + offset,
        "inflammatory_f1": 0.41 + offset,
        "inflammatory_channel": 1,
    }


def _write_smoke_run_and_eval(dataset_name: str, method_name: str, output_dir: str) -> str:
    _ensure_dir(output_dir)
    if dataset_name == "BCData":
        run_summary_path = os.path.join(output_dir, "bcdata_test_run_summary.json")
        eval_summary_path = os.path.join(output_dir, "bcdata_test_eval_summary.json")
        run_summary = {"parameters": {"detection_method": method_name}, "normalize_stain": False}
    elif dataset_name == "HER2-IHC-40x":
        run_summary_path = os.path.join(output_dir, "her2_ihc_40x_test_run_summary.json")
        eval_summary_path = os.path.join(output_dir, "her2_ihc_40x_test_eval_summary.json")
        run_summary = {"parameters": {"detection_method": method_name}, "normalize_stain": False}
    elif dataset_name == "BCSS":
        run_summary_path = os.path.join(output_dir, "bcss_run_summary.json")
        eval_summary_path = os.path.join(output_dir, "bcss_eval_summary.json")
        run_summary = {"parameters": {"method": "threshold"}, "normalize_stain": False}
    elif dataset_name == "PanNuke":
        run_summary_path = os.path.join(output_dir, "pannuke_run_summary.json")
        eval_summary_path = os.path.join(output_dir, "pannuke_eval_summary.json")
        run_summary = {"parameters": {"detection_method": method_name}, "normalize_stain": False}
    else:
        raise ValueError(f"Unsupported smoke dataset: {dataset_name}")

    _write_json(run_summary_path, run_summary)
    _write_json(eval_summary_path, _smoke_metrics(dataset_name, method_name))
    return eval_summary_path


def _validate_real_inputs(args: dict, selected_datasets: tuple[str, ...]) -> None:
    required = []
    if "BCData" in selected_datasets:
        required.append(("bcdata_root", args.get("bcdata_root")))
    if "HER2-IHC-40x" in selected_datasets:
        required.append(("her2_root", args.get("her2_root")))
    if "BCSS" in selected_datasets:
        required.append(("bcss_images_dir", args.get("bcss_images_dir")))
        required.append(("bcss_masks_dir", args.get("bcss_masks_dir")))
    if "PanNuke" in selected_datasets:
        required.append(("pannuke_images_npy", args.get("pannuke_images_npy")))
        required.append(("pannuke_masks_npy", args.get("pannuke_masks_npy")))

    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(f"Missing required dataset paths for phase1 run: {', '.join(missing)}")


def run_phase1(args: dict) -> dict:
    """Run the full phase-1 benchmark workflow and aggregate results."""
    output_dir = _ensure_dir(args["output_dir"])
    selected_datasets = tuple(args.get("datasets") or PHASE1_DATASETS)
    methods = tuple(args.get("methods") or DEFAULT_METHODS)
    smoke = bool(args.get("smoke", False))

    if not smoke:
        _validate_real_inputs(args, selected_datasets)

    eval_json_paths = []
    for dataset_name in selected_datasets:
        if dataset_name == "BCSS":
            method_names = ("threshold",)
        else:
            method_names = methods

        for method_name in method_names:
            dataset_output_dir = _method_dir(output_dir, dataset_name, method_name)
            if smoke:
                eval_json_paths.append(_write_smoke_run_and_eval(dataset_name, method_name, dataset_output_dir))
                continue

            if dataset_name == "BCData":
                eval_json_paths.append(_run_bcdata_real(args, dataset_output_dir, method_name))
            elif dataset_name == "HER2-IHC-40x":
                eval_json_paths.append(_run_her2_real(args, dataset_output_dir, method_name))
            elif dataset_name == "BCSS":
                eval_json_paths.append(_run_bcss_real(args, dataset_output_dir))
            elif dataset_name == "PanNuke":
                eval_json_paths.append(_run_pannuke_real(args, dataset_output_dir, method_name))
            else:
                raise ValueError(f"Unsupported dataset: {dataset_name}")

    summary_dir = _ensure_dir(os.path.join(output_dir, "summary"))
    run_date = args.get("run_date") or dt.datetime.now().isoformat(timespec="seconds")
    long_rows, wide_rows = phase1_summary.aggregate_eval_summaries(
        eval_summary_paths=eval_json_paths,
        commit=args.get("commit", ""),
        run_date=run_date,
    )
    phase1_summary.write_csv(
        long_rows,
        os.path.join(summary_dir, "phase1_metrics_long.csv"),
        phase1_summary.LONG_FIELDNAMES,
    )
    phase1_summary.write_csv(
        wide_rows,
        os.path.join(summary_dir, "phase1_summary.csv"),
        phase1_summary.WIDE_FIELDNAMES,
    )
    phase1_summary.write_markdown(
        phase1_summary.render_markdown_summary(wide_rows),
        os.path.join(summary_dir, "phase1_summary.md"),
    )
    return {
        "output_dir": output_dir,
        "summary_dir": summary_dir,
        "eval_json_paths": eval_json_paths,
        "datasets": list(selected_datasets),
        "methods": list(methods),
        "smoke": smoke,
    }
