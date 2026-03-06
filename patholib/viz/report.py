"""
patholib.viz.report
===================
Generate comprehensive analysis reports for pathology image analysis.
Exports JSON reports, per-cell CSV data, and overlay images.
"""

import json
import os
import datetime
import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

SOFTWARE_VERSION = "patholib 0.1.0"


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        return super().default(obj)


def results_to_dataframe(cell_data):
    """Convert per-cell data list to a pandas DataFrame."""
    if not cell_data:
        return pd.DataFrame() if HAS_PANDAS else []
    rows = []
    for cell in cell_data:
        row = {}
        centroid = cell.get("centroid", (0, 0))
        row["centroid_y"] = centroid[0]
        row["centroid_x"] = centroid[1]
        for key in ["area", "circularity", "eccentricity", "solidity",
                     "label", "cell_type", "grade", "intensity_mean",
                     "intensity_std"]:
            if key in cell:
                row[key] = cell[key]
        rows.append(row)
    return pd.DataFrame(rows) if HAS_PANDAS else rows


def _build_metadata(input_path, params):
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "software": SOFTWARE_VERSION,
        "input_file": str(input_path) if input_path else None,
        "parameters": _sanitize_params(params),
    }


def _sanitize_params(params):
    if params is None:
        return {}
    clean = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            clean[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            clean[k] = int(v)
        elif isinstance(v, (np.floating,)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean


def _save_overlay_image(arr, output_path):
    if HAS_PIL:
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        Image.fromarray(arr, mode).save(output_path)
    else:
        try:
            from skimage.io import imsave
            imsave(output_path, arr)
        except ImportError:
            import warnings
            warnings.warn("Neither PIL nor skimage available; cannot save overlay.")


def _save_csv(cell_data, output_path):
    df = results_to_dataframe(cell_data)
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        df.to_csv(output_path, index=False)
    else:
        import csv
        if not df:
            return
        keys = list(df[0].keys()) if isinstance(df, list) and df else []
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(df)


def generate_ihc_report(results, input_path, params, output_dir,
                        save_overlay=True, save_csv=True):
    """Generate IHC analysis report. Returns path to JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(str(input_path)))[0]
    json_path = os.path.join(output_dir, f"{basename}_ihc_report.json")

    report = {
        "metadata": _build_metadata(input_path, params),
        "analysis_type": "ihc",
        "summary": {},
    }
    for key in ["total_cells", "positive_cells", "negative_cells",
                "h_score", "positive_percentage", "allred_score",
                "grade_counts", "grade_percentages", "stain_type", "marker"]:
        if key in results:
            val = results[key]
            if isinstance(val, np.ndarray):
                val = val.tolist()
            report["summary"][key] = val

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)

    if save_overlay and "overlay" in results:
        _save_overlay_image(results["overlay"],
                            os.path.join(output_dir, f"{basename}_ihc_overlay.png"))
    if save_csv and "cell_data" in results:
        _save_csv(results["cell_data"],
                  os.path.join(output_dir, f"{basename}_ihc_cells.csv"))
    return json_path


def generate_he_report(results, input_path, params, output_dir, mode,
                       save_overlay=True, save_heatmap=False, save_csv=True):
    """Generate H&E analysis report. Returns path to JSON report."""
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(str(input_path)))[0]
    json_path = os.path.join(output_dir, f"{basename}_he_report.json")

    report = {
        "metadata": _build_metadata(input_path, params),
        "analysis_type": f"he_{mode}",
        "summary": {},
    }

    if mode in ("inflammation", "both"):
        inflam = results.get("inflammation", results)
        isummary = {}
        for key in ["total_nuclei", "inflammatory_cells", "parenchymal_cells",
                     "inflammatory_density", "inflammation_score"]:
            if key in inflam:
                v = inflam[key]
                isummary[key] = v.tolist() if isinstance(v, np.ndarray) else v
        if "grid_scores" in inflam:
            isummary["grid_scores"] = inflam["grid_scores"].tolist()
        report["summary"]["inflammation"] = isummary

    if mode in ("area-ratio", "both"):
        area = results.get("area_ratio", results)
        asummary = {}
        for key in ["tissue_area_px", "tissue_area_um2", "tumor_ratio",
                     "necrosis_ratio", "regions"]:
            if key in area:
                v = area[key]
                asummary[key] = v.tolist() if isinstance(v, np.ndarray) else v
        report["summary"]["area_ratio"] = asummary

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, cls=NumpyJSONEncoder, indent=2, ensure_ascii=False)

    if save_overlay:
        if "overlay" in results:
            _save_overlay_image(results["overlay"],
                                os.path.join(output_dir, f"{basename}_he_overlay.png"))
        if "segmentation_overlay" in results:
            _save_overlay_image(results["segmentation_overlay"],
                                os.path.join(output_dir, f"{basename}_he_segmentation.png"))

    if save_heatmap and "heatmap" in results:
        _save_overlay_image(results["heatmap"],
                            os.path.join(output_dir, f"{basename}_he_heatmap.png"))

    if save_csv:
        cd = results.get("cell_data")
        if cd is None:
            inf = results.get("inflammation", {})
            cd = inf.get("cell_data")
        if cd:
            _save_csv(cd, os.path.join(output_dir, f"{basename}_he_cells.csv"))

    return json_path
