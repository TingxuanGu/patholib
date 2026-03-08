import os
import tempfile
import unittest
from unittest import mock

from benchmarks import bcdata


class BCDataBenchmarkTests(unittest.TestCase):
    def test_infer_prediction_label_prefers_cell_type(self):
        self.assertEqual(
            bcdata.infer_prediction_label({"cell_type": "positive", "grade": "0"}),
            "positive",
        )
        self.assertEqual(
            bcdata.infer_prediction_label({"cell_type": "negative", "grade": "3"}),
            "negative",
        )

    def test_infer_prediction_label_falls_back_to_grade(self):
        self.assertEqual(bcdata.infer_prediction_label({"grade": "2"}), "positive")
        self.assertEqual(bcdata.infer_prediction_label({"grade": "0"}), "negative")
        self.assertIsNone(bcdata.infer_prediction_label({"grade": ""}))

    def test_greedy_match_points_returns_expected_counts(self):
        tp, fp, fn = bcdata.greedy_match_points(
            gt_points=[(10.0, 10.0), (30.0, 30.0)],
            pred_points=[(11.0, 10.0), (31.0, 29.0), (80.0, 80.0)],
            radius_px=3.0,
        )
        self.assertEqual((tp, fp, fn), (2, 1, 0))

    def test_load_prediction_cells_reads_patholib_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/cells.csv"
            with open(csv_path, "w", encoding="utf-8") as handle:
                handle.write("centroid_y,centroid_x,cell_type,grade\n")
                handle.write("10,20,positive,2\n")
                handle.write("30,40,,0\n")

            rows = bcdata.load_prediction_cells(csv_path)

        self.assertEqual(rows, [{"x": 20.0, "y": 10.0, "label": "positive"}, {"x": 40.0, "y": 30.0, "label": "negative"}])

    def test_evaluate_patch_reports_class_metrics_and_percentages(self):
        metrics = bcdata.evaluate_patch(
            gt_positive=[(10.0, 10.0), (20.0, 20.0)],
            gt_negative=[(40.0, 40.0)],
            pred_positive=[(10.5, 9.5), (70.0, 70.0)],
            pred_negative=[(39.0, 41.0)],
            radius_px=2.0,
        )

        self.assertEqual(metrics["positive"]["tp"], 1)
        self.assertEqual(metrics["positive"]["fp"], 1)
        self.assertEqual(metrics["positive"]["fn"], 1)
        self.assertEqual(metrics["negative"]["tp"], 1)
        self.assertAlmostEqual(metrics["gt_positive_percentage"], 66.6666666667, places=5)
        self.assertAlmostEqual(metrics["pred_positive_percentage"], 66.6666666667, places=5)
        self.assertAlmostEqual(metrics["positive_percentage_abs_error"], 0.0, places=5)

    def test_aggregate_patch_results_computes_summary_metrics(self):
        rows = [
            {
                "positive": {"tp": 4, "fp": 1, "fn": 1, "precision": 0.8, "recall": 0.8, "f1": 0.8},
                "negative": {"tp": 3, "fp": 1, "fn": 0, "precision": 0.75, "recall": 1.0, "f1": 0.8571428571},
                "gt_positive_percentage": 50.0,
                "pred_positive_percentage": 40.0,
                "positive_percentage_abs_error": 10.0,
            },
            {
                "positive": {"tp": 2, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.6666666667, "f1": 0.8},
                "negative": {"tp": 2, "fp": 1, "fn": 1, "precision": 0.6666666667, "recall": 0.6666666667, "f1": 0.6666666667},
                "gt_positive_percentage": 25.0,
                "pred_positive_percentage": 20.0,
                "positive_percentage_abs_error": 5.0,
            },
        ]

        summary = bcdata.aggregate_patch_results(rows)

        self.assertEqual(summary["images_evaluated"], 2)
        self.assertAlmostEqual(summary["positive"]["f1"], 0.8, places=5)
        self.assertAlmostEqual(summary["positive_percentage_mae"], 7.5, places=5)
        self.assertGreaterEqual(summary["positive_percentage_rmse"], 0.0)

    def test_build_default_ihc_params_matches_bcdata_defaults(self):
        params = bcdata.build_default_ihc_params(detection_method="cellpose", use_gpu=True)
        self.assertEqual(params["stain_type"], "nuclear")
        self.assertEqual(params["marker"], "Ki67")
        self.assertEqual(params["detection_method"], "cellpose")
        self.assertTrue(params["use_gpu"])

    def test_evaluate_bcdata_split_allows_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/images/test"
            os.makedirs(image_dir)
            open(f"{image_dir}/patch1.png", "wb").close()
            summary, rows = bcdata.evaluate_bcdata_split(
                dataset_root=tmpdir,
                split="test",
                predictions_dir=f"{tmpdir}/preds",
                require_predictions=False,
            )

        self.assertEqual(rows, [])
        self.assertEqual(summary["images_evaluated"], 0)
        self.assertEqual(summary["missing_predictions"], ["patch1"])

    def test_run_bcdata_split_writes_summary_with_mocked_analysis(self):
        params = bcdata.build_default_ihc_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/images/test"
            output_dir = f"{tmpdir}/out"
            os.makedirs(image_dir)
            open(f"{image_dir}/patch1.png", "wb").close()

            fake_results = {
                "total_cells": 3,
                "positive_cells": 2,
                "negative_cells": 1,
                "positive_percentage": 66.7,
                "cell_data": [],
            }

            with mock.patch("analyze_ihc.load_image", return_value="image") as load_image, mock.patch(
                "analyze_ihc.run_analysis", return_value=fake_results
            ) as run_analysis, mock.patch(
                "patholib.viz.report.generate_ihc_report", return_value=f"{output_dir}/patch1_ihc_report.json"
            ) as generate_report:
                summary = bcdata.run_bcdata_split(
                    dataset_root=tmpdir,
                    split="test",
                    output_dir=output_dir,
                    params=params,
                )

        self.assertEqual(summary["images_completed"], 1)
        load_image.assert_called_once()
        run_analysis.assert_called_once()
        generate_report.assert_called_once()

    def test_write_per_image_csv_outputs_expected_header(self):
        rows = [
            {
                "image_stem": "1",
                "gt_positive": 1,
                "gt_negative": 2,
                "pred_positive": 1,
                "pred_negative": 1,
                "positive": {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
                "negative": {"tp": 1, "fp": 0, "fn": 1, "precision": 1.0, "recall": 0.5, "f1": 0.6666666667},
                "mean_f1": 0.8333333333,
                "gt_positive_percentage": 33.3,
                "pred_positive_percentage": 50.0,
                "positive_percentage_abs_error": 16.7,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/per_image.csv"
            bcdata.write_per_image_csv(rows, output_path)
            with open(output_path, encoding="utf-8") as handle:
                content = handle.read()

        self.assertIn("image_stem", content)
        self.assertIn("positive_f1", content)
        self.assertIn("negative_f1", content)

    def test_write_summary_json_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/nested/summary.json"
            bcdata.write_summary_json({"dataset": "BCData"}, output_path)
            self.assertTrue(os.path.isfile(output_path))


if __name__ == "__main__":
    unittest.main()
