import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmarks import bcss


class BCSSBenchmarkTests(unittest.TestCase):
    def test_normalize_ground_truth_mask_without_map_accepts_integer_mask(self):
        mask = np.array([[0, 2], [3, 4]], dtype=np.uint8)
        normalized, valid = bcss.normalize_ground_truth_mask(mask)
        self.assertTrue(np.array_equal(normalized, mask))
        self.assertTrue(np.all(valid))

    def test_normalize_ground_truth_mask_with_int_map(self):
        raw_mask = np.array([[1, 2], [3, 255]], dtype=np.uint8)
        normalized, valid = bcss.normalize_ground_truth_mask(
            raw_mask,
            label_map={
                "type": "int",
                "mapping": {
                    "1": "tumor",
                    "2": "stroma",
                    "3": "necrosis",
                    "255": "ignore",
                },
            },
        )
        expected = np.array(
            [
                [bcss.CLASS_ID_BY_NAME["tumor"], bcss.CLASS_ID_BY_NAME["stroma"]],
                [bcss.CLASS_ID_BY_NAME["necrosis"], bcss.IGNORED_LABEL],
            ],
            dtype=np.int32,
        )
        self.assertTrue(np.array_equal(normalized, expected))
        self.assertFalse(valid[1, 1])

    def test_evaluate_mask_pair_reports_region_metrics(self):
        gt_mask = np.array(
            [
                [0, 2, 2],
                [0, 3, 4],
                [0, 0, 4],
            ],
            dtype=np.int32,
        )
        pred_mask = np.array(
            [
                [0, 2, 0],
                [0, 3, 4],
                [0, 0, 4],
            ],
            dtype=np.int32,
        )
        valid_mask = np.ones_like(gt_mask, dtype=bool)
        metrics = bcss.evaluate_mask_pair(pred_mask, gt_mask, valid_mask)

        self.assertAlmostEqual(metrics["per_class"]["tumor"]["dice"], 2.0 / 3.0, places=5)
        self.assertAlmostEqual(metrics["per_class"]["necrosis"]["dice"], 1.0, places=5)
        self.assertGreaterEqual(metrics["tumor_ratio_abs_error"], 0.0)

    def test_aggregate_mask_results_computes_summary(self):
        rows = [
            {
                "per_class": {
                    "tumor": {"intersection": 4, "pred_area": 5, "gt_area": 5, "union": 6},
                    "stroma": {"intersection": 3, "pred_area": 4, "gt_area": 4, "union": 5},
                    "necrosis": {"intersection": 2, "pred_area": 2, "gt_area": 3, "union": 3},
                },
                "tumor_ratio_abs_error": 5.0,
                "necrosis_ratio_abs_error": 2.0,
            },
            {
                "per_class": {
                    "tumor": {"intersection": 1, "pred_area": 2, "gt_area": 3, "union": 4},
                    "stroma": {"intersection": 1, "pred_area": 2, "gt_area": 1, "union": 2},
                    "necrosis": {"intersection": 0, "pred_area": 1, "gt_area": 1, "union": 2},
                },
                "tumor_ratio_abs_error": 3.0,
                "necrosis_ratio_abs_error": 4.0,
            },
        ]
        summary = bcss.aggregate_mask_results(rows)
        self.assertEqual(summary["images_evaluated"], 2)
        self.assertIn("tumor", summary["per_class"])
        self.assertAlmostEqual(summary["tumor_ratio_mae"], 4.0, places=5)

    def test_evaluate_bcss_predictions_allows_missing_predictions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            masks_dir = os.path.join(tmpdir, "masks")
            os.makedirs(masks_dir)
            np.save(os.path.join(masks_dir, "sample.npy"), np.zeros((2, 2), dtype=np.int32))

            summary, rows = bcss.evaluate_bcss_predictions(
                masks_dir=masks_dir,
                predictions_dir=os.path.join(tmpdir, "preds"),
                require_predictions=False,
            )

        self.assertEqual(rows, [])
        self.assertEqual(summary["images_evaluated"], 0)
        self.assertEqual(summary["missing_predictions"], ["sample"])

    def test_run_bcss_images_writes_summary_with_mocked_analysis(self):
        params = bcss.build_default_area_ratio_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = os.path.join(tmpdir, "images")
            output_dir = os.path.join(tmpdir, "out")
            os.makedirs(images_dir)
            np.save(os.path.join(images_dir, "sample.npy"), np.zeros((4, 4, 3), dtype=np.uint8))

            fake_results = {
                "tumor_ratio": 12.5,
                "necrosis_ratio": 4.5,
                "segmentation_mask": np.zeros((4, 4), dtype=np.int32),
            }

            with mock.patch("analyze_he.load_image", return_value=np.zeros((4, 4, 3), dtype=np.uint8)) as load_image, mock.patch(
                "analyze_he.run_area_ratio", return_value=fake_results
            ) as run_area_ratio, mock.patch(
                "patholib.viz.report.generate_he_report", return_value=os.path.join(output_dir, "sample_he_report.json")
            ) as generate_report:
                summary = bcss.run_bcss_images(
                    images_dir=images_dir,
                    output_dir=output_dir,
                    params=params,
                )

        self.assertEqual(summary["images_completed"], 1)
        load_image.assert_called_once()
        run_area_ratio.assert_called_once()
        generate_report.assert_called_once()

    def test_write_summary_json_and_csv_create_outputs(self):
        rows = [
            {
                "image_stem": "sample",
                "per_class": {
                    "tumor": {"dice": 0.9, "iou": 0.8},
                    "stroma": {"dice": 0.8, "iou": 0.7},
                    "necrosis": {"dice": 0.7, "iou": 0.6},
                },
                "gt_tumor_ratio": 20.0,
                "pred_tumor_ratio": 18.0,
                "tumor_ratio_abs_error": 2.0,
                "gt_necrosis_ratio": 5.0,
                "pred_necrosis_ratio": 4.0,
                "necrosis_ratio_abs_error": 1.0,
            }
        ]
        summary = {"dataset": "BCSS", "images_evaluated": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "nested", "per_image.csv")
            json_path = os.path.join(tmpdir, "nested", "summary.json")
            bcss.write_per_image_csv(rows, csv_path)
            bcss.write_summary_json(summary, json_path)
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(json_path))


if __name__ == "__main__":
    unittest.main()
