import json
import os
import tempfile
import unittest

from benchmarks import phase1_summary


class Phase1SummaryTests(unittest.TestCase):
    def test_aggregate_eval_summaries_builds_long_and_wide_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bcss_dir = os.path.join(tmpdir, "bcss-threshold")
            bc_dir = os.path.join(tmpdir, "bcdata-watershed")
            her2_dir = os.path.join(tmpdir, "her2-watershed")
            os.makedirs(bcss_dir)
            os.makedirs(bc_dir)
            os.makedirs(her2_dir)

            bcss_eval = os.path.join(bcss_dir, "bcss_eval_summary.json")
            bcss_run = os.path.join(bcss_dir, "bcss_run_summary.json")
            bc_eval = os.path.join(bc_dir, "bcdata_test_eval_summary.json")
            bc_run = os.path.join(bc_dir, "bcdata_test_run_summary.json")
            her2_eval = os.path.join(her2_dir, "her2_ihc_40x_test_eval_summary.json")
            her2_run = os.path.join(her2_dir, "her2_ihc_40x_test_run_summary.json")

            with open(bcss_eval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": "BCSS",
                        "images_evaluated": 10,
                        "per_class": {
                            "tumor": {"dice": 0.7, "iou": 0.55},
                            "stroma": {"dice": 0.6, "iou": 0.45},
                            "necrosis": {"dice": 0.5, "iou": 0.35},
                        },
                        "tumor_ratio_mae": 8.0,
                        "necrosis_ratio_mae": 12.0,
                        "label_map": "/tmp/bcss_label_map.json",
                    },
                    handle,
                )
            with open(bcss_run, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "parameters": {"method": "threshold"},
                        "normalize_stain": False,
                    },
                    handle,
                )
            with open(bc_eval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": "BCData",
                        "split": "test",
                        "images_evaluated": 12,
                        "positive": {"f1": 0.81},
                        "negative": {"f1": 0.77},
                        "mean_f1": 0.79,
                        "positive_percentage_mae": 4.2,
                        "positive_percentage_rmse": 5.1,
                        "positive_percentage_pearson_r": 0.88,
                        "match_radius_px": 6.0,
                        "annotation_coord_order": "xy",
                    },
                    handle,
                )
            with open(bc_run, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "parameters": {"detection_method": "watershed"},
                        "normalize_stain": False,
                    },
                    handle,
                )
            with open(her2_eval, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "dataset": "HER2-IHC-40x",
                        "split": "test",
                        "images_evaluated": 20,
                        "accuracy": 0.75,
                        "macro_f1": 0.70,
                        "quadratic_weighted_kappa": 0.82,
                        "heuristic": {
                            "zero_cutoff": 1.0,
                            "positive_cutoff": 10.0,
                            "strong_grade_cutoff": 2.5,
                            "strong_fraction_cutoff": 0.3,
                        },
                    },
                    handle,
                )
            with open(her2_run, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "parameters": {"detection_method": "cellpose"},
                        "normalize_stain": True,
                    },
                    handle,
                )

            long_rows, wide_rows = phase1_summary.aggregate_eval_summaries(
                [bcss_eval, bc_eval, her2_eval],
                commit="abc123",
                run_date="2026-03-08T12:00:00",
            )

        self.assertEqual(len(wide_rows), 3)
        self.assertEqual(len(long_rows), 20)
        self.assertEqual(wide_rows[0]["dataset"], "BCSS")
        self.assertEqual(wide_rows[0]["detection_backend"], "threshold")
        self.assertEqual(wide_rows[0]["tumor_dice"], 0.7)
        self.assertEqual(wide_rows[1]["dataset"], "BCData")
        self.assertEqual(wide_rows[1]["detection_backend"], "watershed")
        self.assertEqual(wide_rows[1]["normalization"], "off")
        self.assertEqual(wide_rows[1]["positive_f1"], 0.81)
        self.assertEqual(wide_rows[2]["dataset"], "HER2-IHC-40x")
        self.assertEqual(wide_rows[2]["detection_backend"], "cellpose")
        self.assertEqual(wide_rows[2]["normalization"], "on")
        self.assertEqual(wide_rows[2]["quadratic_weighted_kappa"], 0.82)

    def test_render_markdown_summary_contains_both_datasets(self):
        markdown = phase1_summary.render_markdown_summary(
            [
                {
                    "dataset": "BCSS",
                    "split": "",
                    "method_name": "threshold",
                    "detection_backend": "threshold",
                    "normalization": "off",
                    "images_evaluated": 10,
                    "positive_f1": "",
                    "mean_f1": "",
                    "positive_percentage_mae": "",
                    "tumor_dice": 0.7,
                    "accuracy": "",
                    "macro_f1": "",
                    "quadratic_weighted_kappa": "",
                },
                {
                    "dataset": "BCData",
                    "split": "test",
                    "method_name": "watershed",
                    "detection_backend": "watershed",
                    "normalization": "off",
                    "images_evaluated": 12,
                    "positive_f1": 0.81,
                    "mean_f1": 0.79,
                    "positive_percentage_mae": 4.2,
                    "accuracy": "",
                    "macro_f1": "",
                    "quadratic_weighted_kappa": "",
                },
                {
                    "dataset": "HER2-IHC-40x",
                    "split": "test",
                    "method_name": "cellpose",
                    "detection_backend": "cellpose",
                    "normalization": "on",
                    "images_evaluated": 20,
                    "positive_f1": "",
                    "mean_f1": "",
                    "positive_percentage_mae": "",
                    "accuracy": 0.75,
                    "macro_f1": 0.70,
                    "quadratic_weighted_kappa": 0.82,
                },
            ]
        )

        self.assertIn("BCSS", markdown)
        self.assertIn("BCData", markdown)
        self.assertIn("HER2-IHC-40x", markdown)
        self.assertIn("0.8100", markdown)
        self.assertIn("0.8200", markdown)

    def test_write_csv_and_markdown_create_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "summary.csv")
            md_path = os.path.join(tmpdir, "summary.md")
            phase1_summary.write_csv(
                [{"dataset": "BCData", "split": "test"}],
                csv_path,
                ["dataset", "split"],
            )
            phase1_summary.write_markdown("# ok\n", md_path)

            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(md_path))


if __name__ == "__main__":
    unittest.main()
