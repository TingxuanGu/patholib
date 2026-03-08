import json
import os
import tempfile
import unittest
from unittest import mock

from benchmarks import her2_ihc_40x


class Her2BenchmarkTests(unittest.TestCase):
    def test_normalize_her2_label_handles_common_forms(self):
        self.assertEqual(her2_ihc_40x.normalize_her2_label("0"), "0")
        self.assertEqual(her2_ihc_40x.normalize_her2_label("1"), "1+")
        self.assertEqual(her2_ihc_40x.normalize_her2_label("2+"), "2+")
        self.assertEqual(her2_ihc_40x.normalize_her2_label("3plus"), "3+")
        self.assertIsNone(her2_ihc_40x.normalize_her2_label("unknown"))

    def test_predict_her2_grade_handles_basic_cases(self):
        self.assertEqual(
            her2_ihc_40x.predict_her2_grade({"positive_percentage": 0.0, "grade_counts": {"0": 10}}),
            "0",
        )
        self.assertEqual(
            her2_ihc_40x.predict_her2_grade({"positive_percentage": 5.0, "grade_counts": {"1": 2}}),
            "1+",
        )
        self.assertEqual(
            her2_ihc_40x.predict_her2_grade({"positive_percentage": 20.0, "grade_counts": {"2": 5, "1": 1}}),
            "2+",
        )
        self.assertEqual(
            her2_ihc_40x.predict_her2_grade({"positive_percentage": 30.0, "grade_counts": {"3": 5, "2": 1}}),
            "3+",
        )

    def test_iter_her2_images_detects_labels_from_class_folders(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/Patches/Test/2+"
            os.makedirs(image_dir)
            open(f"{image_dir}/patch1.png", "wb").close()

            rows = her2_ihc_40x.iter_her2_images(tmpdir, "test")

        self.assertEqual(rows, [{"image_path": os.path.join(image_dir, "patch1.png"), "label": "2+"}])

    def test_quadratic_weighted_kappa_is_one_for_perfect_predictions(self):
        rows = [
            {"ground_truth": "0", "predicted_label": "0"},
            {"ground_truth": "1+", "predicted_label": "1+"},
            {"ground_truth": "2+", "predicted_label": "2+"},
            {"ground_truth": "3+", "predicted_label": "3+"},
        ]
        self.assertAlmostEqual(her2_ihc_40x.quadratic_weighted_kappa(rows), 1.0, places=6)
        self.assertAlmostEqual(her2_ihc_40x.macro_f1(rows), 1.0, places=6)

    def test_evaluate_her2_split_allows_missing_reports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/Patches/Test/0"
            os.makedirs(image_dir)
            open(f"{image_dir}/patch1.png", "wb").close()

            summary, rows = her2_ihc_40x.evaluate_her2_split(
                dataset_root=tmpdir,
                split="test",
                reports_dir=f"{tmpdir}/reports",
                require_reports=False,
            )

        self.assertEqual(rows, [])
        self.assertEqual(summary["images_evaluated"], 0)
        self.assertEqual(summary["missing_reports"], ["patch1"])

    def test_evaluate_her2_split_reads_report_and_scores_patch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/Patches/Test/3+"
            reports_dir = f"{tmpdir}/reports"
            os.makedirs(image_dir)
            os.makedirs(reports_dir)
            open(f"{image_dir}/patch1.png", "wb").close()
            with open(f"{reports_dir}/patch1_ihc_report.json", "w", encoding="utf-8") as handle:
                json.dump({"summary": {"positive_percentage": 25.0, "grade_counts": {"3": 8}}}, handle)

            summary, rows = her2_ihc_40x.evaluate_her2_split(
                dataset_root=tmpdir,
                split="test",
                reports_dir=reports_dir,
            )

        self.assertEqual(summary["images_evaluated"], 1)
        self.assertEqual(rows[0]["predicted_label"], "3+")
        self.assertEqual(rows[0]["ground_truth"], "3+")
        self.assertAlmostEqual(summary["accuracy"], 1.0, places=6)

    def test_run_her2_split_writes_summary_with_mocked_analysis(self):
        params = her2_ihc_40x.build_default_membrane_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_dir = f"{tmpdir}/Patches/Test/1+"
            output_dir = f"{tmpdir}/out"
            os.makedirs(image_dir)
            open(f"{image_dir}/patch1.png", "wb").close()

            fake_results = {
                "positive_percentage": 8.0,
                "h_score": 30.0,
                "grade_counts": {0: 8, 1: 2, 2: 0, 3: 0},
            }

            with mock.patch("analyze_ihc.load_image", return_value="image") as load_image, mock.patch(
                "analyze_ihc.run_analysis", return_value=fake_results
            ) as run_analysis, mock.patch(
                "patholib.viz.report.generate_ihc_report", return_value=f"{output_dir}/patch1_ihc_report.json"
            ) as generate_report:
                summary = her2_ihc_40x.run_her2_split(
                    dataset_root=tmpdir,
                    split="test",
                    output_dir=output_dir,
                    params=params,
                )

        self.assertEqual(summary["images_completed"], 1)
        load_image.assert_called_once()
        run_analysis.assert_called_once()
        generate_report.assert_called_once()

    def test_write_summary_json_and_csv_create_outputs(self):
        rows = [
            {
                "image_stem": "patch1",
                "ground_truth": "2+",
                "predicted_label": "3+",
                "positive_percentage": 25.0,
                "h_score": 150.0,
            }
        ]
        summary = {"dataset": "HER2-IHC-40x", "accuracy": 0.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = f"{tmpdir}/nested/per_image.csv"
            json_path = f"{tmpdir}/nested/summary.json"
            her2_ihc_40x.write_per_image_csv(rows, csv_path)
            her2_ihc_40x.write_summary_json(summary, json_path)

            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(json_path))


if __name__ == "__main__":
    unittest.main()
