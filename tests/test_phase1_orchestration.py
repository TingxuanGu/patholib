import os
import tempfile
import unittest

from benchmarks import phase1_orchestration


class Phase1OrchestrationTests(unittest.TestCase):
    def test_run_phase1_smoke_writes_summary_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = phase1_orchestration.run_phase1(
                {
                    "output_dir": tmpdir,
                    "datasets": ["BCData", "HER2-IHC-40x", "BCSS", "PanNuke"],
                    "methods": ["watershed"],
                    "smoke": True,
                    "commit": "smoke123",
                    "run_date": "2026-03-09T10:00:00",
                }
            )

            summary_dir = result["summary_dir"]
            self.assertTrue(os.path.isfile(os.path.join(summary_dir, "phase1_metrics_long.csv")))
            self.assertTrue(os.path.isfile(os.path.join(summary_dir, "phase1_summary.csv")))
            self.assertTrue(os.path.isfile(os.path.join(summary_dir, "phase1_summary.md")))
            self.assertEqual(len(result["eval_json_paths"]), 4)

    def test_run_phase1_real_requires_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                phase1_orchestration.run_phase1(
                    {
                        "output_dir": tmpdir,
                        "datasets": ["BCData"],
                        "methods": ["watershed"],
                        "smoke": False,
                    }
                )


if __name__ == "__main__":
    unittest.main()
