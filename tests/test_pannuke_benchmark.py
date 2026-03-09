import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from benchmarks import pannuke


class PanNukeBenchmarkTests(unittest.TestCase):
    def test_merge_instance_channels_relabels_across_channels(self):
        mask_stack = np.zeros((3, 3, 2), dtype=np.int32)
        mask_stack[0, 0, 0] = 1
        mask_stack[1, 1, 1] = 1
        merged = pannuke.merge_instance_channels(mask_stack)
        self.assertEqual(set(np.unique(merged)), {0, 1, 2})

    def test_match_instances_and_pq_perfect_overlap(self):
        gt = np.array([[0, 1], [0, 2]], dtype=np.int32)
        pred = np.array([[0, 1], [0, 2]], dtype=np.int32)
        match = pannuke.match_instances(gt, pred, iou_threshold=0.5)
        self.assertAlmostEqual(match["f1"], 1.0, places=6)
        self.assertAlmostEqual(match["pq"], 1.0, places=6)

    def test_aji_and_binary_dice_on_simple_masks(self):
        gt = np.array([[0, 1], [0, 0]], dtype=np.int32)
        pred = np.array([[0, 1], [0, 1]], dtype=np.int32)
        self.assertGreaterEqual(pannuke.binary_dice_score(gt, pred), 0.0)
        self.assertGreaterEqual(pannuke.aggregated_jaccard_index(gt, pred), 0.0)

    def test_evaluate_patch_reports_inflammatory_metrics(self):
        gt_stack = np.zeros((3, 3, 3), dtype=np.int32)
        gt_stack[0, 0, 0] = 1
        gt_stack[1, 1, 1] = 1
        pred_all = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int32)
        pred_inflam = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int32)

        metrics = pannuke.evaluate_patch(gt_stack, pred_all, pred_inflam, inflammatory_channel=1)

        self.assertGreaterEqual(metrics["binary_nuclei_dice"], 0.0)
        self.assertGreaterEqual(metrics["aji"], 0.0)
        self.assertGreaterEqual(metrics["inflammatory_f1"], 0.0)

    def test_run_pannuke_images_writes_prediction_arrays_with_mocked_analysis(self):
        params = pannuke.build_default_inflammation_params()
        with tempfile.TemporaryDirectory() as tmpdir:
            images_path = os.path.join(tmpdir, "images.npy")
            output_dir = os.path.join(tmpdir, "out")
            np.save(images_path, np.zeros((2, 4, 4, 3), dtype=np.uint8))

            fake_results = {
                "labels": np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 2, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                    ],
                    dtype=np.int32,
                ),
                "cell_data": [
                    {"label": 1, "cell_type": "parenchymal"},
                    {"label": 2, "cell_type": "inflammatory"},
                ],
                "total_nuclei": 2,
                "inflammatory_cells": 1,
            }

            with mock.patch("analyze_he.run_inflammation", return_value=fake_results) as run_inflammation:
                summary = pannuke.run_pannuke_images(
                    images_npy_path=images_path,
                    output_dir=output_dir,
                    params=params,
                    start_index=0,
                    limit=1,
                )

            self.assertEqual(summary["images_evaluated"], 1)
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "pannuke_pred_instances.npy")))
            self.assertTrue(os.path.isfile(os.path.join(output_dir, "pannuke_pred_inflammatory_instances.npy")))
            run_inflammation.assert_called_once()

    def test_evaluate_pannuke_predictions_reads_saved_arrays(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            masks_path = os.path.join(tmpdir, "masks.npy")
            preds_dir = os.path.join(tmpdir, "preds")
            os.makedirs(preds_dir)

            gt_masks = np.zeros((1, 2, 2, 2), dtype=np.int32)
            gt_masks[0, 0, 0, 0] = 1
            gt_masks[0, 1, 1, 1] = 1
            np.save(masks_path, gt_masks)
            np.save(os.path.join(preds_dir, "pannuke_pred_instances.npy"), np.array([[[1, 0], [0, 2]]], dtype=np.int32))
            np.save(
                os.path.join(preds_dir, "pannuke_pred_inflammatory_instances.npy"),
                np.array([[[0, 0], [0, 1]]], dtype=np.int32),
            )
            with open(os.path.join(preds_dir, "pannuke_run_summary.json"), "w", encoding="utf-8") as handle:
                json.dump({"start_index": 0}, handle)

            summary, rows = pannuke.evaluate_pannuke_predictions(
                masks_npy_path=masks_path,
                predictions_dir=preds_dir,
                inflammatory_channel=1,
            )

        self.assertEqual(summary["images_evaluated"], 1)
        self.assertEqual(rows[0]["patch_index"], 0)

    def test_write_summary_json_and_csv_create_outputs(self):
        rows = [
            {
                "patch_index": 0,
                "binary_nuclei_dice": 0.8,
                "aji": 0.7,
                "pq": 0.6,
                "all_nuclei_f1": 0.9,
                "inflammatory_precision": 0.5,
                "inflammatory_recall": 0.4,
                "inflammatory_f1": 0.44,
            }
        ]
        summary = {"dataset": "PanNuke", "images_evaluated": 1}
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "nested", "per_patch.csv")
            json_path = os.path.join(tmpdir, "nested", "summary.json")
            pannuke.write_per_image_csv(rows, csv_path)
            pannuke.write_summary_json(summary, json_path)
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(json_path))
