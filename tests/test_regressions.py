import importlib.util
import pathlib
import unittest
from unittest import mock

import numpy as np

import analyze_he
import analyze_ihc
from patholib.io import image_loader


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HAS_SKIMAGE = importlib.util.find_spec("skimage") is not None


class CliRegressionTests(unittest.TestCase):
    def test_he_parser_accepts_gpu_and_fail_fast(self):
        parser = analyze_he.build_parser()
        args = parser.parse_args(
            ["--input", "slide.tif", "--mode", "inflammation", "--use-gpu", "--fail-fast", "--grid-size", "128"]
        )

        params = analyze_he.build_inflammation_params(args)

        self.assertTrue(args.use_gpu)
        self.assertTrue(args.fail_fast)
        self.assertEqual(params["grid_size_px"], 128)

    def test_he_parser_rejects_removed_batch_knobs(self):
        parser = analyze_he.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--input", "slide.tif", "--mode", "inflammation", "--tile-size", "512"])

    def test_ihc_parser_accepts_gpu_and_fail_fast(self):
        parser = analyze_ihc.build_parser()
        args = parser.parse_args(
            ["--input", "slide.tif", "--stain-type", "nuclear", "--use-gpu", "--fail-fast"]
        )

        params = analyze_ihc.build_params(args)

        self.assertTrue(params["use_gpu"])
        self.assertTrue(params["fail_fast"])

    def test_ihc_parser_rejects_removed_stain_vector_option(self):
        parser = analyze_ihc.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--input", "slide.tif", "--stain-type", "nuclear", "--stain-vector", "auto"])


class AnalysisRegressionTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SKIMAGE, "scikit-image is not installed")
    def test_grid_size_controls_scoring_without_mpp(self):
        from patholib.analysis import he_inflammation

        tissue_mask = np.ones((100, 100), dtype=bool)
        scores, densities = he_inflammation._grid_scoring(
            (100, 100, 3),
            [],
            {"grid_size_um": 50, "mpp": None},
            tissue_mask,
        )

        self.assertEqual(scores.shape, (2, 2))
        self.assertEqual(densities.shape, (2, 2))

    @unittest.skipUnless(HAS_SKIMAGE, "scikit-image is not installed")
    def test_classifier_mode_raises_not_implemented(self):
        from patholib.analysis import he_area_ratio

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        with self.assertRaises(NotImplementedError):
            he_area_ratio.analyze_area_ratio(
                image,
                {"method": "classifier", "classifier_path": "fake-model.pkl"},
            )


class IoRegressionTests(unittest.TestCase):
    def test_tiff_load_falls_back_to_regular_loader(self):
        sentinel = np.zeros((4, 4, 3), dtype=np.uint8)
        with mock.patch.object(image_loader, "_has_openslide", return_value=True), mock.patch.object(
            image_loader, "_load_wsi", side_effect=RuntimeError("not a slide")
        ), mock.patch.object(image_loader, "_load_regular", return_value=sentinel) as load_regular:
            loaded = image_loader.load_image("sample.tif")

        self.assertIs(loaded, sentinel)
        load_regular.assert_called_once_with("sample.tif", region=None)


class ExampleImportTests(unittest.TestCase):
    def test_example_scripts_are_import_safe(self):
        for rel_path in (
            "examples/test_threshold_sweep.py",
            "examples/test_hod_sweep.py",
            "examples/test_recalibration.py",
        ):
            script_path = REPO_ROOT / rel_path
            spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)


if __name__ == "__main__":
    unittest.main()
