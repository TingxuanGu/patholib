"""
region_segmenter.py - Region-level semantic segmentation for H&E images.

Provides texture-based classification of tissue regions (tumor, necrosis,
normal, stroma) using LBP, GLCM, and color features with a Random Forest
classifier. Also includes simple threshold-based necrosis detection.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

logger = logging.getLogger(__name__)


class TextureClassifier:
    """Texture-based patch classifier for H&E image region segmentation.

    Extracts LBP histogram, GLCM texture features, and color statistics
    from image patches and classifies them using a Random Forest.

    Parameters
    ----------
    n_classes : int
        Number of tissue classes.
    classes : list of str
        Human-readable class names.
    """

    # LBP parameters
    _LBP_RADIUS = 3
    _LBP_N_POINTS = 24  # 8 * radius
    _LBP_METHOD = "uniform"

    # GLCM parameters
    _GLCM_DISTANCES = [1, 3]
    _GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    _GLCM_PROPERTIES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    def __init__(
        self,
        n_classes: int = 4,
        classes: Optional[List[str]] = None,
    ) -> None:
        self.n_classes = n_classes
        if classes is None:
            self.classes = ["normal", "tumor", "necrosis", "stroma"]
        else:
            self.classes = list(classes)
        self.classifier = None  # trained sklearn model
        self._feature_dim: Optional[int] = None

    def extract_features(self, image_patch: np.ndarray) -> np.ndarray:
        """Extract feature vector from a single image patch.

        Combines LBP histogram, GLCM features, and color statistics.

        Parameters
        ----------
        image_patch : np.ndarray
            RGB image patch, uint8, shape (H, W, 3).

        Returns
        -------
        features : np.ndarray
            1-D feature vector (float64).
        """
        features_list = []

        # Convert to grayscale for texture features
        if image_patch.ndim == 3:
            gray = cv2.cvtColor(image_patch, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_patch

        # --- LBP histogram ---
        lbp = local_binary_pattern(
            gray, self._LBP_N_POINTS, self._LBP_RADIUS, method=self._LBP_METHOD
        )
        n_bins = self._LBP_N_POINTS + 2  # uniform LBP produces P+2 bins
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        features_list.append(lbp_hist)

        # --- GLCM features ---
        # Quantize gray levels to 64 to speed up GLCM computation
        gray_q = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(
            gray_q,
            distances=self._GLCM_DISTANCES,
            angles=self._GLCM_ANGLES,
            levels=64,
            symmetric=True,
            normed=True,
        )
        for prop_name in self._GLCM_PROPERTIES:
            prop_vals = graycoprops(glcm, prop_name)  # shape: (n_dist, n_angle)
            features_list.append(prop_vals.ravel())

        # --- Color features ---
        if image_patch.ndim == 3:
            # HSV statistics
            hsv = cv2.cvtColor(image_patch, cv2.COLOR_RGB2HSV).astype(np.float64)
            for ch in range(3):  # H, S, V
                features_list.append(np.array([hsv[:, :, ch].mean(), hsv[:, :, ch].std()]))

            # RGB means
            rgb_means = image_patch.astype(np.float64).mean(axis=(0, 1))
            features_list.append(rgb_means)
        else:
            # Grayscale fallback: 9 zeros for color features
            features_list.append(np.zeros(9))

        feature_vector = np.concatenate(features_list).astype(np.float64)
        # Replace NaN/Inf with 0
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        return feature_vector

    def fit(
        self,
        patches: Sequence[np.ndarray],
        labels: Sequence[int],
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> "TextureClassifier":
        """Train the Random Forest classifier on labeled patches.

        Parameters
        ----------
        patches : sequence of np.ndarray
            List of RGB image patches (uint8).
        labels : sequence of int
            Integer class labels corresponding to each patch.
        n_estimators : int
            Number of trees in the Random Forest.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        self
        """
        from sklearn.ensemble import RandomForestClassifier

        logger.info("Extracting features from %d patches...", len(patches))
        X = np.array([self.extract_features(p) for p in patches])
        y = np.array(labels)
        self._feature_dim = X.shape[1]

        logger.info("Training RandomForest (n_estimators=%d, features=%d)...", n_estimators, self._feature_dim)
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(X, y)
        self.classifier = clf

        train_acc = clf.score(X, y)
        logger.info("Training accuracy: %.4f", train_acc)

        return self

    def predict(
        self,
        image: np.ndarray,
        patch_size: int = 64,
        stride: int = 32,
    ) -> np.ndarray:
        """Predict a segmentation mask for the full image.

        Slides a window across the image, extracts features from each patch,
        classifies it, and assembles the result into a label map.

        Parameters
        ----------
        image : np.ndarray
            RGB image, uint8, shape (H, W, 3).
        patch_size : int
            Size of the sliding window (pixels).
        stride : int
            Step between consecutive windows.

        Returns
        -------
        seg_mask : np.ndarray
            Integer label map of shape (H, W). Each pixel is assigned the
            class predicted for the patch it falls in. Overlapping patches
            are resolved by majority vote.
        """
        if self.classifier is None:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        h, w = image.shape[:2]
        # Accumulator for vote counting: shape (H, W, n_classes)
        vote_map = np.zeros((h, w, self.n_classes), dtype=np.float32)

        # Collect patches and their positions
        patches = []
        positions = []
        for y0 in range(0, h - patch_size + 1, stride):
            for x0 in range(0, w - patch_size + 1, stride):
                patch = image[y0:y0 + patch_size, x0:x0 + patch_size]
                patches.append(patch)
                positions.append((y0, x0))

        if len(patches) == 0:
            logger.warning("Image too small for patch_size=%d", patch_size)
            return np.zeros((h, w), dtype=np.int32)

        # Extract features for all patches
        X = np.array([self.extract_features(p) for p in patches])

        # Predict
        predictions = self.classifier.predict(X)

        # Accumulate votes
        for (y0, x0), pred in zip(positions, predictions):
            cls_idx = int(pred)
            if 0 <= cls_idx < self.n_classes:
                vote_map[y0:y0 + patch_size, x0:x0 + patch_size, cls_idx] += 1.0

        # Majority vote
        seg_mask = np.argmax(vote_map, axis=2).astype(np.int32)

        return seg_mask

    def save_model(self, path: str) -> None:
        """Save the trained classifier to disk via joblib.

        Parameters
        ----------
        path : str
            File path for the saved model (.joblib).
        """
        import joblib

        if self.classifier is None:
            raise RuntimeError("No trained classifier to save. Call fit() first.")

        state = {
            "classifier": self.classifier,
            "n_classes": self.n_classes,
            "classes": self.classes,
            "feature_dim": self._feature_dim,
        }
        joblib.dump(state, path)
        logger.info("Model saved to %s", path)

    def load_model(self, path: str) -> None:
        """Load a trained classifier from disk via joblib.

        Parameters
        ----------
        path : str
            File path to the saved model (.joblib).
        """
        import joblib

        state = joblib.load(path)
        self.classifier = state["classifier"]
        self.n_classes = state["n_classes"]
        self.classes = state["classes"]
        self._feature_dim = state["feature_dim"]
        logger.info("Model loaded from %s (classes=%s)", path, self.classes)


def segment_by_threshold(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    eosin_low: Tuple[int, int, int] = (170, 100, 100),
    eosin_high: Tuple[int, int, int] = (180, 255, 255),
    pink_low: Tuple[int, int, int] = (0, 40, 100),
    pink_high: Tuple[int, int, int] = (15, 255, 255),
    nuclei_darkness_threshold: int = 120,
    min_necrosis_area: int = 1000,
) -> np.ndarray:
    """Simple threshold-based necrosis detection in H&E images.

    Necrotic tissue in H&E stains appears eosinophilic (pink) with
    few intact nuclei (lack of dark blue/purple). This function detects
    regions that are pink but lack nuclear staining.

    Parameters
    ----------
    image : np.ndarray
        RGB image, uint8, shape (H, W, 3).
    tissue_mask : np.ndarray
        Boolean tissue mask (H, W). Only tissue regions are analyzed.
    eosin_low, eosin_high : tuple
        HSV range for eosinophilic (pink/red) color.
    pink_low, pink_high : tuple
        Additional HSV range for pink hues near 0 degrees.
    nuclei_darkness_threshold : int
        Pixels below this grayscale value are considered "dark" (nuclei).
    min_necrosis_area : int
        Minimum area (pixels) for a necrotic region.

    Returns
    -------
    result : np.ndarray
        Integer mask: 0=background, 1=normal tissue, 2=necrosis.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Detect eosinophilic (pink) regions in HSV
    eosin_mask1 = cv2.inRange(hsv, np.array(eosin_low), np.array(eosin_high))
    eosin_mask2 = cv2.inRange(hsv, np.array(pink_low), np.array(pink_high))
    pink_mask = (eosin_mask1 > 0) | (eosin_mask2 > 0)

    # Detect regions lacking nuclei (no dark pixels)
    dark_pixel_mask = gray < nuclei_darkness_threshold

    # For each local region, check if nuclei density is low
    # Use a block-based approach: count dark pixels in 32x32 blocks
    block_size = 32
    kernel = np.ones((block_size, block_size), dtype=np.float32) / (block_size * block_size)
    nuclei_density = cv2.filter2D(dark_pixel_mask.astype(np.float32), -1, kernel)

    # Low nuclei density region (less than 5% dark pixels)
    low_nuclei = nuclei_density < 0.05

    # Necrosis = pink + low nuclei density + within tissue
    necrosis_mask = pink_mask & low_nuclei & tissue_mask

    # Morphological cleaning
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    necrosis_u8 = necrosis_mask.astype(np.uint8) * 255
    necrosis_u8 = cv2.morphologyEx(necrosis_u8, cv2.MORPH_CLOSE, kernel_close)
    necrosis_u8 = cv2.morphologyEx(necrosis_u8, cv2.MORPH_OPEN, kernel_close)
    necrosis_clean = necrosis_u8 > 0

    # Remove small components
    from scipy import ndimage as ndi
    labelled, n_feat = ndi.label(necrosis_clean)
    if n_feat > 0:
        sizes = np.bincount(labelled.ravel())
        too_small = sizes < min_necrosis_area
        too_small[0] = False
        necrosis_clean[too_small[labelled]] = False

    # Build result mask
    result = np.zeros(image.shape[:2], dtype=np.int32)
    result[tissue_mask] = 1  # normal tissue
    result[necrosis_clean] = 2  # necrosis

    return result
