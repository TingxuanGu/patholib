"""
patholib.io.wsi_tiler
=====================
Tile extraction from Whole Slide Images (WSI) with tissue masking.
Supports .svs, .ndpi, .mrxs and other OpenSlide-compatible formats.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class WSITiler:
    """
    Extract tiles from a WSI with automatic tissue detection.

    Parameters
    ----------
    path : str
        Path to WSI file.
    tile_size : int
        Tile dimension in pixels (square tiles).
    level : int
        Pyramid level for tile extraction (0 = highest resolution).
    overlap : int
        Overlap between adjacent tiles in pixels.
    tissue_threshold : float
        Minimum fraction of tissue in a tile to keep it (0-1).
    """

    def __init__(
        self,
        path: str,
        tile_size: int = 512,
        level: int = 0,
        overlap: int = 0,
        tissue_threshold: float = 0.5,
    ):
        import openslide
        self.path = str(path)
        self.slide = openslide.OpenSlide(self.path)
        self.tile_size = tile_size
        self.level = level
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold

        self.dims = self.slide.level_dimensions[level]
        self.downsample = self.slide.level_downsamples[level]

        logger.info(
            "WSITiler: %s, level=%d, dims=%s, tile=%d, overlap=%d",
            Path(path).name, level, self.dims, tile_size, overlap
        )

    def _build_tissue_mask(self, thumb_level: Optional[int] = None) -> np.ndarray:
        """
        Build a binary tissue mask from a low-resolution thumbnail.

        Returns
        -------
        np.ndarray
            Boolean mask at thumbnail resolution.
        """
        from skimage.color import rgb2gray
        from skimage.filters import threshold_otsu
        from scipy.ndimage import binary_fill_holes

        # Use a low-res level for speed
        if thumb_level is None:
            thumb_level = min(self.slide.level_count - 1, self.level + 2)

        thumb_dims = self.slide.level_dimensions[thumb_level]
        thumb = self.slide.read_region((0, 0), thumb_level, thumb_dims)
        thumb_rgb = np.array(thumb.convert('RGB'), dtype=np.uint8)

        gray = rgb2gray(thumb_rgb)
        try:
            thresh = threshold_otsu(gray)
        except ValueError:
            thresh = 0.8
        mask = gray < thresh  # tissue is darker
        mask = binary_fill_holes(mask)
        return mask.astype(bool), thumb_dims

    def tiles(self) -> Generator[Tuple[np.ndarray, int, int], None, None]:
        """
        Yield (tile_rgb, x, y) tuples where x,y are level-0 coordinates.

        Only yields tiles that pass the tissue_threshold filter.
        """
        tissue_mask, thumb_dims = self._build_tissue_mask()
        scale_x = thumb_dims[0] / self.dims[0]
        scale_y = thumb_dims[1] / self.dims[1]

        step = self.tile_size - self.overlap
        w, h = self.dims
        n_tiles = 0
        n_kept = 0

        for y0 in range(0, h, step):
            for x0 in range(0, w, step):
                tw = min(self.tile_size, w - x0)
                th = min(self.tile_size, h - y0)
                if tw < self.tile_size // 2 or th < self.tile_size // 2:
                    continue

                n_tiles += 1

                # Check tissue fraction in mask
                mx0 = int(x0 * scale_x)
                my0 = int(y0 * scale_y)
                mx1 = int(min((x0 + tw) * scale_x, tissue_mask.shape[1]))
                my1 = int(min((y0 + th) * scale_y, tissue_mask.shape[0]))
                if mx1 <= mx0 or my1 <= my0:
                    continue

                region = tissue_mask[my0:my1, mx0:mx1]
                frac = region.mean() if region.size > 0 else 0.0
                if frac < self.tissue_threshold:
                    continue

                # Read tile at level-0 coordinates
                lv0_x = int(x0 * self.downsample)
                lv0_y = int(y0 * self.downsample)
                tile = self.slide.read_region((lv0_x, lv0_y), self.level, (tw, th))
                tile_rgb = np.array(tile.convert('RGB'), dtype=np.uint8)

                n_kept += 1
                yield tile_rgb, lv0_x, lv0_y

        logger.info("Tiling complete: %d/%d tiles passed tissue filter (%.1f%%)",
                     n_kept, n_tiles, 100 * n_kept / max(n_tiles, 1))

    def close(self):
        """Close the underlying OpenSlide handle."""
        self.slide.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
