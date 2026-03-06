"""
patholib.io.image_loader
========================
Unified image loading for both regular microscopy photos and WSI formats.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# WSI extensions handled by OpenSlide
WSI_EXTENSIONS = {'.svs', '.ndpi', '.mrxs', '.tif', '.tiff', '.scn', '.vms', '.vmu', '.bif'}

# Regular image extensions
REGULAR_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def load_image(path: str, level: int = 0, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """
    Load an image from disk. Automatically detects WSI vs regular format.

    Parameters
    ----------
    path : str
        Path to image file.
    level : int
        For WSI: pyramid level (0 = highest resolution). Ignored for regular images.
    region : tuple, optional
        (x, y, width, height) region to extract. If None, loads entire image.
        For WSI, this extracts at the given level.

    Returns
    -------
    np.ndarray
        RGB uint8 image, shape (H, W, 3).
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in WSI_EXTENSIONS and _has_openslide():
        return _load_wsi(str(p), level=level, region=region)
    else:
        return _load_regular(str(p), region=region)


def _has_openslide() -> bool:
    """Check if OpenSlide is available."""
    try:
        import openslide  # noqa: F401
        return True
    except ImportError:
        return False


def _load_regular(path: str, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Load a regular image file via PIL."""
    from PIL import Image

    img = Image.open(path)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    if region is not None:
        x, y, w, h = region
        img = img.crop((x, y, x + w, y + h))

    arr = np.array(img, dtype=np.uint8)
    logger.info("Loaded regular image %s: shape=%s", path, arr.shape)
    return arr


def _load_wsi(path: str, level: int = 0, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Load a WSI file via OpenSlide."""
    import openslide

    slide = openslide.OpenSlide(path)
    dims = slide.level_dimensions[level]
    logger.info("WSI %s: levels=%d, level %d dims=%s", path, slide.level_count, level, dims)

    if region is not None:
        x, y, w, h = region
        # read_region always uses level-0 coordinates for location
        tile = slide.read_region((x, y), level, (w, h))
    else:
        # Load entire level (warning: can be very large)
        tile = slide.read_region((0, 0), level, dims)

    arr = np.array(tile.convert('RGB'), dtype=np.uint8)
    slide.close()
    return arr


def get_wsi_info(path: str) -> dict:
    """
    Get metadata about a WSI file.

    Returns dict with keys: level_count, dimensions (per level), mpp (if available).
    """
    import openslide

    slide = openslide.OpenSlide(path)
    info = {
        'level_count': slide.level_count,
        'dimensions': [slide.level_dimensions[i] for i in range(slide.level_count)],
        'level_downsamples': [slide.level_downsamples[i] for i in range(slide.level_count)],
        'mpp_x': slide.properties.get(openslide.PROPERTY_NAME_MPP_X),
        'mpp_y': slide.properties.get(openslide.PROPERTY_NAME_MPP_Y),
        'vendor': slide.properties.get(openslide.PROPERTY_NAME_VENDOR, 'unknown'),
    }
    slide.close()
    return info
