"""
patholib.viz.heatmap
====================
Generate density and intensity heatmaps for pathology analysis.

Provides Gaussian KDE-based density heatmaps, grid-level heatmaps,
colormap application, and overlay blending.
"""

import numpy as np


def create_density_heatmap(image_shape, cell_coords, sigma=50):
    """
    Create a Gaussian KDE-based density heatmap from cell coordinates.

    Parameters
    ----------
    image_shape : tuple
        (H, W) or (H, W, C) shape of the original image.
    cell_coords : array-like
        Nx2 array of (row, col) coordinates, or list of (row, col) tuples.
    sigma : float
        Standard deviation for Gaussian kernel (pixels).

    Returns
    -------
    ndarray
        Float heatmap in [0, 1] range with shape (H, W).
    """
    from scipy.ndimage import gaussian_filter

    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float64)

    coords = np.asarray(cell_coords)
    if coords.ndim != 2 or coords.shape[0] == 0:
        return heatmap

    for r, c in coords:
        ri, ci = int(round(r)), int(round(c))
        if 0 <= ri < h and 0 <= ci < w:
            heatmap[ri, ci] += 1.0

    heatmap = gaussian_filter(heatmap, sigma=sigma)

    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap


def create_grid_heatmap(grid_values, grid_shape, image_shape):
    """
    Upscale grid-level values to full image resolution.

    Parameters
    ----------
    grid_values : ndarray
        2D array of values at grid resolution.
    grid_shape : tuple
        (n_rows, n_cols) of the grid.
    image_shape : tuple
        (H, W) or (H, W, C) target image shape.

    Returns
    -------
    ndarray
        Float heatmap at image resolution, normalised to [0, 1].
    """
    from scipy.ndimage import zoom

    h, w = image_shape[:2]
    gr, gc = grid_values.shape[:2]

    if gr == 0 or gc == 0:
        return np.zeros((h, w), dtype=np.float64)

    zoom_r = h / gr
    zoom_c = w / gc
    heatmap = zoom(grid_values.astype(np.float64), (zoom_r, zoom_c), order=1)

    # Ensure exact output shape
    heatmap = heatmap[:h, :w]
    if heatmap.shape[0] < h or heatmap.shape[1] < w:
        padded = np.zeros((h, w), dtype=np.float64)
        padded[:heatmap.shape[0], :heatmap.shape[1]] = heatmap
        heatmap = padded

    hmax = heatmap.max()
    if hmax > 0:
        heatmap /= hmax

    return heatmap


def apply_colormap(heatmap, cmap="jet", vmin=None, vmax=None):
    """
    Apply a matplotlib colormap to a heatmap array.

    Parameters
    ----------
    heatmap : ndarray
        2D float array.
    cmap : str
        Matplotlib colormap name.
    vmin : float, optional
        Minimum value for normalisation.
    vmax : float, optional
        Maximum value for normalisation.

    Returns
    -------
    ndarray
        RGB image (H, W, 3), uint8.
    """
    try:
        from matplotlib import cm as mpl_cm
        from matplotlib.colors import Normalize
    except ImportError:
        return _fallback_colormap(heatmap, vmin, vmax)

    if vmin is None:
        vmin = float(heatmap.min())
    if vmax is None:
        vmax = float(heatmap.max())

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    colormap = mpl_cm.get_cmap(cmap)
    mapped = colormap(norm(heatmap))
    rgb = (mapped[:, :, :3] * 255).astype(np.uint8)
    return rgb


def _fallback_colormap(heatmap, vmin=None, vmax=None):
    """Simple jet-like colormap fallback when matplotlib is unavailable."""
    if vmin is None:
        vmin = float(heatmap.min())
    if vmax is None:
        vmax = float(heatmap.max())
    rng = vmax - vmin if vmax > vmin else 1.0
    normed = np.clip((heatmap - vmin) / rng, 0.0, 1.0)

    r = np.clip(1.5 - np.abs(normed - 0.75) * 4, 0, 1)
    g = np.clip(1.5 - np.abs(normed - 0.5) * 4, 0, 1)
    b = np.clip(1.5 - np.abs(normed - 0.25) * 4, 0, 1)

    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(image, heatmap, alpha=0.5, cmap="jet"):
    """
    Blend a heatmap with an original image.

    Parameters
    ----------
    image : ndarray
        RGB image (H, W, 3), uint8.
    heatmap : ndarray
        2D float heatmap.
    alpha : float
        Blending factor for the heatmap (0 = image only, 1 = heatmap only).
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    ndarray
        Blended RGB image (H, W, 3), uint8.
    """
    colored = apply_colormap(heatmap, cmap=cmap)

    img = image[:, :, :3].astype(np.float64)
    hm = colored.astype(np.float64)

    # Only blend where heatmap has non-zero values
    mask = heatmap > 0
    blended = img.copy()
    blended[mask] = img[mask] * (1.0 - alpha) + hm[mask] * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)
