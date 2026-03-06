"""
patholib.viz.overlay
====================
Generate annotated overlay images for pathology analysis results.

Provides functions to create detection overlays (colored nuclei),
segmentation overlays (region masks), blending, and scale bars.
"""

import numpy as np


# Default colour palettes
IHC_GRADE_COLORS = {
    0: (0, 0, 255, 200),      # negative  -> blue
    1: (255, 255, 0, 200),    # weak      -> yellow
    2: (255, 165, 0, 200),    # moderate  -> orange
    3: (255, 0, 0, 200),      # strong    -> red
}

HE_CELL_COLORS = {
    "inflammatory": (255, 0, 0, 200),    # red
    "parenchymal":  (0, 0, 255, 200),    # blue
}

SEGMENTATION_COLORS = {
    "background": (0, 0, 0),
    "normal":     (0, 180, 0),
    "tumor":      (220, 20, 20),
    "necrosis":   (160, 160, 160),
    "stroma":     (30, 30, 220),
}

SEGMENTATION_ID_COLORS = {
    0: (0, 0, 0),
    1: (0, 180, 0),
    2: (220, 20, 20),
    3: (160, 160, 160),
    4: (30, 30, 220),
}


def create_detection_overlay(image, labels, cell_data, overlay_type="ihc"):
    """
    Create an RGBA overlay colouring detected nuclei.

    Parameters
    ----------
    image : ndarray
        Original RGB image (H, W, 3).
    labels : ndarray
        Integer label mask from detection.
    cell_data : list[dict]
        Per-cell information. For IHC, each dict needs a "grade" key (0-3).
        For HE inflammation, each dict needs a "cell_type" key.
    overlay_type : str
        "ihc" or "he".

    Returns
    -------
    ndarray
        RGBA overlay image (H, W, 4), uint8.
    """
    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    label_to_color = {}
    for cell in cell_data:
        lbl = cell.get("label")
        if lbl is None:
            continue
        if overlay_type == "ihc":
            grade = cell.get("grade", 0)
            label_to_color[lbl] = IHC_GRADE_COLORS.get(grade, (128, 128, 128, 150))
        else:
            ctype = cell.get("cell_type", "parenchymal")
            label_to_color[lbl] = HE_CELL_COLORS.get(ctype, (128, 128, 128, 150))

    for lbl, rgba in label_to_color.items():
        mask = labels == lbl
        overlay[mask, 0] = rgba[0]
        overlay[mask, 1] = rgba[1]
        overlay[mask, 2] = rgba[2]
        overlay[mask, 3] = rgba[3]

    return overlay


def create_segmentation_overlay(image, mask, class_colors=None, alpha=0.4):
    """
    Create an RGBA overlay for a segmentation mask.

    Parameters
    ----------
    image : ndarray
        Original RGB image (H, W, 3).
    mask : ndarray
        Integer class labels (H, W).
    class_colors : dict, optional
        Mapping class_id -> (R, G, B). Defaults to SEGMENTATION_ID_COLORS.
    alpha : float
        Overlay transparency (0-1).

    Returns
    -------
    ndarray
        RGBA overlay image (H, W, 4), uint8.
    """
    if class_colors is None:
        class_colors = SEGMENTATION_ID_COLORS

    h, w = image.shape[:2]
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    alpha_byte = int(np.clip(alpha * 255, 0, 255))

    for cls_id, rgb_color in class_colors.items():
        if cls_id == 0:
            continue
        region = mask == cls_id
        if not region.any():
            continue
        overlay[region, 0] = rgb_color[0]
        overlay[region, 1] = rgb_color[1]
        overlay[region, 2] = rgb_color[2]
        overlay[region, 3] = alpha_byte

    return overlay


def blend_overlay(image, overlay, alpha=0.5):
    """
    Alpha-blend an RGBA overlay onto an RGB image.

    Parameters
    ----------
    image : ndarray
        RGB image (H, W, 3), uint8.
    overlay : ndarray
        RGBA overlay (H, W, 4), uint8.
    alpha : float
        Global blending factor (0 = image only, 1 = overlay only).

    Returns
    -------
    ndarray
        Blended RGB image (H, W, 3), uint8.
    """
    img = image[:, :, :3].astype(np.float64)
    ovr = overlay[:, :, :3].astype(np.float64)
    ovr_alpha = overlay[:, :, 3].astype(np.float64) / 255.0 * alpha

    ovr_alpha_3 = ovr_alpha[:, :, np.newaxis]
    blended = img * (1.0 - ovr_alpha_3) + ovr * ovr_alpha_3
    return np.clip(blended, 0, 255).astype(np.uint8)


def add_scale_bar(image, mpp, bar_length_um=100, position="bottom-right",
                  bar_color=(0, 0, 0), bar_height_px=6, margin_px=20,
                  text_color=(0, 0, 0)):
    """
    Draw a scale bar on an image.

    Parameters
    ----------
    image : ndarray
        RGB image (H, W, 3), uint8. Modified in place and returned.
    mpp : float
        Microns per pixel.
    bar_length_um : float
        Desired bar length in micrometers.
    position : str
        "bottom-right", "bottom-left", "top-right", or "top-left".
    bar_color : tuple
        RGB colour of the bar.
    bar_height_px : int
        Height of the bar in pixels.
    margin_px : int
        Margin from image edge.
    text_color : tuple
        RGB colour for the label text (requires PIL for text rendering).

    Returns
    -------
    ndarray
        Image with scale bar drawn.
    """
    if mpp is None or mpp <= 0:
        return image

    result = image.copy()
    h, w = result.shape[:2]
    bar_length_px = int(round(bar_length_um / mpp))
    bar_length_px = min(bar_length_px, w - 2 * margin_px)

    if "right" in position:
        x_end = w - margin_px
        x_start = x_end - bar_length_px
    else:
        x_start = margin_px
        x_end = x_start + bar_length_px

    if "bottom" in position:
        y_end = h - margin_px
        y_start = y_end - bar_height_px
    else:
        y_start = margin_px
        y_end = y_start + bar_height_px

    result[y_start:y_end, x_start:x_end, 0] = bar_color[0]
    result[y_start:y_end, x_start:x_end, 1] = bar_color[1]
    result[y_start:y_end, x_start:x_end, 2] = bar_color[2]

    # Attempt to render text label using PIL
    try:
        from PIL import Image, ImageDraw, ImageFont
        pil_img = Image.fromarray(result)
        draw = ImageDraw.Draw(pil_img)
        label = f"{bar_length_um} um"
        try:
            font = ImageFont.truetype("arial.ttf", max(12, bar_height_px * 2))
        except (IOError, OSError):
            font = ImageFont.load_default()
        text_x = (x_start + x_end) // 2
        text_y = y_start - bar_height_px * 3
        if text_y < 0:
            text_y = y_end + 2
        bbox = draw.textbbox((text_x, text_y), label, font=font)
        tw = bbox[2] - bbox[0]
        text_x = text_x - tw // 2
        draw.text((text_x, text_y), label, fill=text_color, font=font)
        result = np.array(pil_img)
    except ImportError:
        pass

    return result
