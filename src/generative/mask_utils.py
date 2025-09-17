from typing import Tuple
import numpy as np
import cv2
from PIL import Image


def expand_and_feather_mask(mask: np.ndarray, expand_px: int = 12, feather_radius: int = 8) -> np.ndarray:
    """Expand binary mask and apply feather for smoother inpaint context."""
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask, kernel, iterations=max(1, expand_px // 2))
    if feather_radius > 0:
        blurred = cv2.GaussianBlur(dilated, (0, 0), sigmaX=feather_radius)
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        return thresh
    return dilated


def create_overlay(image_bgr, mask: np.ndarray, color=(0, 255, 0), alpha=0.45):
    overlay = image_bgr.copy()
    color_arr = np.zeros_like(image_bgr)
    color_arr[:, :] = color
    mask_bool = mask > 0
    overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + color_arr[mask_bool] * alpha).astype('uint8')
    return overlay


def downscale_for_fast_mode(img, target_max_side: int) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    max_side = max(h, w)
    if max_side <= target_max_side:
        return img, 1.0
    scale = target_max_side / max_side
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def resize_mask(mask: np.ndarray, new_shape) -> np.ndarray:
    return cv2.resize(mask, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_NEAREST)
