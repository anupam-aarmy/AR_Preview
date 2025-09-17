import os
import hashlib
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

# Try to use controlnet-aux depth detectors
try:
    from controlnet_aux import MidasDetector
    _HAS_AUX = True
except Exception:
    _HAS_AUX = False

_DEPTH_MODEL_SINGLETON = None

def _get_cache_name(image: np.ndarray, width: int, height: int) -> str:
    h = hashlib.sha256(image.tobytes() + f"{width}x{height}".encode()).hexdigest()[:16]
    return f"depth_{h}_{width}x{height}.png"

def load_depth_detector(device: str = "cpu"):
    global _DEPTH_MODEL_SINGLETON
    if not _HAS_AUX:
        return None
    if _DEPTH_MODEL_SINGLETON is None:
        try:
            _DEPTH_MODEL_SINGLETON = MidasDetector.from_pretrained("lllyasviel/Annotators")
            _DEPTH_MODEL_SINGLETON.to(device)
        except Exception:
            _DEPTH_MODEL_SINGLETON = None
    return _DEPTH_MODEL_SINGLETON


def compute_depth_map(image_bgr: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Return a uint8 depth map (0-255). Falls back to luminance edges if detector unavailable."""
    detector = load_depth_detector(device)
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    if detector is not None:
        try:
            depth = detector(rgb)
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
            depth_uint8 = (depth * 255).clip(0, 255).astype(np.uint8)
            return depth_uint8
        except Exception:
            pass
    # Fallback: simple pseudo-depth using blurred edges
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges_blur = cv2.GaussianBlur(edges, (9, 9), 0)
    depth_uint8 = cv2.normalize(edges_blur, None, 0, 255, cv2.NORM_MINMAX)
    return depth_uint8


def get_or_create_depth(image_bgr: np.ndarray, out_dir: str, device: str = "cpu") -> Tuple[str, np.ndarray]:
    os.makedirs(out_dir, exist_ok=True)
    h, w = image_bgr.shape[:2]
    name = _get_cache_name(image_bgr, w, h)
    out_path = os.path.join(out_dir, name)
    if os.path.exists(out_path):
        depth = cv2.imread(out_path, cv2.IMREAD_GRAYSCALE)
        return out_path, depth
    depth = compute_depth_map(image_bgr, device=device)
    cv2.imwrite(out_path, depth)
    return out_path, depth


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    if depth.ndim == 2:
        depth_rgb = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    else:
        depth_rgb = depth
    return Image.fromarray(cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB))


def save_depth_overlay(room_image: np.ndarray, depth: np.ndarray, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.resize(depth_color, (room_image.shape[1], room_image.shape[0])), 0.55, room_image, 0.45, 0)
    path = os.path.join(out_dir, "depth_overlay.png")
    cv2.imwrite(path, overlay)
    return path
