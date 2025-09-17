import math
import json
from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter
from datetime import datetime

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    ssim = None  # Will handle gracefully

BASE_TV_PROMPT = (
    "ultra realistic wall mounted flat screen television, thin dark matte bezel, subtle diffuse reflection, professional interior photography, soft natural lighting"
)
NEGATIVE_TV = (
    "text, logo, watermark, ui, people, extra objects, painting frame, low quality, noisy, distorted"
)

@dataclass
class GenerationConfig:
    steps: int = 30
    guidance_scale: float = 8.0
    width: int = None
    height: int = None
    fast: bool = False
    save_overlays: bool = True
    no_fallback: bool = False
    downscale_max_side: int = 896  # for fast mode


def apply_fast_mode(cfg: GenerationConfig):
    if cfg.fast:
        # Lower steps & enable downscale behaviour externally
        if cfg.steps >= 25:
            cfg.steps = 15
    return cfg


def choose_scheduler(pipe, fast: bool):
    """Swap to DPMSolverMultistepScheduler for speed if fast."""
    try:
        from diffusers import DPMSolverMultistepScheduler
        if fast:
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    except Exception:
        pass
    return pipe


def compute_ssim(original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> float:
    if ssim is None:
        return -1.0
    # Focus only inside mask: compute SSIM on bounding box to reduce cost
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return -1.0
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    crop_o = cv2.cvtColor(original[y1:y2+1, x1:x2+1], cv2.COLOR_BGR2GRAY)
    crop_g = cv2.cvtColor(generated[y1:y2+1, x1:x2+1], cv2.COLOR_BGR2GRAY)
    # Normalize sizes (should match)
    if crop_o.shape != crop_g.shape:
        h = min(crop_o.shape[0], crop_g.shape[0])
        w = min(crop_o.shape[1], crop_g.shape[1])
        crop_o = cv2.resize(crop_o, (w, h))
        crop_g = cv2.resize(crop_g, (w, h))
    score = ssim(crop_o, crop_g, data_range=255)
    return float(score)


def synthetic_tv_fallback(base_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render a synthetic TV (matte bezel + panel) inside the mask bbox."""
    out = base_image.copy()
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return out
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    # Slight inset for bezel thickness
    bezel_thickness_ratio = 0.04
    w = x2 - x1 + 1
    h = y2 - y1 + 1
    bezel_px = max(2, int(min(w, h) * bezel_thickness_ratio))

    panel = Image.new("RGB", (w, h), (10, 12, 16))
    draw = ImageDraw.Draw(panel)
    # Outer bezel (darker)
    draw.rectangle([0, 0, w-1, h-1], outline=(25, 27, 30), width=bezel_px)
    # Subtle gradient / reflection
    reflection = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(reflection)
    # Diagonal streaks
    for i in range(2):
        alpha = 28 if i == 0 else 18
        rdraw.rectangle([int(w*0.05)+i*int(w*0.1), 0, int(w*0.15)+i*int(w*0.1), h], fill=(255, 255, 255, alpha))
    reflection = reflection.filter(ImageFilter.GaussianBlur(radius=w * 0.05))
    panel = Image.alpha_composite(panel.convert("RGBA"), reflection).convert("RGB")

    panel_np = np.array(panel)
    # Feathered mask for blending
    feather = np.zeros((h, w), dtype=np.uint8)
    feather[:] = 255
    feather = cv2.GaussianBlur(feather, (0, 0), sigmaX=bezel_px*0.8)
    feather = feather.astype(np.float32) / 255.0

    roi = out[y1:y2+1, x1:x2+1].astype(np.float32)
    blended = (panel_np.astype(np.float32) * feather[..., None] + roi * (1 - feather[..., None]))
    out[y1:y2+1, x1:x2+1] = blended.astype(np.uint8)
    return out


def save_metadata(path: str, data: Dict[str, Any]):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
