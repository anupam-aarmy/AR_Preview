# Copilot Instructions for AR_Preview

## Project Overview
This repository implements an AI-powered product visualization MVP for AR preview applications. The project builds two parallel solutions for realistic wall fitting visualization: a deterministic computer vision approach and a generative AI solution using Stable Diffusion.

### Assignment Context
This is an AI candidate test for Module 2 (Single Wall Fitting) focusing on:
- **Task 1:** Wall segmentation & product placement using classical CV + SAM
- **Task 2:** Stable Diffusion + ControlNet for generative product placement

The goal is to allow users to visualize wall fittings (TVs, paintings, frames) in their own space with realistic scaling, perspective, and lighting.

## Architecture & Data Flow
### Task 1: Deterministic Pipeline (main.py)
- **Wall Segmentation:** Use SAM (Segment Anything Model) for zero-shot wall detection
- **Product Placement:** OpenCV perspective transformation (homography) for realistic scaling
- **Blending:** Alpha blending to merge product PNG onto wall

### Task 2: Generative Pipeline (to be implemented)
- **Stable Diffusion Inpainting:** Guided by ControlNet for context-aware generation
- **ControlNet Conditioning:** Depth/inpainting for proper alignment and scaling
- **Size Variations:** Adjust mask scale and prompts for different product sizes

### Directory Structure
- **Entry Point:** `main.py` orchestrates both pipelines
- **Assets:** Input room photos (`assets/`) and product PNGs with transparency
- **Output:** Composite images and generated results (`output/`)
- **Models:** SAM checkpoints and Stable Diffusion models (`models/`)
- **src/**: Reusable modules for segmentation, placement, and generation logic

## Key Libraries
- `torch`, `torchvision`: Deep learning and tensor operations
- `opencv-python`: Image I/O, perspective transformations, and blending
- `segment-anything`: Meta's SAM for wall segmentation
- `diffusers`: Hugging Face Stable Diffusion pipeline (Task 2)
- `transformers`: ControlNet models and tokenizers
- `matplotlib`: Visualization and debugging
- `numpy`: Array operations and mathematical computations

## Developer Workflows
- **Install dependencies:**
  ```bash
  pip install -r requirements.txt
  ```
- **Run pipeline:**
  ```bash
  python main.py
  ```
- **Add new assets:** Place images in `assets/`.
- **Save outputs:** Write results to `output/`.

## Conventions & Patterns
- Keep all pipeline logic in `main.py` or import from `src/`
- Use `assets/` for input images and `output/` for results
- Prefer using SAM via `segment-anything` for segmentation tasks
- Use `matplotlib` for visualizations and debugging
- Follow Python best practices for module imports and error handling
- Save intermediate results (masks, transformed images) for debugging
- Use PNG format for products with transparency, JPEG/PNG for room photos

## Integration Points
- **External Models:** Place any downloaded or custom models in `models/`
- **Image Assets:** All input images should be referenced from `assets/`
- **Task 1 Pipeline:** SAM → OpenCV homography → Alpha blending
- **Task 2 Pipeline:** Stable Diffusion inpainting + ControlNet conditioning

## Example: Loading and Segmenting an Image
```python
import cv2
from segment_anything import SamAutomaticMaskGenerator

img = cv2.imread('assets/room_wall.png')
mask_generator = SamAutomaticMaskGenerator(...)
masks = mask_generator.generate(img)
```

## Evaluation Criteria
### Task 1 (Deterministic):
- Accuracy of wall segmentation using SAM
- Proper scaling & perspective of product on wall (homography)
- Visual realism and clean blending (no "copy-paste" artifacts)

### Task 2 (Generative):
- Successful Stable Diffusion + ControlNet pipeline setup
- Output quality: alignment, scaling, realistic shadows
- Size variation capability (42" vs 55" TV examples)

## Notes
- No tests or CI/CD scripts are present yet.
- No custom conventions or rules files found.
- Update this file as new workflows or conventions emerge.
