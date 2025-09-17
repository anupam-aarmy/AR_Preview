# AR Preview: AI-Powered Product Visualization

> Current Branch: `feature/AIP-2-generative-pipeline` (EXPERIMENTAL Task 2 work in progress)

An AI-powered product visualization MVP for AR preview applications. It implements a completed deterministic (Task 1) pipeline and a **work‑in‑progress** generative (Task 2) pipeline for realistic wall fitting visualization (Module 2 Single Wall Fitting).

## 🎯 Project Overview

Two parallel approaches:

### ✅ Task 1: Deterministic Pipeline (COMPLETED)
- Wall Segmentation: SAM (Segment Anything Model)
- Product Placement: OpenCV perspective & adaptive scaling
- Alpha Blending: Transparency preservation & background removal
- Performance Modes: Fast vs High Quality
- Entry Point: `main.py`

### 🔧 Task 2: Generative Pipeline (IN PROGRESS)
Implemented so far (this branch):
- Stable Diffusion inpainting pipeline (Hugging Face Diffusers) with GPU/CPU support
- Fast mode (`--fast`): downscale to <=896px longest side, DPMSolver scheduler swap, step reduction (default 30 → 15) and lighter guidance
- Centralized prompt + negative prompt templates for TV (see `src/generative/utils.py`)
- Size variation logic (42" / 55" TV) + painting variants
- Mask expansion & feathering + diagnostic overlays saved to `output/task2_generative/overlays/`
- Run metadata & per-product metadata JSON emission
- CLI flags for flexible experimentation (`--fast --steps --product-type --room-image --width --height --no-save-overlays --no-fallback --delta-threshold`)
- SSIM delta detection + synthetic fallback TV compositor (guarantees visible output if diffusion produces no change)

Pending (next milestones):
- ControlNet Depth integration for structural conditioning (`--use-depth` planned)
- Prompt tuning iteration after stable non-fallback TVs >70% of runs
- Quality metrics expansion (MSE, delta area %) & regression script
- README merge & polish once TVs render consistently
## 📂 Updated Project Structure
```
AR_Preview/
  assets/                 # Input room & product images
  docs/
    assignment/AI_Assignment.md
    plans/AIP-2-PLAN.md
    reports/PROOF_OF_COMPLETION.md
    reports/RELIABILITY_TEST_RESULTS.md
    PROJECT_INDEX.md      # Integration & roadmap index
  models/                 # SAM & (future) SD model cache pointers
  output/
    task1_deterministic/{masks,results,comparisons}
    task2_generative/{masks,generated,comparisons}
  main.py                 # Task 1
  generative_pipeline.py  # Task 2 (draft)
### Current Limitations & Immediate Plan
| Gap | Status | Planned Action |
|-----|--------|----------------|
| Visible product generation | Fallback may still trigger frequently | Integrate ControlNet and prompt refine to reduce fallback rate |
| Fallback rendering | Wired (SSIM-triggered) | Improve synthetic reflection realism (optional) |
| ControlNet depth guidance | Not implemented | Add optional `--use-depth` flag + preprocessing module |
| Prompt tuning | Initial template only | Iterate once structural change appears |
| Evaluation metrics | SSIM + fallback JSON only | Add MSE + changed pixel ratio |
  src/pipeline.py         # Alternate deterministic interface
```

## 🚀 Quick Start (Deterministic Pipeline)
```bash
git clone https://github.com/anupam-aarmy/AR_Preview.git
## 🔄 Next Engineering Steps
1. Integrate optional ControlNet depth (`--use-depth`) preprocessing
2. Prompt refinement & negative prompt balancing (reduce over-suppression)
3. Add MSE + changed pixel percentage to `delta_report.json`
4. Optimize fallback aesthetics (parameterize bezel thickness, reflection strength)
5. Regression pass: ensure Task 1 outputs unchanged
6. Documentation pass then candidate-ready demo set
cd AR_Preview
Produces experimental outputs in `output/task2_generative/` (may still appear blank until delta/fallback stage is added).
ython -m venv venv
# Windows PowerShell:  .\\venv\\Scripts\\Activate.ps1
# Bash (Git Bash):     source venv/Scripts/activate
pip install -r requirements.txt
## 🧬 Generative Pipeline Usage
Basic run (TV + Painting, quality mode):
```bash
python generative_pipeline.py
```
Fast debug (recommended GPU/early tuning):
```bash
python generative_pipeline.py --fast --product-type TV --steps 15
```
TV only, no overlays, custom resolution:
```bash
python generative_pipeline.py --product-type TV --width 768 --height 512 --no-save-overlays
```
Painting only, keep high steps:
```bash
python generative_pipeline.py --product-type Painting --steps 28
```
Inspect overlays & masks under `output/task2_generative/overlays/` and `masks/`.
python main.py
```

## 🧪 Deterministic Usage Notes
- Input room: `assets/room_wall.png` (or other `room_wall_*.png` examples)
- Products: `prod_1_tv.png`, `prod_2_painting.png`
- Results saved under `output/task1_deterministic/`

## 🧬 Generative Pipeline (Current Experimental State)
```bash
python generative_pipeline.py
```
Produces outputs in `output/task2_generative/`. If diffusion fails to alter the masked area (SSIM ≥ threshold), a synthetic fallback TV is composited for continuity.

### SSIM & Fallback Detection (What/Why)
Structural Similarity Index (SSIM) measures how similar two images are (1.0 = identical). We compute SSIM only inside each product mask between the original room and the generated result:
- If `SSIM >= --delta-threshold` (default 0.98) the model likely made no meaningful change → apply synthetic TV fallback.
- If `SSIM < threshold` we keep the diffusion output.
`delta_report.json` summarizes per-size SSIM and whether fallback was applied.

### Why blank / low-change outputs can occur
| Factor | Impact |
|--------|--------|
| CPU-only inference | 30 min per run prevents iterative tuning |
| Small rectangular mask | Too little semantic context for SD to generate TV |
| No ControlNet conditioning | Model drifts to “do nothing” safe output |
| Overly restrictive negative prompt | Suppresses emergent structure |
| Full-resolution generation | Higher compute + slower exploratory loop |

### Mitigation Plan
1. Fast mode (+) implemented
2. Overlays + SSIM delta (+) implemented (MSE pending)
3. Refined prompt template (initial pass) (+) in `utils.py`
4. Mask expansion & feather (+) implemented
5. Synthetic fallback compositor (+) implemented
6. ControlNet depth (pending)

## 🖥️ GPU Migration (Summary)
A full Windows Azure + GCP step-by-step guide will be added at: `docs/guides/GPU_SETUP_WINDOWS.md` (created in this branch). Use 1× NVIDIA 8–12 GB VRAM (e.g. T4/RTX A4000/RTX 3060). See guide for drivers, CUDA Torch install, and validation script `sd_environment_test.py`.

## 📑 Key Documentation
- Assignment Spec: `docs/assignment/AI_Assignment.md`
- Task 2 Plan: `docs/plans/AIP-2-PLAN.md`
- Task 1 Completion Evidence: `docs/reports/PROOF_OF_COMPLETION.md`
- Integration Index: `docs/PROJECT_INDEX.md`

## 🔍 Current Issues
| ID | Issue | Status | Planned Fix |
|----|-------|--------|-------------|
| G1 | Blank TV generations | Open | Fast mode + prompt/mask tuning |
| G2 | No ControlNet | Open | Add after visible baseline |
| G3 | Slow iteration (CPU) | Open | Migrate to GPU / fast mode |
| G4 | README outdated claims | Addressed | Updated to WIP status |
| G5 | Lack of diagnostic overlays | Open | Implement mask overlay save |

## 🔄 Next Engineering Steps
1. Add fast mode CLI args (`--fast`, `--steps`, `--downscale`)
2. Swap scheduler to DPMSolver in fast mode
3. Implement mask overlay & delta detection
4. Prompt refinement + fallback TV compositor
5. Write and commit GPU setup guide

## 🧪 Environment Validation
Run:
```bash
python sd_environment_test.py
```
Will print: device detection, diffusers availability, torch version.

## 🤝 Collaboration & Branch Policy
- Active development: `feature/AIP-2-generative-pipeline`
- Do **not** merge to `main` until TVs visibly render
- Commit small, traceable increments (docs + code separated when possible)

## 📜 License / Assessment Context
This is part of an AI candidate assessment (Module 2). External model weights not redistributed—user expected to download SAM & SD models via official sources.

---
For a full index see `docs/PROJECT_INDEX.md`. GPU setup guide & fast mode changes coming next.
