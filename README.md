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
Current state (this branch):
- Stable Diffusion inpainting pipeline loads and runs on CPU
- Reuses SAM wall mask to restrict product placement region
- Generates two size variants (intended 42" / 55" TV) but output currently unchanged (blank wall)
- Missing ControlNet conditioning & realism enhancements
- No fast mode (30 steps at full resolution → ~30 min per image on CPU)

Planned improvements (next commits):
- Fast debug mode (downscale + 12–15 steps + DPMSolver scheduler)
- Prompt & negative prompt refinement
- Mask enlargement + overlay diagnostics
- Fallback synthetic TV rendering if diffusion fails
- ControlNet depth / canny integration (post fast-mode stabilization)

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
  src/pipeline.py         # Alternate deterministic interface
```

## 🚀 Quick Start (Deterministic Pipeline)
```bash
git clone https://github.com/anupam-aarmy/AR_Preview.git
cd AR_Preview
ython -m venv venv
# Windows PowerShell:  .\\venv\\Scripts\\Activate.ps1
# Bash (Git Bash):     source venv/Scripts/activate
pip install -r requirements.txt
python download_sam.py
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
Produces (currently blank) outputs in `output/task2_generative/`.

### Why outputs are blank now
| Factor | Impact |
|--------|--------|
| CPU-only inference | 30 min per run prevents iterative tuning |
| Small rectangular mask | Too little semantic context for SD to generate TV |
| No ControlNet conditioning | Model drifts to “do nothing” safe output |
| Overly restrictive negative prompt | Suppresses emergent structure |
| Full-resolution generation | Higher compute + slower exploratory loop |

### Mitigation Plan
1. Implement fast mode (downscale copy, 15 steps, DPMSolverMultistep)
2. Add mask overlays & change-detection (SSIM/MSE) logging
3. Introduce refined prompt template with balanced negatives
4. Enlarge + softly feather mask; ensure multiples of 8
5. Add fallback TV compositor (parametric bezel + dark panel) if diffusion delta < threshold
6. After stable visible TVs → integrate ControlNet depth model

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
