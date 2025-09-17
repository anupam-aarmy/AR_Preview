# AIP-2 Continuation Plan (Generative Pipeline Progress)

Commit Reference: 443f3dc (feature/AIP-2-generative-pipeline)
Date: 2025-09-17

## 1. Current Functional State
| Component | Status | Notes |
|-----------|--------|-------|
| Deterministic (Task 1) | Complete | SAM segmentation + placement stable |
| Generative Script | Runs | Inpainting executes but outputs blank wall |
| SAM Integration | Partial | Reuses wall mask; no overlay diagnostics yet |
| ControlNet | Not integrated | Depth / canny outstanding |
| GPU Support | Pending | Guide written; code still CPU-tuned |
| Outputs Structure | Stable | `output/task1_deterministic` & `output/task2_generative` |
| Docs | Updated | README, PROJECT_INDEX, GPU guide done |

## 2. Root Issues (Why TVs Still Blank)
1. Small isolated rectangular mask without structural conditioning.
2. Weak semantic pressure + restrictive negative prompt.
3. High-res + CPU = slow iteration (30 steps ~30m → little experimentation).
4. No delta check or fallback compositing.

## 3. Immediate Next Engineering Tasks (Order)
| Order | Task | Goal | Acceptance Criteria |
|-------|------|------|----------------------|
| 1 | Fast Mode Flag (`--fast`) | Speed iteration | <2m on GPU; <6m CPU | 
| 2 | Scheduler Swap (DPMSolverMultistep) | Better quality/time | Configurable via arg |
| 3 | Downscale Workflow | Performance | Internal working copy <= 896px longest side |
| 4 | Prompt Template Refactor | Stable semantics | Visible TV bezel >80% runs |
| 5 | Mask Expansion + Feather | More context | Expanded mask boundaries + saved overlay |
| 6 | Diagnostic Overlays | Debug visibility | Files: `mask_overlay_*.png`, `delta_report.json` |
| 7 | Delta Change Detection (SSIM/MSE) | Trigger fallback | SSIM < 0.98 considered changed |
| 8 | Fallback Synthetic TV Renderer | Guarantee output | Adds bezel + dark panel if unchanged |
| 9 | ControlNet Depth Integration | Structure guidance | Depth map generated & used |

### 3.1 Implemented (Current Session)
- Added fast mode flag `--fast` (downscale + scheduler swap + step reduction)
- Added CLI args: `--steps`, `--room-image`, `--product-type`, `--width`, `--height`, `--no-save-overlays`, `--no-fallback`, `--output-dir`
- Introduced mask expansion + feathering and overlay saves (`output/task2_generative/overlays/`)
- Added structured metadata JSON per run (`output/task2_generative/run_metadata.json` and per-product metadata files)
- Refactored prompts to centralized templates (`src/generative/utils.py`)
- Scheduler automatically switches to `DPMSolverMultistepScheduler` in fast mode (best-effort)

Pending (next iteration): SSIM delta detection + fallback compositor activation, ControlNet depth integration.

## 4. CLI Arguments To Add
| Arg | Default | Description |
|-----|---------|-------------|
| `--fast` | false | (Implemented) Enables lower steps (<=15) + downscale (<=896px) + DPMSolver |
| `--steps` | 30 | Override inference steps |
| `--width` / `--height` | auto | Optional forced working resolution |
| `--save-overlays` | true | Save diagnostic overlay images |
| `--no-fallback` | false | Disable synthetic TV compositor |

## 5. Prompt Templates (Draft)
```
BASE_TV_PROMPT = (
  "ultra realistic wall mounted flat screen television, thin dark matte bezel, subtle diffuse reflection, professional interior photography, soft natural lighting"
)
NEGATIVE_TV = (
  "text, logo, watermark, ui, people, extra objects, painting frame, low quality, noisy, distorted"
)
```
Painting variant will emphasize frame texture & art style instead.

## 6. Fallback TV Compositor (Spec)
Steps if SSIM >= 0.98 within mask:
1. Draw bezel rectangle (mask box) with subtle outer shadow (Gaussian blur).
2. Inner panel: near-black (#0b0d10) gradient to #121417.
3. Reflection accent: 1–2 diagonal soft white transparent streaks (alpha 8–12%).
4. Blend with original image (overwrite only inside mask).
5. Save `fallback_tv_applied=true` in metadata JSON.

## 7. File Additions Planned
- `src/generative/utils.py` (added)
- `src/generative/mask_utils.py` (added)
- `src/generative/controlnet.py` (later) depth preprocessing
- `docs/guides/CONTROLNET_SETUP.md` (later)

## 8. Performance Targets
| Mode | Resolution (long side) | Steps | Expected GPU (T4) | CPU (Fallback) |
|------|------------------------|-------|-------------------|----------------|
| Fast | 768–896 | 15 | 1–2 min | 6–10 min |
| Quality | Original | 30 | 2.5–4 min | 25–35 min |

## 9. Acceptance Criteria to Exit "Experimental"
- TV & painting generations show distinct product forms (bezel/frame) in 5 consecutive runs.
- Fallback compositor not triggered in >70% runs after tuning.
- ControlNet depth integrated with toggle and readme instructions.
- README updated removing “blank output” notice.

## 10. Quick Start on New GPU VM
```powershell
# After cloning & activating venv
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python download_sam.py
python generative_pipeline.py --fast --steps 15
```

## 11. Known Risks
| Risk | Mitigation |
|------|-----------|
| VRAM OOM on small GPUs | Downscale & reduce steps first |
| ControlNet model size | Lazy load only when flag set |
| Over-fitting prompts | Keep base prompt stable, adjust adjectives incrementally |

## 12. Pending Documentation
- ControlNet depth usage guide
- Troubleshooting matrix for diffusion artifacts
- Benchmark log template

---
This continuation file is the single source of truth for resuming AIP‑2 development on a fresh environment.
