# Project Index & Integration Map

## 1. High-Level Modules
- Deterministic Pipeline (Task 1): `main.py`, helper: `src/pipeline.py`
- Generative Pipeline (Task 2 draft): `generative_pipeline.py`
- Model Utilities: `download_sam.py`, `sd_environment_test.py`

## 2. Documentation Map
| Area | File | Purpose |
|------|------|---------|
| Assignment Spec | `docs/assignment/AI_Assignment.md` | Original task goals (Task 1 & Task 2) |
| Planning (Task 2) | `docs/plans/AIP-2-PLAN.md` | JIRA-style breakdown for generative pipeline |
| Completion Report (Task 1) | `docs/reports/PROOF_OF_COMPLETION.md` | Evidence + metrics for AIP-1 |
| Reliability Tests | `docs/reports/RELIABILITY_TEST_RESULTS.md` | Multi-room segmentation tests |
| Index | `docs/PROJECT_INDEX.md` | This file |

## 3. Data / Assets Flow
1. Input room image(s): `assets/room_wall*.png`
2. Task 1: SAM -> wall mask(s) saved to `output/task1_deterministic/masks/*`
3. Product placement -> `results/` & `comparisons/` under `task1_deterministic`
4. Task 2 (current): loads latest wall mask via `load_sam_wall_mask()` → builds product inpainting mask → (inpainting) → `output/task2_generative/...`

## 4. Output Structure
```
output/
  task1_deterministic/
    masks/           # wall_mask_*.png
    results/         # result_tv_*.png, result_painting_*.png
    comparisons/     # comparison_* side-by-side renders
  task2_generative/
    masks/           # inpaint_mask_* (rectangular product masks)
    generated/       # generated_tv_*.png
    comparisons/     # size_comparison_tv_*.png
```

## 5. Known Gaps (Current Branch)
- Generative outputs are blank (TV not synthesized inside mask)
- No ControlNet (depth / canny) integrated yet
- No fast mode or scheduler optimization
- Missing GPU setup guide (in progress)
- README claims need alignment with reality (will be updated)

## 6. Next Engineering Priorities
| Priority | Item | Rationale |
|----------|------|-----------|
| P0 | Fast CPU Debug Mode | Reduce iteration cycle from 30m → <2m |
| P0 | Prompt & Mask tuning | Produce visible product silhouette |
| P1 | ControlNet depth integration | Strong structural conditioning |
| P1 | Diagnostic overlays | Visual verification of mask usage |
| P2 | Fallback synthetic TV | Guarantees visible output during tuning |
| P2 | GPU migration execution | Performance baseline for Task 2 |

## 7. Key Functions (Entry Points)
| File | Function | Purpose |
|------|----------|---------|
| `main.py` | `main()` | Runs deterministic segmentation & placement |
| `generative_pipeline.py` | `main()` | Runs current draft inpainting flow |
| `generative_pipeline.py` | `load_sam_wall_mask()` | Reuses Task 1 wall mask |
| `src/pipeline.py` | `run_pipeline()` | Alternate deterministic interface |

## 8. Environment Expectations
- Python 3.8–3.11 recommended
- CPU-only works for Task 1; GPU strongly advised for Task 2
- Models stored under `models/` (SAM checkpoint manual download via `download_sam.py`)

## 9. Branching Strategy
- Stable segmentation baseline delivered on prior AIP‑1 branches
- Current experimentation: `feature/AIP-2-generative-pipeline`
- Do NOT merge to `main` until generative pipeline produces non-trivial TVs

## 10. Migration Checklist (Before GPU Work)
- [x] Restructure docs
- [x] Preserve deterministic outputs under new hierarchy
- [ ] Update README (next step)
- [ ] Add GPU setup guide (Azure + GCP)
- [ ] Implement fast mode + improved prompts

## 11. Testing Notes
- Deterministic: visual validation and wall mask correctness
- Generative: will introduce SSIM change detection + mask overlay soon

---
This index will expand as ControlNet & performance features are added.
