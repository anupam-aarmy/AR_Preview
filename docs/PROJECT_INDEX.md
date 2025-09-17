# Project Index & Integration Map
**Updated:** September 17, 2025  
**Status:** Task 1 Complete ✅, Task 2 Partial ⚠️  
**Branch:** `feature/AIP-2-generative-pipeline`

## 1. High-Level Modules

### Core Pipeline Implementations
- **Task 1 (Deterministic)**: `scripts/task1_clean.py` - SAM + OpenCV product placement ✅
- **Task 2 (Generative)**: `scripts/task2_clean.py` - Stable Diffusion + ControlNet ⚠️
- **Main Entry Point**: `main.py` - Orchestration script for both tasks
- **Model Utilities**: `download_sam.py` - SAM model downloader

### Support Files
- **Environment Setup**: `requirements.txt`, `sd_environment_test.py`
- **Asset Creation**: `create_assets.py` - Asset management utilities
- **Legacy Files**: `src/pipeline.py` (preserved), deprecated standalone scripts

## 2. Documentation Map (Post-Cleanup)
| Area | File | Purpose | Status |
|------|------|---------|---------|
| Assignment Spec | `docs/assignment/AI_Assignment.md` | Original task goals (Task 1 & Task 2) | ✅ Complete |
| Progress Tracking | `docs/PROGRESS_TRACKER.md` | Current status, issues & resolution plans | ✅ Current |
| Completion Report | `docs/reports/PROOF_OF_COMPLETION.md` | Evidence + metrics for both tasks | ✅ Updated |
| Reliability Tests | `docs/reports/RELIABILITY_TEST_RESULTS.md` | Multi-pipeline testing results | ✅ Updated |
| GPU Setup | `docs/guides/GPU_SETUP_WINDOWS.md` | Windows GPU configuration | ✅ Available |
| Project Index | `docs/PROJECT_INDEX.md` | This file - project overview | ✅ Current |

## 3. Asset & Data Flow

### Input Assets
```
assets/
├── room_wall.png           # Primary room image
├── room_wall_2.png         # Additional room for testing
├── room_wall_3.png         # Additional room for testing  
├── room_wall_4.png         # Additional room for testing
├── prod_1_tv.png          # TV product (original)
├── prod_2_painting.png    # Painting product
└── prod_3_tv.png          # TV product (new variant)
```

### Processing Flow
1. **Task 1**: Room image → SAM segmentation → Product placement → Output
2. **Task 2**: Room image → Depth estimation → ControlNet conditioning → SD generation → Output

## 4. Output Structure (Current)
```
output/
├── task1_deterministic/           # Task 1 outputs ✅
│   ├── result_*.png              # Final product placements
│   ├── comparison_*.png          # Before/after comparisons
│   └── wall_mask_*.png           # SAM wall segmentation masks
└── task2_controlnet/             # Task 2 outputs ⚠️
    └── run_20250917_190102/      # Timestamped run directory
        ├── depth_map_*.png       # Depth conditioning maps ✅
        ├── generated_42_inch_*.png   # 42" TV generation attempts ❌
        ├── generated_55_inch_*.png   # 55" TV generation attempts ❌
        └── task2_comparison_*.png    # Pipeline analysis ⚠️
```

## 5. Current Status & Issues

### Task 1: Deterministic Pipeline ✅ WORKING
- **Implementation**: `scripts/task1_clean.py`
- **Features**: SAM wall segmentation, OpenCV product placement, alpha blending
- **Performance**: ~8 seconds per run, optimized for Tesla T4
- **Output Quality**: Clean product placement preserving original appearance
- **Products Supported**: TV, painting, additional variants

### Task 2: Generative Pipeline ⚠️ PARTIAL
- **Implementation**: `scripts/task2_clean.py`
- **Working Components**: 
  - ✅ Depth map generation (Intel DPT-large)
  - ✅ ControlNet pipeline setup (SD v1.5)
  - ✅ Size variations framework (42"/55")
- **Current Issues**:
  - ❌ Products disappearing during diffusion
  - ❌ Room morphing/distortion in generated images
  - ❌ Ineffective product-wall integration

## 6. Key Technologies & Dependencies
| Component | Technology | Status | Notes |
|-----------|------------|--------|-------|
| Wall Segmentation | Meta SAM (ViT-H) | ✅ Working | 2.4GB checkpoint, optimized config |
| Computer Vision | OpenCV | ✅ Working | Perspective transformation, blending |
| Generative AI | Stable Diffusion v1.5 | ⚠️ Partial | Pipeline setup complete, diffusion issues |
| Conditioning | ControlNet (depth) | ⚠️ Partial | Depth maps working, integration failing |
| Depth Estimation | Intel DPT-large | ✅ Working | Accurate depth field generation |
| Deep Learning | PyTorch + CUDA | ✅ Working | Tesla T4 optimized, memory management |

## 7. Entry Points & Usage
| Script | Command | Purpose | Status |
|--------|---------|---------|---------|
| Main Orchestrator | `python main.py --task 1` | Run Task 1 (deterministic) | ✅ Working |
| Main Orchestrator | `python main.py --task 2` | Run Task 2 (generative) | ⚠️ Partial |
| Main Orchestrator | `python main.py --task all` | Run both tasks | ⚠️ Task 1 works, Task 2 partial |
| Task 1 Direct | `python scripts/task1_clean.py` | Direct Task 1 execution | ✅ Working |
| Task 2 Direct | `python scripts/task2_clean.py` | Direct Task 2 execution | ⚠️ Partial |
| Model Download | `python download_sam.py` | Download SAM checkpoint | ✅ Working |
| Environment Test | `python sd_environment_test.py` | Test SD environment | ✅ Working |

## 8. AI Assignment Compliance
### Task 1 Requirements ✅ COMPLETE
- [x] Wall segmentation using AI vision model (SAM)
- [x] Realistic product placement with scaling & alignment
- [x] Visual realism without copy-paste appearance
- [x] Clean Python pipeline with error handling

### Task 2 Requirements ⚠️ PARTIAL
- [x] Stable Diffusion pipeline setup (Hugging Face/Diffusers)
- [x] ControlNet conditioning (depth conditioning implemented)
- [x] Size variations (42" vs 55" TV framework)
- [ ] Product diffusion quality (products disappearing)
- [ ] Room preservation (morphing issues)
- [ ] Output quality with proper alignment and shadows

## 9. Environment & Hardware
- **Python Version**: 3.8-3.11 recommended
- **GPU Support**: Tesla T4 with CUDA acceleration (16GB VRAM)
- **CPU Fallback**: Task 1 works on CPU, Task 2 requires GPU
- **Model Storage**: `models/` directory (SAM checkpoint: 2.4GB)
- **Memory Management**: Automatic image resizing for large inputs

## 10. Development Workflow
### Current Branch Strategy
- **Active Branch**: `feature/AIP-2-generative-pipeline`
- **Status**: Task 1 complete, Task 2 needs refinement
- **Merge Policy**: Do NOT merge to `main` until Task 2 issues resolved

### Recent Progress (September 17, 2025)
- ✅ Code recovery after accidental deletion
- ✅ Task 1 blending issues fixed
- ✅ Task 2 pipeline established with ControlNet
- ⚠️ Task 2 diffusion quality issues identified
- ✅ Documentation updated comprehensively

## 11. Next Steps & Priorities
| Priority | Task | Status | Notes |
|----------|------|--------|-------|
| P0 | Fix Task 2 product disappearing | 🔄 Active | Investigate diffusion parameters |
| P0 | Resolve room morphing in Task 2 | 🔄 Active | Adjust guidance scale, conditioning |
| P1 | Alternative ControlNet approaches | 📋 Planned | Test inpainting + depth combination |
| P1 | Mask generation refinement | 📋 Planned | Improve product area targeting |
| P2 | Performance optimization | 📋 Planned | Reduce Task 2 inference time |
| P2 | Additional product support | 📋 Planned | Expand beyond TV/painting |

## 12. Testing & Validation
### Current Test Coverage
- **Task 1**: Multiple room types, product variants, visual quality validation
- **Task 2**: Depth generation, pipeline execution, size variations
- **Integration**: Main orchestrator script, error handling

### Missing Test Areas
- Task 2 output quality metrics
- Automated regression testing
- Performance benchmarking
- Cross-platform compatibility

## 13. File Dependencies
### Critical Files
- `scripts/task1_clean.py` - Core Task 1 implementation
- `scripts/task2_clean.py` - Core Task 2 implementation  
- `main.py` - Entry point orchestration
- `requirements.txt` - Python dependencies
- `models/sam_vit_h_4b8939.pth` - SAM model checkpoint

### Support Files
- `download_sam.py` - Model acquisition
- `create_assets.py` - Asset management
- Documentation in `docs/` - Project knowledge base

---
**Note**: This index reflects the current state as of September 17, 2025. Task 1 is production-ready, Task 2 requires additional development to meet assignment criteria.
