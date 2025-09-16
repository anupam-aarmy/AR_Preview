# AIP-2: Generative Pipeline Implementation Plan

## 🎯 User Story: AIP-2 - Generative Product Placement with Stable Diffusion

### Objective
Implement a Stable Diffusion + ControlNet pipeline for AI-generated product placement that creates realistic, context-aware visualizations of wall fittings.

## 📋 Planned Subtasks

### AIP-9: Stable Diffusion Environment Setup
- [ ] Add Stable Diffusion dependencies to requirements.txt
- [ ] Set up Hugging Face diffusers pipeline
- [ ] Configure CUDA/GPU acceleration if available
- [ ] Test basic SD inpainting functionality

### AIP-10: ControlNet Integration
- [ ] Integrate ControlNet for guided generation
- [ ] Implement depth conditioning for proper alignment
- [ ] Set up inpainting masks based on SAM wall detection
- [ ] Test controlled generation with wall context

### AIP-11: Inpainting Pipeline Implementation
- [ ] Create generative placement pipeline
- [ ] Implement prompt engineering for different products
- [ ] Handle size variations (42" vs 55" TV examples)
- [ ] Ensure realistic lighting and shadows

### AIP-12: Pipeline Integration & Testing
- [ ] Integrate with existing SAM wall detection
- [ ] Create side-by-side comparison: deterministic vs generative
- [ ] Performance optimization and GPU utilization
- [ ] Quality assessment and parameter tuning

### AIP-13: Documentation & Completion
- [ ] Update README with generative pipeline usage
- [ ] Document model requirements and setup
- [ ] Create example outputs and comparisons
- [ ] Final integration testing

## 🛠️ Technical Requirements

### New Dependencies (to be added):
```txt
# Generative AI Pipeline (AIP-2)
diffusers>=0.21.0
transformers>=4.33.0
accelerate>=0.21.0
xformers  # Optional: for memory optimization
controlnet-aux  # For preprocessing
```

### Model Requirements:
- Stable Diffusion 1.5 or XL model
- ControlNet models (depth, inpainting)
- ~10-15GB additional storage for models

### Integration Points:
- Reuse SAM wall segmentation from AIP-1
- Generate masks for inpainting regions
- Compare outputs with deterministic pipeline

## 📈 Success Criteria
- [ ] Successful SD + ControlNet pipeline setup
- [ ] Generated images with proper scaling and alignment
- [ ] Realistic lighting, shadows, and perspective
- [ ] Size variation capability demonstration
- [ ] Performance benchmarking vs deterministic approach

## 🚀 Getting Started
1. Switch to this branch: `git checkout feature/AIP-2-generative-pipeline`
2. Install additional dependencies for generative AI
3. Begin with AIP-9: Environment setup
4. Test basic functionality before full implementation

---
**Note:** This is a planning document. Delete or move to docs/ once implementation begins.