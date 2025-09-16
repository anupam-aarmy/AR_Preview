# AIP-2: Hyper-Realistic Product Generation with AI (HF, Stable Diffusion)

## 🎯 User Story: AIP-2 - Stable Diffusion Fine-Tuning for Product Placement

### Objective
**Assignment Task 2:** Generate hyper-realistic images of products on walls with natural lighting and shadows for better aesthetic fit judgment.

**JIRA Description:** "As a user, I want to generate a hyper-realistic image of a product on my wall, complete with natural lighting and shadows, so I can better judge its aesthetic fit."

**Requirements from AI Assignment Task 2:**
- Use Stable Diffusion + ControlNet (depth or inpaint)
- Generate realistic product placement on wall
- Show at least two product size variations (e.g., 42" vs 55" TV)
- Demonstrate pipeline setup and ControlNet conditioning

## 📋 JIRA Subtasks (Aligned with Assignment Task 2)

### AIP-9: Set up the Stable Diffusion environment using Hugging Face Diffusers library
- [ ] Install Stable Diffusion dependencies (diffusers, transformers, accelerate)
- [ ] Set up Hugging Face Diffusers pipeline
- [ ] Configure CUDA/GPU acceleration if available
- [ ] Test basic SD model loading and inference
- [ ] **Assignment Focus:** Ability to set up Stable Diffusion pipeline

### AIP-10: Configure and load a Stable Diffusion inpainting pipeline with ControlNet Inpaint model
- [ ] Integrate ControlNet Inpaint model for guided generation
- [ ] Set up inpainting pipeline with proper conditioning
- [ ] Test ControlNet conditioning with sample inputs
- [ ] Integrate with existing SAM wall detection from AIP-1
- [ ] **Assignment Focus:** Correct ControlNet conditioning (depth/inpainting)

### AIP-11: Create a script that uses a room image and a wall mask as inputs for the ControlNet
- [ ] Implement script to process room_wall_3.png from AIP-1
- [ ] Use existing SAM wall masks as ControlNet inputs
- [ ] Create proper inpainting masks for product placement regions
- [ ] Test wall context integration and mask processing
- [ ] **Assignment Focus:** Integration with wall segmentation

### AIP-12: Implement logic to generate images based on text prompts that specify the product and size
- [ ] Create prompt engineering system for products (TV, painting)
- [ ] **CRITICAL:** Implement size variation logic (42" vs 55" TV requirement)
- [ ] Generate products with realistic lighting and shadows
- [ ] Ensure proper scaling and alignment within wall context
- [ ] **Assignment Focus:** Output quality (alignment, scaling, shadows)

### AIP-13: Test and compare the output quality for at least two different product size variations
- [ ] Generate 42" TV vs 55" TV comparisons (assignment requirement)
- [ ] Test different painting sizes and orientations
- [ ] Quality assessment: realism, lighting, perspective
- [ ] Compare with deterministic approach from AIP-1
- [ ] **Assignment Focus:** Size variation demonstration and quality evaluation

### AIP-14: Document the generative pipeline, including setup instructions and examples
- [ ] Document Stable Diffusion setup process
- [ ] Create usage examples and prompt templates
- [ ] **Demonstrate understanding of fine-tuning/custom training**
- [ ] Update README with generative pipeline documentation
- [ ] **Assignment Focus:** Understanding of fine-tuning and code quality

## 🛠️ Technical Requirements

### New Dependencies (Assignment Requirements):
```txt
# Generative AI Pipeline (AIP-2) - Assignment Task 2
diffusers>=0.21.0           # Core Stable Diffusion pipeline
transformers>=4.33.0        # Model transformers
accelerate>=0.21.0          # GPU acceleration
controlnet-aux>=0.0.6       # ControlNet preprocessing utilities
torch-audio                 # Audio processing (if needed)
xformers                    # Optional: memory optimization
pillow>=9.0.0              # Enhanced image processing
```

### Model Requirements (Assignment Specific):
- **Stable Diffusion 1.5 or XL model** (as mentioned in assignment)
- **ControlNet models:** depth and inpainting (assignment requirement)
- **Storage:** ~10-15GB additional storage for models
- **Memory:** 8GB+ VRAM recommended for optimal performance

### Integration Points with AIP-1:
- **Reuse SAM wall segmentation** from completed AIP-1 implementation
- **Input:** Use room_wall_3.png (main room from AIP-1)
- **Products:** Use existing prod_1_tv.png and prod_2_painting.png
- **Generate inpainting masks** from SAM wall detection results
- **Compare outputs** with deterministic pipeline results

## 📈 Success Criteria (Assignment Task 2 Evaluation)

### ✅ **Pipeline Setup & Technical Skills:**
- [ ] **Ability to set up Stable Diffusion pipeline** (Hugging Face/Diffusers)
- [ ] Proper dependency management and environment setup
- [ ] GPU acceleration configuration and optimization

### ✅ **ControlNet Implementation:**
- [ ] **Correct ControlNet conditioning** (depth/inpainting as specified)
- [ ] Integration with existing SAM wall detection from AIP-1
- [ ] Proper mask generation for inpainting regions

### ✅ **Output Quality & Realism:**
- [ ] **Generated images with proper alignment and scaling**
- [ ] **Realistic lighting, shadows, and perspective**
- [ ] Natural product placement (not copy-pasted appearance)
- [ ] Integration with room context and lighting

### ✅ **Size Variation Demonstration:**
- [ ] **Show at least two product size variations** (42" vs 55" TV requirement)
- [ ] Consistent quality across different sizes
- [ ] Proper scaling relative to wall and room context

### ✅ **Fine-Tuning Understanding:**
- [ ] **Understanding of fine-tuning/custom training** (prototype level)
- [ ] Documentation of approach and methodology
- [ ] Comparison with deterministic approach from AIP-1

### ✅ **Code Quality & Documentation:**
- [ ] **Clean Python pipeline** (assignment requirement)
- [ ] Proper integration with existing codebase
- [ ] Comprehensive documentation and usage examples

## 🚀 Getting Started (Implementation Roadmap)

### Phase 1: Environment Setup (AIP-9) 
1. **Current Branch:** `feature/AIP-2-generative-pipeline` ✅ (up to date with AIP-1)
2. **Install Dependencies:** Add Stable Diffusion packages to requirements.txt
3. **HuggingFace Setup:** Configure Diffusers library and model access
4. **GPU Configuration:** Set up CUDA acceleration if available

### Phase 2: ControlNet Inpainting (AIP-10)
1. **ControlNet Inpaint Model:** Load and configure inpainting pipeline
2. **SAM Integration:** Use wall masks from AIP-1 as ControlNet inputs
3. **Pipeline Testing:** Verify inpainting functionality with room images
4. **Conditioning Validation:** Ensure proper ControlNet conditioning

### Phase 3: Script Development (AIP-11)
1. **Room Integration:** Process room_wall_3.png with existing SAM masks
2. **Mask Processing:** Convert SAM wall masks to inpainting inputs
3. **Input Pipeline:** Create seamless room → mask → ControlNet workflow
4. **Integration Testing:** Validate with AIP-1 wall detection results

### Phase 4: Prompt Engineering & Size Variations (AIP-12)
1. **Text Prompts:** Develop prompts for realistic product placement
2. **Size Logic:** Implement 42" vs 55" TV generation (assignment critical requirement)
3. **Quality Optimization:** Ensure natural lighting, shadows, and perspective
4. **Product Support:** Handle both TV and painting products effectively

### Phase 5: Testing & Comparison (AIP-13)
1. **Size Variations:** Generate and compare different TV sizes (assignment requirement)
2. **Quality Assessment:** Evaluate realism, alignment, and scaling
3. **Comparative Analysis:** Generative vs deterministic pipeline results
4. **Performance Metrics:** Speed, quality, and consistency evaluation

### Phase 6: Documentation & Delivery (AIP-14)
1. **Setup Documentation:** Complete installation and usage instructions
2. **Fine-tuning Understanding:** Document approach and methodology
3. **Example Gallery:** Create comprehensive output examples
4. **Final Integration:** Update all project documentation

## 🎯 Assignment Task 2 Compliance Checklist

### ✅ **Given Requirements (Inputs):**
- [ ] **Set of product images:** TV/painting variants ✅ (prod_1_tv.png, prod_2_painting.png from AIP-1)
- [ ] **Room photo (wall image):** ✅ (room_wall_3.png from AIP-1)
- [ ] **Wall segmentation:** ✅ (SAM masks from AIP-1 available for reuse)

### ✅ **Task 2 Implementation Requirements:**
- [ ] **Use Stable Diffusion + ControlNet** (depth or inpaint)
- [ ] **Generate realistic product placement** on wall
- [ ] **Show at least two product size variations** (42" vs 55" TV - CRITICAL)
- [ ] **Natural lighting and shadows** for aesthetic fit evaluation

### ✅ **Evaluation Criteria Compliance:**
- [ ] **Pipeline Setup:** Ability to set up Stable Diffusion (HuggingFace/Diffusers)
- [ ] **ControlNet Conditioning:** Correct depth/inpainting implementation
- [ ] **Output Quality:** Alignment, scaling, shadows, realism
- [ ] **Fine-tuning Understanding:** Prototype level demonstration
- [ ] **Code Quality:** Clean Python pipeline integration

### ✅ **JIRA User Story Alignment:**
- [ ] **Hyper-realistic product generation** with natural lighting
- [ ] **Better aesthetic fit judgment** through realistic shadows
- [ ] **Integration with existing wall detection** from AIP-1
- [ ] **Size variation capability** for different product dimensions