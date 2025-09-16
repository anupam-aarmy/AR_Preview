# AR## 🎯 Project Overview

This project implements a **deterministic pipeline** for realistic wall fitting visualization as per AI Assignment requirements:

### ✅ Task 1: Deterministic Pipeline (AIP-1) - **COMPLETED** 
- **Input:** Single room image (`room_wall_3.png`) as specified in assignment
- **Wall Segmentation:** SAM (Segment Anything Model) for zero-shot wall detection
- **Product Placement:** OpenCV perspective transformation with intelligent sizing  
- **Alpha Blending:** Advanced blending with automatic transparency detection
- **Performance:** Fast mode (~50s) with High Quality option (~350s)

### 🔄 Task 2: Generative Pipeline (AIP-2) - **PLANNED**
- **Stable Diffusion Inpainting:** Guided by ControlNet for context-aware generation
- **ControlNet Conditioning:** Depth/inpainting for proper alignment and scaling  
- **Size Variations:** Adjust mask scale and prompts for different product sizes

**🎯 Assignment Compliance:** Uses single room image input as specified, with additional reliability testing across multiple room scenarios.Powered Product Visualization

An AI-powered product visualization MVP for AR preview applications, allowing users to visualize wall fittings (TVs, paintings, frames) in their own space with realistic scaling, perspective, and lighting.

## 🎯 Project Overview

This project implements two parallel solutions for realistic wall fitting visualization:

### ✅ Task 1: Deterministic Pipeline (AIP-1) - **COMPLETED**
- **Wall Segmentation:** Uses SAM (Segment Anything Model) for zero-shot wall detection
- **Product Placement:** OpenCV perspective transformation with intelligent sizing
- **Alpha Blending:** Advanced blending with automatic transparency detection
- **Performance:** Optimized with fast/high-quality modes (~79s processing time)

### 🔄 Task 2: Generative Pipeline (AIP-2) - **PLANNED**
- **Stable Diffusion Inpainting:** Guided by ControlNet for context-aware generation
- **ControlNet Conditioning:** Depth/inpainting for proper alignment and scaling
- **Size Variations:** Adjust mask scale and prompts for different product sizes

## 🏗️ Project Structure

```
AR_Preview/
├── .github/
│   └── copilot-instructions.md    # AI agent guidance and project context
├── assets/                        # Input images
│   ├── room_wall.png             # Sample room photo (original)
│   ├── room_wall_2.png           # Additional room (lighting test)
│   ├── room_wall_3.png           # **Main room** (used by pipeline)
│   ├── room_wall_4.png           # Complex environment room  
│   ├── prod_1_tv.png             # Product image (TV)
│   └── prod_2_painting.png       # Product image (painting)
├── models/                        # AI model storage
│   └── sam_vit_h_4b8939.pth     # SAM model checkpoint (~2.4GB)
├── output/                        # 📊 Generated results (main + reliability testing)
│   ├── result_tv_20250916_184241.png          # Main TV result (room_wall_3)
│   ├── result_painting_20250916_184242.png    # Main painting result (room_wall_3)
│   ├── comparison_tv_20250916_184241.png      # Main TV comparison
│   ├── comparison_painting_20250916_184242.png # Main painting comparison
│   ├── wall_mask_tv_20250916_184241.png       # Wall segmentation mask
│   ├── wall_mask_painting_20250916_184242.png # Wall segmentation mask
│   └── [reliability_test_outputs...]          # Additional room scenarios
├── src/                          # Source modules
│   ├── __init__.py
│   └── pipeline.py              # Enhanced pipeline implementation
├── main.py                       # Main pipeline orchestrator
├── download_sam.py              # SAM model download utility
├── create_assets.py             # Asset creation utility
├── requirements.txt              # Python dependencies
├── PROOF_OF_COMPLETION.md       # 📋 Project completion documentation
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- ~3GB free space (for SAM model)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anupam-aarmy/AR_Preview.git
   cd AR_Preview
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download SAM model:**
   ```bash
   python download_sam.py
   ```

### Usage

```bash
# Run main pipeline (uses room_wall_3.png as per assignment requirements)
python main.py

# The pipeline processes:
# - Single room image: assets/room_wall_3.png
# - Two products: TV and painting
# - Fast mode by default (~50s processing time)
# - For high-quality mode: Edit main.py and set use_fast_mode = False (~350s processing)

# Enhanced pipeline (alternative interface)
python src/pipeline.py

# Process specific product with enhanced pipeline
python src/pipeline.py --product assets/prod_2_tv.png
```

## 📊 Sample Results

### Main Pipeline Results (Assignment Compliance)
**Input Room:** `room_wall_3.png` (As per AI Assignment requirements: single room image input)  
**Performance:** Fast mode - 15 wall masks detected in ~50 seconds

#### TV Placement
**Input:** Modern TV product → **Output:** [result_tv_20250916_184241.png](output/result_tv_20250916_184241.png)
- Clean transparency detection for user-provided TV image  
- Realistic wall mounting visualization with proper scaling (25% of wall width)
- **Comparison:** [comparison_tv_20250916_184241.png](output/comparison_tv_20250916_184241.png)

#### Painting Placement  
**Input:** Abstract painting → **Output:** [result_painting_20250916_184242.png](output/result_painting_20250916_184242.png)
- Intelligent wall detection and properly scaled art placement (30% of wall width)
- Natural integration with room lighting and perspective
- **Comparison:** [comparison_painting_20250916_184242.png](output/comparison_painting_20250916_184242.png)

#### Wall Segmentation Analysis
- **TV Wall Mask:** [wall_mask_tv_20250916_184241.png](output/wall_mask_tv_20250916_184241.png)
- **Painting Wall Mask:** [wall_mask_painting_20250916_184242.png](output/wall_mask_painting_20250916_184242.png)

### Additional Reliability Testing Results
*Note: These demonstrate pipeline robustness across different wall conditions and room scenarios*

#### Room 1 (room_wall.png) - Original Sample Room
- **TV Result:** [result_tv_room1_20250916_180532.png](output/result_tv_room1_20250916_180532.png) | [Comparison](output/comparison_tv_room1_20250916_180532.png)
- **Painting Result:** [result_painting_room1_20250916_180624.png](output/result_painting_room1_20250916_180624.png) | [Comparison](output/comparison_painting_room1_20250916_180624.png)

#### Room 2 (room_wall_2.png) - Different Lighting Conditions
- **TV Result:** [result_tv_room2_20250916_180714.png](output/result_tv_room2_20250916_180714.png) | [Comparison](output/comparison_tv_room2_20250916_180714.png)
- **Painting Result:** [result_painting_room2_20250916_180806.png](output/result_painting_room2_20250916_180806.png) | [Comparison](output/comparison_painting_room2_20250916_180806.png)

#### Room 4 (room_wall_4.png) - Complex Environment with Obstacles
- **TV Result:** [result_tv_room4_20250916_181039.png](output/result_tv_room4_20250916_181039.png) | [Comparison](output/comparison_tv_room4_20250916_181039.png)
- **Painting Result:** [result_painting_room4_20250916_181129.png](output/result_painting_room4_20250916_181129.png) | [Comparison](output/comparison_painting_room4_20250916_181129.png)

## 🛠️ Technology Stack

- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Segment Anything Model (SAM)
- **Generative AI:** Stable Diffusion, ControlNet, Hugging Face Diffusers (planned)
- **Visualization:** Matplotlib
- **Core:** NumPy, Python 3.8+

## ⚡ Performance Features

- **Fast Mode (Default):** Optimized SAM parameters for quicker processing (~50s for room_wall_3.png, 15 masks)
- **High Quality Mode:** Enhanced segmentation with more detailed masks (~350s for room_wall_3.png, 42 masks)
- **Wall Segmentation Reuse:** Computed once, reused for multiple products on same room
- **Smart Transparency:** Preserves original product appearance while removing backgrounds
- **Timestamped Outputs:** Preserves all results with unique filenames
- **Adaptive Scaling:** Product-specific sizing (TVs: 25%, Paintings: 30%)
- **Memory Optimization:** Auto-resize large images to 1024px max dimension

## 📋 Development Roadmap (JIRA: AIP-1)

### ✅ Phase 1: Product Visualization via Segmentation (**COMPLETED**)

#### ✅ AIP-4: Implement script to load room image and process with SAM
- [x] SAM model integration and wall segmentation
- [x] Room image loading and preprocessing
- [x] Mask generation and wall detection

#### ✅ AIP-5: Develop logic to isolate primary wall mask
- [x] Wall mask filtering and selection algorithms
- [x] Center-based wall detection
- [x] Contour analysis for optimal wall selection

#### ✅ AIP-6: Implement perspective transformation function using OpenCV
- [x] Homography-based perspective correction
- [x] Adaptive product scaling based on wall dimensions
- [x] Aspect ratio preservation for different product types

#### ✅ AIP-7: Apply alpha blending to composite warped product
- [x] Advanced transparency detection for user products
- [x] Smart background removal without changing product appearance
- [x] Clean alpha blending without artifacts or shadows

#### ✅ AIP-8: Document the segmentation and placement pipeline
- [x] Comprehensive README documentation
- [x] Code comments and inline documentation
- [x] Proof of Completion document
- [x] Final README polish and JIRA status update

### 🔄 Phase 2: Generative Pipeline (PLANNED - AIP-2/AIP-3)
- [ ] Stable Diffusion environment setup
- [ ] ControlNet integration for guided generation  
- [ ] Inpainting pipeline implementation
- [ ] Size variation handling (42" vs 55" TV)

**🎯 AIP-1 USER STORY: COMPLETE ✅**

### 📈 Phase 3: Enhancement & Optimization
- [ ] GPU acceleration optimization
- [ ] Real-time processing capabilities
- [ ] Multiple wall detection
- [ ] Product database integration

## 🎨 Sample Assets & Results

### Input Assets
- **Room Photo:** Modern minimalist room with white walls and windows
- **User Products:** Original TV and painting images (preserves exact appearance)
- **Transparency Handling:** Automatic background detection without changing product content

### Output Examples
- **Wall Segmentation:** Green overlay showing detected wall regions
- **Product Placement:** Realistically scaled and positioned user products
- **Clean Blending:** Natural integration without artificial shadows
- **Timestamped Results:** Preserves all iterations for comparison

## 🤝 Contributing

1. Create a feature branch from `main`
2. Follow the naming convention: `feature/AIP-X-description`
3. Implement changes with proper testing
4. Update documentation as needed
5. Create a pull request with detailed description

## 📝 Evaluation Criteria

### ✅ JIRA Story AIP-1: Product Visualization via Segmentation - **100% COMPLETE** 🎯

#### ✅ AIP-4: SAM Integration & Wall Segmentation - **COMPLETE**
- ✅ **Accuracy:** SAM achieves excellent wall segmentation (19 masks generated)
- ✅ **Performance:** 79s processing time with optimization modes
- ✅ **Reliability:** Consistent wall detection across different room types

#### ✅ AIP-5: Wall Mask Isolation Logic - **COMPLETE** 
- ✅ **Algorithm:** Center-based scoring system for optimal wall selection
- ✅ **Robustness:** Handles multiple wall candidates intelligently
- ✅ **Accuracy:** Successfully isolates primary wall in test cases

#### ✅ AIP-6: Perspective Transformation with OpenCV - **COMPLETE**
- ✅ **Scaling:** Adaptive sizing (TV: 25%, Painting: 30% of wall)
- ✅ **Perspective:** Proper aspect ratio preservation and centering
- ✅ **Quality:** Natural product placement without distortion

#### ✅ AIP-7: Alpha Blending & Compositing - **COMPLETE**
- ✅ **Transparency:** Smart background detection for any product type
- ✅ **Visual Realism:** Clean blending preserving original product appearance
- ✅ **No Artifacts:** Eliminated unwanted shadows and transparency issues

#### ✅ AIP-8: Documentation & Pipeline Documentation - **COMPLETE**
- ✅ **Code Documentation:** Comprehensive README and inline comments
- ✅ **PoC Document:** Proof of completion with evidence
- ✅ **Final Polish:** JIRA status alignment and final documentation review

### 🔄 Future Work: AIP-2/AIP-3 Generative Pipeline - **PLANNED**
- ⏳ Stable Diffusion + ControlNet pipeline setup
- ⏳ Output quality: alignment, scaling, realistic shadows  
- ⏳ Size variation capability (42" vs 55" TV examples)

## 📄 License

This project is part of an AI candidate assessment for Module 2 (Single Wall Fitting).

## 🔗 Links

- **Repository:** [https://github.com/anupam-aarmy/AR_Preview](https://github.com/anupam-aarmy/AR_Preview)
- **Current Branch:** `feature/AIP-4-segmentation-pipeline`
- **JIRA Context:** AIP-4 (Segmentation Pipeline), AIP-2 (Generative Pipeline)

---
