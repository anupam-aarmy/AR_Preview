# AR Preview - AI-Powered Product Visualization

An AI-powered product visualization MVP for AR preview applications, allowing users to visualize wall fittings (TVs, paintings, frames) in their own space with realistic scaling, perspective, and lighting.

## 🎯 Project Overview

This project implements two parallel solutions for realistic wall fitting visualization:

### Task 1: Deterministic Pipeline (AIP-4)
- **Wall Segmentation:** Uses SAM (Segment Anything Model) for zero-shot wall detection
- **Product Placement:** OpenCV perspective transformation (homography) for realistic scaling
- **Blending:** Alpha blending to merge product PNG onto wall

### Task 2: Generative Pipeline (AIP-2)
- **Stable Diffusion Inpainting:** Guided by ControlNet for context-aware generation
- **ControlNet Conditioning:** Depth/inpainting for proper alignment and scaling
- **Size Variations:** Adjust mask scale and prompts for different product sizes

## 🏗️ Project Structure

```
AR_Preview/
├── .github/
│   └── copilot-instructions.md    # AI agent guidance and project context
├── assets/                        # Input images
│   ├── room_wall.png             # Sample room photo
│   ├── prod_1_paintings.png      # Product image (paintings)
│   └── prod_2_tv.png             # Product image (TV)
├── models/                        # AI model storage (SAM, Stable Diffusion)
├── output/                        # Generated results and composite images
├── src/                          # Source modules
│   └── __init__.py
├── main.py                       # Main pipeline orchestrator
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

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

### Usage

```bash
python main.py
```

## 🛠️ Technology Stack

- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Segment Anything Model (SAM)
- **Generative AI:** Stable Diffusion, ControlNet, Hugging Face Diffusers
- **Visualization:** Matplotlib
- **Core:** NumPy, Python 3.8+

## 📋 Development Roadmap

### Phase 1: Deterministic Pipeline (Current - AIP-4)
- [x] Project setup and structure
- [ ] SAM wall segmentation implementation
- [ ] OpenCV homography for product placement
- [ ] Alpha blending for realistic composition
- [ ] Testing with sample assets

### Phase 2: Generative Pipeline (AIP-2)
- [ ] Stable Diffusion environment setup
- [ ] ControlNet integration for guided generation
- [ ] Inpainting pipeline implementation
- [ ] Size variation handling (42" vs 55" TV)

### Phase 3: Enhancement & Optimization
- [ ] Performance optimization
- [ ] Error handling and edge cases
- [ ] Comprehensive testing
- [ ] Documentation and examples

## 🎨 Sample Assets

The project includes sample assets for testing:
- **Room Photo:** `assets/room_wall.png` - Room with visible wall
- **Products:** TV and painting images with transparency
- **Expected Output:** Realistic product placement on wall

## 🤝 Contributing

1. Create a feature branch from `main`
2. Follow the naming convention: `feature/AIP-X-description`
3. Implement changes with proper testing
4. Update documentation as needed
5. Create a pull request with detailed description

## 📝 Evaluation Criteria

### Task 1 (Deterministic):
- ✅ Accuracy of wall segmentation using SAM
- ✅ Proper scaling & perspective of product on wall
- ✅ Visual realism and clean blending

### Task 2 (Generative):
- ✅ Successful Stable Diffusion + ControlNet pipeline setup
- ✅ Output quality: alignment, scaling, realistic shadows
- ✅ Size variation capability

## 📄 License

This project is part of an AI candidate assessment for Module 2 (Single Wall Fitting).

## 🔗 Links

- **Repository:** [https://github.com/anupam-aarmy/AR_Preview](https://github.com/anupam-aarmy/AR_Preview)
- **Current Branch:** `feature/AIP-4-segmentation-pipeline`
- **JIRA Context:** AIP-4 (Segmentation Pipeline), AIP-2 (Generative Pipeline)

---

*Built with ❤️ for realistic AR product visualization*