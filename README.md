# AR Preview - AI-Powered Product Visualization

An AI-powered product visualization MVP for AR preview applications, allowing users to visualize wall fittings (TVs, paintings, frames) in their own space with realistic scaling, perspective, and lighting.

## 🎯 Project Overview

This project implements two parallel solutions for realistic wall fitting visualization:

### Task 1: Deterministic Pipeline - **COMPLETED**
- **Wall Segmentation:** Uses SAM (Segment Anything Model) for zero-shot wall detection
- **Product Placement:** OpenCV perspective transformation with intelligent sizing
- **Alpha Blending:** Advanced blending with automatic transparency detection

### Task 2: Generative Pipeline - **PLANNED**
- **Stable Diffusion Inpainting:** Guided by ControlNet for context-aware generation
- **ControlNet Conditioning:** Depth/inpainting for proper alignment and scaling
- **Size Variations:** Adjust mask scale and prompts for different product sizes

## 🏗️ Project Structure

```
AR_Preview/
├── assets/                        # Input images
├── models/                        # AI model storage (SAM checkpoint)
├── output/                        # Generated results
├── src/                          # Source modules
├── main.py                       # Main pipeline orchestrator
├── download_sam.py              # SAM model download utility
├── requirements.txt              # Python dependencies
├── PROOF_OF_COMPLETION.md       # Project completion documentation
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
# Run pipeline with both products (fast mode)
python main.py

# Use enhanced pipeline
python src/pipeline.py

# Process specific product
python src/pipeline.py --product assets/prod_2_tv.png
```

## 📊 Sample Results

Check the `output/` directory for example results including:
- Product placement visualizations
- Wall segmentation masks  
- Before/after comparison views

For detailed results and analysis, see [PROOF_OF_COMPLETION.md](PROOF_OF_COMPLETION.md).

## 🛠️ Technology Stack

- **Deep Learning:** PyTorch, TorchVision
- **Computer Vision:** OpenCV, Segment Anything Model (SAM)
- **Generative AI:** Stable Diffusion, ControlNet (planned)
- **Visualization:** Matplotlib
- **Core:** NumPy, Python 3.8+

## 🤝 Contributing

1. Create a feature branch from the main branch
2. Follow the naming convention: `feature/description`
3. Implement changes with proper testing
4. Update documentation as needed
5. Create a pull request with detailed description

## 📄 License

This project is part of an AI candidate assessment for Module 2 (Single Wall Fitting).

## 🔗 Links

- **Repository:** [https://github.com/anupam-aarmy/AR_Preview](https://github.com/anupam-aarmy/AR_Preview)
- **Documentation:** See [PROOF_OF_COMPLETION.md](PROOF_OF_COMPLETION.md) for detailed implementation details and progress tracking

---
