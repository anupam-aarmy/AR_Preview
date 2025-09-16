"""
AIP-9: Stable Diffusion Environment Setup (Basic Version)
This version tests basic functionality without advanced optimizations
"""

import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

def test_basic_imports():
    """Test core dependencies without problematic optimizations"""
    print("🧪 Testing basic imports...")
    
    try:
        import diffusers
        print(f"✅ diffusers {diffusers.__version__}")
        
        import transformers
        print(f"✅ transformers {transformers.__version__}")
        
        print("✅ Core imports successful!")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {str(e)}")
        return False

def test_pipeline_import():
    """Test pipeline import separately"""
    print("🧪 Testing pipeline import...")
    
    try:
        # Import one component at a time
        from diffusers import DiffusionPipeline
        print("✅ DiffusionPipeline imported")
        
        from diffusers import StableDiffusionPipeline
        print("✅ StableDiffusionPipeline imported")
        
        # Try the inpainting pipeline
        from diffusers import StableDiffusionInpaintPipeline
        print("✅ StableDiffusionInpaintPipeline imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline import error: {str(e)}")
        return False

def test_model_loading():
    """Test loading a model in the most basic way"""
    print("🧪 Testing basic model loading...")
    
    try:
        from diffusers import StableDiffusionInpaintPipeline
        
        # Try loading with minimal settings
        print("📦 Loading SD Inpainting model...")
        model_id = "runwayml/stable-diffusion-inpainting"
        
        # Most basic loading approach
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # Use float32 for compatibility
            use_safetensors=False,  # Disable safetensors if causing issues
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print("✅ Model loaded successfully!")
        print(f"   Model type: {type(pipe).__name__}")
        print(f"   Device: CPU (PyTorch {torch.__version__})")
        
        return pipe
        
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        return None

def main():
    print("🎯 AIP-9: Basic Stable Diffusion Environment Test")
    print("-" * 50)
    
    # Step 1: Test basic imports
    if not test_basic_imports():
        return False
    
    # Step 2: Test pipeline imports
    if not test_pipeline_import():
        return False
    
    # Step 3: Test model loading
    pipe = test_model_loading()
    if pipe is None:
        return False
    
    print("\n✅ AIP-9 BASIC TEST COMPLETED!")
    print("🚀 Stable Diffusion environment is working (basic functionality)")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)