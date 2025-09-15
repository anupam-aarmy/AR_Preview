#!/usr/bin/env python3
"""
Download SAM model checkpoint for wall segmentation
"""
import os
import requests
from pathlib import Path

def download_sam_model():
    """Download SAM ViT-H model checkpoint"""
    model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    model_dir = Path("models")
    model_path = model_dir / "sam_vit_h_4b8939.pth"
    
    # Create models directory
    model_dir.mkdir(exist_ok=True)
    
    # Check if model already exists
    if model_path.exists():
        print(f"✅ SAM model already exists at {model_path}")
        return str(model_path)
    
    print(f"📥 Downloading SAM model from {model_url}")
    print("⚠️  This is a large file (~2.4GB), please wait...")
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rProgress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ SAM model downloaded successfully to {model_path}")
        return str(model_path)
        
    except Exception as e:
        print(f"\n❌ Failed to download SAM model: {e}")
        if model_path.exists():
            model_path.unlink()  # Remove partial download
        return None

if __name__ == "__main__":
    download_sam_model()