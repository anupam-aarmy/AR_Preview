"""
Set environment variables to address HuggingFace Hub caching warnings
Run this before executing Task 2 to suppress symlink warnings
"""

import os

def setup_huggingface_environment():
    """Setup environment variables for HuggingFace Hub"""
    # Disable symlink warnings on Windows
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
    
    # Set cache directory if needed
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
    
    print("✅ HuggingFace environment configured")
    print("🔇 Symlink warnings disabled")
    print(f"📁 Cache directory: {os.environ.get('HF_HOME')}")

if __name__ == "__main__":
    setup_huggingface_environment()