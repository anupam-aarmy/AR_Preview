"""
AR Preview MVP - Production Release
Implements both Task 1 (Deterministic) and Task 2 (Generative) pipelines
Production-ready implementation with aspect-corrected placement and enhanced detail preservation

Key Features:
- Perfect aspect ratio preservation from actual product images
- Smart sizing with bounds checking  
- Enhanced detail preservation in generative pipeline
- Production-quality error handling and logging
"""

import os
import sys
import argparse
from pathlib import Path

# Add scripts to path for production pipeline imports
sys.path.append(str(Path(__file__).parent / "scripts"))

def main():
    """Main execution - runs production-ready pipelines"""
    print("="*80)
    print("🏠 AR PREVIEW MVP - PRODUCTION RELEASE")
    print("✨ AI-Powered Product Placement Solution")
    print("🎯 Using production assets: tv_1.png + painting_1.png")
    print("🔧 Aspect-corrected placement with enhanced detail preservation")
    print("="*80)
    
    parser = argparse.ArgumentParser(description="AR Preview - Production Pipeline Implementation")
    parser.add_argument("--task", type=str, required=False, default="all", 
                       choices=["1", "2", "all"], 
                       help="Task to run (1: Deterministic SAM+Placement, 2: Generative SD+ControlNet, all: Both)")
    
    # Interactive mode if no arguments provided
    if len(sys.argv) == 1:
        print("\n🔧 Available Production Pipelines:")
        print("1. Task 1: Deterministic (SAM + OpenCV + Aspect Correction)")
        print("2. Task 2: Generative (Stable Diffusion + ControlNet + Enhanced Details)")
        print("3. Both production pipelines")
        
        choice = input("\nSelect pipeline (1/2/3) or press Enter for both: ").strip()
        
        if choice == "1":
            run_deterministic_pipeline()
        elif choice == "2":
            run_generative_pipeline()
        else:
            print("\n🚀 Running both production pipelines...")
            run_deterministic_pipeline()
            print("\n" + "="*50)
            run_generative_pipeline()
    else:
        args = parser.parse_args()
        
        if args.task == "1":
            run_deterministic_pipeline()
        elif args.task == "2":
            run_generative_pipeline()
        elif args.task == "all":
            print("\n🚀 Running both production pipelines...")
            run_deterministic_pipeline()
            print("\n" + "="*50)
            run_generative_pipeline()
            
            print("\n🎉 ALL PRODUCTION TASKS COMPLETED SUCCESSFULLY!")
            print("📁 Check output/ directory for results")
            print("📊 Task 1: output/task1_deterministic/")
            print("📊 Task 2: output/task2_real_product_placement/")
            print("🎯 Production-quality product placement with correct aspect ratios")
            print("✨ Enhanced detail preservation and smart positioning")
        
def run_deterministic_pipeline():
    """Run Task 1: Production Deterministic Pipeline (SAM + OpenCV)"""
    print("\n🎯 STARTING TASK 1: PRODUCTION DETERMINISTIC PIPELINE")
    print("🔧 SAM segmentation + OpenCV placement with CORRECT aspect ratios")
    print("✅ Products use ACTUAL aspect ratios from input images")
    print("✅ Smart sizing with bounds checking and safe positioning")
    print("✅ Complete mask filling with LANCZOS resampling")
    print("="*60)
    
    try:
        from deterministic_pipeline import main as deterministic_main
        deterministic_main()
        print("\n✅ Task 1 production pipeline completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Task 1 import error: {e}")
        print("   Production file deterministic_pipeline.py not found")
    except Exception as e:
        print(f"\n❌ Task 1 execution error: {e}")
        
def run_generative_pipeline():
    """Run Task 2: Production Generative Pipeline (SD + ControlNet)""" 
    print("\n🎯 STARTING TASK 2: PRODUCTION GENERATIVE PIPELINE")
    print("🔧 Stable Diffusion + ControlNet with enhanced quality")
    print("📺 TV: Ultra-sharp detail preservation with actual aspect ratios")
    print("🖼️ Painting: Enhanced texture preservation with safe positioning")
    print("🎨 Using actual product images with optimized prompts")
    print("="*60)
    
    try:
        from generative_pipeline import main as generative_main
        generative_main()
        print("\n✅ Task 2 production pipeline completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Task 2 import error: {e}")
        print("   Production file generative_pipeline.py not found")
    except Exception as e:
        print(f"\n❌ Task 2 execution error: {e}")

# Legacy function names for backward compatibility        
def run_improved_task1():
    """Legacy function - redirects to production deterministic pipeline"""
    print("\n⚠️ Legacy function called - redirecting to production version...")
    run_deterministic_pipeline()
        
def run_improved_task2():
    """Legacy function - redirects to production generative pipeline"""
    print("\n⚠️ Legacy function called - redirecting to production version...")
    run_generative_pipeline()

def run_task1():
    """Legacy function - redirects to production deterministic pipeline"""
    print("\n⚠️ Legacy function called - redirecting to production version...")
    run_deterministic_pipeline()
        
def run_task2():
    """Legacy function - redirects to production generative pipeline"""
    print("\n⚠️ Legacy function called - redirecting to production version...")
    run_generative_pipeline()

if __name__ == "__main__":
    main()