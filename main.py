"""
AR Preview MVP - Main Pipeline (IMPROVED VERSION)
Implements both Task 1 (Deterministic) and Task 2 (Generative) pipelines
Uses new assets: tv_1.png + painting_1.png

Enhanced Features:
- Better detail preservation 
- Improved alignment and positioning
- Correct aspect ratios maintained
- Enhanced quality settings
"""

import os
import sys
import argparse
from pathlib import Path

# Import task implementations
from task1_fixed_placement import run_task1_with_improvements

def main():
    """Main execution - runs improved pipelines with new assets"""
    print("="*80)
    print("🏠 AR PREVIEW MVP - IMPROVED WALL FITTING VISUALIZATION")
    print("✨ Enhanced AI-Powered Product Placement Solution")
    print("🎯 Using new assets: tv_1.png + painting_1.png")
    print("🔧 Improved detail preservation, alignment, and aspect ratios")
    print("="*80)
    
    parser = argparse.ArgumentParser(description="AR Preview - Improved Pipeline Implementation")
    parser.add_argument("--task", type=str, required=False, default="all", 
                       choices=["1", "2", "all"], 
                       help="Task to run (1: Enhanced SAM+Placement, 2: Enhanced SD+ControlNet, all: Both improved tasks)")
    
    # If no args provided, run interactive mode
    if len(sys.argv) == 1:
        print("\n🔧 Available Improved Pipelines:")
        print("1. Task 1: Enhanced Deterministic (SAM + OpenCV)")
        print("2. Task 2: Enhanced Generative (Stable Diffusion + ControlNet)")
        print("3. Both improved pipelines")
        
        choice = input("\nSelect pipeline (1/2/3) or press Enter for both: ").strip()
        
        if choice == "1":
            run_improved_task1()
        elif choice == "2":
            run_improved_task2()
        else:
            print("\n🚀 Running both improved pipelines...")
            run_improved_task1()
            print("\n" + "="*50)
            run_improved_task2()
    else:
        args = parser.parse_args()
        
        if args.task == "1":
            run_improved_task1()
        elif args.task == "2":
            run_improved_task2()
        elif args.task == "all":
            print("\n🚀 Running both improved pipelines...")
            run_improved_task1()
            print("\n" + "="*50)
            run_improved_task2()
            
            print("\n🎉 ALL IMPROVED TASKS COMPLETED SUCCESSFULLY!")
            print("📁 Check output/ directory for enhanced results")
            print("📊 Task 1: output/task1_deterministic/")
            print("📊 Task 2: output/task2_real_product_placement/")
            print("🎯 Enhanced product placement with tv_1.png + painting_1.png")
            print("✨ Better detail preservation, alignment, and aspect ratios")
        
def run_improved_task1():
    """Run Task 1: ASPECT CORRECTED Deterministic Pipeline"""
    print("\n🎯 STARTING TASK 1: ASPECT CORRECTED DETERMINISTIC PIPELINE")
    print("🔧 SAM segmentation + OpenCV placement with CORRECT aspect ratios")
    print("✅ Products use ACTUAL aspect ratios from input images")
    print("✅ No more hardcoded 16:9 for TVs or 1:1 for paintings")
    print("✅ Placement rectangles match actual product proportions")
    print("="*60)
    
    try:
        from task1_fixed_placement import main as task1_main
        task1_main()
        print("\n✅ Task 1 ASPECT CORRECTED pipeline completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Task 1 import error: {e}")
        print("   Please ensure task1_fixed_placement.py exists in project root")
    except Exception as e:
        print(f"\n❌ Task 1 execution error: {e}")
        
def run_improved_task2():
    """Run Task 2: Enhanced Generative Pipeline""" 
    print("\n🎯 STARTING TASK 2: ENHANCED GENERATIVE PIPELINE")
    print("🔧 Stable Diffusion + ControlNet with quality improvements")
    print("📺 TV: Better detail preservation and aspect ratio accuracy")
    print("🖼️ Painting: Enhanced texture preservation and positioning")
    print("🎨 Using actual product images (not text generation)")
    print("="*60)
    
    try:
        from scripts.task2_improved_placement import main as task2_main
        task2_main()
        print("\n✅ Task 2 enhanced pipeline completed successfully!")
        
    except ImportError as e:
        print(f"\n❌ Task 2 import error: {e}")
        print("   Please ensure task2_improved_placement.py exists in scripts/")
        print("   Falling back to corrected task2_corrected_placement.py...")
        try:
            from task2_corrected_placement import main as task2_fallback
            task2_fallback()
            print("\n✅ Task 2 (fallback) completed!")
        except Exception as e2:
            print(f"❌ Task 2 fallback also failed: {e2}")
    except Exception as e:
        print(f"\n❌ Task 2 execution error: {e}")

# Legacy functions for backward compatibility        
def run_task1():
    """Run Task 1: Legacy wrapper"""
    print("\n⚠️ Legacy function called - redirecting to improved version...")
    run_improved_task1()
        
def run_task2():
    """Run Task 2: Legacy wrapper""" 
    print("\n⚠️ Legacy function called - redirecting to improved version...")
    run_improved_task2()

if __name__ == "__main__":
    main()