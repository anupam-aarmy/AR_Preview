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
    print("🏠 AR Preview - AI Product Placement Solution")
    print("🎯 Production-ready pipelines with aspect ratio preservation")
    
    parser = argparse.ArgumentParser(description="AR Preview - Production Pipeline Implementation")
    parser.add_argument("--task", type=str, required=False, default="all", 
                       choices=["1", "2", "all"], 
                       help="Task to run (1: Deterministic SAM+Placement, 2: Generative SD+ControlNet, all: Both)")
    
    # Interactive mode if no arguments provided
    if len(sys.argv) == 1:
        print("\n🔧 Available Pipelines:")
        print("1. Task 1: Deterministic (SAM + OpenCV)")
        print("2. Task 2: Generative (Stable Diffusion + ControlNet)")
        print("3. Both pipelines")
        
        choice = input("\nSelect pipeline (1/2/3) or press Enter for both: ").strip()
        
        if choice == "1":
            run_deterministic_pipeline()
        elif choice == "2":
            run_generative_pipeline()
        else:
            print("\n🚀 Running both pipelines...")
            run_deterministic_pipeline()
            print("\n" + "="*40)
            run_generative_pipeline()
    else:
        args = parser.parse_args()
        
        if args.task == "1":
            run_deterministic_pipeline()
        elif args.task == "2":
            run_generative_pipeline()
        elif args.task == "all":
            print("\n🚀 Running both pipelines...")
            run_deterministic_pipeline()
            print("\n" + "="*40)
            run_generative_pipeline()
            
            print("\n✅ All pipelines completed successfully!")
            print("📁 Results available in output/ directory")
        
def run_deterministic_pipeline():
    """Run Task 1: Production Deterministic Pipeline (SAM + OpenCV)"""
    try:
        from deterministic_pipeline import main as deterministic_main
        deterministic_main()
        print("✅ Task 1 completed successfully!")
        
    except ImportError as e:
        print(f"❌ Task 1 import error: {e}")
    except Exception as e:
        print(f"❌ Task 1 execution error: {e}")
        
def run_generative_pipeline():
    """Run Task 2: Production Generative Pipeline (SD + ControlNet)""" 
    try:
        from generative_pipeline import main as generative_main
        generative_main()
        print("✅ Task 2 completed successfully!")
        
    except ImportError as e:
        print(f"❌ Task 2 import error: {e}")
    except Exception as e:
        print(f"❌ Task 2 execution error: {e}")

# Legacy function names for backward compatibility        
def run_improved_task1():
    """Legacy function - redirects to production deterministic pipeline"""
    run_deterministic_pipeline()
        
def run_improved_task2():
    """Legacy function - redirects to production generative pipeline"""
    run_generative_pipeline()

def run_task1():
    """Legacy function - redirects to production deterministic pipeline"""
    run_deterministic_pipeline()
        
def run_task2():
    """Legacy function - redirects to production generative pipeline"""
    run_generative_pipeline()

if __name__ == "__main__":
    main()