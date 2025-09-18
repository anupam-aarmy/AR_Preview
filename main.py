"""
AR Preview Pipeline - Main Entry Point
AI Assignment Implem        # Run Task 2
        print("\n📋 TASK 2: Stable Diffusion + ControlNet (AI Assignment)")
        print("-" * 50)
        try:
            from scripts.task2_assignment_compliant import main as task2_main
            task2_main()
            print("✅ Task 2 completed!")
        except Exception as e:
            print(f"❌ Task 2 failed: {e}")
            returnor Module 2 (Single Wall Fitting)

Tasks:
- Task 1: SAM wall segmentation + deterministic product placement
- Task 2: Stable Diffusion + ControlNet generative product placement

Usage:
    python main.py --task 1                    # Run Task 1 (SAM + Product Placement)
    python main.py --task 2                    # Run Task 2 (SD + ControlNet)
    python main.py --task all                  # Run both tasks
"""

import sys
import argparse
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent / "scripts"))

def main():
    parser = argparse.ArgumentParser(description="AR Preview Pipeline - AI Assignment Implementation")
    parser.add_argument("--task", type=str, required=True, choices=["1", "2", "all"], 
                       help="Task to run (1: SAM+Placement, 2: SD+ControlNet, all: Both tasks)")
    
    args = parser.parse_args()
    
    if args.task == "1":
        print("🚀 Running Task 1: SAM + Product Placement")
        print("=" * 50)
        try:
            from task1_clean import main as task1_main
            task1_main()
            print("\n✅ Task 1 completed successfully!")
        except Exception as e:
            print(f"\n❌ Task 1 failed: {e}")
            sys.exit(1)
            
    elif args.task == "2":
        print("🚀 Running Task 2: Stable Diffusion + ControlNet (AI Assignment)")
        print("=" * 60)
        try:
            from scripts.task2_assignment_compliant import main as task2_main
            task2_main()
            print("\n✅ Task 2 completed successfully!")
        except Exception as e:
            print(f"\n❌ Task 2 failed: {e}")
            sys.exit(1)
            
    elif args.task == "all":
        print("🚀 Running Both Tasks: Complete AI Assignment Pipeline")
        print("=" * 60)
        
        # Run Task 1
        print("\n📋 TASK 1: SAM + Product Placement")
        print("-" * 40)
        try:
            from task1_clean import main as task1_main
            task1_main()
            print("✅ Task 1 completed!")
        except Exception as e:
            print(f"❌ Task 1 failed: {e}")
            sys.exit(1)
        
        # Run Task 2
        print("\n📋 TASK 2: Stable Diffusion + ControlNet")
        print("-" * 40)
        try:
            from task2_clean import main as task2_main
            task2_main()
            print("✅ Task 2 completed!")
        except Exception as e:
            print(f"❌ Task 2 failed: {e}")
            sys.exit(1)
            
        print("\n🎉 ALL TASKS COMPLETED SUCCESSFULLY!")
        print("📁 Check output/ directory for results")
        print("📊 Task 1: output/task1_deterministic/")
        print("📊 Task 2: output/task2_controlnet/")

if __name__ == "__main__":
    main()