#!/usr/bin/env python3
"""
Sequential experiment runner for MLLM CXR report generation
Runs all scripts in order with proper error handling and dependency checking
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'h5py', 'tqdm', 'pandas', 'numpy',
        'transformers', 'peft', 'datasets', 'PIL', 'accelerate', 'bitsandbytes',
        'trl'  # Added for SFTTrainer
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {missing_packages}")
        print("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All required packages are installed")
    return True

def check_input_files():
    """Check if required input files exist"""
    required_files = [
        "../data/mimic.h5",
        "../data/chexpert.h5", 
        "../data/chexpert_train.csv",
        "../concepts/cbm_concepts.json",
        "../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required input files found")
    return True

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 STEP: {description}")
    print(f"   Script: {script_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name], 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully in {duration:.1f}s")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            return False
            
    except Exception as e:
        duration = time.time() - start_time
        print(f"❌ Error running {script_name}: {e}")
        print(f"   Duration: {duration:.1f}s")
        return False

def main():
    """Run the complete experiment pipeline"""
    print("🎯 MLLM CXR Report Generation Experiment")
    print("=========================================")
    
    # Check prerequisites
    print("\n🔍 Checking Prerequisites...")
    if not check_dependencies():
        print("❌ Please install missing dependencies first")
        return False
    
    if not check_input_files():
        print("❌ Please ensure all required input files are available")
        return False
    
    # Create results directory
    os.makedirs("../results", exist_ok=True)
    print("✅ Results directory ready")
    
    # Define the experiment pipeline
    pipeline_steps = [
        ("01_compute_concept_scores.py", "Compute concept scores for MIMIC-CXR images"),
        ("02_process_mimic_reports.py", "Process MIMIC reports and create training dataset"),
        ("03_prepare_chexpert_validation.py", "Prepare CheXpert validation dataset (500 samples)"),
        ("04_medgemma_finetune_sft.py", "Fine-tune MedGemma-4B-IT with SFTTrainer and CIDEr-D validation")
    ]
    
    overall_start = time.time()
    successful_steps = 0
    
    # Run each step
    for i, (script, description) in enumerate(pipeline_steps, 1):
        print(f"\n📊 PROGRESS: Step {i}/{len(pipeline_steps)}")
        
        success = run_script(script, description)
        
        if success:
            successful_steps += 1
        else:
            print(f"\n⚠️  Step {i} failed. Do you want to continue? (y/N): ", end="")
            choice = input().strip().lower()
            if choice != 'y':
                break
    
    # Final summary
    overall_duration = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"🏁 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"   ✅ Successful steps: {successful_steps}/{len(pipeline_steps)}")
    print(f"   ⏱️  Total duration: {overall_duration/60:.1f} minutes")
    
    if successful_steps == len(pipeline_steps):
        print(f"   🎉 All steps completed successfully!")
        print(f"   📁 Check ../results/ for trained models and outputs")
        return True
    else:
        print(f"   ⚠️  Some steps failed - check logs above")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 