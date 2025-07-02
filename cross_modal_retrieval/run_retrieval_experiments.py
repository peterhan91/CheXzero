#!/usr/bin/env python3
"""
Simple launcher script for image-report retrieval experiments
"""

import os
import sys
import argparse
import subprocess

def run_experiment(models=None, datasets=None, device='auto'):
    """Run retrieval experiment with specified parameters"""
    
    if models is None:
        models = ['chexzero', 'biomedclip', 'openai_clip']  # Skip LLM for now as it needs trained model
    
    if datasets is None:
        datasets = ['padchest-test', 'indiana-test']
    
    cmd = [
        sys.executable, 
        'cross_modal_retrieval/image_report_retrieval.py',
        '--models'] + models + [
        '--datasets'] + datasets + [
        '--device', device
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Launcher for Image-Report Retrieval Experiments")
    parser.add_argument('--models', nargs='+', 
                        default=['llm_embedding','chexzero', 'biomedclip', 'openai_clip'],
                        choices=['llm_embedding', 'chexzero', 'biomedclip', 'openai_clip'],
                        help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+',
                        default=['padchest-test', 'indiana-test'],
                        choices=['padchest-test', 'indiana-test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with OpenAI CLIP only')
    
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick test with OpenAI CLIP only...")
        models = ['openai_clip']
        datasets = ['padchest-test']
    else:
        models = args.models
        datasets = args.datasets
    
    print(f"Running experiments with:")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Device: {args.device}")
    
    return_code = run_experiment(models, datasets, args.device)
    
    if return_code == 0:
        print("\n🎉 Experiments completed successfully!")
    else:
        print(f"\n❌ Experiments failed with return code: {return_code}")
    
    return return_code

if __name__ == "__main__":
    sys.exit(main()) 