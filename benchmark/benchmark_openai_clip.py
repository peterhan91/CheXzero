#!/usr/bin/env python3
"""
Benchmark script for OpenAI CLIP model.
"""

import os
import sys
import argparse
import torch

# Add the parent directory to Python path to import from the main codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force tqdm to use console mode to avoid jupyter widget issues
os.environ['TQDM_DISABLE_JUPYTER'] = '1'
import tqdm
tqdm.tqdm_notebook = tqdm.tqdm  # Patch to use standard tqdm instead of notebook tqdm

from benchmark_base import setup_test_data, save_results, evaluate_predictions, save_detailed_results
from tqdm import tqdm

def load_openai_clip_model(device):
    """Load OpenAI CLIP model using open_clip library for better compatibility."""
    print("Loading OpenAI CLIP model...")
    
    try:
        import open_clip
        
        # Load the OpenAI CLIP model using open_clip library
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        model = model.to(device)
        
        print("OpenAI CLIP model loaded successfully using open_clip")
        return model, preprocess
        
    except ImportError as e:
        print(f"Error importing open_clip: {e}")
        print("Make sure the open_clip package is installed: pip install open_clip_torch")
        raise
    except Exception as e:
        print(f"Error loading OpenAI CLIP model: {e}")
        raise

def run_openai_clip_evaluation(model, dataloader, y_true, labels, templates, device, context_length=77):
    """
    Run zero-shot evaluation using OpenAI CLIP model with open_clip library.
    """
    import open_clip
    import numpy as np
    import torch
    from tqdm import tqdm
    
    model.eval()
    pos_template, neg_template = templates[0]
    
    print(f"Running OpenAI CLIP evaluation on {len(dataloader.dataset)} images...")
    print(f"Template: ('{pos_template}', '{neg_template}')")
    
    # Generate text embeddings for positive and negative templates
    print("Generating text embeddings...")
    pos_texts = [pos_template.format(label) for label in labels]
    neg_texts = [neg_template.format(label) for label in labels]
    
    with torch.no_grad():
        pos_tokens = open_clip.tokenize(pos_texts).to(device)
        neg_tokens = open_clip.tokenize(neg_texts).to(device)
        
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        
        # Normalize features
        pos_features = pos_features / pos_features.norm(dim=-1, keepdim=True)
        neg_features = neg_features / neg_features.norm(dim=-1, keepdim=True)
    
    # Run image predictions
    print("Running image predictions...")
    pos_predictions = []
    neg_predictions = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Processing images")):
            images = data['img'].to(device)
            batch_size = images.shape[0]
            
            # Get image features
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute logits for positive template
            # Get logit scale from model (open_clip uses logit_scale)
            logit_scale = model.logit_scale.exp()
            pos_logits = (image_features @ pos_features.T) * logit_scale
            pos_logits = pos_logits.cpu().numpy()
            
            # Compute logits for negative template  
            neg_logits = (image_features @ neg_features.T) * logit_scale
            neg_logits = neg_logits.cpu().numpy()
            
            # Handle each image in the batch individually
            for j in range(batch_size):
                pos_predictions.append(pos_logits[j] if pos_logits.ndim > 1 else pos_logits)
                neg_predictions.append(neg_logits[j] if neg_logits.ndim > 1 else neg_logits)
    
    pos_pred = np.array(pos_predictions)
    neg_pred = np.array(neg_predictions)
    
    # Compute probabilities with softmax
    sum_pred = np.exp(pos_pred) + np.exp(neg_pred)
    y_pred = np.exp(pos_pred) / sum_pred
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Evaluate using our common evaluation function
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    return results_df, y_pred

def benchmark_openai_clip(datasets, device):
    """Benchmark OpenAI CLIP on all datasets."""
    print(f"\n{'='*60}")
    print(f"Benchmarking OpenAI CLIP model")
    print(f"{'='*60}")
    
    # Load model
    try:
        model, preprocess = load_openai_clip_model(device)
        model.to(device)
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading OpenAI CLIP model: {e}")
        return
    
    # Test on each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing on {dataset_name} ---")
        
        try:
            # Setup test data using OpenAI CLIP standard resolution (224x224)
            dataloader, y_true, labels, templates = setup_test_data(
                dataset_name, 
                batch_size=32,  # Smaller batch size for external models
                input_resolution=224  # OpenAI CLIP uses 224x224
            )
            
            # Run evaluation
            results_df, y_pred = run_openai_clip_evaluation(
                model, dataloader, y_true, labels, templates, device, context_length=77
            )
            
            # Save detailed results
            config_data = {
                "model_name": "openai_clip",
                "dataset": dataset_name,
                "method": "openai_clip_zero_shot",
                "num_images": len(y_true),
                "num_labels": len(labels),
                "labels": labels,
                "input_resolution": 224
            }
            
            # Get predictions for detailed saving
            save_detailed_results(y_pred, y_true, labels, "openai_clip", dataset_name, config_data)
            
            # Save results
            save_results(results_df, "openai_clip", dataset_name)
            
        except Exception as e:
            print(f"Error evaluating OpenAI CLIP on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenAI CLIP model")
    parser.add_argument('--datasets', nargs='+', 
                        default=['chexpert_test', 'padchest_test', 'vindrcxr_test', 'vindrpcxr_test', 'indiana_test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Benchmark OpenAI CLIP
    benchmark_openai_clip(args.datasets, device)
    
    print(f"\n{'='*60}")
    print("OpenAI CLIP benchmarking completed!")
    print("Results saved in benchmark/results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 