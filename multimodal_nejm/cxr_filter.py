#!/usr/bin/env python3
"""
Filter chest X-ray cases from NEJM Image Challenges using BiomedCLIP
"""

import os
import torch
import shutil
from pathlib import Path
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer

def setup_model():
    """Load BiomedCLIP model and tokenizer"""
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    return model, preprocess, tokenizer, device

def classify_image(image_path, model, preprocess, tokenizer, device):
    """Classify a single image and return chest X-ray probability"""
    template = 'this is a photo of '
    labels = [
        'adenocarcinoma histopathology',
        'brain MRI', 
        'covid line chart',
        'squamous cell carcinoma histopathology',
        'immunohistochemistry histopathology',
        'bone X-ray',
        'chest X-ray',  # This is what we're looking for
        'pie chart',
        'hematoxylin and eosin histopathology'
    ]
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Tokenize text labels
    texts = tokenizer([template + l for l in labels], context_length=256).to(device)
    
    # Get predictions
    with torch.no_grad():
        image_features, text_features, logit_scale = model(image_tensor, texts)
        logits = (logit_scale * image_features @ text_features.t()).detach().softmax(dim=-1)
        
    # Return probability for chest X-ray (index 6)
    chest_xray_prob = logits[0][6].cpu().item()
    predicted_label = labels[torch.argmax(logits[0]).item()]
    
    return chest_xray_prob, predicted_label

def filter_chest_xrays(nejm_path, output_path, threshold=0.5):
    """Filter chest X-ray cases from NEJM dataset"""
    nejm_path = Path(nejm_path)
    output_path = Path(output_path)
    
    # Setup model
    print("Loading BiomedCLIP model...")
    model, preprocess, tokenizer, device = setup_model()
    
    # Get all case directories
    case_dirs = [d for d in nejm_path.iterdir() if d.is_dir() and d.name.startswith('IC')]
    print(f"Found {len(case_dirs)} NEJM cases to process")
    
    chest_xray_cases = []
    
    for i, case_dir in enumerate(case_dirs):
        image_path = case_dir / 'image.png'
        
        if not image_path.exists():
            continue
            
        try:
            # Classify image
            chest_prob, predicted_label = classify_image(image_path, model, preprocess, tokenizer, device)
            
            print(f"[{i+1}/{len(case_dirs)}] {case_dir.name}: {predicted_label} (chest X-ray prob: {chest_prob:.3f})")
            
            # If chest X-ray probability is above threshold, copy to output
            if chest_prob > threshold:
                output_case_dir = output_path / case_dir.name
                output_case_dir.mkdir(exist_ok=True)
                
                # Copy all files from the case
                for file_path in case_dir.iterdir():
                    if file_path.is_file():
                        shutil.copy2(file_path, output_case_dir)
                
                chest_xray_cases.append((case_dir.name, chest_prob, predicted_label))
                print(f"  -> Copied to {output_case_dir}")
                
        except Exception as e:
            print(f"Error processing {case_dir.name}: {e}")
            continue
    
    print(f"\nFiltered {len(chest_xray_cases)} chest X-ray cases (threshold: {threshold})")
    
    # Save summary
    summary_path = output_path / 'filtering_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Chest X-ray filtering results (threshold: {threshold})\n")
        f.write(f"Total cases processed: {len(case_dirs)}\n")
        f.write(f"Chest X-ray cases found: {len(chest_xray_cases)}\n\n")
        
        for case_name, prob, pred_label in chest_xray_cases:
            f.write(f"{case_name}: {pred_label} (chest X-ray prob: {prob:.3f})\n")
    
    print(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    nejm_dataset_path = "/home/than/Datasets/NEJM/NEJM/"
    output_path = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    
    filter_chest_xrays(nejm_dataset_path, output_path, threshold=0.99) 