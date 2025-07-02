#!/usr/bin/env python3
"""
Compute concept scores (dot products) for MIMIC CXR images.
Generates [#CXR, #Concepts] matrix and saves as h5 file.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import h5py
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import zero_shot
import clip

def print_gpu_memory(stage=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"🖥️ GPU Memory {stage}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

def load_concepts():
    """Load the 68 concepts from cbm_concepts.json"""
    concepts_path = "concepts/cbm_concepts.json"
    with open(concepts_path, 'r') as f:
        concepts_data = json.load(f)
    
    # Flatten all concepts from all categories
    all_concepts = []
    for category, concepts_list in concepts_data.items():
        for concept_info in concepts_list:
            all_concepts.append(concept_info['concept'])
    
    print(f"✅ Loaded {len(all_concepts)} concepts from {len(concepts_data)} categories")
    return all_concepts

def compute_mimic_concept_scores():
    """Compute concept scores for MIMIC CXR dataset"""
    
    print("=== Computing MIMIC CXR Concept Scores ===")
    
    print("\n=== Step 1: Load CLIP Model ===")
    model = load_clip(
        model_path="checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ CLIP model loaded")
    print_gpu_memory("after CLIP model load")
    
    print("\n=== Step 2: Setup MIMIC Dataset ===")
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    mimic_dataset = zero_shot.CXRTestDataset(
        img_path="data/mimic.h5",
        transform=transform,
    )
    
    mimic_loader = torch.utils.data.DataLoader(
        mimic_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    
    print(f"✅ MIMIC dataset loaded: {len(mimic_dataset)} images")
    
    print("\n=== Step 3: Load and Encode Concepts ===")
    concepts = load_concepts()
    
    # Encode concepts in batches
    concept_batch_size = 512
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_features = torch.cat(all_concept_features).to('cuda')
    print(f"✅ Concept features shape: {concept_features.shape}")
    
    print("\n=== Step 4: Encode Images and Compute Scores ===")
    all_concept_scores = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(mimic_loader, desc="Processing images")):
            imgs = data['img'].to('cuda')
            
            # Encode images
            img_features = model.encode_image(imgs)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            # Compute dot products (cosine similarity with normalized features)
            concept_scores = img_features @ concept_features.T
            all_concept_scores.append(concept_scores.cpu().numpy())
            
            # Clear GPU cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
    
    # Concatenate all scores
    concept_scores_matrix = np.concatenate(all_concept_scores, axis=0)
    print(f"✅ Concept scores matrix shape: {concept_scores_matrix.shape}")
    print(f"✅ Score range: [{concept_scores_matrix.min():.3f}, {concept_scores_matrix.max():.3f}]")
    
    print("\n=== Step 5: Save Results ===")
    os.makedirs("reports/data", exist_ok=True)
    output_path = "reports/data/mimic_concept_scores.h5"
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('concept_scores', data=concept_scores_matrix, compression='gzip')
        f.create_dataset('concepts', data=[c.encode('utf-8') for c in concepts])
        
        # Save original indices for alignment (index-based pairing)
        # These correspond to positions in mimic_paths.csv and mimic.h5
        original_indices = np.arange(concept_scores_matrix.shape[0])
        f.create_dataset('original_indices', data=original_indices)
        print(f"   💾 Saved original indices for index-based alignment")
        
        # Try to save study IDs if available (for reference only, not alignment)
        try:
            with h5py.File("data/mimic.h5", 'r') as mimic_f:
                if 'study_id' in mimic_f:
                    study_ids = mimic_f['study_id'][:]
                    f.create_dataset('study_ids', data=study_ids)
                    print(f"   💾 Saved study IDs for reference")
                else:
                    print(f"   ℹ️  No study_ids in original MIMIC file (index-based alignment used)")
        except Exception as e:
            print(f"   ℹ️  Using index-based alignment: {e}")
        
        # Save metadata
        f.attrs['num_images'] = len(mimic_dataset)
        f.attrs['num_concepts'] = len(concepts)
        f.attrs['image_resolution'] = 448
        f.attrs['normalization'] = 'L2 normalized (cosine similarity)'
        f.attrs['score_range_min'] = float(concept_scores_matrix.min())
        f.attrs['score_range_max'] = float(concept_scores_matrix.max())
    
    print(f"✅ Concept scores saved to: {output_path}")
    
    # Also save concepts list as text for reference
    concepts_list_path = "reports/data/mimic_concepts_list.txt"
    with open(concepts_list_path, 'w') as f:
        for i, concept in enumerate(concepts):
            f.write(f"{i:2d}: {concept}\n")
    print(f"✅ Concepts list saved to: {concepts_list_path}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   📊 Images processed: {len(mimic_dataset)}")
    print(f"   🧠 Concepts used: {len(concepts)}")
    print(f"   📈 Matrix shape: {concept_scores_matrix.shape}")
    print(f"   💾 Saved to: {output_path}")
    
    return concept_scores_matrix, concepts

if __name__ == "__main__":
    print("🚀 Starting MIMIC Concept Scores Computation")
    
    try:
        concept_scores, concepts = compute_mimic_concept_scores()
        print("\n✅ MIMIC concept scores computation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise 