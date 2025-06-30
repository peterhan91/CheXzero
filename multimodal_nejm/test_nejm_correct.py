#!/usr/bin/env python3
"""
Corrected NEJM image encoding test - following EXACT h5 preprocessing pipeline
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor
from tqdm import tqdm
from PIL import Image
import json
import cv2

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import clip

def preprocess_like_h5(img, desired_size=448):
    """Preprocess image EXACTLY like the h5 pipeline in data_process.py"""
    # Convert to RGB first (like cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.convert('RGB')
    
    # Apply same resize logic as h5 pipeline
    old_size = img.size
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create new GRAYSCALE image and paste (critical difference!)
    new_img = Image.new('L', (desired_size, desired_size))  # 'L' = grayscale!
    new_img.paste(img, ((desired_size - new_size[0]) // 2,
                        (desired_size - new_size[1]) // 2))
    
    return new_img

def convert_grayscale_to_rgb_tensor(img_pil):
    """Convert grayscale PIL image to RGB tensor like CXRTestDataset does"""
    # Convert PIL to numpy array
    img_array = np.array(img_pil)  # Shape: (448, 448)
    
    # Add channel dimension and repeat for RGB (like CXRTestDataset)
    img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 448, 448)
    img_array = np.repeat(img_array, 3, axis=0)    # Shape: (3, 448, 448)
    
    # Convert to torch tensor
    img_tensor = torch.from_numpy(img_array).float()
    
    return img_tensor

def test_nejm_correct_encoding():
    """Test NEJM encoding with EXACT h5 pipeline"""
    
    print("=== CORRECTED NEJM Image Encoding Test ===")
    print("Following EXACT h5 preprocessing pipeline")
    
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
    
    print("\n=== Step 2: Load NEJM Images ===")
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    case_dirs = [d for d in os.listdir(cases_dir) if d.startswith('IC')]
    case_dirs.sort()
    
    image_paths = []
    case_ids = []
    
    for case_dir in case_dirs:
        image_path = os.path.join(cases_dir, case_dir, 'image.png')
        if os.path.exists(image_path):
            image_paths.append(image_path)
            case_ids.append(case_dir)
    
    print(f"📂 Found {len(image_paths)} NEJM case images")
    
    print("\n=== Step 3: Load Concepts ===")
    concepts_df = pd.read_csv("concepts/mimic_concepts.csv")
    concepts = concepts_df['concept'].tolist()
    print(f"✅ Loaded {len(concepts)} concepts")
    
    print("\n=== Step 4: Encode Concepts ===")
    concept_batch_size = 1024
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), desc="Encoding concepts"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            # L2 normalize
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_features = torch.cat(all_concept_features).to('cuda')
    print(f"✅ Concept features: {concept_features.shape}")
    
    print("\n=== Step 5: Encode Images (H5 Pipeline) ===")
    # Use EXACT transform as CXRTestDataset
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    all_img_features = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Encoding images (H5 style)"):
            # Step 1: Load with CV2 (like h5 pipeline)
            img_cv2 = cv2.imread(img_path)
            img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_cv2_rgb)
            
            # Step 2: Preprocess to grayscale (like h5 pipeline)
            img_gray = preprocess_like_h5(img_pil, 448)
            
            # Step 3: Convert grayscale to RGB tensor (like CXRTestDataset)
            img_tensor = convert_grayscale_to_rgb_tensor(img_gray)
            
            # Step 4: Apply transforms and move to GPU
            img_tensor = transform(img_tensor).unsqueeze(0).to('cuda')
            
            # Step 5: Encode
            img_features = model.encode_image(img_tensor)
            # L2 normalize
            img_features /= img_features.norm(dim=-1, keepdim=True)
            all_img_features.append(img_features.cpu())
    
    img_features = torch.cat(all_img_features)
    print(f"✅ Image features: {img_features.shape}")
    
    print("\n=== Step 6: Compute Dot Products ===")
    img_features_cuda = img_features.to('cuda')
    dot_products = img_features_cuda @ concept_features.T
    dot_products = dot_products.cpu().numpy()
    
    print(f"✅ Dot products shape: {dot_products.shape}")
    print(f"✅ Range: [{dot_products.min():.6f}, {dot_products.max():.6f}]")
    
    print("\n=== Step 7: Analyze Results ===")
    
    # Check multiple test cases
    test_cases = [0, 10, 20, 30] if len(case_ids) > 30 else [0, 5, 10]
    
    print("Analyzing different cases:")
    for i in test_cases:
        if i < len(case_ids):
            case_similarities = dot_products[i]
            top_indices = np.argsort(case_similarities)[-5:][::-1]
            top_values = case_similarities[top_indices]
            
            print(f"\nCase {i} ({case_ids[i]}):")
            print(f"  Mean: {case_similarities.mean():.6f}, Std: {case_similarities.std():.6f}")
            print(f"  Top 5 concepts:")
            for j, (idx, val) in enumerate(zip(top_indices, top_values)):
                print(f"    {j+1}. {concepts[idx][:50]}: {val:.6f}")
    
    # Check differences between cases
    print(f"\nDifference analysis:")
    if len(test_cases) >= 3:
        diff_01 = np.abs(dot_products[test_cases[0]] - dot_products[test_cases[1]]).max()
        diff_02 = np.abs(dot_products[test_cases[0]] - dot_products[test_cases[2]]).max()
        print(f"  Max diff case 0 vs 1: {diff_01:.8f}")
        print(f"  Max diff case 0 vs 2: {diff_02:.8f}")
        
        # Check if top concepts are same
        top_0 = np.argsort(dot_products[test_cases[0]])[-10:][::-1]
        top_1 = np.argsort(dot_products[test_cases[1]])[-10:][::-1]
        top_2 = np.argsort(dot_products[test_cases[2]])[-10:][::-1]
        
        print(f"  Top 10 concepts case 0 vs 1 same? {np.array_equal(top_0, top_1)}")
        print(f"  Top 10 concepts case 0 vs 2 same? {np.array_equal(top_0, top_2)}")
        
        # Show actual top concept names for comparison
        print(f"\nTop 3 concepts comparison:")
        for case_idx, case_num in enumerate(test_cases[:3]):
            if case_num < len(case_ids):
                top_3 = np.argsort(dot_products[case_num])[-3:][::-1]
                print(f"  Case {case_num}: {[concepts[idx][:30] for idx in top_3]}")
    
    print("\n=== Step 8: Save Results ===")
    os.makedirs("multimodal_nejm/test_results", exist_ok=True)
    
    # Save corrected dot products
    np.save("multimodal_nejm/test_results/nejm_dot_products_corrected.npy", dot_products)
    
    # Save analysis
    analysis_results = {
        'method': 'h5_pipeline_corrected',
        'case_ids': case_ids,
        'image_paths': image_paths,
        'num_cases': len(case_ids),
        'num_concepts': len(concepts),
        'preprocessing': 'CV2 -> RGB -> Grayscale -> 3-channel repeat',
        'resolution': 448,
        'differences': {}
    }
    
    # Add difference analysis to results
    if len(test_cases) >= 3:
        analysis_results['differences'] = {
            'max_diff_case_0_vs_1': float(diff_01),
            'max_diff_case_0_vs_2': float(diff_02),
            'top_concepts_identical_0_1': bool(np.array_equal(top_0, top_1)),
            'top_concepts_identical_0_2': bool(np.array_equal(top_0, top_2))
        }
    
    with open("multimodal_nejm/test_results/analysis_corrected.json", 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    print("✅ Results saved to multimodal_nejm/test_results/")
    print(f"\n🔍 SUMMARY:")
    print(f"  📊 Method: H5 preprocessing pipeline")
    print(f"  🖼️ Images: {len(case_ids)}")
    print(f"  🧠 Concepts: {len(concepts)}")
    print(f"  📈 Dot products range: [{dot_products.min():.6f}, {dot_products.max():.6f}]")
    
    return dot_products, case_ids, concepts

if __name__ == "__main__":
    dot_products, case_ids, concepts = test_nejm_correct_encoding() 