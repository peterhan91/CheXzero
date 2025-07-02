#!/usr/bin/env python3
"""
Prepare CheXpert validation set:
- Sample 500 CXR-report pairs from CheXpert training set
- Compute concept scores for the sampled CXRs
- Process reports similar to MIMIC (FINDINGS + IMPRESSION)
- Ensure proper pairing between images, reports, and concept scores
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import h5py
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from tqdm import tqdm
import json
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import zero_shot
import clip

def load_concepts():
    """Load the 68 concepts from cbm_concepts.json"""
    concepts_path = "concepts/cbm_concepts.json"
    with open(concepts_path, 'r') as f:
        concepts_data = json.load(f)
    
    all_concepts = []
    for category, concepts_list in concepts_data.items():
        for concept_info in concepts_list:
            all_concepts.append(concept_info['concept'])
    
    return all_concepts

def clean_text(text):
    """Clean report text"""
    if pd.isna(text) or text.strip() == "":
        return ""
    
    import re
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'[^\w\s\.,;:\-\(\)]', ' ', text)
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def getIndexOfLast(lst, item):
    """Helper function to get last occurrence index (from original codebase)"""
    for i in range(len(lst) - 1, -1, -1):
        if lst[i] == item:
            return i
    return -1

def extract_sections(report_text):
    """Extract FINDINGS and IMPRESSION sections following original codebase approach"""
    if pd.isna(report_text) or report_text.strip() == "":
        return "", ""
    
    # Split text by whitespace (following original approach)
    s_split = report_text.split()
    
    findings = ""
    impression = ""
    
    # Extract FINDINGS section (similar to original IMPRESSION logic)
    if "FINDINGS:" in s_split:
        begin = getIndexOfLast(s_split, "FINDINGS:") + 1
        # Find stopping points
        end_candidates = []
        
        for stop_word in ["IMPRESSION:", "CONCLUSION:", "RECOMMENDATION:", "RECOMMENDATIONS:", "RECOMMENDATION(S):", "NOTIFICATION:", "NOTIFICATIONS:"]:
            if stop_word in s_split:
                idx = s_split.index(stop_word)
                if idx > begin:  # Only consider if it comes after FINDINGS
                    end_candidates.append(idx)
        
        # Use earliest stopping point
        end = min(end_candidates) if end_candidates else None
        
        if end is None:
            findings = " ".join(s_split[begin:])
        else:
            findings = " ".join(s_split[begin:end])
    
    # Extract IMPRESSION section (following original codebase exactly)
    if "IMPRESSION:" in s_split:
        begin = getIndexOfLast(s_split, "IMPRESSION:") + 1
        end = None
        end_cand1 = None
        end_cand2 = None
        
        # Remove recommendation(s) and notification (exact original logic)
        if "RECOMMENDATION(S):" in s_split:
            end_cand1 = s_split.index("RECOMMENDATION(S):")
        elif "RECOMMENDATION:" in s_split:
            end_cand1 = s_split.index("RECOMMENDATION:")
        elif "RECOMMENDATIONS:" in s_split:
            end_cand1 = s_split.index("RECOMMENDATIONS:")

        if "NOTIFICATION:" in s_split:
            end_cand2 = s_split.index("NOTIFICATION:")
        elif "NOTIFICATIONS:" in s_split:
            end_cand2 = s_split.index("NOTIFICATIONS:")

        if end_cand1 and end_cand2:
            end = min(end_cand1, end_cand2)
        elif end_cand1:
            end = end_cand1
        elif end_cand2:
            end = end_cand2

        if end is None:
            impression = " ".join(s_split[begin:])
        else:
            impression = " ".join(s_split[begin:end])
    else:
        impression = 'NO IMPRESSION'
    
    # Clean the extracted sections but preserve original whitespace handling
    findings = findings.strip() if findings else ""
    impression = impression.strip() if impression else ""
    
    return findings, impression

def sample_chexpert_validation():
    """Sample 500 CXR-report pairs from CheXpert for validation"""
    
    print("=== Preparing CheXpert Validation Set ===")
    
    print("\n=== Step 1: Load CheXpert Reports ===")
    reports_path = "data/chexpert_train.csv"
    reports_df = pd.read_csv(reports_path)
    print(f"✅ Loaded {len(reports_df)} CheXpert reports")
    print(f"   Columns: {list(reports_df.columns)}")
    
    # Find text column
    text_columns = ['Report Impression', 'Impression', 'report', 'text']
    text_col = None
    for col in text_columns:
        if col in reports_df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {list(reports_df.columns)}")
    
    print(f"   Using text column: {text_col}")
    
    print("\n=== Step 2: Filter Reports with IMPRESSION ===")
    valid_reports = []
    
    for idx, row in tqdm(reports_df.iterrows(), desc="Filtering reports"):
        report_text = row[text_col] if not pd.isna(row[text_col]) else ""
        findings, impression = extract_sections(report_text)
        
        # Keep only reports with IMPRESSION
        if impression.strip():
            valid_reports.append({
                'original_index': idx,
                'path': row.get('Path', ''),
                'findings': findings,
                'impression': impression,
                'original_text': report_text
            })
    
    print(f"✅ Found {len(valid_reports)} reports with IMPRESSION")
    
    print("\n=== Step 3: Sample 500 Cases ===")
    random.seed(42)  # For reproducibility
    sample_size = min(500, len(valid_reports))
    sampled_reports = random.sample(valid_reports, sample_size)
    
    print(f"✅ Sampled {sample_size} cases for validation")
    
    # Get the original indices for image extraction
    sampled_indices = [r['original_index'] for r in sampled_reports]
    
    print("\n=== Step 4: Load CLIP Model ===")
    model = load_clip(
        model_path="checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ CLIP model loaded")
    
    print("\n=== Step 5: Load and Encode Concepts ===")
    concepts = load_concepts()
    
    # Encode concepts
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
    
    print("\n=== Step 6: Load CheXpert Images and Compute Concept Scores ===")
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    # Load full CheXpert dataset
    chexpert_dataset = zero_shot.CXRTestDataset(
        img_path="data/chexpert.h5",
        transform=transform,
    )
    
    print(f"✅ CheXpert dataset loaded: {len(chexpert_dataset)} images")
    
    # Extract sampled images and compute concept scores
    sampled_concept_scores = []
    
    with torch.no_grad():
        for idx in tqdm(sampled_indices, desc="Computing concept scores"):
            if idx < len(chexpert_dataset):
                # Get the specific image
                data = chexpert_dataset[idx]
                img = data['img'].unsqueeze(0).to('cuda')  # Add batch dimension
                
                # Encode image
                img_features = model.encode_image(img)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                
                # Compute concept scores
                concept_scores = img_features @ concept_features.T
                sampled_concept_scores.append(concept_scores.cpu().numpy()[0])  # Remove batch dim
            else:
                # Fallback: use zero scores
                sampled_concept_scores.append(np.zeros(len(concepts)))
    
    concept_scores_matrix = np.array(sampled_concept_scores)
    print(f"✅ Computed concept scores: {concept_scores_matrix.shape}")
    
    print("\n=== Step 7: Create Validation Dataset ===")
    validation_data = []
    
    for i, report_data in enumerate(sampled_reports):
        # Combine findings and impression
        findings = report_data['findings']
        impression = report_data['impression']
        
        if findings.strip():
            combined_text = f"FINDINGS: {findings} IMPRESSION: {impression}"
        else:
            combined_text = f"IMPRESSION: {impression}"
        
        validation_data.append({
            'sample_id': i,
            'original_index': report_data['original_index'],
            'path': report_data['path'],
            'findings': findings,
            'impression': impression,
            'combined_report': combined_text,
            'concept_scores': concept_scores_matrix[i]
        })
    
    print("\n=== Step 8: Save Validation Dataset ===")
    
    # Ensure output directory exists
    os.makedirs("reports/data", exist_ok=True)
    
    # Save as H5 file
    output_h5 = "reports/data/chexpert_validation_dataset.h5"
    with h5py.File(output_h5, 'w') as f:
        # Save reports
        f.create_dataset('reports', data=[d['combined_report'].encode('utf-8') for d in validation_data])
        f.create_dataset('findings', data=[d['findings'].encode('utf-8') for d in validation_data])
        f.create_dataset('impressions', data=[d['impression'].encode('utf-8') for d in validation_data])
        
        # Save concept scores
        f.create_dataset('concept_scores', data=concept_scores_matrix, compression='gzip')
        f.create_dataset('concept_names', data=[c.encode('utf-8') for c in concepts])
        
        # Save indices for image retrieval
        f.create_dataset('original_indices', data=[d['original_index'] for d in validation_data])
        
        # Save metadata
        f.attrs['num_samples'] = len(validation_data)
        f.attrs['num_concepts'] = len(concepts)
        f.attrs['image_source'] = 'chexpert.h5'
        f.attrs['image_resolution'] = 448
        f.attrs['description'] = 'CheXpert validation dataset (500 samples) with concept scores'
        f.attrs['sampling_seed'] = 42
    
    print(f"✅ Saved validation dataset to: {output_h5}")
    
    # Save as CSV for inspection
    validation_df = pd.DataFrame([{
        'sample_id': d['sample_id'],
        'original_index': d['original_index'],
        'path': d['path'],
        'combined_report': d['combined_report'],
        'report_length': len(d['combined_report'])
    } for d in validation_data])
    
    output_csv = "reports/data/chexpert_validation_dataset.csv"
    validation_df.to_csv(output_csv, index=False)
    print(f"✅ Saved validation metadata to: {output_csv}")
    
    print(f"\n🎯 VALIDATION DATASET STATISTICS:")
    print(f"   📊 Total samples: {len(validation_data)}")
    print(f"   📝 Avg report length: {validation_df['report_length'].mean():.1f} chars")
    print(f"   📈 Report length range: {validation_df['report_length'].min()}-{validation_df['report_length'].max()}")
    print(f"   🧠 Concept features: {len(concepts)}")
    print(f"   📈 Concept scores range: [{concept_scores_matrix.min():.3f}, {concept_scores_matrix.max():.3f}]")
    
    # Show some examples
    print(f"\n📋 SAMPLE VALIDATION REPORTS:")
    for i in range(min(3, len(validation_data))):
        report = validation_data[i]['combined_report']
        print(f"   {i+1}. {report[:150]}...")
    
    return validation_data, concepts

if __name__ == "__main__":
    print("🚀 Starting CheXpert Validation Set Preparation")
    
    try:
        validation_data, concepts = sample_chexpert_validation()
        print("\n✅ CheXpert validation set preparation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise 