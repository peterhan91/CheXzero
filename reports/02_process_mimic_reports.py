#!/usr/bin/env python3
"""
Process MIMIC-CXR reports according to paper requirements:
- Extract FINDINGS and IMPRESSION sections only
- Remove redundant whitespaces and line breaks
- Filter out cases without IMPRESSION
- Pair with CXR images and concept scores
"""

import os
import pandas as pd
import numpy as np
import h5py
import re
from tqdm import tqdm
import json

def clean_text(text):
    """Remove redundant whitespaces and line breaks"""
    if pd.isna(text) or text.strip() == "":
        return ""
    
    # Remove extra whitespaces and line breaks
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might interfere
    text = re.sub(r'[^\w\s\.,;:\-\(\)]', ' ', text)
    # Remove extra spaces again
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

# Compile regex patterns for prior exam references (optimized for performance)
_PRIOR_PATTERNS = [
    r"compared\s+to\s+(the\s+)?(prior|previous)\s+(examination|exam|study|film|radiograph|imaging)",
    r"since\s+(the\s+)?(prior|previous)\s+(exam|study|imaging)",
    r"(previous|prior)\s+(exam|study|film|radiograph|imaging)",
    r"compared\s+with\s+(a\s+)?(previous|prior)\s+(study|exam)",
    r"on\s+(the\s+)?(prior|previous)\s+(study|exam)"
]
_PRIOR_REGEX = re.compile("|".join(_PRIOR_PATTERNS), re.IGNORECASE)

def has_prior_references(text):
    """
    Check if report contains references to previous studies/scans
    Following Nature paper requirement to remove training examples with prior references
    Uses optimized compiled regex for better performance
    """
    if pd.isna(text) or text.strip() == "":
        return False
    
    # Use compiled regex for efficient pattern matching
    return bool(_PRIOR_REGEX.search(text))

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

def process_mimic_reports():
    """Process MIMIC-CXR reports and create training dataset"""
    
    print("=== Processing MIMIC-CXR Reports ===")
    
    print("\n=== Step 1: Load MIMIC Data Using Index-Based Pairing ===")
    # Load the image paths which are linked by index to the H5 file
    paths_csv = "data/mimic_paths.csv"
    reports_base_path = "/home/than/physionet.org/files/mimic-cxr-jpg/2.0.0/reports/files"
    
    if not os.path.exists(paths_csv):
        raise FileNotFoundError(f"MIMIC paths CSV not found: {paths_csv}")
    
    if not os.path.exists(reports_base_path):
        raise FileNotFoundError(f"MIMIC reports directory not found: {reports_base_path}")
    
    # Load image paths
    paths_df = pd.read_csv(paths_csv)
    print(f"✅ Loaded MIMIC image paths: {len(paths_df)} rows from {paths_csv}")
    
    print("\n=== Step 2: Extract Study Info and Read Original Reports ===")
    
    def extract_info_from_path(image_path):
        """Extract patient_id and study_id from image path structure"""
        # Path format: .../p15/p15342059/s54865292/filename.jpg
        path_parts = image_path.split('/')
        patient_id = None
        study_id = None
        
        # Find the longest patient ID (full patient ID, not just the group)
        for part in path_parts:
            if part.startswith('p') and len(part) > 1 and part[1:].isdigit():
                if patient_id is None or len(part) > len(f"p{patient_id}"):
                    patient_id = part[1:]  # Remove 'p' prefix
            elif part.startswith('s') and len(part) > 1 and part[1:].isdigit():
                study_id = part[1:]  # Remove 's' prefix
                
        return patient_id, study_id
    
    def get_report_path(patient_id, study_id):
        """Construct report file path from patient_id and study_id"""
        if not patient_id or not study_id:
            return None
            
        patient_id_padded = str(patient_id).zfill(8)  # Pad to 8 digits
        patient_group = f"p{patient_id_padded[:2]}"  # First 2 digits for group
        patient_dir = f"p{patient_id_padded}"
        report_filename = f"s{study_id}.txt"
        
        return os.path.join(reports_base_path, patient_group, patient_dir, report_filename)
    
    def read_report_file(filepath):
        """Read and return content of report file"""
        if not filepath:
            return ""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except (FileNotFoundError, UnicodeDecodeError, PermissionError) as e:
            return ""
    
    # Process all image paths and read corresponding reports
    print("   Reading original report files for FINDINGS + IMPRESSION extraction...")
    combined_data = []
    missing_reports = 0
    
    for idx, path in enumerate(tqdm(paths_df['Path'], desc="Reading reports")):
        patient_id, study_id = extract_info_from_path(path)
        report_path = get_report_path(patient_id, study_id)
        report_text = read_report_file(report_path)
        
        if not report_text.strip():
            missing_reports += 1
        
        combined_data.append({
            'index': idx,
            'image_path': path,
            'patient_id': patient_id,
            'study_id': study_id,
            'report_path': report_path,
            'report_text': report_text
        })
    
    combined_df = pd.DataFrame(combined_data)
    print(f"✅ Read original reports: {len(combined_df) - missing_reports}/{len(combined_df)} found")
    print(f"   Missing reports: {missing_reports}")
    
    print("\n=== Step 3: Process Reports Following Nature Paper Requirements ===")
    
    processed_reports = []
    valid_count = 0
    filtered_no_impression = 0
    filtered_prior_refs = 0
    filtered_missing_text = 0
    
    for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Processing reports"):
        report_text = row['report_text'] if not pd.isna(row['report_text']) else ""
        
        # Filter 0: Skip if no report text available
        if not report_text.strip():
            filtered_missing_text += 1
            continue
        
        # Extract both FINDINGS and IMPRESSION sections from full report
        findings, impression = extract_sections(report_text)
        
        # Filter 1: Keep only reports with IMPRESSION section (Nature paper requirement)
        if impression.strip() == "" or impression == "NO IMPRESSION":
            filtered_no_impression += 1
            continue
        
        # Filter 2: Remove reports with references to previous studies (Nature paper requirement)
        if has_prior_references(impression) or has_prior_references(findings):
            filtered_prior_refs += 1
            continue
        
        # Combine FINDINGS and IMPRESSION following Nature paper format
        combined_text = ""
        if findings.strip():
            combined_text += f"FINDINGS: {findings}\n\n"
        combined_text += f"IMPRESSION: {impression}"
        
        processed_reports.append({
            'index': row['index'],
            'image_path': row['image_path'],
            'study_id': row['study_id'],
            'findings': findings,
            'impression': impression,
            'combined_report': combined_text,
            'original_length': len(report_text),
            'processed_length': len(combined_text)
        })
        
        valid_count += 1
    
    print(f"✅ Processed {valid_count} valid reports (following Nature paper preprocessing)")
    print(f"   Filtered out: {filtered_missing_text} reports without text")
    print(f"   Filtered out: {filtered_no_impression} reports without IMPRESSION")
    print(f"   Filtered out: {filtered_prior_refs} reports with prior study references")
    print(f"   Total filtered: {len(combined_df) - valid_count} of {len(combined_df)} reports")
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_reports)
    
    print("\n=== Step 4: Load Concept Scores ===")
    concept_scores_path = "reports/data/mimic_concept_scores.h5"
    
    if not os.path.exists(concept_scores_path):
        print(f"❌ Concept scores file not found: {concept_scores_path}")
        print("   Please run 01_compute_concept_scores.py first")
        return None
    
    with h5py.File(concept_scores_path, 'r') as f:
        concept_scores = f['concept_scores'][:]
        concepts = [c.decode('utf-8') for c in f['concepts'][:]]
        
        print(f"✅ Loaded concept scores: {concept_scores.shape}")
        print(f"   Concepts: {len(concepts)}")
    
    # Align reports with concept scores using index-based pairing
    # The concept scores are computed for the original H5 file in the same order as mimic_paths.csv
    # Key insight: index i in mimic_paths.csv = index i in mimic.h5 = index i in concept_scores
    aligned_data = []
    missing_concept_scores = 0
    
    for idx, row in processed_df.iterrows():
        original_index = row['index']  # This is the index from the original mimic_paths.csv
        
        if original_index < len(concept_scores):
            # Use the concept scores from the corresponding index
            scores = concept_scores[original_index]
            row_data = row.to_dict()
            
            # Add concept scores
            for i, concept in enumerate(concepts):
                row_data[f'concept_{i:02d}'] = scores[i]
            
            aligned_data.append(row_data)
        else:
            missing_concept_scores += 1
            print(f"   Warning: Index {original_index} out of range for concept scores")
    
    print(f"✅ Aligned {len(aligned_data)} reports with concept scores using index-based pairing")
    print(f"   Missing concept scores: {missing_concept_scores}")
    
    if len(aligned_data) == 0:
        raise ValueError("No reports could be aligned with concept scores!")
    
    # Create final DataFrame
    final_df = pd.DataFrame(aligned_data)
    
    print("\n=== Step 5: Save Processed Dataset ===")
    
    # Ensure output directory exists
    os.makedirs("reports/data", exist_ok=True)
    
    # Save as CSV
    output_csv = "reports/data/mimic_processed_reports.csv"
    final_df.to_csv(output_csv, index=False)
    print(f"✅ Saved processed reports to: {output_csv}")
    
    # Save as H5 for efficient loading during training
    output_h5 = "reports/data/mimic_train_dataset.h5"
    with h5py.File(output_h5, 'w') as f:
        # Save reports
        f.create_dataset('reports', data=[r.encode('utf-8') for r in final_df['combined_report']])
        f.create_dataset('findings', data=[r.encode('utf-8') for r in final_df['findings']])
        f.create_dataset('impressions', data=[r.encode('utf-8') for r in final_df['impression']])
        
        # Save concept scores
        concept_cols = [col for col in final_df.columns if col.startswith('concept_')]
        if concept_cols:
            concept_scores_aligned = final_df[concept_cols].values
            f.create_dataset('concept_scores', data=concept_scores_aligned, compression='gzip')
            f.create_dataset('concept_names', data=[c.encode('utf-8') for c in concepts])
            
            # Save original indices for alignment (same as concept scores)
            f.create_dataset('original_indices', data=final_df['index'].values)
            # Save study IDs for reference only
            f.create_dataset('study_ids', data=[str(sid).encode('utf-8') for sid in final_df['study_id']])
        
        # Save metadata
        f.attrs['num_samples'] = len(final_df)
        f.attrs['num_concepts'] = len(concepts) if concepts else 0
        f.attrs['has_concept_scores'] = len(concept_cols) > 0
        f.attrs['alignment_method'] = 'index_based_pairing'
        f.attrs['filtered_no_impression'] = filtered_no_impression
        f.attrs['filtered_prior_refs'] = filtered_prior_refs
        f.attrs['preprocessing_method'] = 'nature_paper_compliant'
        f.attrs['description'] = 'MIMIC-CXR processed training dataset with IMPRESSION, filtered for prior study references (Nature paper), and index-based aligned concept scores'
    
    print(f"✅ Saved training dataset to: {output_h5}")
    
    # Statistics
    print(f"\n🎯 DATASET STATISTICS:")
    print(f"   📊 Total samples: {len(final_df)}")
    print(f"   📝 Avg report length: {final_df['processed_length'].mean():.1f} chars")
    print(f"   📈 Report length range: {final_df['processed_length'].min()}-{final_df['processed_length'].max()}")
    print(f"   🧠 Concept features: {len(concepts) if concepts else 0}")
    print(f"   ✅ Data alignment: Index-based pairing (original codebase approach)")
    print(f"   🔍 Filtering applied:")
    print(f"     • No IMPRESSION: {filtered_no_impression} reports")
    print(f"     • Prior references: {filtered_prior_refs} reports")
    print(f"     • Retention rate: {len(final_df)/len(combined_df)*100:.1f}%")
    
    # Show some examples
    print(f"\n📋 SAMPLE REPORTS:")
    for i in range(min(3, len(final_df))):
        report = final_df.iloc[i]['combined_report']
        print(f"   {i+1}. {report[:150]}...")
    
    return final_df

if __name__ == "__main__":
    print("🚀 Starting MIMIC Report Processing")
    
    try:
        dataset = process_mimic_reports()
        if dataset is not None and len(dataset) > 0:
            print("\n✅ MIMIC report processing completed successfully!")
            print(f"   Final dataset size: {len(dataset)} samples")
        else:
            print("\n❌ Processing failed - check input files and alignment")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise 