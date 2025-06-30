#!/usr/bin/env python3
"""
Visual Question Answering (VQA) for NEJM Chest X-ray Cases
Using BiomedCLIP for direct image and text encoding

IMPORTANT: This script ensures NO ANSWER LEAKAGE by only using:
- Clinical question/context for embedding
- Multiple choice options for comparison
- Ground truth answers ONLY for final evaluation
"""

import os
import sys
import json
import re
import hashlib
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import time
import pickle
from tqdm import tqdm
import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor

# BiomedCLIP imports
from open_clip import create_model_from_pretrained, get_tokenizer


def preprocess_image_biomedclip(img, target_size=448):
    """
    Preprocess image for BiomedCLIP with custom resolution (448x448 input size)
    """
    old_size = img.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = img.resize(new_size, Image.LANCZOS)
    
    # Create a new RGB image and paste the resized image on it
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))  # Black padding
    new_img.paste(img, ((target_size - new_size[0]) // 2, (target_size - new_size[1]) // 2))
    return new_img


def create_custom_transform_448():
    """Create custom transform for 448x448 resolution"""
    # BiomedCLIP normalization values (from the model)
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    transform = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
    return transform


def get_cache_path(cache_type: str, identifier: str) -> str:
    """Generate cache file path"""
    cache_dir = "multimodal_nejm/cache_biomedclip"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the identifier for filename safety
    hash_obj = hashlib.md5(identifier.encode())
    filename = f"{cache_type}_{hash_obj.hexdigest()}.pkl"
    return os.path.join(cache_dir, filename)


def save_cache(cache_type: str, identifier: str, data: Any):
    """Save data to cache"""
    cache_path = get_cache_path(cache_type, identifier)
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Cached {cache_type}: {cache_path}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")


def load_cache(cache_type: str, identifier: str) -> Any:
    """Load data from cache"""
    cache_path = get_cache_path(cache_type, identifier)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            print(f"📂 Loaded from cache: {cache_path}")
            return data
        except Exception as e:
            print(f"⚠️ Failed to load cache: {e}")
    return None


def validate_no_answer_leakage(clinical_context: str, answer: str, choices: List[str]) -> bool:
    """
    Validate that the clinical context doesn't contain answer information
    Returns True if no leakage detected, False otherwise
    """
    context_lower = clinical_context.lower().strip()
    answer_lower = answer.lower().strip()
    
    # Check if the exact answer appears in the context
    if answer_lower in context_lower:
        return False
    
    # Check if any choice appears directly in the context (except as medical terms)
    for choice in choices:
        choice_lower = choice.lower().strip()
        # Allow common medical terms but flag if choice appears as conclusion
        if choice_lower in context_lower and len(choice_lower) > 10:  # Only check longer, specific terms
            return False
    
    return True


def parse_nejm_case(content_text: str) -> Dict[str, Any]:
    """
    Parse NEJM case content to extract clinical context and options
    ENSURES NO ANSWER LEAKAGE by only extracting question and choices
    """
    case_data = {}
    
    # Extract ID
    id_match = re.search(r'ID:\s*(\S+)', content_text)
    case_data['id'] = id_match.group(1) if id_match else ""
    
    # Extract question (clinical context) - CRITICAL: Only the question, no answer info
    question_match = re.search(r'Question:\s*(.+?)(?=Choices:|$)', content_text, re.DOTALL)
    if question_match:
        case_data['clinical_context'] = question_match.group(1).strip()
    else:
        case_data['clinical_context'] = ""
    
    # Extract choices - these are the options to compare against
    choices_section = re.search(r'Choices:\s*(.+?)(?=Answer:|$)', content_text, re.DOTALL)
    if choices_section:
        choices_text = choices_section.group(1).strip()
        choices = []
        for line in choices_text.split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                # Remove the number prefix
                choice = re.sub(r'^\d+\.\s*', '', line)
                choices.append(choice)
        case_data['choices'] = choices
    else:
        case_data['choices'] = []
    
    # Extract answer (ONLY for evaluation - not used in prediction)
    answer_match = re.search(r'Answer:\s*(.+?)(?=Reason:|$)', content_text, re.DOTALL)
    if answer_match:
        case_data['answer'] = answer_match.group(1).strip()
    else:
        case_data['answer'] = ""
    
    return case_data


def load_nejm_cases(cases_dir: str) -> List[Dict[str, Any]]:
    """Load all NEJM cases from the directory with validation"""
    cases = []
    
    case_dirs = [d for d in os.listdir(cases_dir) if d.startswith('IC') and os.path.isdir(os.path.join(cases_dir, d))]
    case_dirs.sort()
    
    print(f"📂 Found {len(case_dirs)} NEJM case directories")
    
    # Add progress bar for case loading
    leakage_detected = 0
    for case_dir in tqdm(case_dirs, desc="🔍 Loading and validating cases"):
        case_path = os.path.join(cases_dir, case_dir)
        
        # Handle both content.txt and context.txt naming conventions
        content_file = os.path.join(case_path, 'content.txt')
        context_file = os.path.join(case_path, 'context.txt')
        image_file = os.path.join(case_path, 'image.png')
        
        # Check for either content.txt or context.txt
        if os.path.exists(content_file):
            text_file = content_file
        elif os.path.exists(context_file):
            text_file = context_file
        else:
            text_file = None
        
        if text_file and os.path.exists(image_file):
            with open(text_file, 'r') as f:
                content_text = f.read()
            
            case_data = parse_nejm_case(content_text)
            case_data['case_dir'] = case_dir
            case_data['image_path'] = image_file
            
            # Only include cases with valid questions and choices
            if case_data['clinical_context'] and len(case_data['choices']) > 0:
                # CRITICAL: Validate no answer leakage
                if validate_no_answer_leakage(case_data['clinical_context'], 
                                            case_data['answer'], 
                                            case_data['choices']):
                    cases.append(case_data)
                else:
                    leakage_detected += 1
                    print(f"⚠️ LEAKAGE DETECTED in {case_dir}: answer info found in clinical context")
            else:
                print(f"⚠️ Skipping case {case_dir}: missing context or choices")
    
    print(f"✅ Loaded {len(cases)} valid NEJM cases")
    if leakage_detected > 0:
        print(f"⚠️ WARNING: {leakage_detected} cases had potential answer leakage")
    
    return cases


def load_biomedclip_model():
    """Load the BiomedCLIP model"""
    print("🤖 Loading BiomedCLIP Model...")
    
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()
    
    print(f"✅ BiomedCLIP model loaded successfully on {device}")
    return model, preprocess, tokenizer, device


def encode_images_biomedclip(model, preprocess, image_paths: List[str], device, use_448=True) -> torch.Tensor:
    """Encode chest X-ray images using BiomedCLIP with caching"""
    resolution = "448x448" if use_448 else "224x224"
    print(f"🖼️ Encoding {len(image_paths)} chest X-ray images with BiomedCLIP at {resolution}...")
    
    # Create cache identifier based on image paths and resolution
    paths_str = "|".join(sorted(image_paths))
    cache_id = f"biomedclip_image_features_{len(image_paths)}_paths_{resolution}"
    
    # Try to load from cache
    cached_features = load_cache("biomedclip_image_features", cache_id)
    if cached_features is not None and cached_features.shape[0] == len(image_paths):
        print(f"✅ Using cached BiomedCLIP image features ({resolution}): {cached_features.shape}")
        return cached_features
    
    # Create custom transform for 448x448 or use default
    if use_448:
        custom_transform = create_custom_transform_448()
        target_size = 448
        print(f"🔧 Using custom 448x448 preprocessing")
    else:
        custom_transform = None
        target_size = 224
        print(f"🔧 Using default 224x224 preprocessing")
    
    all_img_features = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="📷 Encoding images", unit="image"):
            try:
                # Load and preprocess image
                image = Image.open(img_path).convert('RGB')
                
                if use_448:
                    # Use custom 448x448 preprocessing
                    preprocessed_image = preprocess_image_biomedclip(image, target_size=target_size)
                    img_tensor = custom_transform(preprocessed_image).unsqueeze(0).to(device)
                else:
                    # Use BiomedCLIP's default preprocess function (224x224)
                    img_tensor = preprocess(image).unsqueeze(0).to(device)
                
                # Encode image
                img_features = model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                all_img_features.append(img_features.cpu())
                
            except Exception as e:
                print(f"❌ Error encoding image {img_path}: {e}")
                # Use zero features for failed images
                feature_dim = 512  # BiomedCLIP image feature dimension
                all_img_features.append(torch.zeros(1, feature_dim))
    
    img_features = torch.cat(all_img_features)
    print(f"✅ BiomedCLIP image features encoded ({resolution}): {img_features.shape}")
    
    # Cache the results
    save_cache("biomedclip_image_features", cache_id, img_features)
    
    return img_features


def encode_texts_biomedclip(model, tokenizer, texts: List[str], device, context_length=256) -> torch.Tensor:
    """Encode texts using BiomedCLIP with caching"""
    print(f"📝 Encoding {len(texts)} texts with BiomedCLIP...")
    
    # Create cache identifier based on texts
    texts_str = "|".join(texts)
    cache_id = f"biomedclip_text_features_{len(texts)}_texts"
    
    # Try to load from cache
    cached_features = load_cache("biomedclip_text_features", cache_id)
    if cached_features is not None and cached_features.shape[0] == len(texts):
        print(f"✅ Using cached BiomedCLIP text features: {cached_features.shape}")
        return cached_features
    
    all_text_features = []
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="📝 Encoding text batches", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            
            try:
                # Tokenize texts
                text_tokens = tokenizer(batch_texts, context_length=context_length).to(device)
                
                # Encode texts
                text_features = model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                all_text_features.append(text_features.cpu())
                
            except Exception as e:
                print(f"❌ Error encoding text batch: {e}")
                # Use zero features for failed texts
                feature_dim = 512  # BiomedCLIP text feature dimension
                all_text_features.append(torch.zeros(len(batch_texts), feature_dim))
    
    text_features = torch.cat(all_text_features)
    print(f"✅ BiomedCLIP text features encoded: {text_features.shape}")
    
    # Cache the results
    save_cache("biomedclip_text_features", cache_id, text_features)
    
    return text_features


def predict_answers_multimodal_biomedclip(img_features: torch.Tensor, 
                                         context_features: torch.Tensor,
                                         choice_features: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers by combining image + context features and comparing with choices"""
    print(f"🎯 Making multimodal predictions for {len(img_features)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(img_features)), desc="🔮 Multimodal predictions", unit="case"):
        # Get image feature for this case
        img_feat = img_features[i:i+1]  # [1, 512]
        
        # Get context feature for this case
        context_feat = context_features[i:i+1]  # [1, 512]
        
        # Combine image and context features (simple addition + normalize)
        combined_feat = img_feat + context_feat  # [1, 512]
        combined_feat /= combined_feat.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_feat in choice_features[i]:  # Each choice feature [1, 512]
            similarity = torch.cosine_similarity(combined_feat, choice_feat, dim=1)
            choice_scores.append(similarity.item())
        
        # Predict the choice with highest similarity
        predicted_choice = np.argmax(choice_scores)
        predictions.append(predicted_choice)
    
    print(f"✅ Multimodal predictions completed")
    return np.array(predictions)


def predict_answers_image_only_biomedclip(img_features: torch.Tensor,
                                         choice_features: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers using only image features"""
    print(f"🖼️ Making image-only predictions for {len(img_features)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(img_features)), desc="📷 Image-only predictions", unit="case"):
        # Get image feature for this case
        img_feat = img_features[i:i+1]  # [1, 512]
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_feat in choice_features[i]:  # Each choice feature [1, 512]
            similarity = torch.cosine_similarity(img_feat, choice_feat, dim=1)
            choice_scores.append(similarity.item())
        
        # Predict the choice with highest similarity
        predicted_choice = np.argmax(choice_scores)
        predictions.append(predicted_choice)
    
    print(f"✅ Image-only predictions completed")
    return np.array(predictions)


def predict_answers_context_only_biomedclip(context_features: torch.Tensor,
                                           choice_features: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers using only clinical context features"""
    print(f"📝 Making context-only predictions for {len(context_features)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(context_features)), desc="📄 Context-only predictions", unit="case"):
        # Get context feature for this case
        context_feat = context_features[i:i+1]  # [1, 512]
        context_feat /= context_feat.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_feat in choice_features[i]:  # Each choice feature [1, 512]
            similarity = torch.cosine_similarity(context_feat, choice_feat, dim=1)
            choice_scores.append(similarity.item())
        
        # Predict the choice with highest similarity
        predicted_choice = np.argmax(choice_scores)
        predictions.append(predicted_choice)
    
    print(f"✅ Context-only predictions completed")
    return np.array(predictions)


def evaluate_predictions(predictions: np.ndarray, cases: List[Dict[str, Any]], 
                        method_name: str) -> Tuple[float, List[Dict[str, Any]]]:
    """Evaluate predictions and return accuracy and detailed results"""
    correct_predictions = 0
    detailed_results = []
    
    for i, case in enumerate(cases):
        predicted_choice = predictions[i]
        predicted_answer = case['choices'][predicted_choice]
        correct_answer = case['answer']
        
        is_correct = predicted_answer.lower().strip() == correct_answer.lower().strip()
        if is_correct:
            correct_predictions += 1
        
        result = {
            'case_id': case['case_dir'],
            'predicted_option': predicted_answer,
            'ground_truth_option': correct_answer,
            'correct': is_correct,
            'method': method_name
        }
        detailed_results.append(result)
        
        status = '✅' if is_correct else '❌'
        print(f"{status} {case['case_dir']}: '{predicted_answer}' vs '{correct_answer}'")
    
    accuracy = (correct_predictions / len(cases)) * 100
    return accuracy, detailed_results


def run_nejm_vqa_biomedclip():
    """Run VQA evaluation on NEJM chest X-ray cases using BiomedCLIP"""
    print("🚀 Starting NEJM VQA Evaluation with BiomedCLIP")
    print("=" * 80)
    print("🔒 ANSWER LEAKAGE PREVENTION: Only using clinical questions and choices")
    print("🎯 Ground truth answers used ONLY for final evaluation")
    print("🏥 Using BiomedCLIP for direct image and text encoding")
    print("=" * 80)
    
    # Load NEJM cases
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    cases = load_nejm_cases(cases_dir)
    
    if len(cases) == 0:
        print("❌ No valid cases found!")
        return 0.0
    
    print(f"\n📊 Processing {len(cases)} valid NEJM cases")
    
    # Load BiomedCLIP model
    model, preprocess, tokenizer, device = load_biomedclip_model()
    
    # Encode images with 224x224 resolution (BiomedCLIP native)
    image_paths = [case['image_path'] for case in cases]
    img_features = encode_images_biomedclip(model, preprocess, image_paths, device, use_448=False)
    
    # Encode clinical contexts (NO ANSWER LEAKAGE - only questions)
    print("\n📝 Encoding clinical contexts...")
    clinical_contexts = [case['clinical_context'] for case in cases]
    print(f"Sample context: '{clinical_contexts[0][:100]}...'")
    
    context_features = encode_texts_biomedclip(model, tokenizer, clinical_contexts, device)
    print(f"✅ Context features: {context_features.shape}")
    
    # Encode choice options
    print("\n🔤 Encoding choice options...")
    
    all_choice_features = []
    all_choices_flat = []
    
    # First, collect all choices for batch encoding
    for case in cases:
        all_choices_flat.extend(case['choices'])
    
    # Encode all choices in batches
    choice_features_flat = encode_texts_biomedclip(model, tokenizer, all_choices_flat, device)
    
    # Reconstruct the nested structure
    choice_idx = 0
    for case in cases:
        case_choice_features = []
        for _ in case['choices']:
            choice_feature = choice_features_flat[choice_idx:choice_idx+1]  # [1, 512]
            case_choice_features.append(choice_feature)
            choice_idx += 1
        all_choice_features.append(case_choice_features)
    
    print("✅ All choice features encoded")
    
    # Run all three prediction scenarios
    print("\n" + "=" * 80)
    print("🧪 RUNNING ABLATION STUDIES - Comparing Different Modalities")
    print("=" * 80)
    
    all_results = []
    all_accuracies = {}
    
    # 1. Multimodal (Image + Context)
    print("\n🔬 1. MULTIMODAL (Image + Clinical Context)")
    print("-" * 50)
    multimodal_predictions = predict_answers_multimodal_biomedclip(img_features, context_features, all_choice_features)
    multimodal_accuracy, multimodal_results = evaluate_predictions(multimodal_predictions, cases, "biomedclip_multimodal")
    all_results.extend(multimodal_results)
    all_accuracies["multimodal"] = multimodal_accuracy
    
    # 2. Image Only
    print("\n🔬 2. IMAGE ONLY (CXR Only)")
    print("-" * 50)
    image_only_predictions = predict_answers_image_only_biomedclip(img_features, all_choice_features)
    image_only_accuracy, image_only_results = evaluate_predictions(image_only_predictions, cases, "biomedclip_image_only")
    all_results.extend(image_only_results)
    all_accuracies["image_only"] = image_only_accuracy
    
    # 3. Context Only
    print("\n🔬 3. CONTEXT ONLY (Clinical Context Only)")
    print("-" * 50)
    context_only_predictions = predict_answers_context_only_biomedclip(context_features, all_choice_features)
    context_only_accuracy, context_only_results = evaluate_predictions(context_only_predictions, cases, "biomedclip_context_only")
    all_results.extend(context_only_results)
    all_accuracies["context_only"] = context_only_accuracy
    
    # Compare results
    print("\n" + "=" * 80)
    print("📊 BIOMEDCLIP COMPARATIVE RESULTS:")
    print("=" * 80)
    print(f"Total cases processed: {len(cases)}")
    print()
    
    # Sort methods by accuracy for better display
    sorted_methods = sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, accuracy) in enumerate(sorted_methods):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
        method_display = {
            "multimodal": "Multimodal (Image + Context)",
            "image_only": "Image Only (CXR)",
            "context_only": "Context Only (Text)"
        }[method]
        
        correct = int(accuracy * len(cases) / 100)
        incorrect = len(cases) - correct
        
        print(f"{rank_emoji} {method_display}:")
        print(f"   Accuracy: {accuracy:.2f}% ({correct}/{len(cases)} correct)")
        print(f"   Incorrect: {incorrect}")
        print()
    
    # Calculate improvements/differences
    best_accuracy = max(all_accuracies.values())
    multimodal_acc = all_accuracies["multimodal"]
    image_acc = all_accuracies["image_only"]
    context_acc = all_accuracies["context_only"]
    
    print("🔍 ANALYSIS:")
    if multimodal_acc > max(image_acc, context_acc):
        improvement = multimodal_acc - max(image_acc, context_acc)
        better_than = "image-only" if image_acc > context_acc else "context-only"
        print(f"✅ Multimodal approach improves by {improvement:.2f}% over best single modality ({better_than})")
    else:
        print("⚠️ Single modality outperforms multimodal approach")
    
    image_vs_context = image_acc - context_acc
    if image_vs_context > 0:
        print(f"🖼️ Images are {image_vs_context:.2f}% more informative than context alone")
    else:
        print(f"📝 Context is {abs(image_vs_context):.2f}% more informative than images alone")
    
    print("=" * 80)
    
    # Save results to results/ directory
    os.makedirs("multimodal_nejm/results", exist_ok=True)
    
    # Save comprehensive results
    results_df = pd.DataFrame(all_results)
    results_file = "multimodal_nejm/results/nejm_vqa_biomedclip_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"📁 Comprehensive results saved to: {results_file}")
    
    # Save accuracy summary
    summary_data = []
    for method, accuracy in all_accuracies.items():
        correct = int(accuracy * len(cases) / 100)
        summary_data.append({
            'method': f'biomedclip_{method}',
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_cases': len(cases)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = "multimodal_nejm/results/nejm_vqa_biomedclip_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📁 Accuracy summary saved to: {summary_file}")
    
    return multimodal_acc


if __name__ == "__main__":
    accuracy = run_nejm_vqa_biomedclip()
    print(f"\n🎯 Final BiomedCLIP Accuracy: {accuracy:.2f}%") 