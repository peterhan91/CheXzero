#!/usr/bin/env python3
"""
Visual Question Answering (VQA) for NEJM Chest X-ray Cases
Using concept-based approach with CXR + Text embeddings

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
import shutil
import cv2

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import clip
from concepts.get_embed import RadiologyEmbeddingGenerator


def preprocess_image_h5_style(img, desired_size=448, threshold=200):
    """
    Improved preprocessing: Remove white borders + H5-style grayscale conversion
    Combines border removal with the critical grayscale conversion needed for consistency
    """
    # Step 1: Convert to RGB and remove white borders
    img = img.convert("RGB")
    np_img = np.array(img)

    # Compute mask of non-white pixels
    mask = np.any(np_img < threshold, axis=-1)  # Anything not almost white

    if not np.any(mask):
        print("Warning: No content found (image is mostly white).")
        # Fallback to simple resize if all white
        img_resized = img.resize((desired_size, desired_size), Image.LANCZOS)
    else:
        # Remove white borders by cropping to content
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1  # slicing is non-inclusive
        cropped_img = img.crop((x0, y0, x1, y1))

        # Center-crop to square
        width, height = cropped_img.size
        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        cropped_img = cropped_img.crop((left, top, left + min_side, top + min_side))

        # Resize to target size
        img_resized = cropped_img.resize((desired_size, desired_size), Image.LANCZOS)
    
    # Step 2: CRITICAL - Convert to grayscale like H5 pipeline
    # This step is essential for consistency with the reference preprocessing
    img_gray = img_resized.convert('L')  # Convert RGB to grayscale
    
    return img_gray

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

def get_cache_path(cache_type: str, identifier: str) -> str:
    """Generate cache file path"""
    cache_dir = "multimodal_nejm/cache"
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


class LLMEmbeddingGenerator:
    """LLM Embedding Generator using sfr-mistral for text embeddings"""
    
    def __init__(self):
        self._local_generators = {}
        
    def get_embeddings_batch(self, texts: List[str], model_name: str = "sfr_mistral") -> np.ndarray:
        """Get embeddings for a batch of texts using specified model"""
        if model_name not in self._local_generators:
            print(f"🔄 Loading {model_name} model...")
            torch.cuda.empty_cache()
            
            if model_name == "sfr_mistral":
                generator = RadiologyEmbeddingGenerator(
                    embedding_type="local",
                    local_model_name='Salesforce/SFR-Embedding-Mistral',
                    batch_size=16
                )
            else:
                raise ValueError(f"Unknown model: {model_name}")
            
            self._local_generators[model_name] = generator
            print(f"✅ {model_name} model loaded and cached")
        
        generator = self._local_generators[model_name]
        embeddings = generator.get_local_embeddings_batch(texts)
        return np.array(embeddings)
    
    def cleanup(self):
        """Clean up cached models to free GPU memory"""
        model_names = list(self._local_generators.keys())  # Create a copy of keys
        for model_name in model_names:
            print(f"🧹 Cleaning up {model_name} model...")
            del self._local_generators[model_name]
        self._local_generators.clear()
        torch.cuda.empty_cache()


def load_concept_embeddings(model_name: str = "sfr_mistral") -> Tuple[Dict[int, np.ndarray], int]:
    """Load concept embeddings for the specified model"""
    embeddings_file = f"/home/than/DeepLearning/cxr_concept/CheXzero/concepts/embeddings/concepts_embeddings_{model_name}.pickle"
    
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Concept embeddings file not found: {embeddings_file}")
    
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    # Get embedding dimension from first embedding
    embedding_dim = len(list(embeddings_data.values())[0])
    return embeddings_data, embedding_dim


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
    
    # Extract reason/explanation
    reason_match = re.search(r'Reason:\s*(.+?)$', content_text, re.DOTALL)
    if reason_match:
        case_data['reason'] = reason_match.group(1).strip()
    else:
        case_data['reason'] = ""
    
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


def load_clip_model():
    """Load the CLIP model"""
    print("🤖 Loading CLIP Model...")
    model = load_clip(
        model_path="checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("✅ CLIP model loaded successfully")
    return model


def get_concept_features(model, concepts: List[str]) -> torch.Tensor:
    """Get CLIP text features for diagnostic concepts with caching"""
    print(f"📝 Encoding {len(concepts)} diagnostic concepts...")
    
    # Create cache identifier based on concepts
    concepts_str = "|".join(concepts)
    cache_id = f"concept_features_{len(concepts)}_concepts"
    
    # Try to load from cache
    cached_features = load_cache("concept_features", cache_id)
    if cached_features is not None and cached_features.shape[0] == len(concepts):
        print(f"✅ Using cached concept features: {cached_features.shape}")
        return cached_features.to('cuda')
    
    concept_batch_size = 512
    all_concept_features = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(concepts), concept_batch_size), 
                     desc="🧠 Encoding concept batches", 
                     unit="batch"):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()
    
    concept_features = torch.cat(all_concept_features)
    print(f"✅ Concept features encoded: {concept_features.shape}")
    
    # Cache the results
    save_cache("concept_features", cache_id, concept_features)
    
    return concept_features.to('cuda')


def encode_images(model, image_paths: List[str]) -> torch.Tensor:
    """Encode chest X-ray images using CLIP with H5-style preprocessing"""
    print(f"🖼️ Encoding {len(image_paths)} chest X-ray images (H5 style)...")
    
    # Create cache identifier based on image paths (use different cache for H5 style)
    paths_str = "|".join(sorted(image_paths))
    cache_id = f"image_features_h5_{len(image_paths)}_paths"
    
    # Try to load from cache
    cached_features = load_cache("image_features", cache_id)
    if cached_features is not None and cached_features.shape[0] == len(image_paths):
        print(f"✅ Using cached H5-style image features: {cached_features.shape}")
        return cached_features
    
    target_size = 448
    
    # Use EXACT transform as CXRTestDataset
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    
    all_img_features = []
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="📷 Encoding images (H5 style)", unit="image"):
            try:
                # Step 1: Load with CV2 (like h5 pipeline)
                img_cv2 = cv2.imread(img_path)
                img_cv2_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_cv2_rgb)
                
                # Step 2: Preprocess to grayscale (like h5 pipeline)
                img_gray = preprocess_image_h5_style(img_pil, target_size)
                
                # Step 3: Convert grayscale to RGB tensor (like CXRTestDataset)
                img_tensor = convert_grayscale_to_rgb_tensor(img_gray)
                
                # Step 4: Apply transforms and move to GPU
                img_tensor = transform(img_tensor).unsqueeze(0).to('cuda')
                
                # Step 5: Encode
                img_features = model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                all_img_features.append(img_features.cpu())
                
            except Exception as e:
                print(f"❌ Error encoding image {img_path}: {e}")
                # Use zero features for failed images
                try:
                    dummy_tensor = torch.zeros(1, 3, target_size, target_size).to('cuda')
                    dummy_features = model.encode_image(dummy_tensor)
                    feature_dim = dummy_features.shape[1]
                except:
                    feature_dim = 768  # Default fallback
                all_img_features.append(torch.zeros(1, feature_dim))
    
    img_features = torch.cat(all_img_features)
    print(f"✅ H5-style image features encoded: {img_features.shape}")
    
    # Cache the results
    save_cache("image_features", cache_id, img_features)
    
    return img_features


def compute_concept_similarities(img_features: torch.Tensor, concept_features: torch.Tensor) -> torch.Tensor:
    """Compute similarities between images and diagnostic concepts with caching"""
    print(f"🔄 Computing concept similarities for {len(img_features)} images...")
    
    # Create cache identifier based on input shapes
    cache_id = f"concept_similarities_{img_features.shape}_{concept_features.shape}"
    
    # Try to load from cache
    cached_similarities = load_cache("concept_similarities", cache_id)
    if cached_similarities is not None and cached_similarities.shape[0] == len(img_features):
        print(f"✅ Using cached concept similarities: {cached_similarities.shape}")
        return cached_similarities
    
    img_batch_size = 25  # Smaller batch size for stability
    all_similarities = []
    
    for i in tqdm(range(0, len(img_features), img_batch_size), 
                 desc="⚡ Computing similarities", 
                 unit="batch"):
        batch_img_features = img_features[i:i+img_batch_size].to('cuda')
        batch_similarity = batch_img_features @ concept_features.T
        all_similarities.append(batch_similarity.cpu())
        torch.cuda.empty_cache()
    
    concept_similarity = torch.cat(all_similarities)
    print(f"✅ Concept similarities computed: {concept_similarity.shape}")
    
    # Cache the results
    save_cache("concept_similarities", cache_id, concept_similarity)
    
    return concept_similarity


def get_llm_representations(concept_similarity: torch.Tensor, concept_embeddings: torch.Tensor) -> torch.Tensor:
    """Transform concept similarities to LLM embedding space with caching"""
    print("🔀 Transforming to LLM embedding space...")
    
    # Create cache identifier based on input shapes
    cache_id = f"llm_representations_{concept_similarity.shape}_{concept_embeddings.shape}"
    
    # Try to load from cache
    cached_representations = load_cache("llm_representations", cache_id)
    if cached_representations is not None and cached_representations.shape[0] == len(concept_similarity):
        print(f"✅ Using cached LLM representations: {cached_representations.shape}")
        return cached_representations.to('cuda')
    
    concept_embeddings = concept_embeddings.to('cuda')
    concept_similarity = concept_similarity.to('cuda')
    
    # Matrix multiplication: [#images, #concepts] @ [#concepts, embedding_dim] -> [#images, embedding_dim]
    print(f"   Matrix multiplication: {concept_similarity.shape} @ {concept_embeddings.shape}")
    llm_representation = concept_similarity @ concept_embeddings
    llm_representation /= llm_representation.norm(dim=-1, keepdim=True)
    
    print(f"✅ LLM representations: {llm_representation.shape}")
    
    # Cache the results
    save_cache("llm_representations", cache_id, llm_representation.cpu())
    
    return llm_representation


def predict_answers_multimodal(cxr_representations: torch.Tensor, 
                              context_embeddings: torch.Tensor,
                              choice_embeddings: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers by combining CXR + context and comparing with choices"""
    print(f"🎯 Making multimodal predictions for {len(cxr_representations)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(cxr_representations)), desc="🔮 Multimodal predictions", unit="case"):
        # Get CXR representation for this image
        cxr_repr = cxr_representations[i:i+1]  # [1, 4096]
        
        # Get context embedding for this case
        context_repr = context_embeddings[i:i+1]  # [1, 4096]
        
        # Combine CXR and context representations
        combined_repr = cxr_repr + context_repr  # [1, 4096]
        combined_repr /= combined_repr.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_emb in choice_embeddings[i]:  # Each choice embedding [1, 4096]
            choice_emb = choice_emb.to(combined_repr.device)
            similarity = torch.cosine_similarity(combined_repr, choice_emb, dim=1)
            choice_scores.append(similarity.item())
        
        # Predict the choice with highest similarity
        predicted_choice = np.argmax(choice_scores)
        predictions.append(predicted_choice)
    
    print(f"✅ Multimodal predictions completed")
    return np.array(predictions)


def predict_answers_image_only(cxr_representations: torch.Tensor,
                              choice_embeddings: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers using only CXR representations"""
    print(f"🖼️ Making image-only predictions for {len(cxr_representations)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(cxr_representations)), desc="📷 Image-only predictions", unit="case"):
        # Get CXR representation for this image
        cxr_repr = cxr_representations[i:i+1]  # [1, 4096]
        cxr_repr /= cxr_repr.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_emb in choice_embeddings[i]:  # Each choice embedding [1, 4096]
            choice_emb = choice_emb.to(cxr_repr.device)
            similarity = torch.cosine_similarity(cxr_repr, choice_emb, dim=1)
            choice_scores.append(similarity.item())
        
        # Predict the choice with highest similarity
        predicted_choice = np.argmax(choice_scores)
        predictions.append(predicted_choice)
    
    print(f"✅ Image-only predictions completed")
    return np.array(predictions)


def predict_answers_context_only(context_embeddings: torch.Tensor,
                                choice_embeddings: List[List[torch.Tensor]]) -> np.ndarray:
    """Predict answers using only clinical context embeddings"""
    print(f"📝 Making context-only predictions for {len(context_embeddings)} cases...")
    
    predictions = []
    
    for i in tqdm(range(len(context_embeddings)), desc="📄 Context-only predictions", unit="case"):
        # Get context embedding for this case
        context_repr = context_embeddings[i:i+1]  # [1, 4096]
        context_repr /= context_repr.norm(dim=-1, keepdim=True)
        
        # Compute similarities with all choices for this case
        choice_scores = []
        for choice_emb in choice_embeddings[i]:  # Each choice embedding [1, 4096]
            choice_emb = choice_emb.to(context_repr.device)
            similarity = torch.cosine_similarity(context_repr, choice_emb, dim=1)
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


def save_correct_multimodal_cases(cases: List[Dict[str, Any]], 
                                 multimodal_results: List[Dict[str, Any]],
                                 concept_similarity: torch.Tensor,
                                 concepts: List[str]) -> None:
    """
    Save data for correctly answered multimodal cases including:
    - Input image
    - Context, choices, answers
    - Concept similarity scores (dot products)
    """
    # Create main directory for correct cases
    correct_cases_dir = "multimodal_nejm/results/correct_multimodal_cases"
    os.makedirs(correct_cases_dir, exist_ok=True)
    
    # Filter for correct multimodal predictions
    correct_indices = []
    for i, result in enumerate(multimodal_results):
        if result['method'] == 'multimodal' and result['correct']:
            correct_indices.append(i)
    
    print(f"\n💾 Saving data for {len(correct_indices)} correctly answered multimodal cases...")
    
    # Convert concept_similarity to CPU for saving
    concept_similarity_cpu = concept_similarity.cpu().numpy()
    
    for idx in tqdm(correct_indices, desc="💿 Saving correct cases", unit="case"):
        case = cases[idx]
        case_id = case['case_dir']
        
        # Create subdirectory for this case
        case_dir = os.path.join(correct_cases_dir, case_id)
        os.makedirs(case_dir, exist_ok=True)
        
        try:
            # 1. Copy input image
            original_image_path = case['image_path']
            target_image_path = os.path.join(case_dir, 'image.png')
            shutil.copy2(original_image_path, target_image_path)
            
            # 2. Save context, choices, answers to text file
            case_info = {
                'case_id': case_id,
                'clinical_context': case['clinical_context'],
                'choices': case['choices'],
                'ground_truth_answer': case['answer'],
                'predicted_answer': multimodal_results[idx]['predicted_option'],
                'reason': case.get('reason', '')
            }
            
            case_info_path = os.path.join(case_dir, 'case_info.json')
            with open(case_info_path, 'w') as f:
                json.dump(case_info, f, indent=2)
            
            # Also save as readable text file
            case_text_path = os.path.join(case_dir, 'case_info.txt')
            with open(case_text_path, 'w') as f:
                f.write(f"Case ID: {case_id}\n")
                f.write("=" * 50 + "\n\n")
                f.write("CLINICAL CONTEXT:\n")
                f.write(case['clinical_context'] + "\n\n")
                f.write("CHOICES:\n")
                for i, choice in enumerate(case['choices']):
                    f.write(f"{i+1}. {choice}\n")
                f.write(f"\nGROUND TRUTH ANSWER: {case['answer']}\n")
                f.write(f"PREDICTED ANSWER: {multimodal_results[idx]['predicted_option']}\n")
                if case.get('reason'):
                    f.write(f"\nREASON/EXPLANATION:\n{case['reason']}\n")
            
            # 3. Save concept similarity scores (dot products) - use correct image index
            concept_scores = concept_similarity_cpu[idx]  # Shape: [num_concepts] - idx is the original case position
            
            # Save as numpy array for easy loading
            concept_scores_path = os.path.join(case_dir, 'concept_similarities.npy')
            np.save(concept_scores_path, concept_scores)
            
            # Create DataFrame and filter for positive similarities only
            concept_similarity_df = pd.DataFrame({
                'concept': concepts,
                'similarity_score': concept_scores
            })
            
            # Filter for positive similarities only
            positive_concepts_df = concept_similarity_df[concept_similarity_df['similarity_score'] > 0].copy()
            
            # Sort by similarity score (descending)
            positive_concepts_df = positive_concepts_df.sort_values('similarity_score', ascending=False)
            
            # Save all positive concepts
            concept_similarity_csv_path = os.path.join(case_dir, 'concept_similarities_positive.csv')
            positive_concepts_df.to_csv(concept_similarity_csv_path, index=False)
            
            # Get top 50 positive concepts
            top_50_positive = positive_concepts_df.head(50)
            
            # Save top 50 positive concepts for quick inspection (text format)
            top_concepts_path = os.path.join(case_dir, 'top_50_concepts.txt')
            with open(top_concepts_path, 'w') as f:
                f.write(f"Top 50 most similar POSITIVE concepts for case {case_id}:\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total positive concepts: {len(positive_concepts_df)}\n")
                f.write("=" * 50 + "\n\n")
                for i, (_, row) in enumerate(top_50_positive.iterrows()):
                    f.write(f"{i+1:2d}. {row['concept']:50s} {row['similarity_score']:.6f}\n")
            
            # Save top 50 positive concepts in CSV format for easier analysis
            top_50_df = top_50_positive.copy()
            top_50_df.reset_index(drop=True, inplace=True)
            top_50_df['rank'] = range(1, min(51, len(top_50_df)+1))
            top_50_df = top_50_df[['rank', 'concept', 'similarity_score']]  # Reorder columns
            top_50_csv_path = os.path.join(case_dir, 'top_50_concepts.csv')
            top_50_df.to_csv(top_50_csv_path, index=False)
            
        except Exception as e:
            print(f"❌ Error saving case {case_id}: {e}")
            continue
    
    print(f"✅ Successfully saved data for {len(correct_indices)} correct multimodal cases")
    print(f"📁 Data saved to: {correct_cases_dir}")
    
    # Save summary of all correct cases
    summary_path = os.path.join(correct_cases_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Summary of Correctly Answered Multimodal Cases\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total correct cases: {len(correct_indices)}\n")
        f.write(f"Total concepts: {len(concepts)}\n\n")
        f.write("Case IDs:\n")
        for idx in correct_indices:
            f.write(f"- {cases[idx]['case_dir']}\n")


def run_nejm_vqa():
    """Run VQA evaluation on NEJM chest X-ray cases"""
    print("🚀 Starting NEJM VQA Evaluation")
    print("=" * 80)
    print("🔒 ANSWER LEAKAGE PREVENTION: Only using clinical questions and choices")
    print("🎯 Ground truth answers used ONLY for final evaluation")
    print("=" * 80)
    
    # Load NEJM cases
    cases_dir = "/home/than/DeepLearning/cxr_concept/CheXzero/multimodal_nejm/cxr_cases/"
    cases = load_nejm_cases(cases_dir)
    
    if len(cases) == 0:
        print("❌ No valid cases found!")
        return 0.0
    
    print(f"\n📊 Processing {len(cases)} valid NEJM cases")
    
    # Load CLIP model
    clip_model = load_clip_model()
    
    # Load diagnostic concepts
    print("\n📖 Loading diagnostic concepts...")
    concepts_file = "/home/than/DeepLearning/cxr_concept/CheXzero/concepts/mimic_concepts.csv"
    if not os.path.exists(concepts_file):
        print(f"❌ Concepts file not found: {concepts_file}")
        return 0.0
        
    concepts_df = pd.read_csv(concepts_file)
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()
    print(f"✅ Loaded {len(concepts)} diagnostic concepts")
    
    # Load concept embeddings for sfr-mistral
    print("\n🧠 Loading SFR-Mistral concept embeddings...")
    try:
        embeddings_data, embedding_dim = load_concept_embeddings("sfr_mistral")
        print(f"✅ Embedding dimension: {embedding_dim}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return 0.0
    
    # Process concept embeddings with proper alignment
    concept_embeddings = np.zeros((len(concepts), embedding_dim))
    missing_count = 0
    
    print("🔗 Aligning concept embeddings...")
    for pos, concept_idx in enumerate(tqdm(concept_indices, desc="Aligning embeddings")):
        if concept_idx in embeddings_data:
            concept_embeddings[pos] = embeddings_data[concept_idx]
        else:
            concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01
            missing_count += 1
    
    if missing_count > 0:
        print(f"⚠️ Warning: {missing_count} concepts missing embeddings, used random vectors")
    
    concept_embeddings = torch.tensor(concept_embeddings).float()
    print(f"✅ Concept embeddings ready: {concept_embeddings.shape}")
    
    # Get concept features from CLIP
    concept_features = get_concept_features(clip_model, concepts)
    
    # Encode images
    image_paths = [case['image_path'] for case in cases]
    img_features = encode_images(clip_model, image_paths)
    
    # Compute concept similarities
    concept_similarity = compute_concept_similarities(img_features, concept_features)
    
    # Get LLM representations for images
    cxr_representations = get_llm_representations(concept_similarity, concept_embeddings)
    
    # Initialize LLM embedding generator
    print("\n🤖 Initializing SFR-Mistral for text embeddings...")
    embedding_generator = LLMEmbeddingGenerator()
    
    # Encode clinical contexts (NO ANSWER LEAKAGE - only questions)
    print("\n📝 Encoding clinical contexts...")
    clinical_contexts = [case['clinical_context'] for case in cases]
    print(f"Sample context: '{clinical_contexts[0][:100]}...'")
    
    # Create cache identifier for clinical contexts
    contexts_str = "|".join(clinical_contexts)
    context_cache_id = f"clinical_contexts_{len(clinical_contexts)}_sfr_mistral"
    
    # Try to load from cache
    cached_context_embeddings = load_cache("context_embeddings", context_cache_id)
    if cached_context_embeddings is not None and len(cached_context_embeddings) == len(clinical_contexts):
        print(f"✅ Using cached context embeddings: {cached_context_embeddings.shape}")
        context_embeddings = torch.tensor(cached_context_embeddings).float().to('cuda')
        context_embeddings /= context_embeddings.norm(dim=-1, keepdim=True)
    else:
        context_embeddings_np = embedding_generator.get_embeddings_batch(clinical_contexts, "sfr_mistral")
        context_embeddings = torch.tensor(context_embeddings_np).float().to('cuda')
        context_embeddings /= context_embeddings.norm(dim=-1, keepdim=True)
        
        # Cache the results
        save_cache("context_embeddings", context_cache_id, context_embeddings_np)
    
    print(f"✅ Context embeddings: {context_embeddings.shape}")
    
    # Encode choice options
    print("\n🔤 Encoding choice options...")
    
    # Create cache identifier for all choice options
    all_choices = []
    for case in cases:
        all_choices.extend(case['choices'])
    choices_str = "|".join(all_choices)
    choices_cache_id = f"choice_embeddings_{len(all_choices)}_sfr_mistral"
    
    # Try to load from cache
    cached_choice_embeddings = load_cache("choice_embeddings", choices_cache_id)
    if cached_choice_embeddings is not None and len(cached_choice_embeddings) == len(all_choices):
        print(f"✅ Using cached choice embeddings for {len(all_choices)} choices")
        
        # Reconstruct the nested structure from cached flat embeddings
        all_choice_embeddings = []
        choice_idx = 0
        for case in cases:
            case_choice_embeddings = []
            for _ in case['choices']:
                choice_embedding = torch.tensor(cached_choice_embeddings[choice_idx]).float().unsqueeze(0)
                choice_embedding /= choice_embedding.norm(dim=-1, keepdim=True)
                case_choice_embeddings.append(choice_embedding)
                choice_idx += 1
            all_choice_embeddings.append(case_choice_embeddings)
    else:
        all_choice_embeddings = []
        choice_embeddings_to_cache = []
        
        total_choices = sum(len(case['choices']) for case in cases)
        with tqdm(total=total_choices, desc="Encoding choices", unit="choice") as pbar:
            for case in cases:
                case_choice_embeddings = []
                for choice in case['choices']:
                    choice_embedding_np = embedding_generator.get_embeddings_batch([choice], "sfr_mistral")
                    choice_embedding = torch.tensor(choice_embedding_np).float()
                    choice_embedding /= choice_embedding.norm(dim=-1, keepdim=True)
                    case_choice_embeddings.append(choice_embedding)
                    choice_embeddings_to_cache.append(choice_embedding_np[0])  # Store flat for caching
                    pbar.update(1)
                all_choice_embeddings.append(case_choice_embeddings)
        
        # Cache the results
        save_cache("choice_embeddings", choices_cache_id, choice_embeddings_to_cache)
    
    print("✅ All choice embeddings encoded")
    
    # Run all three prediction scenarios
    print("\n" + "=" * 80)
    print("🧪 RUNNING ABLATION STUDIES - Comparing Different Modalities")
    print("=" * 80)
    
    all_results = []
    all_accuracies = {}
    
    # 1. Multimodal (CXR + Context)
    print("\n🔬 1. MULTIMODAL (CXR + Clinical Context)")
    print("-" * 50)
    multimodal_predictions = predict_answers_multimodal(cxr_representations, context_embeddings, all_choice_embeddings)
    multimodal_accuracy, multimodal_results = evaluate_predictions(multimodal_predictions, cases, "multimodal")
    all_results.extend(multimodal_results)
    all_accuracies["multimodal"] = multimodal_accuracy
    
    # Save data for correctly answered multimodal cases
    save_correct_multimodal_cases(cases, multimodal_results, concept_similarity, concepts)
    
    # 2. Image Only
    print("\n🔬 2. IMAGE ONLY (CXR Only)")
    print("-" * 50)
    image_only_predictions = predict_answers_image_only(cxr_representations, all_choice_embeddings)
    image_only_accuracy, image_only_results = evaluate_predictions(image_only_predictions, cases, "image_only")
    all_results.extend(image_only_results)
    all_accuracies["image_only"] = image_only_accuracy
    
    # 3. Context Only
    print("\n🔬 3. CONTEXT ONLY (Clinical Context Only)")
    print("-" * 50)
    context_only_predictions = predict_answers_context_only(context_embeddings, all_choice_embeddings)
    context_only_accuracy, context_only_results = evaluate_predictions(context_only_predictions, cases, "context_only")
    all_results.extend(context_only_results)
    all_accuracies["context_only"] = context_only_accuracy
    
    # Compare results
    print("\n" + "=" * 80)
    print("📊 COMPARATIVE RESULTS:")
    print("=" * 80)
    print(f"Total cases processed: {len(cases)}")
    print()
    
    # Sort methods by accuracy for better display
    sorted_methods = sorted(all_accuracies.items(), key=lambda x: x[1], reverse=True)
    
    for i, (method, accuracy) in enumerate(sorted_methods):
        rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
        method_display = {
            "multimodal": "Multimodal (CXR + Context)",
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
    results_file = "multimodal_nejm/results/nejm_vqa_comparative_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"📁 Comprehensive results saved to: {results_file}")
    
    # Save accuracy summary
    summary_data = []
    for method, accuracy in all_accuracies.items():
        correct = int(accuracy * len(cases) / 100)
        summary_data.append({
            'method': method,
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_cases': len(cases)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = "multimodal_nejm/results/nejm_vqa_accuracy_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"📁 Accuracy summary saved to: {summary_file}")
    
    # Cleanup
    embedding_generator.cleanup()
    
    return multimodal_acc


if __name__ == "__main__":
    accuracy = run_nejm_vqa()
    print(f"\n🎯 Final Accuracy: {accuracy:.2f}%")
