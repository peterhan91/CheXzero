#!/usr/bin/env python3
"""
Embedding Model Selection Experiment on CheXpert Validation Set

This script evaluates different LLM embedding models on the CheXpert validation set
to select the best embedding model for CLEAR. The selection is performed on validation
data to avoid data leakage to the test sets.

Usage:
    python exp_embedding_model_selection.py

Output:
    - results/embedding_model_selection/summary.csv
    - results/embedding_model_selection/detailed_results.json
"""

import os
import sys
import datetime
import json
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from scipy import stats

import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from sklearn.metrics import roc_auc_score, roc_curve

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import zero_shot
import clip
from eval import evaluate
from get_embed import RadiologyEmbeddingGenerator


class LLMEmbeddingGenerator:
    """Generator for LLM embeddings supporting multiple models."""

    def __init__(self):
        self._local_generators = {}

    def get_embeddings_batch(self, texts: List[str], model_name: str) -> np.ndarray:
        if model_name in ["sfr_mistral", "qwen3_8b", "biomedbert"]:
            if model_name not in self._local_generators:
                print(f"Loading {model_name} model...")
                torch.cuda.empty_cache()

                if model_name == "sfr_mistral":
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='Salesforce/SFR-Embedding-Mistral',
                        batch_size=16
                    )
                elif model_name == "qwen3_8b":
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='Qwen/Qwen3-Embedding-8B',
                        batch_size=4
                    )
                elif model_name == "biomedbert":
                    generator = RadiologyEmbeddingGenerator(
                        embedding_type="local",
                        local_model_name='microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract',
                        batch_size=16
                    )

                self._local_generators[model_name] = generator

            generator = self._local_generators[model_name]
            embeddings = generator.get_local_embeddings_batch(texts)
            return np.array(embeddings)
        elif model_name == "openai_small":
            # OpenAI API (standard, not Azure)
            from openai import OpenAI
            import time

            # Use environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            client = OpenAI(api_key=api_key)

            embeddings = []
            for text in tqdm(texts, desc="Getting OpenAI embeddings"):
                cleaned_text = text.strip().replace("\\n", " ").replace("\n", " ")
                try:
                    response = client.embeddings.create(
                        input=[cleaned_text],
                        model="text-embedding-3-small"
                    )
                    embeddings.append(np.array(response.data[0].embedding))
                    time.sleep(0.05)  # Rate limiting
                except Exception as e:
                    print(f"Error: {e}")
                    embeddings.append(np.zeros(1536))
            return np.array(embeddings)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def cleanup(self):
        for model_name in list(self._local_generators.keys()):
            del self._local_generators[model_name]
        self._local_generators.clear()
        torch.cuda.empty_cache()


def get_chexpert_valid_config():
    """Get CheXpert validation set configuration."""
    return {
        'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.h5",
        'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.csv",
        'labels': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                   'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                   'Lung Opacity', 'No Finding', 'Pleural Effusion',
                   'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'],
        'core_conditions': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    }


def load_concept_embeddings(model_name: str) -> Tuple[Dict[int, np.ndarray], int]:
    """Load pre-computed concept embeddings for a specific model."""
    embeddings_file = f"/home/than/DeepLearning/cxr_concept/CheXzero/concepts/embeddings/concepts_embeddings_{model_name}.pickle"

    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)

    embedding_dim = len(list(embeddings_data.values())[0])
    return embeddings_data, embedding_dim


def bootstrap_auroc(y_true, y_pred, n_bootstrap=1000, seed=42):
    """Compute AUROC with 95% CI via bootstrap."""
    np.random.seed(seed)

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    auroc = roc_auc_score(y_true, y_pred)

    bootstrap_aurocs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        if len(np.unique(y_true_boot)) < 2:
            continue

        try:
            bootstrap_aurocs.append(roc_auc_score(y_true_boot, y_pred_boot))
        except:
            continue

    if len(bootstrap_aurocs) < 100:
        return auroc, np.nan, np.nan

    ci_lower = np.percentile(bootstrap_aurocs, 2.5)
    ci_upper = np.percentile(bootstrap_aurocs, 97.5)

    return auroc, ci_lower, ci_upper


def run_evaluation_for_model(llm_model: str, config: dict, model, concepts_df,
                              test_loader, y_true, test_labels) -> Dict:
    """Run zero-shot evaluation for a single embedding model."""

    print(f"\n{'='*60}")
    print(f"Evaluating: {llm_model.upper()}")
    print('='*60)

    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()

    # Load concept embeddings
    embeddings_data, embedding_dim = load_concept_embeddings(llm_model)

    concept_embeddings = np.zeros((len(concepts), embedding_dim))
    for pos, concept_idx in enumerate(concept_indices):
        if concept_idx in embeddings_data:
            concept_embeddings[pos] = embeddings_data[concept_idx]
        else:
            concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01

    concept_embeddings = torch.tensor(concept_embeddings).float()

    # Encode concepts
    concept_batch_size = 512 if llm_model == "qwen3_8b" else 1024
    all_concept_features = []

    with torch.no_grad():
        for i in range(0, len(concepts), concept_batch_size):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77).to('cuda')
            concept_features = model.encode_text(concept_tokens)
            concept_features /= concept_features.norm(dim=-1, keepdim=True)
            all_concept_features.append(concept_features.cpu())
            torch.cuda.empty_cache()

    concept_features = torch.cat(all_concept_features).to('cuda')

    # Encode images
    all_img_features = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Encoding images"):
            imgs = data['img'].to('cuda')
            img_features = model.encode_image(imgs)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            all_img_features.append(img_features.cpu())

    img_features = torch.cat(all_img_features)

    # Compute concept similarities
    img_batch_size = 50 if llm_model == "qwen3_8b" else 100
    all_similarities = []

    for i in range(0, len(img_features), img_batch_size):
        batch_img_features = img_features[i:i+img_batch_size].to('cuda')
        batch_similarity = batch_img_features @ concept_features.T
        all_similarities.append(batch_similarity.cpu())
        torch.cuda.empty_cache()

    concept_similarity = torch.cat(all_similarities)

    # Project to LLM space
    concept_embeddings = concept_embeddings.to('cuda')
    concept_similarity = concept_similarity.to('cuda')

    llm_representation = concept_similarity @ concept_embeddings
    llm_representation /= llm_representation.norm(dim=-1, keepdim=True)

    # Generate class embeddings
    if llm_model == "qwen3_8b":
        model = model.cpu()
        torch.cuda.empty_cache()

    embedding_generator = LLMEmbeddingGenerator()

    pos_prompts = [f"{label.lower()}" for label in test_labels]
    neg_prompts = [f"no {label.lower()}" for label in test_labels]

    pos_embeddings = embedding_generator.get_embeddings_batch(pos_prompts, llm_model)
    neg_embeddings = embedding_generator.get_embeddings_batch(neg_prompts, llm_model)

    embedding_generator.cleanup()

    if llm_model == "qwen3_8b":
        model = model.to('cuda')

    pos_class_embeddings = torch.tensor(np.array(pos_embeddings)).float().to('cuda')
    neg_class_embeddings = torch.tensor(np.array(neg_embeddings)).float().to('cuda')
    pos_class_embeddings /= pos_class_embeddings.norm(dim=-1, keepdim=True)
    neg_class_embeddings /= neg_class_embeddings.norm(dim=-1, keepdim=True)

    # Compute predictions
    with torch.no_grad():
        logits_pos = llm_representation @ pos_class_embeddings.T
        logits_neg = llm_representation @ neg_class_embeddings.T

        exp_logits_pos = torch.exp(logits_pos)
        exp_logits_neg = torch.exp(logits_neg)
        probabilities = exp_logits_pos / (exp_logits_pos + exp_logits_neg)

        y_pred = probabilities.cpu().numpy()

    # Compute metrics with bootstrap CI
    results = {
        'model': llm_model,
        'per_label': {},
        'macro_auroc': None,
        'macro_auroc_ci_lower': None,
        'macro_auroc_ci_upper': None
    }

    aurocs = []
    for i, label in enumerate(test_labels):
        auroc, ci_lower, ci_upper = bootstrap_auroc(y_true[:, i], y_pred[:, i])
        results['per_label'][label] = {
            'auroc': auroc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_positive': int(np.sum(y_true[:, i]))
        }
        if not np.isnan(auroc):
            aurocs.append(auroc)

    results['macro_auroc'] = np.mean(aurocs)
    results['macro_auroc_std'] = np.std(aurocs)

    print(f"\n{llm_model.upper()} Results:")
    print(f"  Macro AUROC: {results['macro_auroc']:.4f} (+/- {results['macro_auroc_std']:.4f})")

    return results, model


def main():
    """Main function to run embedding model selection experiment."""

    print("="*70)
    print("EMBEDDING MODEL SELECTION ON CHEXPERT VALIDATION SET")
    print("="*70)

    # Configuration
    config = get_chexpert_valid_config()
    test_labels = config['labels']

    # LLM models to evaluate
    llm_models = ["sfr_mistral", "qwen3_8b", "openai_small", "biomedbert"]

    # Load CLIP model
    print("\n=== Loading CLIP Model ===")
    model = load_clip(
        model_path="../checkpoints/dinov2-multi-v1.0_vitb/best_model.pt",
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    )
    model = model.to('cuda').eval()
    print("CLIP model loaded")

    # Setup dataset
    print("\n=== Loading CheXpert Validation Set ===")
    y_true = zero_shot.make_true_labels(
        cxr_true_labels_path=config['labels_path'],
        cxr_labels=test_labels,
        cutlabels=True
    )

    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])

    test_dataset = zero_shot.CXRTestDataset(
        img_path=config['cxr_filepath'],
        transform=transform,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Validation set: {len(test_dataset)} images")
    print(f"Labels: {len(test_labels)}")

    # Load concepts
    concepts_df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/mimic_concepts.csv")
    print(f"Loaded {len(concepts_df)} concepts")

    # Run evaluation for each model
    all_results = {}

    for llm_model in llm_models:
        try:
            results, model = run_evaluation_for_model(
                llm_model, config, model, concepts_df,
                test_loader, y_true, test_labels
            )
            all_results[llm_model] = results
        except Exception as e:
            print(f"Error evaluating {llm_model}: {e}")
            continue

    # Create summary table
    print("\n" + "="*70)
    print("SUMMARY: EMBEDDING MODEL SELECTION RESULTS")
    print("="*70)

    summary_data = []
    for model_name, results in all_results.items():
        summary_data.append({
            'Model': model_name,
            'Macro AUROC': f"{results['macro_auroc']:.4f}",
            'Std': f"{results['macro_auroc_std']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Macro AUROC', ascending=False)
    print(summary_df.to_string(index=False))

    # Identify best model
    best_model = summary_df.iloc[0]['Model']
    print(f"\n>>> SELECTED MODEL: {best_model} (highest Macro AUROC on validation set)")

    # Save results
    results_dir = "results/embedding_model_selection"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary CSV
    summary_df.to_csv(f"{results_dir}/summary_{timestamp}.csv", index=False)

    # Save detailed results JSON
    with open(f"{results_dir}/detailed_results_{timestamp}.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        json.dump(convert(all_results), f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    return all_results, best_model


if __name__ == "__main__":
    all_results, best_model = main()
