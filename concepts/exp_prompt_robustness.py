#!/usr/bin/env python3
"""
Prompt Robustness Experiment

This script evaluates the robustness of CLEAR to different prompt formulations,
particularly focusing on negation handling. We test multiple variants of
positive/negative prompt pairs to assess stability.

Prompt Variants Tested:
- Standard: "{disease}" vs "no {disease}"
- Variant A: "{disease}" vs "absence of {disease}"
- Variant B: "{disease} present" vs "{disease} absent"
- Variant C: "{disease}" vs "{disease} not present"
- Variant D: "findings consistent with {disease}" vs "no evidence of {disease}"

Usage:
    python exp_prompt_robustness.py --dataset vindrcxr --model sfr_mistral

Output:
    - results/prompt_robustness/robustness_summary.csv
    - results/prompt_robustness/detailed_results.json
"""

import os
import sys
import datetime
import json
import argparse
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from sklearn.metrics import roc_auc_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import load_clip
import zero_shot
import clip
from get_embed import RadiologyEmbeddingGenerator


# Define prompt variants for robustness testing
PROMPT_VARIANTS = {
    'standard': {
        'pos': lambda d: f"{d.lower()}",
        'neg': lambda d: f"no {d.lower()}",
        'description': 'Standard: "{disease}" vs "no {disease}"'
    },
    'absence': {
        'pos': lambda d: f"{d.lower()}",
        'neg': lambda d: f"absence of {d.lower()}",
        'description': 'Absence: "{disease}" vs "absence of {disease}"'
    },
    'present_absent': {
        'pos': lambda d: f"{d.lower()} present",
        'neg': lambda d: f"{d.lower()} absent",
        'description': 'Present/Absent: "{disease} present" vs "{disease} absent"'
    },
    'not_present': {
        'pos': lambda d: f"{d.lower()}",
        'neg': lambda d: f"{d.lower()} not present",
        'description': 'Not Present: "{disease}" vs "{disease} not present"'
    },
    'clinical': {
        'pos': lambda d: f"findings consistent with {d.lower()}",
        'neg': lambda d: f"no evidence of {d.lower()}",
        'description': 'Clinical: "findings consistent with {disease}" vs "no evidence of {disease}"'
    },
    'negative_prefix': {
        'pos': lambda d: f"{d.lower()}",
        'neg': lambda d: f"negative for {d.lower()}",
        'description': 'Negative: "{disease}" vs "negative for {disease}"'
    }
}


class LLMEmbeddingGenerator:
    """Generator for LLM embeddings."""

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


def get_dataset_config(dataset_name):
    """Get dataset-specific configuration."""
    configs = {
        'chexpert_valid': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_valid.csv",
            'labels': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'No Finding', 'Pleural Effusion',
                       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'],
        },
        'chexpert': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_test.csv",
            'labels': ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
                       'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                       'Lung Opacity', 'No Finding', 'Pleural Effusion',
                       'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'],
        },
        'vindrcxr': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/vindrcxr_test.csv",
            'labels': None,  # Will be loaded from CSV
        },
        'padchest': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/padchest_test.csv",
            'labels': None,
        },
        'indiana': {
            'cxr_filepath': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.h5",
            'labels_path': "/home/than/DeepLearning/cxr_concept/CheXzero/data/indiana_test.csv",
            'labels': None,
        }
    }

    config = configs[dataset_name]

    if config['labels'] is None:
        df = pd.read_csv(config['labels_path'])
        exclude_cols = ['image_id', 'ImageID', 'name', 'Path', 'is_test', 'uid', 'filename',
                        'projection', 'MeSH', 'Problems', 'image', 'indication',
                        'comparison', 'findings', 'impression']
        config['labels'] = [col for col in df.columns if col not in exclude_cols]

    return config


def load_concept_embeddings(model_name: str) -> Tuple[Dict[int, np.ndarray], int]:
    """Load pre-computed concept embeddings."""
    embeddings_file = f"/home/than/DeepLearning/cxr_concept/CheXzero/concepts/embeddings/concepts_embeddings_{model_name}.pickle"
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    embedding_dim = len(list(embeddings_data.values())[0])
    return embeddings_data, embedding_dim


def evaluate_prompt_variant(llm_representation, test_labels, y_true,
                            variant_name, llm_model, embedding_generator):
    """Evaluate a single prompt variant."""

    variant = PROMPT_VARIANTS[variant_name]

    # Generate prompts
    pos_prompts = [variant['pos'](label) for label in test_labels]
    neg_prompts = [variant['neg'](label) for label in test_labels]

    print(f"\n  Variant: {variant_name}")
    print(f"    Sample pos: {pos_prompts[0]}")
    print(f"    Sample neg: {neg_prompts[0]}")

    # Get embeddings
    pos_embeddings = embedding_generator.get_embeddings_batch(pos_prompts, llm_model)
    neg_embeddings = embedding_generator.get_embeddings_batch(neg_prompts, llm_model)

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

    # Compute AUROCs
    results = {'variant': variant_name, 'description': variant['description']}
    aurocs = []

    for i, label in enumerate(test_labels):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        try:
            auroc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results[label] = auroc
            aurocs.append(auroc)
        except:
            results[label] = np.nan

    results['macro_auroc'] = np.mean(aurocs) if aurocs else np.nan

    return results


def run_robustness_experiment(dataset_name: str, llm_model: str):
    """Run prompt robustness experiment."""

    print("="*70)
    print(f"PROMPT ROBUSTNESS EXPERIMENT")
    print(f"Dataset: {dataset_name}, Embedding Model: {llm_model}")
    print("="*70)

    config = get_dataset_config(dataset_name)
    test_labels = config['labels']

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

    # Setup dataset
    print("\n=== Loading Dataset ===")
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

    print(f"Test set: {len(test_dataset)} images, {len(test_labels)} labels")

    # Load concepts
    concepts_df = pd.read_csv("/home/than/DeepLearning/cxr_concept/CheXzero/concepts/mimic_concepts.csv")
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
    print("\n=== Encoding Concepts ===")
    concept_batch_size = 512 if llm_model == "qwen3_8b" else 1024
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

    # Encode images
    print("\n=== Encoding Images ===")
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

    # Move model to CPU for large models
    if llm_model == "qwen3_8b":
        model = model.cpu()
        torch.cuda.empty_cache()

    # Initialize embedding generator
    embedding_generator = LLMEmbeddingGenerator()

    # Evaluate each prompt variant
    print("\n=== Evaluating Prompt Variants ===")
    all_results = []

    for variant_name in PROMPT_VARIANTS.keys():
        try:
            results = evaluate_prompt_variant(
                llm_representation, test_labels, y_true,
                variant_name, llm_model, embedding_generator
            )
            all_results.append(results)
            print(f"    Macro AUROC: {results['macro_auroc']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")

    embedding_generator.cleanup()

    # Create summary
    print("\n" + "="*70)
    print("PROMPT ROBUSTNESS SUMMARY")
    print("="*70)

    summary_data = []
    for r in all_results:
        summary_data.append({
            'Variant': r['variant'],
            'Description': r['description'],
            'Macro AUROC': f"{r['macro_auroc']:.4f}"
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Compute robustness metrics
    aurocs = [r['macro_auroc'] for r in all_results if not np.isnan(r['macro_auroc'])]
    print(f"\nRobustness Statistics:")
    print(f"  Mean AUROC across variants: {np.mean(aurocs):.4f}")
    print(f"  Std AUROC across variants:  {np.std(aurocs):.4f}")
    print(f"  Range: [{np.min(aurocs):.4f}, {np.max(aurocs):.4f}]")
    print(f"  Coefficient of Variation:   {np.std(aurocs)/np.mean(aurocs)*100:.2f}%")

    # Save results
    results_dir = f"results/prompt_robustness/{dataset_name}_{llm_model}"
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_df.to_csv(f"{results_dir}/summary_{timestamp}.csv", index=False)

    # Save detailed results
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

    with open(f"{results_dir}/detailed_{timestamp}.json", 'w') as f:
        json.dump({
            'dataset': dataset_name,
            'llm_model': llm_model,
            'robustness_stats': {
                'mean_auroc': float(np.mean(aurocs)),
                'std_auroc': float(np.std(aurocs)),
                'cv_percent': float(np.std(aurocs)/np.mean(aurocs)*100),
                'min_auroc': float(np.min(aurocs)),
                'max_auroc': float(np.max(aurocs))
            },
            'variant_results': convert(all_results)
        }, f, indent=2)

    print(f"\nResults saved to {results_dir}/")

    return all_results, summary_df


def main():
    parser = argparse.ArgumentParser(description='Prompt Robustness Experiment')
    parser.add_argument('--dataset', default='vindrcxr',
                        choices=['chexpert_valid', 'chexpert', 'vindrcxr', 'padchest', 'indiana'])
    parser.add_argument('--model', default='sfr_mistral',
                        choices=['sfr_mistral', 'qwen3_8b', 'openai_small', 'biomedbert'])
    parser.add_argument('--all-models', action='store_true',
                        help='Run for all embedding models')
    args = parser.parse_args()

    if args.all_models:
        models = ['sfr_mistral', 'qwen3_8b', 'openai_small', 'biomedbert']
        all_summaries = {}
        for model in models:
            print(f"\n\n{'#'*70}")
            print(f"# RUNNING FOR MODEL: {model}")
            print(f"{'#'*70}\n")
            try:
                results, summary = run_robustness_experiment(args.dataset, model)
                all_summaries[model] = summary
            except Exception as e:
                print(f"Error with {model}: {e}")

        # Print comparative summary
        print("\n\n" + "="*70)
        print("COMPARATIVE ROBUSTNESS SUMMARY ACROSS MODELS")
        print("="*70)
        for model, summary in all_summaries.items():
            aurocs = [float(r) for r in summary['Macro AUROC'].tolist()]
            print(f"\n{model}:")
            print(f"  Mean: {np.mean(aurocs):.4f}, Std: {np.std(aurocs):.4f}")
    else:
        run_robustness_experiment(args.dataset, args.model)


if __name__ == "__main__":
    main()
