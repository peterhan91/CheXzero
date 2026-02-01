#!/usr/bin/env python3
"""
Experiment B: Curated Concept Intervention for EC Classification
================================================================
Uses clinically correct mediastinal concepts instead of the learned
atelectasis-dominated concepts for Enlarged Cardiomediastinum classification.

Hypothesis: Using mediastinal-specific concepts will produce more clinically
valid predictions and improve generalization.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
import pickle
from tqdm import tqdm
import h5py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths - matching exp_linear.py
CONCEPTS_CSV = "/home/than/DeepLearning/CheXzero/data/mimic_concepts.csv"
EMBEDDINGS_PATH = "/home/than/DeepLearning/CheXzero/embeddings_output/cxr_embeddings_sfr_mistral.pickle"
MIMIC_FEATURES_PATH = "/home/than/DeepLearning/conceptqa_vip/data/mimic_train_448.h5"
MIMIC_LABELS_PATH = "/home/than/DeepLearning/conceptqa_vip/data/mimic_cxr.csv"
CLIP_CONCEPT_CACHE = "cache/clip_concept_features.pkl"

# Clinically correct EC concepts (manually curated)
# These describe mediastinal widening, NOT atelectasis
CURATED_EC_CONCEPTS = [
    "mediastinal widening",
    "widened mediastinum",
    "enlarged mediastinum",
    "mediastinal enlargement",
    "prominent mediastinum",
    "mediastinal mass",
    "enlarged cardiac silhouette",
    "prominent aortic knob",
    "tortuous aorta",
    "aortic enlargement",
    "hilar enlargement",
    "prominent hila",
    "superior mediastinal widening",
    "paratracheal widening",
    "mediastinal contour abnormality"
]

# Original (confounded) concepts from cbm_concepts.json
ORIGINAL_EC_CONCEPTS = [
    "bilateral atelectasis has slightly improved on the right and substantially worsened on the left",
    "right middle and lower lobe atelectasis worsened",
    "left lower lung atelectasis has worsened",
    "right lower lobe atelectasis has worsened substantially, perhaps lobar collapse",
    "bilateral lower lobe atelectasis has improved substantially"
]


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def find_concept_indices(concepts, target_concepts):
    """Find indices of target concepts in the concept list (fuzzy match)"""
    indices = []
    matched = []

    for target in target_concepts:
        target_lower = target.lower()
        best_match = None
        best_score = 0

        for i, concept in enumerate(concepts):
            concept_lower = concept.lower()
            # Check if target is substring of concept
            if target_lower in concept_lower:
                score = len(target_lower) / len(concept_lower)
                if score > best_score:
                    best_score = score
                    best_match = (i, concept)

        if best_match:
            indices.append(best_match[0])
            matched.append((target, best_match[1]))

    return indices, matched


def load_data():
    """Load concepts and embeddings"""
    print("Loading concepts...")
    concepts_df = pd.read_csv(CONCEPTS_CSV)
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()

    print("Loading embeddings...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings_data = pickle.load(f)

    embedding_dim = len(list(embeddings_data.values())[0])
    concept_embeddings = np.zeros((len(concepts), embedding_dim))
    for pos, cidx in enumerate(concept_indices):
        if cidx in embeddings_data:
            concept_embeddings[pos] = embeddings_data[cidx]

    return concepts, torch.tensor(concept_embeddings).float()


def compute_cbm_features(img_features, clip_concept_features, concept_indices, device):
    """Compute CBM-style features using only selected concepts"""
    # Select only the specified concept features
    selected_clip = clip_concept_features[concept_indices]

    # Compute similarities for selected concepts only
    similarities = img_features @ selected_clip.T  # [N, num_selected]

    # For CBM, we use the similarity scores directly as features
    return similarities


def run_cbm_experiment(concepts, concept_embeddings, clip_concept_features,
                       train_features, train_labels, concept_list, name, seed=42):
    """Run CBM experiment with specified concepts"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Find concept indices
    indices, matched = find_concept_indices(concepts, concept_list)
    print(f"\n{name}: Found {len(indices)}/{len(concept_list)} concepts")
    for target, match in matched[:5]:
        print(f"  '{target}' -> '{match[:60]}...'")

    if len(indices) == 0:
        print("  WARNING: No concepts found!")
        return None

    # Compute CBM features
    cbm_features = compute_cbm_features(
        train_features.to(device),
        clip_concept_features.to(device),
        indices, device
    ).cpu()

    # Train model
    model = LogisticRegressionModel(len(indices)).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    dataset = TensorDataset(cbm_features, torch.tensor(train_labels).unsqueeze(1))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)

    for epoch in range(100):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(cbm_features.to(device)).cpu().numpy()
    auc = roc_auc_score(train_labels, preds)

    # Get concept importance (weights)
    weights = model.linear.weight.detach().cpu().numpy().flatten()
    importance = [(concepts[indices[i]], float(weights[i])) for i in range(len(indices))]
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'name': name,
        'num_concepts': len(indices),
        'auc': float(auc),
        'concept_importance': importance[:10]
    }


def main():
    results_dir = 'results/intervention_B_curated_concepts'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT B: Curated Concept Intervention")
    print("="*60)

    # Load data
    concepts, concept_embeddings = load_data()

    with open(CLIP_CONCEPT_CACHE, 'rb') as f:
        cache = pickle.load(f)
    clip_concept_features = cache['concept_features']

    # Load training data
    print("\nLoading training features...")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        train_features = torch.tensor(f['cxr_feature'][:]).float()
    train_features = train_features / train_features.norm(dim=-1, keepdim=True)

    df_train = pd.read_csv(MIMIC_LABELS_PATH)
    train_labels = (df_train['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # Run experiments
    print("\n" + "-"*40)
    print("Running CBM with ORIGINAL (confounded) concepts...")
    original_result = run_cbm_experiment(
        concepts, concept_embeddings, clip_concept_features,
        train_features, train_labels,
        ORIGINAL_EC_CONCEPTS, "Original (Atelectasis-dominated)"
    )

    print("\n" + "-"*40)
    print("Running CBM with CURATED (mediastinal) concepts...")
    curated_result = run_cbm_experiment(
        concepts, concept_embeddings, clip_concept_features,
        train_features, train_labels,
        CURATED_EC_CONCEPTS, "Curated (Mediastinal)"
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    if original_result:
        print(f"\nOriginal Concepts (Atelectasis-dominated):")
        print(f"  AUC: {original_result['auc']:.4f}")
        print(f"  Top concepts by importance:")
        for c, w in original_result['concept_importance'][:5]:
            print(f"    {w:+.4f}: {c[:50]}...")

    if curated_result:
        print(f"\nCurated Concepts (Mediastinal):")
        print(f"  AUC: {curated_result['auc']:.4f}")
        print(f"  Top concepts by importance:")
        for c, w in curated_result['concept_importance'][:5]:
            print(f"    {w:+.4f}: {c[:50]}...")

    if original_result and curated_result:
        print(f"\nDifference: {curated_result['auc'] - original_result['auc']:+.4f}")

    # Save results
    results = {
        'original': original_result,
        'curated': curated_result,
        'curated_concept_list': CURATED_EC_CONCEPTS,
        'original_concept_list': ORIGINAL_EC_CONCEPTS
    }
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
