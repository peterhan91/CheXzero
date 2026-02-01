#!/usr/bin/env python3
"""
Experiment C: Sample Reweighting Intervention for EC Confounder
================================================================
Downweights training samples where EC co-occurs with Atelectasis to reduce
the influence of confounded samples during training.

Hypothesis: Reducing the influence of EC+Atelectasis co-occurring samples
will force the model to learn true mediastinal features rather than
atelectasis shortcuts.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
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


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def compute_sample_weights(df, downweight_factor=0.1):
    """
    Compute sample weights to downweight EC+Atelectasis co-occurring samples.

    Args:
        df: DataFrame with label columns
        downweight_factor: Weight for co-occurring samples (0.1 = 10x less influence)

    Returns:
        weights: Array of sample weights
    """
    ec_labels = (df['Enlarged Cardiomediastinum'] == 1).values
    ate_labels = (df['Atelectasis'] == 1).values

    # Samples where both EC and Atelectasis are positive
    co_occurring = ec_labels & ate_labels

    # Base weight = 1, downweight co-occurring samples
    weights = np.ones(len(df))
    weights[co_occurring] = downweight_factor

    stats = {
        'total_samples': int(len(df)),
        'ec_positive': int(ec_labels.sum()),
        'atelectasis_positive': int(ate_labels.sum()),
        'co_occurring': int(co_occurring.sum()),
        'co_occurrence_rate_in_ec': float(co_occurring.sum() / max(ec_labels.sum(), 1))
    }

    return weights, stats


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


def compute_llm_features(img_features, clip_concept_features, concept_embeddings, device):
    """Compute LLM-projected features"""
    similarities = img_features @ clip_concept_features.T
    llm_features = similarities @ concept_embeddings
    llm_features = llm_features / llm_features.norm(dim=-1, keepdim=True)
    return llm_features


def run_experiment(train_features, train_labels, sample_weights=None, name="Baseline", seed=42):
    """Run training experiment with optional sample weighting"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_features.shape[1]
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = nn.BCELoss(reduction='none')  # Per-sample loss for weighting
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    # Create dataset
    dataset = TensorDataset(
        train_features,
        torch.tensor(train_labels).unsqueeze(1).float()
    )

    if sample_weights is not None:
        # Use weighted random sampler
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights),
            num_samples=len(dataset),
            replacement=True
        )
        loader = DataLoader(dataset, batch_size=512, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Training
    for epoch in range(50):
        model.train()
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets).mean()
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(train_features.to(device)).cpu().numpy()
    auc = roc_auc_score(train_labels, preds)

    return {'name': name, 'auc': float(auc)}


def main():
    results_dir = 'results/intervention_C_sample_reweighting'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT C: Sample Reweighting Intervention")
    print("="*60)

    # Load data
    concepts, concept_embeddings = load_data()

    with open(CLIP_CONCEPT_CACHE, 'rb') as f:
        cache = pickle.load(f)
    clip_concept_features = cache['concept_features'].to(device)

    # Load training data
    print("\nLoading training features...")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        img_features = torch.tensor(f['cxr_feature'][:]).float()
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)

    df_train = pd.read_csv(MIMIC_LABELS_PATH)
    train_labels = (df_train['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # Compute LLM features
    print("Computing LLM features...")
    batch_size = 1000
    features_list = []
    for i in tqdm(range(0, len(img_features), batch_size)):
        batch = img_features[i:i+batch_size].to(device)
        feat = compute_llm_features(batch, clip_concept_features,
                                    concept_embeddings.to(device), device)
        features_list.append(feat.cpu())
    train_features = torch.cat(features_list)

    # Compute sample weights
    print("\nAnalyzing label co-occurrence...")
    downweight_factors = [1.0, 0.5, 0.2, 0.1, 0.05]
    results = []

    for factor in downweight_factors:
        weights, stats = compute_sample_weights(df_train, downweight_factor=factor)

        if factor == 1.0:
            print(f"\nCo-occurrence statistics:")
            print(f"  EC positive samples: {stats['ec_positive']}")
            print(f"  Atelectasis positive samples: {stats['atelectasis_positive']}")
            print(f"  EC+Atelectasis co-occurring: {stats['co_occurring']}")
            print(f"  Co-occurrence rate in EC: {stats['co_occurrence_rate_in_ec']:.1%}")

        name = f"Factor={factor}" if factor < 1.0 else "Baseline"
        print(f"\n--- Running {name} ---")

        result = run_experiment(
            train_features, train_labels,
            sample_weights=weights if factor < 1.0 else None,
            name=name
        )
        result['downweight_factor'] = factor
        result['stats'] = stats
        results.append(result)
        print(f"  AUC: {result['auc']:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_auc = results[0]['auc']
    print(f"\nBaseline AUC: {baseline_auc:.4f}")
    print("\nReweighting results:")
    for r in results[1:]:
        improvement = r['auc'] - baseline_auc
        print(f"  Factor={r['downweight_factor']}: AUC={r['auc']:.4f} ({improvement:+.4f})")

    # Find best
    best = max(results, key=lambda x: x['auc'])
    print(f"\nBest: {best['name']} with AUC={best['auc']:.4f}")

    # Save results
    save_results = {
        'experiments': results,
        'baseline_auc': baseline_auc,
        'best_factor': best['downweight_factor'],
        'best_auc': best['auc']
    }
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
