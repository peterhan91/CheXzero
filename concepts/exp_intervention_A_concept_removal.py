#!/usr/bin/env python3
"""
Experiment A: Concept Removal Intervention for EC Confounder
============================================================
Removes atelectasis-related concepts from the embedding space before training
linear probe for Enlarged Cardiomediastinum classification.

Hypothesis: Removing confounder concepts will improve EC classification and
shift concept attributions from atelectasis to mediastinal concepts.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from sklearn.metrics import roc_auc_score
import pickle
import re
import random
from tqdm import tqdm
import h5py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths - matching exp_linear.py
CONCEPTS_CSV = "/home/than/DeepLearning/CheXzero/data/mimic_concepts.csv"
EMBEDDINGS_PATH = "/home/than/DeepLearning/CheXzero/embeddings_output/cxr_embeddings_sfr_mistral.pickle"
MIMIC_FEATURES_PATH = "/home/than/DeepLearning/conceptqa_vip/data/mimic_train_448.h5"
MIMIC_LABELS_PATH = "/home/than/DeepLearning/conceptqa_vip/data/mimic_cxr.csv"
CHEXPERT_VAL_PATH = "/home/than/DeepLearning/CheXzero/data/chexpert_valid.h5"
CHEXPERT_VAL_LABELS = "/home/than/DeepLearning/CheXzero/data/chexpert_valid.csv"
CHEXPERT_TEST_PATH = "/home/than/DeepLearning/CheXzero/data/chexpert_test.h5"
CHEXPERT_TEST_LABELS = "/home/than/DeepLearning/CheXzero/data/chexpert_test.csv"
CLIP_CONCEPT_CACHE = "cache/clip_concept_features.pkl"
CLIP_MODEL_PATH = "/home/than/DeepLearning/cxr_concept/CheXzero/checkpoints/dinov2-multi-v1.0_vitb/best_model.pt"


class CXRDataset(Dataset):
    def __init__(self, h5_path, num_samples, transform=None):
        self.h5_path = h5_path
        self.num_samples = num_samples
        self.transform = transform
        self.h5_file = h5py.File(h5_path, 'r')

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_data = self.h5_file['cxr'][idx]
        img_data = np.expand_dims(img_data, axis=0)
        img_data = np.repeat(img_data, 3, axis=0)
        img = torch.from_numpy(img_data).float()
        if self.transform:
            img = self.transform(img)
        return img

    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def identify_atelectasis_concepts(concepts):
    """Identify concepts related to atelectasis"""
    atelectasis_patterns = [
        r'atelectasis', r'atelectatic',
    ]
    atelectasis_indices = []
    for i, concept in enumerate(concepts):
        concept_lower = concept.lower()
        for pattern in atelectasis_patterns:
            if re.search(pattern, concept_lower):
                atelectasis_indices.append(i)
                break
    return atelectasis_indices


def load_concepts_and_embeddings():
    """Load concepts and embeddings"""
    print("Loading concepts...")
    concepts_df = pd.read_csv(CONCEPTS_CSV)
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()

    print("Loading concept embeddings...")
    with open(EMBEDDINGS_PATH, 'rb') as f:
        embeddings_data = pickle.load(f)

    embedding_dim = len(list(embeddings_data.values())[0])
    concept_embeddings = np.zeros((len(concepts), embedding_dim))
    for pos, cidx in enumerate(concept_indices):
        if cidx in embeddings_data:
            concept_embeddings[pos] = embeddings_data[cidx]

    return concepts, torch.tensor(concept_embeddings).float()


def compute_llm_features(img_features, clip_concept_features, concept_embeddings, keep_mask=None):
    """Compute LLM-projected features, optionally filtering concepts"""
    if keep_mask is not None:
        clip_concept_features = clip_concept_features[keep_mask]
        concept_embeddings = concept_embeddings[keep_mask]

    similarities = img_features @ clip_concept_features.T
    llm_features = similarities @ concept_embeddings
    llm_features = llm_features / llm_features.norm(dim=-1, keepdim=True)
    return llm_features


@torch.no_grad()
def extract_clip_features(model, h5_path, num_samples, batch_size=64):
    """Extract CLIP image features from raw images"""
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(448, interpolation=InterpolationMode.BICUBIC),
    ])
    dataset = CXRDataset(h5_path, num_samples, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_features = []
    for batch_imgs in tqdm(dataloader, desc="Extracting CLIP features"):
        batch_imgs = batch_imgs.to(device)
        features = model.encode_image(batch_imgs)
        features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu())
        torch.cuda.empty_cache()

    return torch.cat(all_features)


def extract_all_features(clip_model, concepts, concept_embeddings, clip_concept_features, keep_mask=None):
    """Extract LLM features for train/val/test sets"""
    features = {}
    labels = {}

    # MIMIC Train (pre-encoded)
    print("\n--- Loading MIMIC Train ---")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        img_features = torch.tensor(f['cxr_feature'][:]).float()
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    df = pd.read_csv(MIMIC_LABELS_PATH)

    llm_features = []
    for i in tqdm(range(0, len(img_features), 500), desc="Computing LLM features"):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device), keep_mask)
        llm_features.append(feat.cpu())
    features['train'] = torch.cat(llm_features).numpy()
    labels['train'] = (df['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # CheXpert Val
    print("\n--- Extracting CheXpert Val ---")
    df_val = pd.read_csv(CHEXPERT_VAL_LABELS)
    img_features = extract_clip_features(clip_model, CHEXPERT_VAL_PATH, len(df_val))

    llm_features = []
    for i in range(0, len(img_features), 500):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device), keep_mask)
        llm_features.append(feat.cpu())
    features['val'] = torch.cat(llm_features).numpy()
    labels['val'] = (df_val['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # CheXpert Test
    print("\n--- Extracting CheXpert Test ---")
    df_test = pd.read_csv(CHEXPERT_TEST_LABELS)
    img_features = extract_clip_features(clip_model, CHEXPERT_TEST_PATH, len(df_test))

    llm_features = []
    for i in range(0, len(img_features), 500):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device), keep_mask)
        llm_features.append(feat.cpu())
    features['test'] = torch.cat(llm_features).numpy()
    labels['test'] = (df_test['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    return features, labels


def train_and_evaluate(features, labels, seed=42):
    """Train with early stopping and evaluate on test set"""
    set_random_seed(seed)

    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(features['train']).float(),
        torch.tensor(labels['train']).unsqueeze(1).float()
    )
    val_dataset = TensorDataset(
        torch.tensor(features['val']).float(),
        torch.tensor(labels['val']).unsqueeze(1).float()
    )
    test_dataset = TensorDataset(
        torch.tensor(features['test']).float(),
        torch.tensor(labels['test']).unsqueeze(1).float()
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Model
    input_dim = features['train'].shape[1]
    model = LogisticRegressionModel(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-8)

    # Training with early stopping
    best_val_auc = 0
    best_model_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(200):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_preds = []
            val_targets = []
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                val_preds.append(model(inputs).cpu())
                val_targets.append(targets)
            val_preds = torch.cat(val_preds).numpy()
            val_targets = torch.cat(val_targets).numpy()
        val_auc = roc_auc_score(val_targets, val_preds)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    # Evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        test_preds = []
        test_targets = []
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            test_preds.append(model(inputs).cpu())
            test_targets.append(targets)
        test_preds = torch.cat(test_preds).numpy()
        test_targets = torch.cat(test_targets).numpy()
    test_auc = roc_auc_score(test_targets, test_preds)

    return {'val_auc': float(best_val_auc), 'test_auc': float(test_auc)}


def main():
    results_dir = 'results/intervention_A_concept_removal'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT A: Concept Removal Intervention (20 seeds)")
    print("="*60)

    # Load CLIP model
    print("\nLoading CLIP model...")
    clip_model = load_clip(
        model_path=CLIP_MODEL_PATH,
        pretrained=False,
        context_length=77,
        use_dinov2=True,
        dinov2_model_name='dinov2_vitb14'
    ).to(device).eval()

    # Load concepts and embeddings
    concepts, concept_embeddings = load_concepts_and_embeddings()

    # Load CLIP concept features
    with open(CLIP_CONCEPT_CACHE, 'rb') as f:
        cache = pickle.load(f)
    clip_concept_features = cache['concept_features'].to(device)

    # Identify atelectasis concepts
    atelectasis_indices = identify_atelectasis_concepts(concepts)
    print(f"\nFound {len(atelectasis_indices)} atelectasis-related concepts")

    keep_mask = torch.ones(len(concepts), dtype=torch.bool)
    keep_mask[atelectasis_indices] = False

    # Extract features once (features are deterministic, only training varies)
    print("\n" + "="*60)
    print("Extracting BASELINE features (with atelectasis concepts)")
    print("="*60)
    features_baseline, labels = extract_all_features(
        clip_model, concepts, concept_embeddings, clip_concept_features, keep_mask=None
    )

    print("\n" + "="*60)
    print("Extracting INTERVENTION features (atelectasis removed)")
    print("="*60)
    features_intervention, _ = extract_all_features(
        clip_model, concepts, concept_embeddings, clip_concept_features, keep_mask=keep_mask.to(device)
    )

    # Run with 20 seeds
    print("\n" + "="*60)
    print("Running experiments with 20 seeds")
    print("="*60)

    seeds = list(range(42, 62))
    baseline_results = []
    intervention_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/20) ---")

        baseline_result = train_and_evaluate(features_baseline, labels, seed=seed)
        intervention_result = train_and_evaluate(features_intervention, labels, seed=seed)

        baseline_results.append(baseline_result)
        intervention_results.append(intervention_result)

        print(f"  Baseline Test AUC: {baseline_result['test_auc']:.4f}")
        print(f"  Intervention Test AUC: {intervention_result['test_auc']:.4f}")
        print(f"  Improvement: {intervention_result['test_auc'] - baseline_result['test_auc']:+.4f}")

    # Aggregate results
    baseline_test_aucs = [r['test_auc'] for r in baseline_results]
    intervention_test_aucs = [r['test_auc'] for r in intervention_results]
    improvements = [i - b for i, b in zip(intervention_test_aucs, baseline_test_aucs)]

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (20 seeds)")
    print("="*60)
    print(f"Concepts removed: {len(atelectasis_indices)}")
    print(f"\nBaseline Test AUC:     {np.mean(baseline_test_aucs):.4f} ± {np.std(baseline_test_aucs):.4f}")
    print(f"Intervention Test AUC: {np.mean(intervention_test_aucs):.4f} ± {np.std(intervention_test_aucs):.4f}")
    print(f"Improvement:           {np.mean(improvements):+.4f} ± {np.std(improvements):.4f}")

    # Save results
    results = {
        'num_seeds': 20,
        'seeds': seeds,
        'num_concepts_removed': len(atelectasis_indices),
        'baseline': {
            'test_auc_mean': float(np.mean(baseline_test_aucs)),
            'test_auc_std': float(np.std(baseline_test_aucs)),
            'test_aucs': baseline_test_aucs,
            'all_results': baseline_results
        },
        'intervention': {
            'test_auc_mean': float(np.mean(intervention_test_aucs)),
            'test_auc_std': float(np.std(intervention_test_aucs)),
            'test_aucs': intervention_test_aucs,
            'all_results': intervention_results
        },
        'improvement': {
            'mean': float(np.mean(improvements)),
            'std': float(np.std(improvements)),
            'all': improvements
        }
    }
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
