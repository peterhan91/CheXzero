#!/usr/bin/env python3
"""
Experiment B: Curated Concept Intervention for EC Classification
================================================================
Compares atelectasis-based concepts (confounded) vs mediastinal concepts
(clinically correct) for Enlarged Cardiomediastinum classification.

Two modes:
- 'curated': Hand-crafted concept lists (small, precise)
- 'searched': Pattern-matched from full 368k vocabulary (comprehensive)

Usage:
    python exp_intervention_B_curated_concepts.py --mode curated
    python exp_intervention_B_curated_concepts.py --mode searched
"""

import os
import json
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
from sklearn.metrics import roc_auc_score
import pickle
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

# ============================================================================
# MODE 1: Hand-crafted curated concepts (small, precise)
# ============================================================================
CURATED_ATELECTASIS_CONCEPTS = [
    "bilateral atelectasis",
    "bibasilar atelectasis",
    "lower lobe atelectasis",
    "right lower lobe atelectasis",
    "left lower lobe atelectasis",
    "subsegmental atelectasis",
    "linear atelectasis",
    "basilar atelectasis",
    "atelectatic changes",
    "atelectasis worsened",
]

CURATED_MEDIASTINAL_CONCEPTS = [
    "mediastinal widening",
    "widened mediastinum",
    "enlarged mediastinum",
    "cardiomediastinal silhouette",
    "mediastinal contour",
    "prominent mediastinum",
    "cardiomegaly",
    "enlarged heart",
    "heart size enlarged",
    "tortuous aorta",
    "aortic enlargement",
    "prominent aortic knob",
    "hilar enlargement",
    "prominent hila",
    "cardiac silhouette enlarged",
]

# ============================================================================
# MODE 2: Search patterns for mimic_concepts.csv (comprehensive)
# ============================================================================
SEARCH_ATELECTASIS_PATTERNS = [r'atelectasis', r'atelectatic']

SEARCH_MEDIASTINAL_PATTERNS = [
    r'mediastin',                    # mediastinum, mediastinal, cardiomediastinal
    r'cardiomegaly',                 # cardiomegaly
    r'heart.*(enlarg|size)',         # heart enlargement, heart size
    r'(enlarg|prominent).*heart',
    r'aort',                         # aorta, aortic
    r'hilar',                        # hilar
    r'\bhila\b',                     # hila (word boundary)
]


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


def search_concepts_by_patterns(concepts, patterns):
    """Search for concepts matching any of the given regex patterns"""
    indices = []
    matched_concepts = []

    for i, concept in enumerate(concepts):
        concept_lower = concept.lower()
        for pattern in patterns:
            if re.search(pattern, concept_lower):
                indices.append(i)
                matched_concepts.append(concept)
                break  # Only add once per concept

    return indices, matched_concepts


def find_concepts_by_substring(concepts, target_list):
    """Find concepts containing any of the target substrings (fuzzy match)"""
    indices = []
    matched_concepts = []

    for target in target_list:
        target_lower = target.lower()
        best_match = None
        best_score = 0

        for i, concept in enumerate(concepts):
            concept_lower = concept.lower()
            if target_lower in concept_lower:
                score = len(target_lower) / len(concept_lower)
                if score > best_score:
                    best_score = score
                    best_match = (i, concept)

        if best_match and best_match[0] not in indices:
            indices.append(best_match[0])
            matched_concepts.append(best_match[1])

    return indices, matched_concepts


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


def compute_cbm_features(img_features, clip_concept_features, concept_indices, batch_size=1000):
    """Compute CBM-style features using only selected concepts (memory efficient)"""
    selected_clip = clip_concept_features[concept_indices]

    # Process in batches to avoid OOM
    all_similarities = []
    for i in range(0, len(img_features), batch_size):
        batch_img = img_features[i:i+batch_size].to(device)
        batch_sim = batch_img @ selected_clip.T
        all_similarities.append(batch_sim.cpu())
        torch.cuda.empty_cache()

    return torch.cat(all_similarities, dim=0)


def extract_cbm_features(clip_model, clip_concept_features, concept_indices):
    """Extract CBM features for train/val/test sets"""
    features = {}
    labels = {}

    # MIMIC Train (pre-encoded)
    print("\n--- Loading MIMIC Train ---")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        img_features = torch.tensor(f['cxr_feature'][:]).float()
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    df = pd.read_csv(MIMIC_LABELS_PATH)

    features['train'] = compute_cbm_features(
        img_features.to(device), clip_concept_features, concept_indices
    ).cpu().numpy()
    labels['train'] = (df['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # CheXpert Val
    print("\n--- Extracting CheXpert Val ---")
    df_val = pd.read_csv(CHEXPERT_VAL_LABELS)
    img_features = extract_clip_features(clip_model, CHEXPERT_VAL_PATH, len(df_val))
    features['val'] = compute_cbm_features(
        img_features.to(device), clip_concept_features, concept_indices
    ).cpu().numpy()
    labels['val'] = (df_val['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # CheXpert Test
    print("\n--- Extracting CheXpert Test ---")
    df_test = pd.read_csv(CHEXPERT_TEST_LABELS)
    img_features = extract_clip_features(clip_model, CHEXPERT_TEST_PATH, len(df_test))
    features['test'] = compute_cbm_features(
        img_features.to(device), clip_concept_features, concept_indices
    ).cpu().numpy()
    labels['test'] = (df_test['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    return features, labels


def train_and_evaluate(features, labels, concepts, concept_indices, lr=1e-3, seed=42):
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
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)

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

    # Get concept importance (top 20)
    weights = model.linear.weight.detach().cpu().numpy().flatten()
    importance = [(concepts[concept_indices[i]], float(weights[i])) for i in range(len(concept_indices))]
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'concept_importance': importance[:20]
    }


def run_experiment(mode='searched'):
    """Run experiment with specified mode"""
    # Set output directory based on mode
    if mode == 'curated':
        results_dir = 'results/intervention_B_curated_concepts_handcrafted'
        lr = 1e-3  # Higher lr for small concept set
    else:  # searched
        results_dir = 'results/intervention_B_curated_concepts_searched'
        lr = 1e-3  # Same lr, but more features

    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print(f"EXPERIMENT B: Curated Concept Intervention - {mode.upper()} mode")
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

    # Get concept indices based on mode
    print("\nFinding concepts...")
    if mode == 'curated':
        atelectasis_indices, atelectasis_matched = find_concepts_by_substring(
            concepts, CURATED_ATELECTASIS_CONCEPTS
        )
        mediastinal_indices, mediastinal_matched = find_concepts_by_substring(
            concepts, CURATED_MEDIASTINAL_CONCEPTS
        )
        atelectasis_spec = CURATED_ATELECTASIS_CONCEPTS
        mediastinal_spec = CURATED_MEDIASTINAL_CONCEPTS
    else:  # searched
        atelectasis_indices, atelectasis_matched = search_concepts_by_patterns(
            concepts, SEARCH_ATELECTASIS_PATTERNS
        )
        mediastinal_indices, mediastinal_matched = search_concepts_by_patterns(
            concepts, SEARCH_MEDIASTINAL_PATTERNS
        )
        atelectasis_spec = SEARCH_ATELECTASIS_PATTERNS
        mediastinal_spec = SEARCH_MEDIASTINAL_PATTERNS

    print(f"Atelectasis concepts found: {len(atelectasis_indices)}")
    print(f"Mediastinal concepts found: {len(mediastinal_indices)}")

    # Show some examples
    print(f"\nAtelectasis examples: {atelectasis_matched[:3]}")
    print(f"Mediastinal examples: {mediastinal_matched[:3]}")

    # Extract features once
    print("\n" + "="*60)
    print("Extracting ATELECTASIS concept features (confounded)")
    print("="*60)
    features_atelectasis, labels = extract_cbm_features(clip_model, clip_concept_features, atelectasis_indices)

    print("\n" + "="*60)
    print("Extracting MEDIASTINAL concept features (curated)")
    print("="*60)
    features_mediastinal, _ = extract_cbm_features(clip_model, clip_concept_features, mediastinal_indices)

    # Run with 20 seeds
    print("\n" + "="*60)
    print("Running experiments with 20 seeds")
    print("="*60)

    seeds = list(range(42, 62))
    atelectasis_results = []
    mediastinal_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/20) ---")

        atelectasis_result = train_and_evaluate(
            features_atelectasis, labels, concepts, atelectasis_indices, lr=lr, seed=seed
        )
        mediastinal_result = train_and_evaluate(
            features_mediastinal, labels, concepts, mediastinal_indices, lr=lr, seed=seed
        )

        atelectasis_results.append(atelectasis_result)
        mediastinal_results.append(mediastinal_result)

        print(f"  Atelectasis (confounded) Test AUC: {atelectasis_result['test_auc']:.4f}")
        print(f"  Mediastinal (curated) Test AUC: {mediastinal_result['test_auc']:.4f}")
        print(f"  Improvement: {mediastinal_result['test_auc'] - atelectasis_result['test_auc']:+.4f}")

    # Aggregate results
    atelectasis_test_aucs = [r['test_auc'] for r in atelectasis_results]
    mediastinal_test_aucs = [r['test_auc'] for r in mediastinal_results]
    improvements = [m - a for m, a in zip(mediastinal_test_aucs, atelectasis_test_aucs)]

    # Summary
    print("\n" + "="*60)
    print(f"SUMMARY - {mode.upper()} mode (20 seeds)")
    print("="*60)
    print(f"\nAtelectasis Concepts (confounded, {len(atelectasis_indices)} concepts):")
    print(f"  Test AUC: {np.mean(atelectasis_test_aucs):.4f} +/- {np.std(atelectasis_test_aucs):.4f}")

    print(f"\nMediastinal Concepts (curated, {len(mediastinal_indices)} concepts):")
    print(f"  Test AUC: {np.mean(mediastinal_test_aucs):.4f} +/- {np.std(mediastinal_test_aucs):.4f}")

    print(f"\nImprovement: {np.mean(improvements):+.4f} +/- {np.std(improvements):.4f}")

    # Save results
    results = {
        'mode': mode,
        'num_seeds': 20,
        'seeds': seeds,
        'learning_rate': lr,
        'atelectasis_spec': atelectasis_spec,
        'mediastinal_spec': mediastinal_spec,
        'num_atelectasis_concepts': len(atelectasis_indices),
        'num_mediastinal_concepts': len(mediastinal_indices),
        'atelectasis_examples': atelectasis_matched[:10],
        'mediastinal_examples': mediastinal_matched[:10],
        'atelectasis': {
            'test_auc_mean': float(np.mean(atelectasis_test_aucs)),
            'test_auc_std': float(np.std(atelectasis_test_aucs)),
            'test_aucs': atelectasis_test_aucs,
            'all_results': atelectasis_results
        },
        'mediastinal': {
            'test_auc_mean': float(np.mean(mediastinal_test_aucs)),
            'test_auc_std': float(np.std(mediastinal_test_aucs)),
            'test_aucs': mediastinal_test_aucs,
            'all_results': mediastinal_results
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
    return results


def main():
    parser = argparse.ArgumentParser(description='Experiment B: Curated Concept Intervention')
    parser.add_argument('--mode', type=str, choices=['curated', 'searched', 'both'],
                        default='both',
                        help='Mode: curated (hand-crafted), searched (pattern-matched), or both')
    args = parser.parse_args()

    if args.mode == 'both':
        print("\n" + "#"*60)
        print("# Running CURATED mode")
        print("#"*60)
        run_experiment('curated')

        print("\n" + "#"*60)
        print("# Running SEARCHED mode")
        print("#"*60)
        run_experiment('searched')
    else:
        run_experiment(args.mode)


if __name__ == "__main__":
    main()
