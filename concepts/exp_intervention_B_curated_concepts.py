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

# Clinically correct EC concepts (manually curated)
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
            if target_lower in concept_lower:
                score = len(target_lower) / len(concept_lower)
                if score > best_score:
                    best_score = score
                    best_match = (i, concept)

        if best_match:
            indices.append(best_match[0])
            matched.append((target, best_match[1]))

    return indices, matched


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


def compute_cbm_features(img_features, clip_concept_features, concept_indices):
    """Compute CBM-style features using only selected concepts"""
    selected_clip = clip_concept_features[concept_indices]
    similarities = img_features @ selected_clip.T
    return similarities


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


def train_and_evaluate(features, labels, concepts, concept_indices, seed=42):
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

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

    # Get concept importance
    weights = model.linear.weight.detach().cpu().numpy().flatten()
    importance = [(concepts[concept_indices[i]], float(weights[i])) for i in range(len(concept_indices))]
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        'val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'concept_importance': importance[:10]
    }


def main():
    results_dir = 'results/intervention_B_curated_concepts'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT B: Curated Concept Intervention (20 seeds)")
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

    # Find concept indices for both sets
    original_indices, original_matched = find_concept_indices(concepts, ORIGINAL_EC_CONCEPTS)
    curated_indices, curated_matched = find_concept_indices(concepts, CURATED_EC_CONCEPTS)

    print(f"\nOriginal concepts: Found {len(original_indices)}/{len(ORIGINAL_EC_CONCEPTS)}")
    print(f"Curated concepts: Found {len(curated_indices)}/{len(CURATED_EC_CONCEPTS)}")

    # Extract features once
    print("\n" + "="*60)
    print("Extracting ORIGINAL concept features")
    print("="*60)
    features_original, labels = extract_cbm_features(clip_model, clip_concept_features, original_indices)

    print("\n" + "="*60)
    print("Extracting CURATED concept features")
    print("="*60)
    features_curated, _ = extract_cbm_features(clip_model, clip_concept_features, curated_indices)

    # Run with 20 seeds
    print("\n" + "="*60)
    print("Running experiments with 20 seeds")
    print("="*60)

    seeds = list(range(42, 62))
    original_results = []
    curated_results = []

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/20) ---")

        original_result = train_and_evaluate(features_original, labels, concepts, original_indices, seed=seed)
        curated_result = train_and_evaluate(features_curated, labels, concepts, curated_indices, seed=seed)

        original_results.append(original_result)
        curated_results.append(curated_result)

        print(f"  Original Test AUC: {original_result['test_auc']:.4f}")
        print(f"  Curated Test AUC: {curated_result['test_auc']:.4f}")
        print(f"  Difference: {curated_result['test_auc'] - original_result['test_auc']:+.4f}")

    # Aggregate results
    original_test_aucs = [r['test_auc'] for r in original_results]
    curated_test_aucs = [r['test_auc'] for r in curated_results]
    differences = [c - o for c, o in zip(curated_test_aucs, original_test_aucs)]

    # Summary
    print("\n" + "="*60)
    print("SUMMARY (20 seeds)")
    print("="*60)
    print(f"\nOriginal Concepts ({len(original_indices)} concepts):")
    print(f"  Test AUC: {np.mean(original_test_aucs):.4f} ± {np.std(original_test_aucs):.4f}")

    print(f"\nCurated Concepts ({len(curated_indices)} concepts):")
    print(f"  Test AUC: {np.mean(curated_test_aucs):.4f} ± {np.std(curated_test_aucs):.4f}")

    print(f"\nDifference: {np.mean(differences):+.4f} ± {np.std(differences):.4f}")

    # Save results
    results = {
        'num_seeds': 20,
        'seeds': seeds,
        'original_concept_list': ORIGINAL_EC_CONCEPTS,
        'curated_concept_list': CURATED_EC_CONCEPTS,
        'num_original_concepts': len(original_indices),
        'num_curated_concepts': len(curated_indices),
        'original': {
            'test_auc_mean': float(np.mean(original_test_aucs)),
            'test_auc_std': float(np.std(original_test_aucs)),
            'test_aucs': original_test_aucs,
            'all_results': original_results
        },
        'curated': {
            'test_auc_mean': float(np.mean(curated_test_aucs)),
            'test_auc_std': float(np.std(curated_test_aucs)),
            'test_aucs': curated_test_aucs,
            'all_results': curated_results
        },
        'difference': {
            'mean': float(np.mean(differences)),
            'std': float(np.std(differences)),
            'all': differences
        }
    }
    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
