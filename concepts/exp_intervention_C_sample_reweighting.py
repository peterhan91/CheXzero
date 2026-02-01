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
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
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


def compute_sample_weights(df, downweight_factor=0.1):
    """Compute sample weights to downweight EC+Atelectasis co-occurring samples."""
    ec_labels = (df['Enlarged Cardiomediastinum'] == 1).values
    ate_labels = (df['Atelectasis'] == 1).values
    co_occurring = ec_labels & ate_labels

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


def compute_llm_features(img_features, clip_concept_features, concept_embeddings):
    """Compute LLM-projected features"""
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


def extract_all_features(clip_model, concept_embeddings, clip_concept_features):
    """Extract LLM features for train/val/test sets"""
    features = {}
    labels = {}
    df_train = None

    # MIMIC Train (pre-encoded)
    print("\n--- Loading MIMIC Train ---")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        img_features = torch.tensor(f['cxr_feature'][:]).float()
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    df_train = pd.read_csv(MIMIC_LABELS_PATH)

    llm_features = []
    for i in tqdm(range(0, len(img_features), 500), desc="Computing LLM features"):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device))
        llm_features.append(feat.cpu())
    features['train'] = torch.cat(llm_features).numpy()
    labels['train'] = (df_train['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    # CheXpert Val
    print("\n--- Extracting CheXpert Val ---")
    df_val = pd.read_csv(CHEXPERT_VAL_LABELS)
    img_features = extract_clip_features(clip_model, CHEXPERT_VAL_PATH, len(df_val))

    llm_features = []
    for i in range(0, len(img_features), 500):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device))
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
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings.to(device))
        llm_features.append(feat.cpu())
    features['test'] = torch.cat(llm_features).numpy()
    labels['test'] = (df_test['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    return features, labels, df_train


def train_and_evaluate(features, labels, sample_weights=None, seed=42):
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

    # Create data loaders
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights).double(),
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=512, sampler=sampler)
    else:
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
    results_dir = 'results/intervention_C_sample_reweighting'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT C: Sample Reweighting Intervention (20 seeds)")
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

    # Extract features once
    print("\n" + "="*60)
    print("Extracting features")
    print("="*60)
    features, labels, df_train = extract_all_features(
        clip_model, concept_embeddings, clip_concept_features
    )

    # Compute sample weights and stats
    _, stats = compute_sample_weights(df_train, downweight_factor=0.1)
    print(f"\nCo-occurrence statistics:")
    print(f"  EC positive samples: {stats['ec_positive']}")
    print(f"  Atelectasis positive samples: {stats['atelectasis_positive']}")
    print(f"  EC+Atelectasis co-occurring: {stats['co_occurring']}")
    print(f"  Co-occurrence rate in EC: {stats['co_occurrence_rate_in_ec']:.1%}")

    # Run experiments with 20 seeds for each downweight factor
    print("\n" + "="*60)
    print("Running experiments with 20 seeds")
    print("="*60)

    seeds = list(range(42, 62))
    downweight_factors = [1.0, 0.5, 0.2, 0.1, 0.05]
    all_results = {factor: [] for factor in downweight_factors}

    for i, seed in enumerate(seeds):
        print(f"\n--- Seed {seed} ({i+1}/20) ---")

        for factor in downweight_factors:
            weights, _ = compute_sample_weights(df_train, downweight_factor=factor)
            result = train_and_evaluate(
                features, labels,
                sample_weights=weights if factor < 1.0 else None,
                seed=seed
            )
            all_results[factor].append(result)

        # Print progress for this seed
        baseline_auc = all_results[1.0][-1]['test_auc']
        best_factor = max(downweight_factors, key=lambda f: all_results[f][-1]['test_auc'])
        best_auc = all_results[best_factor][-1]['test_auc']
        print(f"  Baseline: {baseline_auc:.4f}, Best (factor={best_factor}): {best_auc:.4f}")

    # Aggregate results
    print("\n" + "="*60)
    print("SUMMARY (20 seeds)")
    print("="*60)

    baseline_aucs = [r['test_auc'] for r in all_results[1.0]]
    print(f"\nBaseline Test AUC: {np.mean(baseline_aucs):.4f} ± {np.std(baseline_aucs):.4f}")

    print("\nReweighting results:")
    best_mean_improvement = -float('inf')
    best_factor = 1.0

    for factor in downweight_factors[1:]:  # Skip baseline
        factor_aucs = [r['test_auc'] for r in all_results[factor]]
        improvements = [f - b for f, b in zip(factor_aucs, baseline_aucs)]
        mean_improvement = np.mean(improvements)

        print(f"  Factor={factor}: {np.mean(factor_aucs):.4f} ± {np.std(factor_aucs):.4f} "
              f"(improvement: {mean_improvement:+.4f} ± {np.std(improvements):.4f})")

        if mean_improvement > best_mean_improvement:
            best_mean_improvement = mean_improvement
            best_factor = factor

    print(f"\nBest factor: {best_factor} with improvement: {best_mean_improvement:+.4f}")

    # Save results
    results = {
        'num_seeds': 20,
        'seeds': seeds,
        'downweight_factors': downweight_factors,
        'co_occurrence_stats': stats,
        'results_by_factor': {}
    }

    for factor in downweight_factors:
        factor_aucs = [r['test_auc'] for r in all_results[factor]]
        improvements = [f - b for f, b in zip(factor_aucs, baseline_aucs)] if factor != 1.0 else [0.0] * 20

        results['results_by_factor'][str(factor)] = {
            'test_auc_mean': float(np.mean(factor_aucs)),
            'test_auc_std': float(np.std(factor_aucs)),
            'test_aucs': factor_aucs,
            'improvement_mean': float(np.mean(improvements)),
            'improvement_std': float(np.std(improvements)),
            'improvements': improvements,
            'all_results': all_results[factor]
        }

    results['best_factor'] = best_factor
    results['best_improvement'] = float(best_mean_improvement)

    with open(f'{results_dir}/results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
