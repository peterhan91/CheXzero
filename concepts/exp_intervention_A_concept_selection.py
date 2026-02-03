#!/usr/bin/env python3
"""
Experiment A: Concept Selection Intervention for EC Classification
===================================================================
Two modes for addressing the atelectasis confounder in Enlarged Cardiomediastinum:

1. EXCLUDE mode: Remove atelectasis-related concepts from the full concept space
   - Tests: "Does removing the confounder help?"

2. PRESERVE mode: Keep ONLY mediastinal-related concepts (clinically correct for EC)
   - Tests: "Can we build a model using only clinically correct concepts?"

Usage:
    python exp_intervention_A_concept_selection.py --mode exclude
    python exp_intervention_A_concept_selection.py --mode preserve
    python exp_intervention_A_concept_selection.py --mode both
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
import argparse
from tqdm import tqdm
import h5py
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import load_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
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
# Concept patterns
# ============================================================================

# Atelectasis patterns (for EXCLUDE mode)
ATELECTASIS_PATTERNS = [
    r'atelectasis',
    r'atelectatic',
]

# Mediastinal patterns (for PRESERVE mode) - comprehensive list for EC
MEDIASTINAL_PATTERNS = [
    # Mediastinum terms
    r'mediastin',                      # mediastinum, mediastinal, cardiomediastinal

    # Cardiomegaly and heart enlargement
    r'cardiomegaly',
    r'cardiac\s*(enlargement|silhouette|contour|shadow)',
    r'heart.*(enlarg|size|border|shadow|silhouette)',
    r'(enlarg|prominent|increased).*heart',
    r'(large|enlarged)\s*heart',

    # Aorta
    r'aort',                           # aorta, aortic, tortuous aorta
    r'thoracic\s*aorta',

    # Hilar
    r'hilar',                          # hilar enlargement, hilar prominence
    r'\bhila\b',                       # hila (word boundary)
    r'hilum',

    # Specific findings related to mediastinal widening
    r'widened?\s*(mediastin|superior)',
    r'mediastinal\s*(width|widen|mass|contour|shift)',
    r'superior\s*mediastin',
    r'paratracheal',
    r'retrosternal',
    r'prevascular',

    # Vascular structures in mediastinum
    r'vena\s*cava',
    r'pulmonary\s*(artery|trunk|vessel)',
    r'azygos',

    # Pericardium (related to cardiac silhouette)
    r'pericardi',                      # pericardial, pericardium

    # Thymus/thyroid (anterior mediastinum)
    r'thymus',
    r'thyroid',
    r'goiter',

    # Lymphadenopathy (mediastinal)
    r'lymph\s*node',
    r'adenopathy',
    r'lymphadenopathy',
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


def find_concepts_by_patterns(concepts, patterns):
    """Find concepts matching any of the given regex patterns"""
    indices = []
    matched_concepts = []

    for i, concept in enumerate(concepts):
        concept_lower = concept.lower()
        for pattern in patterns:
            if re.search(pattern, concept_lower):  # Already lowercase, no need for IGNORECASE
                indices.append(i)
                matched_concepts.append(concept)
                break  # Only add once per concept

    return indices, matched_concepts


def create_concept_mask(concepts, mode):
    """
    Create a boolean mask for concept selection.

    Args:
        concepts: List of concept strings
        mode: 'baseline' (all), 'exclude' (remove atelectasis), or 'preserve' (keep mediastinal)

    Returns:
        mask: Boolean tensor (True = keep concept)
        stats: Dictionary with statistics
    """
    num_concepts = len(concepts)

    if mode == 'baseline':
        mask = torch.ones(num_concepts, dtype=torch.bool)
        stats = {
            'mode': 'baseline',
            'total_concepts': num_concepts,
            'kept_concepts': num_concepts,
            'removed_concepts': 0,
        }

    elif mode == 'exclude':
        # Remove atelectasis-related concepts
        atelectasis_indices, atelectasis_matched = find_concepts_by_patterns(
            concepts, ATELECTASIS_PATTERNS
        )
        mask = torch.ones(num_concepts, dtype=torch.bool)
        mask[atelectasis_indices] = False

        stats = {
            'mode': 'exclude',
            'total_concepts': num_concepts,
            'kept_concepts': int(mask.sum()),
            'removed_concepts': len(atelectasis_indices),
            'removed_pattern': 'atelectasis',
            'example_removed': atelectasis_matched[:5],
        }

    elif mode == 'preserve':
        # Keep ONLY mediastinal-related concepts
        mediastinal_indices, mediastinal_matched = find_concepts_by_patterns(
            concepts, MEDIASTINAL_PATTERNS
        )
        mask = torch.zeros(num_concepts, dtype=torch.bool)
        mask[mediastinal_indices] = True

        stats = {
            'mode': 'preserve',
            'total_concepts': num_concepts,
            'kept_concepts': len(mediastinal_indices),
            'removed_concepts': num_concepts - len(mediastinal_indices),
            'preserved_pattern': 'mediastinal',
            'example_preserved': mediastinal_matched[:10],
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return mask, stats


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

    # Move tensors to device once (not in every batch loop)
    if keep_mask is not None:
        keep_mask = keep_mask.to(device)
    concept_embeddings_device = concept_embeddings.to(device)

    # MIMIC Train (pre-encoded)
    print("\n--- Loading MIMIC Train ---")
    with h5py.File(MIMIC_FEATURES_PATH, 'r') as f:
        img_features = torch.tensor(f['cxr_feature'][:]).float()
    img_features = img_features / img_features.norm(dim=-1, keepdim=True)
    df = pd.read_csv(MIMIC_LABELS_PATH)

    llm_features = []
    for i in tqdm(range(0, len(img_features), 500), desc="Computing LLM features"):
        batch = img_features[i:i+500].to(device)
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings_device, keep_mask)
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
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings_device, keep_mask)
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
        feat = compute_llm_features(batch, clip_concept_features, concept_embeddings_device, keep_mask)
        llm_features.append(feat.cpu())
    features['test'] = torch.cat(llm_features).numpy()
    labels['test'] = (df_test['Enlarged Cardiomediastinum'] == 1).values.astype(np.float32)

    return features, labels


def train_and_evaluate(features, labels, lr=2e-4, seed=42):
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
        test_preds = torch.cat(test_preds).numpy().flatten()
        test_targets = torch.cat(test_targets).numpy().flatten()
    test_auc = roc_auc_score(test_targets, test_preds)

    return {
        'val_auc': float(best_val_auc),
        'test_auc': float(test_auc),
        'y_true': test_targets,
        'y_pred': test_preds
    }


def run_experiment(mode, clip_model, concepts, concept_embeddings, clip_concept_features,
                   num_seeds=20, lr=2e-4):
    """Run experiment for a specific mode"""

    # Create concept mask
    keep_mask, mask_stats = create_concept_mask(concepts, mode)

    print(f"\n{'='*60}")
    print(f"Mode: {mode.upper()}")
    print(f"{'='*60}")
    print(f"Total concepts: {mask_stats['total_concepts']}")
    print(f"Kept concepts: {mask_stats['kept_concepts']}")
    print(f"Removed concepts: {mask_stats['removed_concepts']}")

    if 'example_removed' in mask_stats:
        print(f"Example removed: {mask_stats['example_removed']}")
    if 'example_preserved' in mask_stats:
        print(f"Example preserved: {mask_stats['example_preserved']}")

    # Extract features
    print(f"\nExtracting features for {mode} mode...")
    features, labels = extract_all_features(
        clip_model, concepts, concept_embeddings, clip_concept_features,
        keep_mask=keep_mask if mode != 'baseline' else None  # Device transfer handled inside
    )

    # Run with multiple seeds
    print(f"\nRunning {num_seeds} seeds with lr={lr}...")
    seeds = list(range(42, 42 + num_seeds))
    results = []

    for i, seed in enumerate(seeds):
        result = train_and_evaluate(features, labels, lr=lr, seed=seed)
        results.append(result)
        print(f"  Seed {seed}: Val AUC={result['val_auc']:.4f}, Test AUC={result['test_auc']:.4f}")

    # Aggregate
    test_aucs = [r['test_auc'] for r in results]
    val_aucs = [r['val_auc'] for r in results]

    # Prepare JSON-safe results (without numpy arrays)
    results_json = [
        {'val_auc': r['val_auc'], 'test_auc': r['test_auc']}
        for r in results
    ]

    summary = {
        'mode': mode,
        'mask_stats': mask_stats,
        'learning_rate': lr,
        'num_seeds': num_seeds,
        'seeds': seeds,
        'test_auc_mean': float(np.mean(test_aucs)),
        'test_auc_std': float(np.std(test_aucs)),
        'test_aucs': test_aucs,
        'val_auc_mean': float(np.mean(val_aucs)),
        'val_auc_std': float(np.std(val_aucs)),
        'val_aucs': val_aucs,
        'all_results': results_json,
    }

    # Collect predictions for all seeds
    predictions = {
        'y_true': results[0]['y_true'],  # same for all seeds
        'y_preds': np.stack([r['y_pred'] for r in results]),  # (num_seeds, n_samples)
        'aucs': np.array(test_aucs),
    }

    print(f"\n{mode.upper()} Summary:")
    print(f"  Test AUC: {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
    print(f"  Val AUC: {summary['val_auc_mean']:.4f} ± {summary['val_auc_std']:.4f}")

    return summary, predictions, features, labels


def load_existing_baseline():
    """Load baseline results from previous experiment if available"""
    baseline_file = 'results/intervention_A_concept_removal/results.json'
    if os.path.exists(baseline_file):
        with open(baseline_file, 'r') as f:
            data = json.load(f)
        if 'baseline' in data:
            baseline = data['baseline']
            # Total concepts = kept + removed in intervention
            total_concepts = 368294  # From mimic_concepts.csv
            return {
                'mode': 'baseline',
                'mask_stats': {
                    'mode': 'baseline',
                    'total_concepts': total_concepts,
                    'kept_concepts': total_concepts,
                    'removed_concepts': 0,
                },
                'learning_rate': 2e-4,
                'num_seeds': data.get('num_seeds', 20),
                'seeds': data.get('seeds', list(range(42, 62))),
                'test_auc_mean': baseline['test_auc_mean'],
                'test_auc_std': baseline['test_auc_std'],
                'test_aucs': baseline['test_aucs'],
                'val_auc_mean': float(np.mean([r['val_auc'] for r in baseline['all_results']])),
                'val_auc_std': float(np.std([r['val_auc'] for r in baseline['all_results']])),
                'val_aucs': [r['val_auc'] for r in baseline['all_results']],
                'all_results': baseline['all_results'],
            }
    return None


def main():
    parser = argparse.ArgumentParser(description='Concept Selection Intervention Experiment')
    parser.add_argument('--mode', type=str, choices=['exclude', 'preserve', 'both', 'all'],
                        default='both',
                        help='Mode: exclude (remove atelectasis), preserve (keep mediastinal), '
                             'both (exclude + preserve), or all (baseline + exclude + preserve)')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate (default: 2e-4, same as baseline)')
    parser.add_argument('--num_seeds', type=int, default=20,
                        help='Number of random seeds (default: 20)')
    parser.add_argument('--lr_search', action='store_true',
                        help='Run learning rate search for preserve mode')
    parser.add_argument('--recompute_baseline', action='store_true',
                        help='Force recompute baseline even if cached results exist')
    args = parser.parse_args()

    results_dir = 'results/intervention_A_concept_selection'
    os.makedirs(results_dir, exist_ok=True)

    print("="*60)
    print("EXPERIMENT: Concept Selection Intervention for EC Classification")
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
    print(f"Loaded {len(concepts)} concepts")

    # Load CLIP concept features
    with open(CLIP_CONCEPT_CACHE, 'rb') as f:
        cache = pickle.load(f)
    clip_concept_features = cache['concept_features'].to(device)

    # Determine which modes to run
    if args.mode == 'both':
        modes = ['exclude', 'preserve']
    elif args.mode == 'all':
        modes = ['baseline', 'exclude', 'preserve']
    else:
        modes = [args.mode]

    all_results = {}
    all_predictions = {}

    # Always need baseline for comparison - load cached or compute
    if 'baseline' not in modes:
        cached_baseline = None if args.recompute_baseline else load_existing_baseline()

        if cached_baseline is not None:
            print("\n" + "="*60)
            print("Loading CACHED BASELINE from previous experiment...")
            print("="*60)
            print(f"  Test AUC: {cached_baseline['test_auc_mean']:.4f} ± {cached_baseline['test_auc_std']:.4f}")
            print(f"  Loaded from: results/intervention_A_concept_removal/results.json")
            all_results['baseline'] = cached_baseline
            # Try to load cached baseline predictions
            baseline_pred_file = 'results/intervention_A_concept_removal/predictions.npz'
            if os.path.exists(baseline_pred_file):
                baseline_pred = np.load(baseline_pred_file)
                all_predictions['baseline'] = {
                    'y_true': baseline_pred['y_true'],
                    'y_preds': baseline_pred['baseline_y_preds'],
                    'aucs': baseline_pred['baseline_aucs'],
                }
        else:
            print("\n" + "="*60)
            print("Running BASELINE for comparison...")
            print("="*60)
            baseline_summary, baseline_preds, _, _ = run_experiment(
                'baseline', clip_model, concepts, concept_embeddings,
                clip_concept_features, num_seeds=args.num_seeds, lr=args.lr
            )
            all_results['baseline'] = baseline_summary
            all_predictions['baseline'] = baseline_preds

    # Run requested modes
    for mode in modes:
        summary, preds, _, _ = run_experiment(
            mode, clip_model, concepts, concept_embeddings,
            clip_concept_features, num_seeds=args.num_seeds, lr=args.lr
        )
        all_results[mode] = summary
        all_predictions[mode] = preds

    # Learning rate search for preserve mode (if requested)
    if args.lr_search and 'preserve' in modes:
        print("\n" + "="*60)
        print("Learning Rate Search for PRESERVE mode")
        print("="*60)

        lr_results = {}
        for lr in [1e-4, 2e-4, 5e-4, 1e-3, 2e-3]:
            print(f"\nTrying lr={lr}...")
            summary, _, _, _ = run_experiment(
                'preserve', clip_model, concepts, concept_embeddings,
                clip_concept_features, num_seeds=5, lr=lr  # Fewer seeds for search
            )
            lr_results[str(lr)] = {
                'val_auc_mean': summary['val_auc_mean'],
                'test_auc_mean': summary['test_auc_mean'],
            }
            print(f"  lr={lr}: Val={summary['val_auc_mean']:.4f}, Test={summary['test_auc_mean']:.4f}")

        # Find best LR based on validation
        best_lr = max(lr_results.keys(), key=lambda k: lr_results[k]['val_auc_mean'])
        print(f"\nBest LR (by validation): {best_lr}")

        all_results['lr_search'] = {
            'results': lr_results,
            'best_lr': best_lr,
        }

        # Re-run preserve with best LR if different from default
        if float(best_lr) != args.lr:
            print(f"\nRe-running PRESERVE with best lr={best_lr}...")
            summary, preds, _, _ = run_experiment(
                'preserve', clip_model, concepts, concept_embeddings,
                clip_concept_features, num_seeds=args.num_seeds, lr=float(best_lr)
            )
            all_results['preserve_best_lr'] = summary
            all_predictions['preserve_best_lr'] = preds

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    baseline_auc = all_results['baseline']['test_auc_mean']
    print(f"\nBaseline (all {all_results['baseline']['mask_stats']['total_concepts']} concepts):")
    print(f"  Test AUC: {baseline_auc:.4f} ± {all_results['baseline']['test_auc_std']:.4f}")

    for mode in modes:
        if mode == 'baseline':
            continue
        summary = all_results[mode]
        improvement = summary['test_auc_mean'] - baseline_auc
        print(f"\n{mode.upper()} ({summary['mask_stats']['kept_concepts']} concepts):")
        print(f"  Test AUC: {summary['test_auc_mean']:.4f} ± {summary['test_auc_std']:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")

    # Save results
    output_file = f'{results_dir}/results_{args.mode}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Save predictions for ROC curve plotting (all seeds)
    if all_predictions:
        pred_data = {
            'seeds': np.array(list(range(42, 42 + args.num_seeds))),
        }
        # Get y_true from any available mode (same for all)
        for mode_name in all_predictions:
            if 'y_true' in all_predictions[mode_name]:
                pred_data['y_true'] = all_predictions[mode_name]['y_true']
                break

        # Add predictions for each mode
        for mode_name, preds in all_predictions.items():
            pred_data[f'{mode_name}_y_preds'] = preds['y_preds']
            pred_data[f'{mode_name}_aucs'] = preds['aucs']

        pred_file = f'{results_dir}/predictions_{args.mode}.npz'
        np.savez(pred_file, **pred_data)
        print(f"Predictions saved to: {pred_file}")
        print(f"  Contains: {list(pred_data.keys())}")


if __name__ == "__main__":
    main()
