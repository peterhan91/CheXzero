#!/usr/bin/env python3
"""
Plot ROC curves for Intervention A: Concept Selection Experiment
Compares baseline (all concepts) vs preserve (mediastinal concepts only)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import json
import pickle
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("deep")


def load_baseline_from_linear_probing():
    """Load EC baseline from concept_based_linear_probing_torch"""
    baseline_dir = Path("results/concept_based_linear_probing_torch")

    with open(baseline_dir / "aggregated_results.json", 'r') as f:
        data = json.load(f)

    ec_stats = data['per_label_stats']['Enlarged Cardiomediastinum']
    seeds = data['seeds']
    ec_idx = 4  # Index of 'Enlarged Cardiomediastinum'

    # Load predictions
    y_true = None
    y_preds = []

    for seed in seeds:
        pred_file = baseline_dir / "predictions" / f"seed_{seed}_predictions.pkl"
        with open(pred_file, 'rb') as f:
            pred_data = pickle.load(f)
        if y_true is None:
            y_true = np.array(pred_data['test']['y_true'])[:, ec_idx]
        y_preds.append(np.array(pred_data['test']['y_pred'])[:, ec_idx])

    return y_true, np.stack(y_preds), np.array(ec_stats['aucs'])


def plot_roc_with_seed_variability():
    """Plot ROC curves comparing baseline vs preserve (mediastinal only)"""

    # Load preserve data
    results_dir = Path("results/intervention_A_concept_selection")
    data = np.load(results_dir / "predictions_preserve.npz", allow_pickle=True)

    with open(results_dir / "results_preserve.json", 'r') as f:
        results_json = json.load(f)

    seeds = data['seeds']
    y_true = data['y_true']
    preserve_y_preds = data['preserve_y_preds']  # (20, 500)
    preserve_aucs = data['preserve_aucs']

    # Load baseline - use predictions_preserve.npz if it has baseline, otherwise fall back
    if 'baseline_y_preds' in data and 'baseline_aucs' in data:
        baseline_y_preds = data['baseline_y_preds']
        baseline_aucs = data['baseline_aucs']
    else:
        _, baseline_y_preds, baseline_aucs = load_baseline_from_linear_probing()

    print(f"Loaded data: {len(seeds)} seeds, {len(y_true)} samples")
    print(f"Baseline AUC: {np.mean(baseline_aucs):.4f} ± {np.std(baseline_aucs):.4f}")
    print(f"Preserve (mediastinal) AUC: {np.mean(preserve_aucs):.4f} ± {np.std(preserve_aucs):.4f}")
    print(f"Improvement: {np.mean(preserve_aucs) - np.mean(baseline_aucs):+.4f}")
    print(f"\nConcept stats:")
    print(f"  Total concepts: {results_json['preserve']['mask_stats']['total_concepts']:,}")
    print(f"  Kept concepts: {results_json['preserve']['mask_stats']['kept_concepts']:,}")
    print(f"  Removed concepts: {results_json['preserve']['mask_stats']['removed_concepts']:,}")

    # Create figure with white background and 1:1 aspect ratio
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # Colors
    colors = ['steelblue', 'indianred']

    # Common FPR points for interpolation
    fpr_grid = np.linspace(0, 1, 100)

    # Calculate ROC curves for baseline and preserve
    baseline_tprs = []
    preserve_tprs = []
    for i in range(len(seeds)):
        # Baseline
        fpr_b, tpr_b, _ = roc_curve(y_true, baseline_y_preds[i])
        tpr_interp_b = np.interp(fpr_grid, fpr_b, tpr_b)
        baseline_tprs.append(tpr_interp_b)

        # Preserve
        fpr_p, tpr_p, _ = roc_curve(y_true, preserve_y_preds[i])
        tpr_interp_p = np.interp(fpr_grid, fpr_p, tpr_p)
        preserve_tprs.append(tpr_interp_p)

    baseline_tprs = np.array(baseline_tprs)
    preserve_tprs = np.array(preserve_tprs)

    # Calculate mean and confidence intervals
    baseline_tpr_mean = np.mean(baseline_tprs, axis=0)
    baseline_tpr_lower = np.percentile(baseline_tprs, 2.5, axis=0)
    baseline_tpr_upper = np.percentile(baseline_tprs, 97.5, axis=0)

    preserve_tpr_mean = np.mean(preserve_tprs, axis=0)
    preserve_tpr_lower = np.percentile(preserve_tprs, 2.5, axis=0)
    preserve_tpr_upper = np.percentile(preserve_tprs, 97.5, axis=0)

    # AUC statistics
    baseline_auc_mean = np.mean(baseline_aucs)
    preserve_auc_mean = np.mean(preserve_aucs)

    # Plot baseline ROC
    ax.plot(fpr_grid, baseline_tpr_mean, color=colors[0], linewidth=3, alpha=1,
            label=f'CLEAR (all concepts AUC = {baseline_auc_mean:.3f})')
    ax.fill_between(fpr_grid, baseline_tpr_lower, baseline_tpr_upper,
                    color=colors[0], alpha=0.2)

    # Plot preserve ROC
    ax.plot(fpr_grid, preserve_tpr_mean, color=colors[1], linewidth=3, alpha=1,
            label=f'CLEAR (mediastinal only AUC = {preserve_auc_mean:.3f})')
    ax.fill_between(fpr_grid, preserve_tpr_lower, preserve_tpr_upper,
                    color=colors[1], alpha=0.2)

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    # Customize plot
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False Positive Rate', fontsize=30)
    ax.set_ylabel('True Positive Rate', fontsize=30)
    ax.set_title('Enlarged Cardiomediastinum\n(concept selection intervention)', fontsize=30, pad=10)

    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Customize legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             fontsize=20, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure
    plt.savefig(results_dir / "roc_curves_preserve.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.savefig(results_dir / "roc_curves_preserve.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nPlots saved to {results_dir}/roc_curves_preserve.png and .pdf")


if __name__ == "__main__":
    plot_roc_with_seed_variability()
