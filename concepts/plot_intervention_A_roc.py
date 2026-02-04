#!/usr/bin/env python3
"""
Plot ROC curves for Intervention A: Concept Removal Experiment
Compares baseline (with atelectasis concepts) vs intervention (atelectasis removed)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('default')
sns.set_palette("deep")


def plot_roc_with_seed_variability():
    """Plot ROC curves comparing baseline vs intervention with seed variability"""

    # Load data
    results_dir = Path("results/intervention_A_concept_removal_lr5e-4")
    data = np.load(results_dir / "predictions.npz", allow_pickle=True)

    seeds = data['seeds']
    y_true = data['y_true']
    baseline_y_preds = data['baseline_y_preds']  # (20, 500)
    baseline_aucs = data['baseline_aucs']
    intervention_y_preds = data['intervention_y_preds']  # (20, 500)
    intervention_aucs = data['intervention_aucs']
    best_seed_idx = int(data['best_seed_idx'])

    print(f"Loaded data: {len(seeds)} seeds, {len(y_true)} samples")
    print(f"Baseline AUC: {np.mean(baseline_aucs):.4f} ± {np.std(baseline_aucs):.4f}")
    print(f"Intervention AUC: {np.mean(intervention_aucs):.4f} ± {np.std(intervention_aucs):.4f}")
    print(f"Improvement: {np.mean(intervention_aucs) - np.mean(baseline_aucs):+.4f}")

    # Create figure with white background and 1:1 aspect ratio
    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # Colors
    colors = ['steelblue', 'indianred']

    # Common FPR points for interpolation
    fpr_grid = np.linspace(0, 1, 100)

    # Calculate ROC curves for each seed and interpolate
    baseline_tprs = []
    intervention_tprs = []

    for i in range(len(seeds)):
        # Baseline
        fpr_b, tpr_b, _ = roc_curve(y_true, baseline_y_preds[i])
        tpr_interp_b = np.interp(fpr_grid, fpr_b, tpr_b)
        baseline_tprs.append(tpr_interp_b)

        # Intervention
        fpr_i, tpr_i, _ = roc_curve(y_true, intervention_y_preds[i])
        tpr_interp_i = np.interp(fpr_grid, fpr_i, tpr_i)
        intervention_tprs.append(tpr_interp_i)

    baseline_tprs = np.array(baseline_tprs)
    intervention_tprs = np.array(intervention_tprs)

    # Calculate mean and confidence intervals
    baseline_tpr_mean = np.mean(baseline_tprs, axis=0)
    baseline_tpr_lower = np.percentile(baseline_tprs, 2.5, axis=0)
    baseline_tpr_upper = np.percentile(baseline_tprs, 97.5, axis=0)

    intervention_tpr_mean = np.mean(intervention_tprs, axis=0)
    intervention_tpr_lower = np.percentile(intervention_tprs, 2.5, axis=0)
    intervention_tpr_upper = np.percentile(intervention_tprs, 97.5, axis=0)

    # AUC statistics
    baseline_auc_mean = np.mean(baseline_aucs)
    intervention_auc_mean = np.mean(intervention_aucs)

    # Plot baseline ROC
    ax.plot(fpr_grid, baseline_tpr_mean, color=colors[0], linewidth=3, alpha=1,
            label=f'Baseline (AUC = {baseline_auc_mean:.3f})')
    ax.fill_between(fpr_grid, baseline_tpr_lower, baseline_tpr_upper,
                    color=colors[0], alpha=0.2)

    # Plot intervention ROC
    ax.plot(fpr_grid, intervention_tpr_mean, color=colors[1], linewidth=3, alpha=1,
            label=f'Atelectasis Removed (AUC = {intervention_auc_mean:.3f})')
    ax.fill_between(fpr_grid, intervention_tpr_lower, intervention_tpr_upper,
                    color=colors[1], alpha=0.2)

    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)

    # Customize plot
    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.05))
    ax.set_xlabel('False positive rate', fontsize=34)
    ax.set_ylabel('True positive rate', fontsize=34)
    ax.set_title('Enlarged Cardiomediastinum', fontsize=34, pad=10)

    # Larger tick labels
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Customize legend
    ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True,
             fontsize=22, framealpha=0.9)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure
    plt.savefig(results_dir / "roc_curves.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', format='png')
    plt.savefig(results_dir / "roc_curves.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"\nPlots saved to {results_dir}/roc_curves.png and .pdf")


if __name__ == "__main__":
    plot_roc_with_seed_variability()
