#!/usr/bin/env python3
"""
Calibration Analysis for CXR Models
Computes Expected Calibration Error (ECE), Brier Score, and generates reliability diagrams.

This script analyzes model calibration using saved predictions and ground truth.

Usage:
    python calibration_analysis.py --dataset vindrcxr --model sfr_mistral
    python calibration_analysis.py --dataset chexpert --model chexzero
    python calibration_analysis.py --all  # Run for all available model/dataset combinations
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the average gap between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.

    Args:
        y_true: Ground truth binary labels (n_samples,)
        y_prob: Predicted probabilities (n_samples,)
        n_bins: Number of bins for calibration

    Returns:
        ece: Expected Calibration Error
        bin_accuracies: Accuracy in each bin
        bin_confidences: Mean confidence in each bin
        bin_counts: Number of samples in each bin
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        bin_counts[i] = in_bin.sum()

        if bin_counts[i] > 0:
            bin_accuracies[i] = y_true[in_bin].mean()
            bin_confidences[i] = y_prob[in_bin].mean()

    # ECE is weighted average of |accuracy - confidence| per bin
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / len(y_true)

    return ece, bin_accuracies, bin_confidences, bin_counts


def maximum_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).

    MCE is the maximum gap between predicted confidence and actual accuracy across all bins.
    """
    _, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)

    # Only consider bins with samples
    valid_bins = bin_counts > 0
    if not valid_bins.any():
        return 0.0

    mce = np.max(np.abs(bin_accuracies[valid_bins] - bin_confidences[valid_bins]))
    return mce


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Score.

    Brier score is the mean squared error between predicted probabilities and true outcomes.
    Lower is better. Range: [0, 1]
    """
    return np.mean((y_prob - y_true) ** 2)


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier Skill Score (BSS).

    BSS compares model's Brier score to a reference (climatology) forecast.
    BSS = 1 - (BS_model / BS_reference)
    Range: (-inf, 1], where 1 is perfect, 0 is no skill, negative is worse than reference.
    """
    bs_model = brier_score(y_true, y_prob)
    # Reference is predicting the base rate for all samples
    base_rate = y_true.mean()
    bs_ref = brier_score(y_true, np.full_like(y_prob, base_rate))

    if bs_ref == 0:
        return 0.0
    return 1 - (bs_model / bs_ref)


def plot_reliability_diagram(y_true: np.ndarray, y_prob: np.ndarray,
                             n_bins: int = 10,
                             title: str = "Reliability Diagram",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot reliability diagram (calibration curve).

    A perfectly calibrated model would have all points on the diagonal.
    """
    ece, bin_accuracies, bin_confidences, bin_counts = expected_calibration_error(y_true, y_prob, n_bins)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

    # Main reliability diagram
    bin_centers = np.linspace(0.05, 0.95, n_bins)
    valid_bins = bin_counts > 0

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.bar(bin_centers, bin_accuracies, width=0.08, alpha=0.7,
            color='steelblue', edgecolor='black', label='Model')

    # Add gap visualization
    for i, (center, acc, conf, count) in enumerate(zip(bin_centers, bin_accuracies, bin_confidences, bin_counts)):
        if count > 0:
            gap = acc - conf
            color = 'green' if gap >= 0 else 'red'
            ax1.plot([center, center], [conf, acc], color=color, linewidth=2, alpha=0.7)

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title(f'{title}\nECE = {ece:.4f}', fontsize=14)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), alpha=0.7,
             color='steelblue', edgecolor='black')
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xlim([0, 1])
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")

    return fig


def load_predictions_and_ground_truth(results_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the most recent predictions and ground truth from a results directory.
    """
    pred_files = sorted(Path(results_dir).glob("predictions_*.csv"))
    gt_files = sorted(Path(results_dir).glob("ground_truth_*.csv"))

    if not pred_files or not gt_files:
        raise FileNotFoundError(f"No prediction/ground truth files found in {results_dir}")

    # Use the most recent files
    pred_df = pd.read_csv(pred_files[-1])
    gt_df = pd.read_csv(gt_files[-1])

    return pred_df, gt_df


def analyze_calibration_for_model(results_dir: str,
                                  output_dir: str,
                                  model_name: str,
                                  dataset_name: str,
                                  n_bins: int = 10) -> Dict:
    """
    Analyze calibration for a single model on a dataset.
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {model_name} on {dataset_name}")
    print('='*60)

    try:
        pred_df, gt_df = load_predictions_and_ground_truth(results_dir)
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    # Extract label names
    pred_cols = [col.replace('_pred', '') for col in pred_df.columns if col.endswith('_pred')]

    results = {
        'model': model_name,
        'dataset': dataset_name,
        'labels': {},
        'overall': {}
    }

    all_y_true = []
    all_y_prob = []

    # Compute calibration metrics for each label
    for label in pred_cols:
        pred_col = f"{label}_pred"
        true_col = f"{label}_true"

        if pred_col not in pred_df.columns or true_col not in gt_df.columns:
            continue

        y_prob = pred_df[pred_col].values
        y_true = gt_df[true_col].values

        # Skip if no positive samples
        if y_true.sum() == 0:
            continue

        # Compute metrics
        ece, _, _, _ = expected_calibration_error(y_true, y_prob, n_bins)
        mce = maximum_calibration_error(y_true, y_prob, n_bins)
        bs = brier_score(y_true, y_prob)
        bss = brier_skill_score(y_true, y_prob)

        results['labels'][label] = {
            'n_positive': int(y_true.sum()),
            'n_total': len(y_true),
            'prevalence': float(y_true.mean()),
            'ece': float(ece),
            'mce': float(mce),
            'brier_score': float(bs),
            'brier_skill_score': float(bss)
        }

        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

        print(f"  {label}: ECE={ece:.4f}, Brier={bs:.4f}")

    # Compute overall metrics (pooled across all labels)
    if all_y_true:
        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)

        overall_ece, _, _, _ = expected_calibration_error(all_y_true, all_y_prob, n_bins)
        overall_mce = maximum_calibration_error(all_y_true, all_y_prob, n_bins)
        overall_bs = brier_score(all_y_true, all_y_prob)
        overall_bss = brier_skill_score(all_y_true, all_y_prob)

        results['overall'] = {
            'ece': float(overall_ece),
            'mce': float(overall_mce),
            'brier_score': float(overall_bs),
            'brier_skill_score': float(overall_bss),
            'n_predictions': len(all_y_true)
        }

        print(f"\n  OVERALL: ECE={overall_ece:.4f}, MCE={overall_mce:.4f}, Brier={overall_bs:.4f}")

        # Generate reliability diagram
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f"reliability_diagram_{model_name}_{dataset_name}.png")
        plot_reliability_diagram(
            all_y_true, all_y_prob, n_bins,
            title=f"{model_name} on {dataset_name}",
            save_path=plot_path
        )

    return results


def get_available_models_and_datasets(base_dir: str) -> List[Tuple[str, str, str]]:
    """
    Scan the results directory and find all available model/dataset combinations.

    Returns list of (model_name, dataset_name, results_dir) tuples.
    """
    results_dir = Path(base_dir) / "results"
    available = []

    for subdir in results_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("benchmark_evaluation_"):
            # Parse directory name: benchmark_evaluation_{dataset}_{model}
            parts = subdir.name.replace("benchmark_evaluation_", "").split("_test_")
            if len(parts) == 2:
                dataset = parts[0]
                model = parts[1]

                # Check if predictions exist
                pred_files = list(subdir.glob("predictions_*.csv"))
                gt_files = list(subdir.glob("ground_truth_*.csv"))

                if pred_files and gt_files:
                    available.append((model, dataset, str(subdir)))

    # Also check for CLEAR (concept-based) results
    concepts_results = Path(base_dir).parent / "concepts" / "results"
    # Known LLM model suffixes for CLEAR
    llm_models = ["sfr_mistral", "qwen3_8b", "openai_small", "biomedbert"]

    if concepts_results.exists():
        for subdir in concepts_results.iterdir():
            if subdir.is_dir() and subdir.name.startswith("concept_based_evaluation_"):
                name = subdir.name.replace("concept_based_evaluation_", "")

                # Find which LLM model this is
                model = None
                dataset = None
                for llm in llm_models:
                    if name.endswith(f"_{llm}"):
                        model = llm
                        dataset = name[:-len(f"_{llm}")]
                        break

                if model and dataset:
                    pred_files = list(subdir.glob("predictions_*.csv"))
                    gt_files = list(subdir.glob("ground_truth_*.csv"))

                    if pred_files and gt_files:
                        # Use "CLEAR" for sfr_mistral (main model), otherwise specify variant
                        display_name = "CLEAR" if model == "sfr_mistral" else f"CLEAR_{model}"
                        available.append((display_name, dataset, str(subdir)))

    return available


def create_summary_table(all_results: List[Dict], output_dir: str) -> pd.DataFrame:
    """
    Create a summary table of calibration metrics across all models and datasets.
    """
    rows = []

    for result in all_results:
        if result is None:
            continue

        row = {
            'Model': result['model'],
            'Dataset': result['dataset'],
            'ECE': result['overall'].get('ece', np.nan),
            'MCE': result['overall'].get('mce', np.nan),
            'Brier Score': result['overall'].get('brier_score', np.nan),
            'Brier Skill Score': result['overall'].get('brier_skill_score', np.nan),
            'N Predictions': result['overall'].get('n_predictions', 0)
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save summary
    summary_path = os.path.join(output_dir, "calibration_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to: {summary_path}")

    return df


def create_latex_table(summary_df: pd.DataFrame, output_dir: str) -> str:
    """
    Create a LaTeX table of calibration results for the paper.
    """
    # Pivot table: rows = models, columns = datasets
    datasets = summary_df['Dataset'].unique()
    models = summary_df['Model'].unique()

    latex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Calibration metrics across models and datasets. ECE = Expected Calibration Error, Brier = Brier Score. Lower values indicate better calibration.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{l" + "cc" * len(datasets) + "}",
        r"\toprule",
    ]

    # Header row
    header = r"\textbf{Model}"
    for dataset in datasets:
        header += f" & \\multicolumn{{2}}{{c}}{{\\textbf{{{dataset}}}}}"
    header += r" \\"
    latex_lines.append(header)

    # Sub-header
    subheader = ""
    for _ in datasets:
        subheader += " & ECE & Brier"
    subheader += r" \\"
    latex_lines.append(subheader)
    latex_lines.append(r"\midrule")

    # Data rows
    for model in models:
        row = model.replace("_", r"\_")
        for dataset in datasets:
            mask = (summary_df['Model'] == model) & (summary_df['Dataset'] == dataset)
            if mask.any():
                ece = summary_df.loc[mask, 'ECE'].values[0]
                brier = summary_df.loc[mask, 'Brier Score'].values[0]
                row += f" & {ece:.3f} & {brier:.3f}"
            else:
                row += " & -- & --"
        row += r" \\"
        latex_lines.append(row)

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}"
    ])

    latex_table = "\n".join(latex_lines)

    # Save LaTeX table
    latex_path = os.path.join(output_dir, "calibration_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"LaTeX table saved to: {latex_path}")

    return latex_table


def main():
    parser = argparse.ArgumentParser(description='Calibration Analysis for CXR Models')
    parser.add_argument('--dataset', type=str, help='Dataset name (e.g., vindrcxr, chexpert, padchest, indiana)')
    parser.add_argument('--model', type=str, help='Model name (e.g., sfr_mistral, chexzero, biomedclip)')
    parser.add_argument('--all', action='store_true', help='Run for all available model/dataset combinations')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins for ECE calculation')
    parser.add_argument('--output_dir', type=str, default='results/calibration', help='Output directory')

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    if args.all:
        # Run for all available combinations
        available = get_available_models_and_datasets(base_dir)

        if not available:
            print("No model/dataset combinations found with predictions!")
            print("\nExpected structure:")
            print("  benchmark/results/benchmark_evaluation_{dataset}_test_{model}/")
            print("    - predictions_*.csv")
            print("    - ground_truth_*.csv")
            return

        print(f"Found {len(available)} model/dataset combinations:")
        for model, dataset, _ in available:
            print(f"  - {model} on {dataset}")

        for model, dataset, results_dir in available:
            result = analyze_calibration_for_model(
                results_dir=results_dir,
                output_dir=output_dir,
                model_name=model,
                dataset_name=dataset,
                n_bins=args.n_bins
            )
            if result:
                all_results.append(result)

    elif args.dataset and args.model:
        # Run for specific model/dataset
        # Try benchmark results first
        results_dir = os.path.join(base_dir, "results",
                                   f"benchmark_evaluation_{args.dataset}_test_{args.model}")

        if not os.path.exists(results_dir):
            # Try concept-based results
            results_dir = os.path.join(base_dir, "..", "concepts", "results",
                                      f"concept_based_evaluation_{args.dataset}_{args.model}")

        if not os.path.exists(results_dir):
            print(f"ERROR: Results directory not found!")
            print(f"Tried:")
            print(f"  - benchmark/results/benchmark_evaluation_{args.dataset}_test_{args.model}")
            print(f"  - concepts/results/concept_based_evaluation_{args.dataset}_{args.model}")
            return

        result = analyze_calibration_for_model(
            results_dir=results_dir,
            output_dir=output_dir,
            model_name=args.model,
            dataset_name=args.dataset,
            n_bins=args.n_bins
        )
        if result:
            all_results.append(result)

    else:
        parser.print_help()
        print("\n\nAvailable model/dataset combinations:")
        available = get_available_models_and_datasets(base_dir)
        for model, dataset, _ in available:
            print(f"  --dataset {dataset} --model {model}")
        return

    # Create summary
    if all_results:
        print("\n" + "="*60)
        print("CALIBRATION ANALYSIS SUMMARY")
        print("="*60)

        summary_df = create_summary_table(all_results, output_dir)
        print("\n" + summary_df.to_string(index=False))

        # Create LaTeX table
        latex_table = create_latex_table(summary_df, output_dir)

        # Save detailed results as JSON
        json_path = os.path.join(output_dir, f"calibration_results_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {json_path}")


if __name__ == "__main__":
    main()
