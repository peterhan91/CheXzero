#!/usr/bin/env python3
"""
Statistical Comparison Analysis for CXR Models
Computes 95% confidence intervals for AUROC, F1, AUPRC and p-values
comparing SFR-Mistral concept model against benchmark models.
"""

import os
import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Dataset configurations
DATASET_CONFIGS = {
    'padchest': {
        'csv_path': '../data/padchest_test.csv',
        'phenotypes_path': '../data/pheotypes_padchest.json',
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_padchest_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_padchest_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_padchest_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_padchest_test_openai_clip",
            "CXR_Foundation": "results/benchmark_evaluation_padchest_test_cxr_foundation"
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_padchest_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_padchest_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_padchest_openai_small"
        },
        'exclude_columns': ['Study', 'Path', 'image_id', 'ImageID', 'name', 'is_test']
    },
    'indiana': {
        'csv_path': '../data/indiana_test.csv',
        'phenotypes_path': None,  # Will use all available labels above threshold
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_indiana_test_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_indiana_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_indiana_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_indiana_test_openai_clip",
            "CXR_Foundation": "results/benchmark_evaluation_indiana_test_cxr_foundation"
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_indiana_test_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_indiana_test_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_indiana_test_openai_small"
        },
        'exclude_columns': ['uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 
                           'indication', 'comparison', 'findings', 'impression']
    },
    'vindrcxr': {
        'csv_path': '../data/vindrcxr_test.csv',
        'phenotypes_path': None,  # Will use all available labels above threshold
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_vindrcxr_sfr_mistral',
        'benchmark_dirs': {
            "CheXzero": "results/benchmark_evaluation_vindrcxr_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_vindrcxr_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_vindrcxr_test_openai_clip",
            "CXR_Foundation": "results/benchmark_evaluation_vindrcxr_test_cxr_foundation"
        },
        'llm_dirs': {
            "BiomedBERT": "../concepts/results/concept_based_evaluation_vindrcxr_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_vindrcxr_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_vindrcxr_openai_small"
        },
        'exclude_columns': ['image_id']
    },
    'vindrpcxr': {
        'csv_path': '../data/vindrpcxr_test.csv',
        'phenotypes_path': None,  # Will use all available labels above threshold
        'sfr_results_dir': '../concepts/results/concept_based_evaluation_vindrpcxr_sfr_mistral',  # Note: May not exist yet
        'benchmark_dirs': {
            # Note: These might need to be created/verified
            "CheXzero": "results/benchmark_evaluation_vindrpcxr_test_chexzero",
            "BiomedCLIP": "results/benchmark_evaluation_vindrpcxr_test_biomedclip", 
            "OpenAI_CLIP": "results/benchmark_evaluation_vindrpcxr_test_openai_clip",
            "CXR_Foundation": "results/benchmark_evaluation_vindrpcxr_test_cxr_foundation"
        },
        'llm_dirs': {
            # Note: These might need to be created/verified for vindrpcxr
            "BiomedBERT": "../concepts/results/concept_based_evaluation_vindrpcxr_biomedbert",
            "Qwen3_8B": "../concepts/results/concept_based_evaluation_vindrpcxr_qwen3_8b",
            "OpenAI_Small": "../concepts/results/concept_based_evaluation_vindrpcxr_openai_small"
        },
        'exclude_columns': ['image_id']
    }
}

def load_phenotypes(phenotypes_path):
    """Load diagnostic phenotypes from JSON file."""
    if phenotypes_path is None or not os.path.exists(phenotypes_path):
        return None
    
    with open(phenotypes_path, 'r') as f:
        data = json.load(f)
    return data['diagnostic_phenotypes']

def load_original_dataset_data(csv_path, exclude_columns):
    """Load original dataset CSV to get label counts."""
    print(f"Loading original dataset data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Get label columns (exclude metadata columns)
    label_columns = [col for col in df.columns if col not in exclude_columns]
    
    # Count positive cases for each label
    label_counts = {}
    for label in label_columns:
        if label in df.columns:
            positive_count = df[label].sum() if df[label].dtype in ['int64', 'float64'] else 0
            label_counts[label] = positive_count
    
    return label_counts, label_columns

def filter_labels_by_threshold_and_phenotypes(label_counts, phenotypes, threshold=30):
    """Filter labels based on positive count threshold and diagnostic phenotypes."""
    filtered_labels = []
    
    print(f"\nFiltering labels with threshold >= {threshold}")
    if phenotypes:
        print("and in diagnostic phenotypes:")
    else:
        print("(using all available labels):")
    print("-" * 80)
    
    # If no phenotypes provided, use all labels that meet threshold
    labels_to_check = phenotypes if phenotypes else list(label_counts.keys())
    
    for label in labels_to_check:
        if label in label_counts:
            count = label_counts[label]
            if count >= threshold:
                filtered_labels.append(label)
                print(f"✓ {label}: {count} positive cases")
            else:
                print(f"✗ {label}: {count} positive cases (below threshold)")
        else:
            print(f"✗ {label}: not found in dataset")
    
    total_labels = len(phenotypes) if phenotypes else len(label_counts)
    print(f"\nSelected {len(filtered_labels)} labels out of {total_labels} available labels")
    return filtered_labels

def load_model_results(results_dir, model_name):
    """Load model predictions and ground truth from results directory."""
    print(f"\nLoading {model_name} results from: {results_dir}")
    
    if not os.path.exists(results_dir):
        print(f"Warning: {results_dir} not found")
        return None, None, None
    
    # Find the latest results files
    pred_files = list(Path(results_dir).glob("predictions_*.csv"))
    gt_files = list(Path(results_dir).glob("ground_truth_*.csv"))
    
    if not pred_files or not gt_files:
        print(f"Warning: No prediction or ground truth files found in {results_dir}")
        return None, None, None
    
    # Use the latest files
    pred_file = sorted(pred_files)[-1]
    gt_file = sorted(gt_files)[-1]
    
    print(f"Loading predictions: {pred_file}")
    print(f"Loading ground truth: {gt_file}")
    
    predictions = pd.read_csv(pred_file)
    ground_truth = pd.read_csv(gt_file)
    
    # Extract label names by removing _pred and _true suffixes
    pred_labels = []
    for col in predictions.columns:
        if col.endswith('_pred'):
            label_name = col[:-5]  # Remove '_pred'
            pred_labels.append(label_name)
    
    gt_labels = []
    for col in ground_truth.columns:
        if col.endswith('_true'):
            label_name = col[:-5]  # Remove '_true'
            gt_labels.append(label_name)
    
    # Find common labels
    common_labels = list(set(pred_labels) & set(gt_labels))
    common_labels.sort()
    
    print(f"Found {len(common_labels)} common labels")
    
    # Create DataFrames with clean label names
    pred_clean = pd.DataFrame()
    gt_clean = pd.DataFrame()
    
    for label in common_labels:
        pred_clean[label] = predictions[f"{label}_pred"]
        gt_clean[label] = ground_truth[f"{label}_true"]
    
    return pred_clean, gt_clean, common_labels

def bootstrap_metric(y_true, y_pred, metric_func, n_bootstrap=1000, random_state=42):
    """Compute bootstrap confidence interval for a metric."""
    np.random.seed(random_state)
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if all labels are the same
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            score = metric_func(y_true_boot, y_pred_boot)
            bootstrap_scores.append(score)
        except:
            continue
    
    if len(bootstrap_scores) == 0:
        return np.nan, np.nan, np.nan
    
    bootstrap_scores = np.array(bootstrap_scores)
    mean_score = np.mean(bootstrap_scores)
    ci_lower = np.percentile(bootstrap_scores, 2.5)
    ci_upper = np.percentile(bootstrap_scores, 97.5)
    
    return mean_score, ci_lower, ci_upper

def compute_optimal_f1(y_true, y_pred):
    """Compute F1 score using optimal threshold that maximizes sens^2 + spec^2."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    # Use ROC curve thresholds and grid search for optimization
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    
    best_criterion = -1
    best_f1 = np.nan
    
    # Search over ROC curve thresholds
    all_thresholds = list(roc_thresholds)
    
    # Add fine grid search for completeness
    grid_thresholds = np.linspace(0.01, 0.99, 99)  # 0.01 to 0.99 with 0.01 steps
    all_thresholds.extend(grid_thresholds)
    
    for threshold in all_thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        # Calculate confusion matrix components
        tp = np.sum((y_true == 1) & (y_pred_binary == 1))
        tn = np.sum((y_true == 0) & (y_pred_binary == 0))
        fp = np.sum((y_true == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true == 1) & (y_pred_binary == 0))
        
        # Calculate sensitivity and specificity
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        
        # Optimize sens^2 + spec^2
        criterion = sens**2 + spec**2
        
        # If this is the best criterion so far, calculate F1 at this threshold
        if criterion > best_criterion:
            try:
                f1_at_threshold = f1_score(y_true, y_pred_binary)
                best_criterion = criterion
                best_f1 = f1_at_threshold
            except:
                continue
    
    return best_f1

def compute_auprc(y_true, y_pred):
    """Compute Area Under Precision-Recall Curve."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    return auc(recall, precision)

def compute_metrics_with_ci(y_true, y_pred, label_name, n_bootstrap=1000):
    """Compute AUROC, F1, AUPRC with 95% confidence intervals."""
    
    # Skip if all labels are the same
    if len(np.unique(y_true)) < 2:
        return {
            'label': label_name,
            'n_positive': int(np.sum(y_true)),
            'n_total': len(y_true),
            'auroc': np.nan, 'auroc_ci_lower': np.nan, 'auroc_ci_upper': np.nan,
            'f1': np.nan, 'f1_ci_lower': np.nan, 'f1_ci_upper': np.nan,
            'auprc': np.nan, 'auprc_ci_lower': np.nan, 'auprc_ci_upper': np.nan
        }
    
    # AUROC with CI
    auroc_mean, auroc_ci_lower, auroc_ci_upper = bootstrap_metric(
        y_true, y_pred, roc_auc_score, n_bootstrap
    )
    
    # F1 with CI (using optimal threshold)
    f1_mean, f1_ci_lower, f1_ci_upper = bootstrap_metric(
        y_true, y_pred, compute_optimal_f1, n_bootstrap
    )
    
    # AUPRC with CI
    auprc_mean, auprc_ci_lower, auprc_ci_upper = bootstrap_metric(
        y_true, y_pred, compute_auprc, n_bootstrap
    )
    
    return {
        'label': label_name,
        'n_positive': int(np.sum(y_true)),
        'n_total': len(y_true),
        'auroc': auroc_mean,
        'auroc_ci_lower': auroc_ci_lower,
        'auroc_ci_upper': auroc_ci_upper,
        'f1': f1_mean,
        'f1_ci_lower': f1_ci_lower,
        'f1_ci_upper': f1_ci_upper,
        'auprc': auprc_mean,
        'auprc_ci_lower': auprc_ci_lower,
        'auprc_ci_upper': auprc_ci_upper
    }

def permutation_test(y_true, y_pred1, y_pred2, metric_func, n_permutations=10000, random_state=42):
    """Perform permutation test to compare two models."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    
    np.random.seed(random_state)
    
    # Observed difference
    try:
        score1 = metric_func(y_true, y_pred1)
        score2 = metric_func(y_true, y_pred2)
        observed_diff = score1 - score2
    except:
        return np.nan
    
    # Permutation test
    permuted_diffs = []
    for _ in range(n_permutations):
        # Randomly swap predictions
        mask = np.random.choice([True, False], size=len(y_true))
        y_pred1_perm = np.where(mask, y_pred1, y_pred2)
        y_pred2_perm = np.where(mask, y_pred2, y_pred1)
        
        try:
            score1_perm = metric_func(y_true, y_pred1_perm)
            score2_perm = metric_func(y_true, y_pred2_perm)
            permuted_diffs.append(score1_perm - score2_perm)
        except:
            continue
    
    if len(permuted_diffs) == 0:
        return np.nan
    
    # Two-tailed p-value
    permuted_diffs = np.array(permuted_diffs)
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff))
    
    return p_value

def analyze_model_performance(model_data, filtered_labels, model_name):
    """Analyze performance for a single model."""
    if model_data is None:
        return None
    
    predictions, ground_truth, available_labels = model_data
    
    # Filter to only include labels that are in both filtered_labels and available_labels
    analysis_labels = [label for label in filtered_labels if label in available_labels]
    
    print(f"\nAnalyzing {model_name} performance on {len(analysis_labels)} labels...")
    
    results = []
    for label in tqdm(analysis_labels, desc=f"Analyzing {model_name}", leave=False):
        y_true = ground_truth[label].values
        y_pred = predictions[label].values
        
        metrics = compute_metrics_with_ci(y_true, y_pred, label, n_bootstrap=1000)
        results.append(metrics)
        
        if not np.isnan(metrics['auroc']):
            print(f"  {label}: AUROC={metrics['auroc']:.3f} "
                  f"[{metrics['auroc_ci_lower']:.3f}, {metrics['auroc_ci_upper']:.3f}]")
    
    return pd.DataFrame(results)

def compare_models_statistical(sfr_data, benchmark_data, filtered_labels, sfr_name, benchmark_name):
    """Perform statistical comparison between SFR-Mistral and benchmark model."""
    if sfr_data is None or benchmark_data is None:
        return None
    
    sfr_pred, sfr_gt, sfr_labels = sfr_data
    bench_pred, bench_gt, bench_labels = benchmark_data
    
    # Find common labels
    common_labels = list(set(filtered_labels) & set(sfr_labels) & set(bench_labels))
    common_labels.sort()
    
    print(f"\nStatistical comparison: {sfr_name} vs {benchmark_name}")
    print(f"Comparing on {len(common_labels)} common labels...")
    
    comparison_results = []
    
    for label in tqdm(common_labels, desc="Statistical comparison", leave=False):
        y_true = sfr_gt[label].values  # Use SFR ground truth as reference
        y_pred_sfr = sfr_pred[label].values
        y_pred_bench = bench_pred[label].values
        
        # Skip if all labels are the same
        if len(np.unique(y_true)) < 2:
            continue
        
        # Compute p-values for different metrics
        auroc_p = permutation_test(y_true, y_pred_sfr, y_pred_bench, roc_auc_score, n_permutations=1000)
        f1_p = permutation_test(y_true, y_pred_sfr, y_pred_bench, compute_optimal_f1, n_permutations=1000)
        auprc_p = permutation_test(y_true, y_pred_sfr, y_pred_bench, compute_auprc, n_permutations=1000)
        
        # Compute actual scores
        try:
            sfr_auroc = roc_auc_score(y_true, y_pred_sfr)
            bench_auroc = roc_auc_score(y_true, y_pred_bench)
            sfr_f1 = compute_optimal_f1(y_true, y_pred_sfr)
            bench_f1 = compute_optimal_f1(y_true, y_pred_bench)
            sfr_auprc = compute_auprc(y_true, y_pred_sfr)
            bench_auprc = compute_auprc(y_true, y_pred_bench)
        except:
            continue
        
        comparison_results.append({
            'label': label,
            'n_positive': int(np.sum(y_true)),
            'sfr_auroc': sfr_auroc,
            'benchmark_auroc': bench_auroc,
            'auroc_diff': sfr_auroc - bench_auroc,
            'auroc_p_value': auroc_p,
            'sfr_f1': sfr_f1,
            'benchmark_f1': bench_f1,
            'f1_diff': sfr_f1 - bench_f1,
            'f1_p_value': f1_p,
            'sfr_auprc': sfr_auprc,
            'benchmark_auprc': bench_auprc,
            'auprc_diff': sfr_auprc - bench_auprc,
            'auprc_p_value': auprc_p
        })
    
    return pd.DataFrame(comparison_results)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Statistical Comparison Analysis for CXR Models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dataset', 
        choices=['padchest', 'indiana', 'vindrcxr', 'vindrpcxr'],
        default='padchest',
        help='Dataset to analyze'
    )
    parser.add_argument(
        '--threshold', 
        type=int, 
        default=30,
        help='Minimum positive cases threshold'
    )
    parser.add_argument(
        '--compare-with',
        choices=['clip', 'llm', 'both'],
        default='clip',
        help='Choose which models to compare SFR-Mistral against: "clip" for external CLIP models, "llm" for internal LLM models, "both" for both types'
    )
    
    args = parser.parse_args()
    
    # Get dataset configuration
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Dataset '{args.dataset}' not supported. Choose from: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[args.dataset]
    THRESHOLD = args.threshold
    
    # Dataset-specific paths
    PHENOTYPES_PATH = config['phenotypes_path']
    DATASET_CSV_PATH = config['csv_path']
    SFR_RESULTS_DIR = config['sfr_results_dir']
    BENCHMARK_DIRS = config['benchmark_dirs']
    LLM_DIRS = config['llm_dirs']
    EXCLUDE_COLUMNS = config['exclude_columns']
    
    # Use the existing results directory with relative path
    results_dir = f"results/statistic_{args.dataset}_t{args.threshold}_{args.compare_with}"
    os.makedirs(results_dir, exist_ok=True)
    
    print("="*80)
    print(f"STATISTICAL COMPARISON ANALYSIS FOR CXR MODELS - {args.dataset.upper()}")
    print("="*80)
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"📊 Dataset: {args.dataset}")
    print(f"🔢 Threshold: {THRESHOLD}")
    print(f"🔄 Comparing with: {args.compare_with}")
    print("="*80)
    
    # Initialize overall progress bar
    total_steps = 6  # Main analysis phases
    progress_bar = tqdm(total=total_steps, desc="Overall Progress", position=0, leave=True)
    
    # Step 1: Load phenotypes and original data
    progress_bar.set_description("Loading phenotypes and data")
    phenotypes = load_phenotypes(PHENOTYPES_PATH)
    label_counts, all_labels = load_original_dataset_data(DATASET_CSV_PATH, EXCLUDE_COLUMNS)
    
    # Filter labels by threshold and phenotypes
    filtered_labels = filter_labels_by_threshold_and_phenotypes(
        label_counts, phenotypes, THRESHOLD
    )
    progress_bar.update(1)
    
    # Step 2: Load SFR-Mistral results
    print("\n" + "="*60)
    print("LOADING MODEL RESULTS")
    print("="*60)
    progress_bar.set_description("Loading model results")
    
    sfr_data = load_model_results(SFR_RESULTS_DIR, "SFR-Mistral")
    
    # Load models based on comparison choice
    benchmark_data = {}
    llm_data = {}
    
    if args.compare_with in ['clip', 'both']:
        # Load benchmark results (external CLIP-based models)
        print("Loading external CLIP-based models...")
        for model_name, model_results_dir in BENCHMARK_DIRS.items():
            benchmark_data[model_name] = load_model_results(model_results_dir, model_name)
    
    if args.compare_with in ['llm', 'both']:
        # Load LLM-based results
        print("Loading internal LLM-based models...")
        for model_name, model_results_dir in LLM_DIRS.items():
            llm_data[model_name] = load_model_results(model_results_dir, model_name)
    
    progress_bar.update(1)
    
    # Step 3: Analyze individual model performance
    print("\n" + "="*60)
    print("INDIVIDUAL MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    progress_bar.set_description("Analyzing model performance")
    
    # Analyze SFR-Mistral
    sfr_results = analyze_model_performance(sfr_data, filtered_labels, "SFR-Mistral")
    
    # Analyze models based on comparison choice
    benchmark_results = {}
    llm_results = {}
    
    if args.compare_with in ['clip', 'both']:
        # Analyze benchmark models (external CLIP-based)
        for model_name, data in benchmark_data.items():
            if data is not None:
                benchmark_results[model_name] = analyze_model_performance(
                    data, filtered_labels, model_name
                )
    
    if args.compare_with in ['llm', 'both']:
        # Analyze LLM-based models
        for model_name, data in llm_data.items():
            if data is not None:
                llm_results[model_name] = analyze_model_performance(
                    data, filtered_labels, model_name
                )
    
    progress_bar.update(1)
    
    # Step 4: Find best performing models in each category
    print("\n" + "="*60)
    print("FINDING BEST MODELS")
    print("="*60)
    progress_bar.set_description("Finding best models")
    
    # Find best benchmark model
    best_benchmark_model = None
    best_benchmark_auroc = -1
    
    if args.compare_with in ['clip', 'both'] and benchmark_results:
        print("External CLIP-based models:")
        for model_name, results in benchmark_results.items():
            if results is not None and len(results) > 0:
                mean_auroc = results['auroc'].mean()
                print(f"  {model_name}: Mean AUROC = {mean_auroc:.4f}")
                
                if mean_auroc > best_benchmark_auroc:
                    best_benchmark_auroc = mean_auroc
                    best_benchmark_model = model_name
        
        if best_benchmark_model:
            print(f"\nBest benchmark model: {best_benchmark_model} (Mean AUROC: {best_benchmark_auroc:.4f})")
    
    # Find best LLM-based model
    best_llm_model = None
    best_llm_auroc = -1
    
    if args.compare_with in ['llm', 'both'] and llm_results:
        print("\nLLM-based models:")
        for model_name, results in llm_results.items():
            if results is not None and len(results) > 0:
                mean_auroc = results['auroc'].mean()
                print(f"  {model_name}: Mean AUROC = {mean_auroc:.4f}")
                
                if mean_auroc > best_llm_auroc:
                    best_llm_auroc = mean_auroc
                    best_llm_model = model_name
        
        if best_llm_model:
            print(f"\nBest LLM model: {best_llm_model} (Mean AUROC: {best_llm_auroc:.4f})")
    
    progress_bar.update(1)
    
    # Step 5: Statistical comparisons
    progress_bar.set_description("Performing statistical comparisons")
    print("\n" + "="*60)
    print("STATISTICAL COMPARISONS")
    print("="*60)
    
    comparison_results = {}
    
    # Compare with best benchmark model
    if args.compare_with in ['clip', 'both'] and best_benchmark_model and sfr_data is not None:
        print(f"\nSFR-Mistral vs Best Benchmark ({best_benchmark_model}):")
        print("-" * 50)
        
        comparison_results[f'sfr_vs_{best_benchmark_model.lower()}'] = compare_models_statistical(
            sfr_data, benchmark_data[best_benchmark_model], filtered_labels,
            "SFR-Mistral", best_benchmark_model
        )
    
    # Compare with best LLM model
    if args.compare_with in ['llm', 'both'] and best_llm_model and sfr_data is not None:
        print(f"\nSFR-Mistral vs Best LLM ({best_llm_model}):")
        print("-" * 50)
        
        comparison_results[f'sfr_vs_{best_llm_model.lower()}'] = compare_models_statistical(
            sfr_data, llm_data[best_llm_model], filtered_labels,
            "SFR-Mistral", best_llm_model
        )
    
    # Compare with all LLM models (optional detailed comparison)
    if args.compare_with in ['llm', 'both']:
        for model_name, data in llm_data.items():
            if data is not None and model_name != best_llm_model and sfr_data is not None:
                print(f"\nSFR-Mistral vs {model_name}:")
                print("-" * 30)
                
                comparison_results[f'sfr_vs_{model_name.lower()}'] = compare_models_statistical(
                    sfr_data, data, filtered_labels,
                    "SFR-Mistral", model_name
                )
    
    progress_bar.update(1)
    
    # Step 6: Save individual model results
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    progress_bar.set_description("Saving results")
    
    if sfr_results is not None:
        sfr_file = f"{results_dir}/sfr_mistral_performance_analysis.csv"
        sfr_results.to_csv(sfr_file, index=False, float_format='%.8f')
        print(f"SFR-Mistral results saved to: {sfr_file}")
    
    # Save comparison results with summary statistics
    for comparison_name, comp_results in comparison_results.items():
        if comp_results is not None and len(comp_results) > 0:
            # Save detailed comparison results
            comparison_file = f"{results_dir}/statistical_comparison_{comparison_name}.csv"
            comp_results.to_csv(comparison_file, index=False, float_format='%.8f')
            print(f"Detailed comparison saved to: {comparison_file}")
            
            # Print summary statistics
            print(f"\nSUMMARY STATISTICS for {comparison_name} (n={len(comp_results)} labels):")
            print("-" * 50)
            
            # AUROC comparison
            auroc_wins = (comp_results['auroc_diff'] > 0).sum()
            auroc_sig_wins = ((comp_results['auroc_diff'] > 0) & 
                             (comp_results['auroc_p_value'] < 0.05)).sum()
            
            print(f"AUROC:")
            print(f"  SFR-Mistral wins: {auroc_wins}/{len(comp_results)} labels")
            print(f"  Significant wins (p<0.05): {auroc_sig_wins}/{len(comp_results)} labels")
            print(f"  Mean AUROC difference: {comp_results['auroc_diff'].mean():.4f}")
            
            # F1 comparison
            f1_wins = (comp_results['f1_diff'] > 0).sum()
            f1_sig_wins = ((comp_results['f1_diff'] > 0) & 
                          (comp_results['f1_p_value'] < 0.05)).sum()
            
            print(f"F1 Score:")
            print(f"  SFR-Mistral wins: {f1_wins}/{len(comp_results)} labels")
            print(f"  Significant wins (p<0.05): {f1_sig_wins}/{len(comp_results)} labels")
            print(f"  Mean F1 difference: {comp_results['f1_diff'].mean():.4f}")
            
            # AUPRC comparison
            auprc_wins = (comp_results['auprc_diff'] > 0).sum()
            auprc_sig_wins = ((comp_results['auprc_diff'] > 0) & 
                             (comp_results['auprc_p_value'] < 0.05)).sum()
            
            print(f"AUPRC:")
            print(f"  SFR-Mistral wins: {auprc_wins}/{len(comp_results)} labels")
            print(f"  Significant wins (p<0.05): {auprc_sig_wins}/{len(comp_results)} labels")
            print(f"  Mean AUPRC difference: {comp_results['auprc_diff'].mean():.4f}")
    
    if args.compare_with in ['clip', 'both']:
        for model_name, results in benchmark_results.items():
            if results is not None:
                filename = f"{results_dir}/{model_name.lower()}_performance_analysis.csv"
                results.to_csv(filename, index=False, float_format='%.8f')
                print(f"{model_name} results saved to: {filename}")
    
    if args.compare_with in ['llm', 'both']:
        for model_name, results in llm_results.items():
            if results is not None:
                filename = f"{results_dir}/{model_name.lower()}_performance_analysis.csv"
                results.to_csv(filename, index=False, float_format='%.8f')
                print(f"{model_name} results saved to: {filename}")
    
    progress_bar.update(1)
    progress_bar.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"📁 All results saved to: {results_dir}")
    print(f"📊 Dataset analyzed: {args.dataset}")
    print(f"🔄 Comparison type: {args.compare_with}")
    print("\n📊 Generated files:")
    
    # List all files in the results directory
    if os.path.exists(results_dir):
        for file in sorted(os.listdir(results_dir)):
            if file.endswith('.csv'):
                print(f"   📈 {file}")
    
    # Instructions for missing models
    missing_models = []
    
    if args.compare_with in ['clip', 'both']:
        for model_name, data in benchmark_data.items():
            if data is None:
                missing_models.append(f"{model_name} (CLIP)")
    
    if args.compare_with in ['llm', 'both']:
        for model_name, data in llm_data.items():
            if data is None:
                missing_models.append(f"{model_name} (LLM)")
    
    if missing_models:
        print(f"\nNOTE: {', '.join(missing_models)} results not found for {args.dataset}.")
        print("Once these models complete, re-run this script to include their comparison.")

if __name__ == "__main__":
    main() 