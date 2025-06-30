# Statistical Comparison Analysis for Multiple CXR Datasets

This directory contains scripts for performing statistical comparison analysis between SFR-Mistral concept model and benchmark models across multiple chest X-ray datasets.

## Overview

The updated `statistical_comparison_analysis.py` script now supports analysis on four different datasets:
- **PadChest** - Spanish chest X-ray dataset
- **Indiana** - Indiana University chest X-ray dataset  
- **VinDR-CXR** - Vietnamese chest X-ray dataset (adult)
- **VinDR-PC-CXR** - Vietnamese chest X-ray dataset (pediatric)

## Features

- **Multi-dataset Support**: Analyze performance across different CXR datasets
- **Statistical Rigor**: Bootstrap confidence intervals and permutation tests
- **Comprehensive Metrics**: AUROC, F1-score, and AUPRC with 95% CI
- **Automated Comparison**: Identifies best benchmark model and performs statistical testing
- **Flexible Thresholds**: Configurable minimum positive case thresholds

## Usage

### Option 1: Direct Script Usage

Run analysis for a specific dataset:
```bash
# PadChest (default)
python benchmark/statistical_comparison_analysis.py --dataset padchest --threshold 30

# Indiana
python benchmark/statistical_comparison_analysis.py --dataset indiana --threshold 30

# VinDR-CXR
python benchmark/statistical_comparison_analysis.py --dataset vindrcxr --threshold 30

# VinDR-PC-CXR  
python benchmark/statistical_comparison_analysis.py --dataset vindrpcxr --threshold 30
```

### Option 2: Using the Runner Script

The `run_statistical_analysis.py` wrapper provides convenient batch processing:

```bash
# Run single dataset
python benchmark/run_statistical_analysis.py --datasets padchest --threshold 30

# Run multiple datasets
python benchmark/run_statistical_analysis.py --datasets padchest indiana vindrcxr --threshold 30

# Run all datasets
python benchmark/run_statistical_analysis.py --datasets all --threshold 30
```

### Command Line Arguments

**Main Script (`statistical_comparison_analysis.py`)**:
- `--dataset`: Dataset to analyze (`padchest`, `indiana`, `vindrcxr`, `vindrpcxr`)
- `--threshold`: Minimum positive cases threshold (default: 30)

**Runner Script (`run_statistical_analysis.py`)**:
- `--datasets`: List of datasets or 'all' (default: `padchest`)
- `--threshold`: Minimum positive cases threshold (default: 30)

## Dataset Configurations

The script automatically handles dataset-specific configurations:

### PadChest
- **Data**: `../data/padchest_test.csv`
- **Phenotypes**: Uses curated diagnostic phenotypes from `pheotypes_padchest.json`
- **Labels**: 114+ Spanish medical terms
- **Excludes**: Study, Path, image_id, ImageID, name, is_test

### Indiana
- **Data**: `../data/indiana_test.csv`
- **Phenotypes**: Uses all available labels above threshold
- **Labels**: 70+ English medical conditions 
- **Excludes**: uid, filename, projection, MeSH, Problems, image, indication, comparison, findings, impression

### VinDR-CXR
- **Data**: `../data/vindrcxr_test.csv`
- **Phenotypes**: Uses all available labels above threshold
- **Labels**: 28 diagnostic findings
- **Excludes**: image_id

### VinDR-PC-CXR
- **Data**: `../data/vindrpcxr_test.csv`
- **Phenotypes**: Uses all available labels above threshold
- **Labels**: 15 pediatric conditions
- **Excludes**: image_id

## Output Structure

Results are saved to timestamped directories:
```
benchmark/results/statistic_{dataset}_{timestamp}/
├── sfr_mistral_performance_analysis.csv
├── chexzero_performance_analysis.csv
├── biomedclip_performance_analysis.csv
├── openai_clip_performance_analysis.csv
├── cxr_foundation_performance_analysis.csv
└── statistical_comparison_sfr_vs_{best_model}.csv
```

### Output Files

1. **Individual Model Performance**: `{model}_performance_analysis.csv`
   - Columns: label, n_positive, n_total, auroc, auroc_ci_lower, auroc_ci_upper, f1, f1_ci_lower, f1_ci_upper, auprc, auprc_ci_lower, auprc_ci_upper

2. **Statistical Comparison**: `statistical_comparison_sfr_vs_{best_model}.csv`
   - Columns: label, n_positive, sfr_auroc, benchmark_auroc, auroc_diff, auroc_p_value, sfr_f1, benchmark_f1, f1_diff, f1_p_value, sfr_auprc, benchmark_auprc, auprc_diff, auprc_p_value

## Statistical Methods

### Bootstrap Confidence Intervals
- 1000 bootstrap samples for each metric
- 95% confidence intervals (2.5th and 97.5th percentiles)
- Handles imbalanced datasets gracefully

### Permutation Testing
- 1000 permutations for statistical significance testing
- Two-tailed p-values for difference in performance
- Robust to dataset-specific biases

### F1 Score Optimization
- Optimal threshold selection using sen² + spe² criterion
- Grid search over ROC curve thresholds
- Maximizes both sensitivity and specificity

## Requirements

### Python Dependencies
```
pandas
numpy
scikit-learn
tqdm
argparse
```

### Data Requirements
- Test dataset CSV files in `../data/` directory
- Model prediction files in appropriate results directories
- Ground truth files with matching timestamps

### Results Directory Structure
Model results should follow this pattern:
- **SFR-Mistral**: `../concepts/results/concept_based_evaluation_{dataset}_{model}/`
- **Benchmarks**: `results/benchmark_evaluation_{dataset}_{model}/`

## Troubleshooting

### Common Issues

1. **Missing Results**: 
   - Ensure model evaluation has been completed
   - Check results directory paths in dataset configurations

2. **Low Label Count**:
   - Adjust `--threshold` to include more labels
   - Some datasets have fewer positive cases

3. **Phenotypes Not Found**:
   - For non-PadChest datasets, phenotypes are auto-generated
   - Create custom phenotype files if needed

### Debugging

Run with verbose output:
```bash
python -u benchmark/statistical_comparison_analysis.py --dataset indiana --threshold 10
```

## Examples

### Quick Start - PadChest Analysis
```bash
python benchmark/statistical_comparison_analysis.py --dataset padchest
```

### Batch Analysis - All Datasets
```bash
python benchmark/run_statistical_analysis.py --datasets all --threshold 25
```

### Custom Threshold Analysis
```bash
python benchmark/statistical_comparison_analysis.py --dataset indiana --threshold 50
```

## Notes

- Analysis can take 5-15 minutes per dataset depending on size
- Results include comprehensive statistical testing with multiple comparison correction
- Missing benchmark results are handled gracefully with informative messages
- All output files use high precision (8 decimal places) for p-values 