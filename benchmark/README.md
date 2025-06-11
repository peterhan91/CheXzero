# Zero-Shot CXR Benchmark Suite

A comprehensive benchmarking framework for evaluating vision-language models on chest X-ray (CXR) pathology classification tasks.

## Overview

This benchmark suite evaluates multiple state-of-the-art models on four CXR testing datasets using zero-shot classification with ROC-AUC as the primary metric.

### Models Evaluated

1. **Our Trained Models**: DinoV2-based CLIP models trained on CXR-report pairs
   - Located in `./checkpoints/` (dinov2-multi-v1.0_vitb/vits/vitl)
   - Tested at both 224×224 and 448×448 resolutions (trained at 448×448)

2. **External SOTA Models**:
   - ✅ **CheXzero**: MIMIC-CXR finetuned CLIP model (conda env: `chexzero2`)
     - Uses existing checkpoint: `external_sota/chexzero/best_64_0.0002_original_23000_0.854.pt`
   - ✅ **BiomedCLIP**: PMC image-caption trained CLIP (conda env: `open_clip`)
     - Model: `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
   - ✅ **CXR-Foundation**: Google's foundation model (conda env: `tensorflow`)
     - Auto-downloads from HuggingFace: `google/cxr-foundation`
     - Two-step ELIXR-C → QFormer pipeline for zero-shot classification

### Datasets

- `chexpert_test`: 14 pathology labels (Atelectasis, Cardiomegaly, etc.)
- `padchest_test`: Key pathology subset (5 labels)
- `vindrcxr_test`: 28 pathology labels
- `vindrpcxr_test`: 15 pediatric pathology labels

All datasets are stored as H5 image files with corresponding CSV labels in the `data/` folder.

## Quick Start

### Run All Benchmarks
```bash
./benchmark/run_all_benchmarks.sh
```

### Run Individual Model Benchmarks
```bash
# Our trained models (base environment)
python benchmark/benchmark_our_models.py

# CheXzero (requires chexzero2 environment)
conda activate chexzero2
python benchmark/benchmark_chexzero.py

# BiomedCLIP (requires open_clip environment)  
conda activate open_clip
python benchmark/benchmark_biomedclip.py

# CXR-Foundation (requires tensorflow environment)
conda activate tensorflow
python benchmark/benchmark_cxr_foundation.py
```

### Generate Summary Report
```bash
python benchmark/generate_summary.py --results_dir benchmark/results
```

## File Structure

```
benchmark/
├── benchmark_base.py          # Core infrastructure & shared functions
├── benchmark_our_models.py    # Script for our trained models
├── benchmark_chexzero.py      # Script for CheXzero model
├── benchmark_biomedclip.py    # Script for BiomedCLIP model
├── benchmark_cxr_foundation.py # Explains CXR-Foundation incompatibility
├── run_all_benchmarks.sh      # Orchestration script
├── generate_summary.py        # Summary report generator
├── results/                   # Output directory for CSV results
└── README.md                  # This file
```

## Technical Details

### Configuration
- **Input Resolution**: 
  - Our models: Both 224×224 and 448×448 (trained at 448×448)
  - CheXzero, BiomedCLIP, CXR-Foundation: 224×224
- **Batch Size**: 64 (our models), 32 (external models)
- **Context Length**: 77 tokens (CheXzero), 256 tokens (BiomedCLIP)
- **Normalization**: CXR-specific values (101.48761, 83.43944)

### Dual Resolution Testing
Our models are tested at both resolutions to enable:
- **Fair comparison** at 224×224 with all other SOTA models
- **Optimal performance** at 448×448 (native training resolution)
- **Resolution impact analysis** on zero-shot CXR classification

This dual testing approach provides comprehensive insights:
- How much performance is lost when using lower resolution (224×224)
- Whether our models maintain advantages at standardized resolution
- Optimal vs. fair comparison trade-offs in benchmarking

### Zero-Shot Templates
- **Positive**: `"{pathology}"`
- **Negative**: `"no {pathology}"`

### Evaluation Metrics
- **Primary**: ROC-AUC per pathology + mean AUC
- **Output**: CSV files with detailed per-pathology results
- **Error Handling**: Graceful handling of missing labels/data

## Environment Setup

### Required Conda Environments

1. **Base Environment** (for our models):
   ```bash
   # Should have torch, clip, h5py, sklearn, pandas, tqdm
   # Uses existing codebase infrastructure
   ```

2. **chexzero2** (for CheXzero):
   ```bash
   conda create -n chexzero2 python=3.8
   conda activate chexzero2
   # Install same dependencies as base environment
   # Uses checkpoint: external_sota/chexzero/best_64_0.0002_original_23000_0.854.pt
   ```

3. **open_clip** (for BiomedCLIP):
   ```bash
   conda create -n open_clip python=3.8
   conda activate open_clip
   pip install open_clip_torch>=2.23.0 timm>=0.9.8
   ```

4. **tensorflow** (for CXR-Foundation - informational only):
   ```bash
   # Not used for benchmarking due to architectural incompatibility
   # CXR-Foundation requires TensorFlow/JAX and cannot do zero-shot classification
   ```

## Important Notes

### CXR-Foundation Limitation
**CXR-Foundation is NOT a CLIP-style model** and cannot perform zero-shot classification:
- It's a TF-Keras embedding model (EfficientNet-L2 + BERT)
- Requires two-step inference: ELIXR-C → QFormer
- Outputs embeddings, not classification logits
- Would need additional linear classifier training for each dataset
- Uses `tf.Example` input format, not PyTorch tensors

For this reason, CXR-Foundation is **excluded** from the zero-shot benchmark suite. To use CXR-Foundation for classification, one would need to:
1. Extract embeddings for all images
2. Train dataset-specific linear classifiers
3. Evaluate using supervised metrics

### Model Verification
All implementations have been verified against examples in `external_sota/`:
- ✅ **BiomedCLIP**: Verified against `external_sota/biomedclip/biomedclip_example.py`
- ✅ **CheXzero**: Uses checkpoint `external_sota/chexzero/best_64_0.0002_original_23000_0.854.pt`
- ❌ **CXR-Foundation**: Confirmed incompatible via `external_sota/cxr_foundation/` examples

## Usage Examples

### Evaluate Specific Models
```bash
# Only evaluate specific trained models
python benchmark/benchmark_our_models.py --models dinov2-multi-v1.0_vitb

# Only evaluate on specific datasets
python benchmark/benchmark_our_models.py --datasets chexpert_test vindrcxr_test

# Force CPU usage
python benchmark/benchmark_our_models.py --device cpu
```

### Evaluate External Models
```bash
# BiomedCLIP with specific datasets
conda activate open_clip
python benchmark/benchmark_biomedclip.py --datasets chexpert_test --device cuda

# CheXzero evaluation
conda activate chexzero2  
python benchmark/benchmark_chexzero.py --datasets vindrcxr_test
```

## Results

Results are saved as CSV files in `benchmark/results/` with naming convention:
```
{model_name}_{dataset_name}_results.csv
```

Each file contains:
- Per-pathology ROC-AUC scores
- Mean AUC across all pathologies
- Number of samples evaluated
- Model and dataset metadata

### Expected Models in Results:
- `our_models_*_res224_*_results.csv` - Our DinoV2-CLIP models at 224×224 (for comparison)
- `our_models_*_res448_*_results.csv` - Our DinoV2-CLIP models at 448×448 (native resolution)
- `chexzero_*_results.csv` - CheXzero results
- `biomedclip_*_results.csv` - BiomedCLIP results
- `cxr_foundation_*_results.csv` - CXR-Foundation results

## Troubleshooting

### Common Issues
1. **Missing conda environments**: Install required environments as shown above
2. **Model checkpoint not found**: Ensure `external_sota/chexzero/` contains the checkpoint
3. **Import errors**: Activate the correct conda environment for each model
4. **Out of memory**: Reduce batch size with `--batch_size` argument

### Performance Tips
- Use GPU for faster inference: `--device cuda`
- Reduce batch size for memory-constrained systems
- Run individual model scripts for debugging specific models

## Citation

If you use this benchmark suite, please cite the relevant papers for each model and dataset. 