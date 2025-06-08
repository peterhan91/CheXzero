# DinoV2 Integration for CLIP Training

This is a minimal integration of DinoV2 vision encoders into the CLIP training pipeline.

## Requirements

Install the additional dependency:
```bash
pip install timm>=0.9.0
```

## Usage

### Basic DinoV2 Training
```bash
python run_train_improved.py --use_dinov2 --dinov2_model_name dinov2_vitb14
```

### Available DinoV2 Models
- `dinov2_vits14` - Small (22M parameters)
- `dinov2_vitb14` - Base (87M parameters) 
- `dinov2_vitl14` - Large (304M parameters)
- `dinov2_vitg14` - Giant (1.1B parameters)

### Advanced Options
```bash
# Use larger model with frozen backbone
python run_train_improved.py --use_dinov2 --dinov2_model_name dinov2_vitl14 --freeze_dinov2

# Adjust batch size for larger models
python run_train_improved.py --use_dinov2 --dinov2_model_name dinov2_vitg14 --batch_size 16
```

## What Changed

**Minimal modifications to support DinoV2:**

1. **`train.py`** & **`zero_shot.py`**: Added DinoV2 parameters to `load_clip()` function
2. **`run_train_improved.py`**: Added command-line arguments for DinoV2
3. **`model.py`**: No changes needed - DinoV2 is integrated at runtime

The integration dynamically replaces the vision encoder when `--use_dinov2` is specified, keeping the original codebase intact. 