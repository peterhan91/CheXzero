#!/usr/bin/env python3
"""
Benchmark script for CXR-Foundation model.
Implements zero-shot classification using the official ELIXR-C → QFormer pipeline.
Based on: https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/classify_images_with_natural_language.ipynb
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import io
from PIL import Image
import h5py

# TensorFlow imports (will be checked at runtime)
try:
    import tensorflow as tf
    import tensorflow_hub as tf_hub
    import tensorflow_text as tf_text  # Required for SentencepieceOp
    tf_available = True
    print("✅ TensorFlow and TensorFlow Text available")
except ImportError:
    tf_available = False
    print("❌ TensorFlow or TensorFlow Text not available")

# Additional required imports
import io
try:
    import png  # pypng package
except ImportError:
    print("❌ pypng is required for CXR-Foundation")
    print("Install with: pip install pypng")

# Add the parent directory to Python path to import from the main codebase
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from benchmark_base to use proper saving functions like other models
from benchmark_base import evaluate_predictions, save_results, save_detailed_results, make_true_labels

def png_to_tfexample(image_array: np.ndarray):
    """Creates a tf.train.Example from a NumPy array (from official notebook)."""
    # Convert the image to float32 and shift the minimum value to zero
    image = image_array.astype(np.float32)
    image -= image.min()

    if image_array.dtype == np.uint8:
        # For uint8 images, no rescaling is needed
        pixel_array = image.astype(np.uint8)
        bitdepth = 8
    else:
        # For other data types, scale image to use the full 16-bit range
        max_val = image.max()
        if max_val > 0:
            image *= 65535 / max_val  # Scale to 16-bit range
        pixel_array = image.astype(np.uint16)
        bitdepth = 16

    # Ensure the array is 2-D (grayscale image)
    if pixel_array.ndim != 2:
        raise ValueError(f'Array must be 2-D. Actual dimensions: {pixel_array.ndim}')

    # Encode the array as a PNG image
    output = io.BytesIO()
    png.Writer(
        width=pixel_array.shape[1],
        height=pixel_array.shape[0],
        greyscale=True,
        bitdepth=bitdepth
    ).write(output, pixel_array.tolist())
    png_bytes = output.getvalue()

    # Create a tf.train.Example and assign the features
    if not tf_available:
        raise ImportError("TensorFlow is required for CXR-Foundation")
    example = tf.train.Example()
    features = example.features.feature
    features['image/encoded'].bytes_list.value.append(png_bytes)
    features['image/format'].bytes_list.value.append(b'png')

    return example

def bert_tokenize(text):
    """Tokenizes input text and returns token IDs and padding masks (from official notebook)."""
    preprocessor = tf_hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    out = preprocessor(tf.constant([text.lower()]))
    ids = out['input_word_ids'].numpy().astype(np.int32)
    masks = out['input_mask'].numpy().astype(np.float32)
    paddings = 1.0 - masks
    end_token_idx = ids == 102
    ids[end_token_idx] = 0
    paddings[end_token_idx] = 1.0
    ids = np.expand_dims(ids, axis=1)
    paddings = np.expand_dims(paddings, axis=1)
    assert ids.shape == (1, 1, 128)
    assert paddings.shape == (1, 1, 128)
    return ids, paddings

def compute_image_text_similarity(image_emb, txt_emb):
    """Compute similarity between image and text embeddings (from official notebook)."""
    image_emb = np.reshape(image_emb, (32, 128))
    similarities = []
    for i in range(32):
        # cosine similarity
        similarity = np.dot(image_emb[i], txt_emb)/(np.linalg.norm(image_emb[i]) * np.linalg.norm(txt_emb))
        similarities.append(similarity)
    np_sm_similarities = np.array((similarities))
    return np.max(np_sm_similarities)

def zero_shot(image_emb, pos_txt_emb, neg_txt_emb):
    """Compute zero-shot similarity score (from official notebook)."""
    pos_cosine = compute_image_text_similarity(image_emb, pos_txt_emb)
    neg_cosine = compute_image_text_similarity(image_emb, neg_txt_emb)
    return pos_cosine - neg_cosine

class CXRFoundationModel:
    """CXR-Foundation model wrapper for zero-shot classification (following official notebook patterns)."""
    
    def __init__(self, model_dir="./external_sota/cxr_foundation/models"):
        """Initialize CXR-Foundation models."""
        self.model_dir = model_dir
        self.elixrc_model = None
        self.qformer_model = None
        self.text_embeddings_cache = {}
        
        # Model paths (following official pattern)
        self.elixrc_path = os.path.join(model_dir, "elixr-c-v2-pooled")
        self.qformer_path = os.path.join(model_dir, "pax-elixr-b-text")
        
        print(f"CXR-Foundation model directory: {model_dir}")
        print(f"ELIXR-C path: {self.elixrc_path}")
        print(f"QFormer path: {self.qformer_path}")
    
    def download_models(self):
        """Download CXR-Foundation models from Hugging Face (following official pattern)."""
        try:
            print("Downloading CXR-Foundation models from Hugging Face...")
            
            # Import huggingface_hub
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("❌ Error: huggingface_hub is required for model download")
                print("Install with: pip install huggingface_hub")
                return False
            
            # Create model directory
            os.makedirs(self.model_dir, exist_ok=True)
            
            # Download models (following official pattern from quick_start_with_hugging_face.py)
            print("Downloading from repo: google/cxr-foundation...")
            snapshot_download(
                repo_id="google/cxr-foundation",
                local_dir=self.model_dir,
                allow_patterns=['elixr-c-v2-pooled/*', 'pax-elixr-b-text/*']
            )
            
            print("✅ Models downloaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading models: {e}")
            print("Please check your internet connection and Hugging Face access.")
            return False
        
    def load_models(self, auto_download=True):
        """Load ELIXR-C and QFormer models (following official pattern)."""
        try:
            # Check if models exist, download if not (following official pattern)
            if not os.path.exists(self.elixrc_path) or not os.path.exists(self.qformer_path):
                if auto_download:
                    print("Models not found locally, downloading from Hugging Face...")
                    if not self.download_models():
                        return False
                else:
                    print(f"❌ Models not found and auto_download=False")
                    print(f"ELIXR-C: {self.elixrc_path} (exists: {os.path.exists(self.elixrc_path)})")
                    print(f"QFormer: {self.qformer_path} (exists: {os.path.exists(self.qformer_path)})")
                    return False
            
            # Load ELIXR-C model (following official pattern)
            if os.path.exists(self.elixrc_path):
                print("Loading ELIXR-C model...")
                self.elixrc_model = tf.saved_model.load(self.elixrc_path)
                self.elixrc_infer = self.elixrc_model.signatures['serving_default']
                print("✅ ELIXR-C model loaded successfully")
            else:
                print(f"❌ ELIXR-C model not found at: {self.elixrc_path}")
                return False
            
            # Load QFormer model (following official pattern)
            if os.path.exists(self.qformer_path):
                print("Loading QFormer model...")
                self.qformer_model = tf.saved_model.load(self.qformer_path)
                print("✅ QFormer model loaded successfully")
            else:
                print(f"❌ QFormer model not found at: {self.qformer_path}")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def get_image_embedding(self, image_array):
        """Get image embedding using ELIXR-C → QFormer pipeline with preprocessed HDF5 images."""
        # Convert to PNG bytes like original pipeline
        if image_array.dtype != np.uint16:
            image = image_array.astype(np.float32)
            image -= image.min()
            image *= 65535.0 / (image.max() + 1e-5)
            image = image.astype(np.uint16)
        else:
            image = image_array

        output = io.BytesIO()
        png.Writer(
            width=image.shape[1],
            height=image.shape[0],
            greyscale=True,
            bitdepth=16
        ).write(output, image.tolist())
        png_bytes = output.getvalue()

        # Create tf.train.Example
        example = tf.train.Example()
        features = example.features.feature
        features['image/encoded'].bytes_list.value.append(png_bytes)
        features['image/format'].bytes_list.value.append(b'png')
        serialized_example = example.SerializeToString()

        # ELIXR-C inference
        elixrc_output = self.elixrc_infer(input_example=tf.constant([serialized_example]))
        elixrc_embedding = elixrc_output['feature_maps_0'].numpy()

        # QFormer
        qformer_input = {
            'image_feature': elixrc_embedding.tolist(),
            'ids': np.zeros((1, 1, 128), dtype=np.int32).tolist(),
            'paddings': np.zeros((1, 1, 128), dtype=np.float32).tolist(),
        }

        qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
        image_embeddings = qformer_output['all_contrastive_img_emb'].numpy()

        return image_embeddings

    
    def get_text_embedding(self, text):
        """Get text embedding using QFormer (from official notebook)."""
        if text in self.text_embeddings_cache:
            return self.text_embeddings_cache[text]
        
        # Tokenize text
        tokens, paddings = bert_tokenize(text)
        
        # QFormer with text input
        qformer_input = {
            'image_feature': np.zeros([1, 8, 8, 1376], dtype=np.float32).tolist(),
            'ids': tokens.tolist(),
            'paddings': paddings.tolist(),
        }
        
        qformer_output = self.qformer_model.signatures['serving_default'](**qformer_input)
        text_embeddings = qformer_output['contrastive_txt_emb'].numpy()
        
        # Cache the result
        self.text_embeddings_cache[text] = text_embeddings
        
        return text_embeddings

def setup_cxr_foundation_data(dataset_name):
    """Setup test data for CXR-Foundation evaluation."""
    
    # Dataset configurations (paths only)
    dataset_configs = {
        'chexpert_test': {
            'img_path': '../data/chexpert_test.h5',
            'label_path': '../data/chexpert_test.csv',
        },
        'padchest_test': {
            'img_path': '../data/padchest_test.h5',
            'label_path': '../data/padchest_test.csv',
        },
        'vindrcxr_test': {
            'img_path': '../data/vindrcxr_test.h5',
            'label_path': '../data/vindrcxr_test.csv',
        },
        'vindrpcxr_test': {
            'img_path': '../data/vindrpcxr_test.h5',
            'label_path': '../data/vindrpcxr_test.csv',
        },
        'indiana_test': {
            'img_path': '../data/indiana_test.h5',
            'label_path': '../data/indiana_test.csv',
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = dataset_configs[dataset_name]
    
    # Check if files exist
    if not os.path.exists(config['img_path']):
        raise FileNotFoundError(f"Image file not found: {config['img_path']}")
    if not os.path.exists(config['label_path']):
        raise FileNotFoundError(f"Label file not found: {config['label_path']}")
    
    # Dynamically read labels from CSV header
    df = pd.read_csv(config['label_path'], nrows=0)  # Read only header
    
    # Get all columns except non-label columns
    exclude_columns = {'Study', 'Path', 'image_id', 'ImageID', 'name', 'is_test', 'uid', 'filename', 'projection', 'MeSH', 'Problems', 'image', 'indication', 'comparison', 'findings', 'impression'}
    all_columns = set(df.columns)
    label_columns = all_columns - exclude_columns
    labels = sorted(list(label_columns))  # Sort for consistency
    
    # Use simple template format like benchmark_base.py for fair comparison
    # Create templates dictionary for each label using standardized format
    templates = {}
    for label in labels:
        templates[label] = ("{}", "no {}")
    
    print(f"Loaded {dataset_name}:")
    print(f"  - Found {len(labels)} label columns in CSV")
    print(f"  - Labels: {labels}")
    
    # Load ground truth
    y_true, actual_labels = make_true_labels(config['label_path'], labels)
    
    print(f"  - Ground truth shape: {y_true.shape}")
    print(f"  - Final labels used: {actual_labels}")
    
    # Add templates to config for consistency with benchmark_base.py
    config['labels'] = actual_labels
    config['templates'] = templates
    
    return config, y_true, actual_labels

def run_cxr_foundation_evaluation(model, dataset_name, save_detailed=False, model_name=None):
    """Run zero-shot evaluation using CXR-Foundation."""
    
    config, y_true, labels = setup_cxr_foundation_data(dataset_name)
    
    print(f"Running CXR-Foundation evaluation on {dataset_name}...")
    
    # Load images
    with h5py.File(config['img_path'], 'r') as f:
        images = f['cxr'][:]
    
    print(f"Loaded {len(images)} images")
    
    # Pre-compute text embeddings for all labels
    print("Computing text embeddings...")
    text_embeddings = {}
    for label in labels:
        if label in config['templates']:
            pos_template, neg_template = config['templates'][label]
            # Apply template to label name
            pos_text = pos_template.format(label.lower())
            neg_text = neg_template.format(label.lower())
            pos_emb = model.get_text_embedding(pos_text)
            neg_emb = model.get_text_embedding(neg_text)
            text_embeddings[label] = (pos_emb, neg_emb)
            print(f"  ✅ {label}: '{pos_text}' vs '{neg_text}'")
        else:
            print(f"  ⚠️  No template for {label}, using generic")
            pos_emb = model.get_text_embedding(label.lower())
            neg_emb = model.get_text_embedding(f"no {label.lower()}")
            text_embeddings[label] = (pos_emb, neg_emb)
    
    # Compute predictions
    print("Computing image-text similarities...")
    predictions = []
    
    for i, image in enumerate(tqdm(images, desc="Processing images")):
        # Get image embedding
        img_emb = model.get_image_embedding(image)
        
        # Compute similarities for each label
        img_pred = []
        for label in labels:
            if label in text_embeddings:
                pos_emb, neg_emb = text_embeddings[label]
                # Compute zero-shot similarity score (following official pattern)
                # The official implementation uses direct embeddings (128-dim vectors)
                # Our QFormer outputs: text=(1, 128), image=(1, 32, 128)
                pos_emb_flat = pos_emb[0]     # Extract 128-dim vector from (1, 128)
                neg_emb_flat = neg_emb[0]     # Extract 128-dim vector from (1, 128)
                img_emb_flat = img_emb[0]     # Extract (32, 128) from (1, 32, 128)
                
                similarity = zero_shot(img_emb_flat, pos_emb_flat, neg_emb_flat)
                # Convert to probability using sigmoid (following official pattern)
                prob = 1 / (1 + np.exp(-similarity))
                img_pred.append(prob)
            else:
                img_pred.append(0.5)  # Default probability
        
        predictions.append(img_pred)
    
    y_pred = np.array(predictions)
    
    print(f"Prediction shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # Evaluate
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    # Save detailed results if requested
    if save_detailed and model_name:
        config_data = {
            "model_name": model_name,
            "dataset": dataset_name,
            "method": "cxr_foundation_zero_shot",
            "num_images": len(y_pred),
            "num_labels": len(labels),
            "labels": labels,
            "input_resolution": 448
        }
        save_detailed_results(y_pred, y_true, labels, model_name, dataset_name, config_data)
        return results_df, y_pred
    
    return results_df

def check_dependencies():
    """Check if required dependencies are available."""
    global tf_available
    if not tf_available:
        print("❌ Error: TensorFlow and TensorFlow Text are required for CXR-Foundation")
        print("Install with: pip install tensorflow tensorflow-hub tensorflow-text pypng")
        return False
    
    try:
        import png
        return True
    except ImportError:
        print("❌ Error: pypng is required for CXR-Foundation")
        print("Install with: pip install pypng")
        return False

def benchmark_cxr_foundation(datasets):
    """Benchmark CXR-Foundation on all datasets."""
    print(f"\n{'='*60}")
    print(f"Benchmarking CXR-Foundation model")
    print(f"Following official notebook patterns")
    print(f"{'='*60}")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Initialize model
    model = CXRFoundationModel()
    
    # Try to load models (with automatic download following official pattern)
    if not model.load_models(auto_download=True):
        print("\n❌ CXR-Foundation models could not be loaded!")
        print("This might be due to:")
        print("1. Missing huggingface_hub: pip install huggingface_hub")
        print("2. Network connectivity issues")
        print("3. Hugging Face authentication (for private repos)")
        print("\nSkipping CXR-Foundation evaluation.")
        return
    
    # Test on each dataset
    for dataset_name in datasets:
        print(f"\n--- Testing on {dataset_name} ---")
        
        try:
            # Run evaluation
            results_df, y_pred = run_cxr_foundation_evaluation(
                model, dataset_name, save_detailed=True, model_name="cxr_foundation"
            )
            
            # Save results
            save_results(results_df, "cxr_foundation", dataset_name)
            
        except Exception as e:
            print(f"Error evaluating CXR-Foundation on {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

def main():
    parser = argparse.ArgumentParser(description="Benchmark CXR-Foundation model")
    parser.add_argument('--datasets', nargs='+', 
                        default=['chexpert_test', 'padchest_test', 'vindrcxr_test', 'indiana_test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (not used for TensorFlow models)')
    
    args = parser.parse_args()
    
    print("CXR-Foundation Zero-Shot Classification")
    print("Based on official notebooks:")
    print("  - https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/classify_images_with_natural_language.ipynb")
    print("  - https://colab.research.google.com/github/google-health/cxr-foundation/blob/master/notebooks/quick_start_with_hugging_face.ipynb")
    
    # Benchmark CXR-Foundation
    benchmark_cxr_foundation(args.datasets)
    
    print(f"\n{'='*60}")
    print("CXR-Foundation benchmarking completed!")
    print("Results saved in benchmark/results/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 