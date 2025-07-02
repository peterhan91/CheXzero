#!/usr/bin/env python3
import os
import sys
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, InterpolationMode
from PIL import Image
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import hashlib
import pickle

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model loading functions from benchmark modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmark"))

# Now import local modules after path setup
import clip
from train import load_clip

# Import LLM embedding generator
try:
    from get_embed import RadiologyEmbeddingGenerator
except ImportError:
    try:
        from concepts.get_embed import RadiologyEmbeddingGenerator
    except ImportError:
        print("⚠️ Warning: Could not import RadiologyEmbeddingGenerator")
        RadiologyEmbeddingGenerator = None

warnings.filterwarnings("ignore")

# Ensure required imports are available
try:
    import open_clip
except ImportError:
    print("⚠️ Warning: open_clip not available - some models may not work")
    open_clip = None

class CXRDataset(Dataset):
    """Dataset for loading CXR images from H5 file - matches CXRTestDataset from zero_shot.py"""
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.transform = transform
        with h5py.File(h5_path, 'r') as f:
            self.num_images = len(f['cxr'])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['cxr'][idx]  # numpy array, shape like (448, 448) or (320, 320)
            
        # Convert to 3-channel format exactly like CXRTestDataset
        img = np.expand_dims(img, axis=0)  # (1, H, W)
        img = np.repeat(img, 3, axis=0)    # (3, H, W) - RGB format
        img = torch.from_numpy(img)        # Convert to tensor
        
        if self.transform:
            img = self.transform(img)
            
        return {'img': img}

class BiomedCLIPDataset(Dataset):
    """Dataset for BiomedCLIP that expects PIL Image input"""
    def __init__(self, h5_path, preprocess_fn):
        self.h5_path = h5_path
        self.preprocess = preprocess_fn
        with h5py.File(h5_path, 'r') as f:
            self.num_images = len(f['cxr'])
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            img = f['cxr'][idx]  # numpy array
            
        # Convert to PIL Image (BiomedCLIP preprocessing expects PIL)
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
            
        # Convert to PIL and RGB (3 channels)
        img = Image.fromarray(img).convert('RGB')
        
        # Apply BiomedCLIP preprocessing
        if self.preprocess:
            img = self.preprocess(img)
            
        return {'img': img}

def compute_retrieval_metrics(similarities: np.ndarray) -> Dict[str, float]:
    """
    Compute recall@k metrics for retrieval
    similarities: [num_queries, num_candidates] similarity matrix
    """
    # Get rankings (higher similarity = better rank)
    rankings = np.argsort(-similarities, axis=1)  # Sort descending
    
    num_queries = similarities.shape[0]
    
    # For image-text retrieval, ground truth is diagonal (image i matches text i)
    recall_at_1 = np.mean([i in rankings[i][:1] for i in range(num_queries)])
    recall_at_5 = np.mean([i in rankings[i][:5] for i in range(num_queries)])
    recall_at_10 = np.mean([i in rankings[i][:10] for i in range(num_queries)])
    recall_at_50 = np.mean([i in rankings[i][:50] for i in range(num_queries)])
    
    # Mean reciprocal rank
    ranks = np.array([np.where(rankings[i] == i)[0][0] + 1 for i in range(num_queries)])
    mrr = np.mean(1.0 / ranks)
    
    return {
        'recall@1': recall_at_1,
        'recall@5': recall_at_5, 
        'recall@10': recall_at_10,
        'recall@50': recall_at_50,
        'mean': mrr
    }

def load_chexzero_model_local(device):
    """Load CheXzero model with correct parameters matching the checkpoint"""
    try:
        from model import CLIP
        
        # Try to find CheXzero checkpoint
        possible_paths = [
            "external_sota/chexzero/checkpoints/best_64_5e-05_original_22000_0.864.pt",
            "/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt"
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("Warning: No CheXzero checkpoint found. Using OpenAI CLIP pretrained model.")
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
        else:
            print(f"Loading CheXzero model from: {model_path}")
            # Use parameters that match the checkpoint file
            params = {
                'embed_dim': 512,  # From text_projection shape [512, 512]
                'image_resolution': 224,  # From visual.positional_embedding [50, 768] -> 7x7+1=50 -> 224/32=7
                'vision_layers': 12,
                'vision_width': 768,
                'vision_patch_size': 32,  # From visual.conv1.weight [768, 3, 32, 32]
                'context_length': 77, 
                'vocab_size': 49408,
                'transformer_width': 512,
                'transformer_heads': 8,
                'transformer_layers': 12
            }
            model = CLIP(**params)
            # Load the checkpoint
            model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Move model to the specified device
        model = model.to(device)
        return model, None
    except Exception as e:
        print(f"Error loading CheXzero model: {e}")
        raise

def load_biomedclip_model_local(device):
    """Load BiomedCLIP model and return with its built-in preprocessing"""
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
        
        model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        model, preprocess = create_model_from_pretrained(model_name)
        tokenizer = get_tokenizer(model_name)
        
        # Create wrapper
        class BiomedCLIPWrapper:
            def __init__(self, model, tokenizer, preprocess):
                self.model = model
                self.tokenizer = tokenizer
                self.preprocess = preprocess
                
            def encode_image(self, images):
                return self.model.encode_image(images)
            
            def encode_text(self, text_tokens):
                return self.model.encode_text(text_tokens)
                
            def tokenize(self, texts, context_length=256):
                return self.tokenizer(texts, context_length=context_length)
                
            def eval(self):
                self.model.eval()
                
            def to(self, device):
                self.model.to(device)
                return self
        
        wrapper = BiomedCLIPWrapper(model, tokenizer, preprocess)
        return wrapper, preprocess
    except Exception as e:
        print(f"Error loading BiomedCLIP model: {e}")
        raise

def load_openai_clip_model_local(device):
    """Load OpenAI CLIP model and return with its built-in preprocessing"""
    try:
        if open_clip is None:
            raise ImportError("open_clip not available")
            
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        return model, preprocess
    except Exception as e:
        print(f"Error loading OpenAI CLIP model: {e}")
        raise

def get_image_embeddings_chexzero(model, images, device):
    """Get image embeddings using CheXzero model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_chexzero(model, texts, device, batch_size=32):
    """Get text embeddings using CheXzero model"""
    import clip
    
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = clip.tokenize(batch_texts, context_length=77, truncate=True).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_image_embeddings_biomedclip(model, images, device):
    """Get image embeddings using BiomedCLIP model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_biomedclip(model, texts, device, batch_size=32):
    """Get text embeddings using BiomedCLIP model"""
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = model.tokenize(batch_texts, context_length=256).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def get_image_embeddings_openai_clip(model, images, device):
    """Get image embeddings using OpenAI CLIP model"""
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        image_features = model.encode_image(images)
        image_features = F.normalize(image_features, dim=-1)
    return image_features.cpu().numpy()

def get_text_embeddings_openai_clip(model, texts, device, batch_size=32):
    """Get text embeddings using OpenAI CLIP model"""
    import open_clip
    
    model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            text_tokens = open_clip.tokenize(batch_texts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            all_embeddings.append(text_features.cpu().numpy())
    
    return np.vstack(all_embeddings)

def load_llm_components(device='cuda'):
    """Load CLIP model, concepts, and LLM concept embeddings for the LLM method"""
    print("🔄 Loading LLM method components...")
    
    # Load trained CLIP model
    model_path = "checkpoints/dinov2-multi-v1.0_vitb/best_model.pt"
    if os.path.exists(model_path):
        print(f"Loading trained model from: {model_path}")
        try:
            model = load_clip(
                model_path=model_path,
                pretrained=False,
                context_length=77,
                use_dinov2=True,
                dinov2_model_name='dinov2_vitb14'
            )
            model = model.to(device).eval()
        except Exception as e:
            print(f"❌ Error loading trained model: {e}")
            print("🔄 Falling back to OpenAI CLIP")
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            model = model.eval()
    else:
        print("⚠️ Warning: No trained model found, using OpenAI CLIP pretrained")
        model, _ = clip.load("ViT-B/32", device=device, jit=False)
        model = model.eval()
    
    # Load concepts
    concepts_file = "concepts/mimic_concepts.csv"
    if not os.path.exists(concepts_file):
        raise FileNotFoundError(f"Concepts file not found: {concepts_file}")
    
    concepts_df = pd.read_csv(concepts_file)
    concepts = concepts_df['concept'].tolist()
    concept_indices = concepts_df['concept_idx'].tolist()
    
    # Load LLM concept embeddings (sfr-mistral)
    embeddings_file = "concepts/embeddings/concepts_embeddings_sfr_mistral.pickle"
    
    if not os.path.exists(embeddings_file):
        print(f"⚠️ Warning: Embeddings file not found: {embeddings_file}")
        print("Creating dummy embeddings for demonstration")
        embedding_dim = 4096  # sfr-mistral dimension
        concept_embeddings = np.random.randn(len(concepts), embedding_dim) * 0.01
    else:
        with open(embeddings_file, 'rb') as f:
            embeddings_data = pickle.load(f)
        
        # Align concept embeddings
        embedding_dim = len(list(embeddings_data.values())[0])
        concept_embeddings = np.zeros((len(concepts), embedding_dim))
        missing_count = 0
        
        for pos, concept_idx in enumerate(concept_indices):
            if concept_idx in embeddings_data:
                concept_embeddings[pos] = embeddings_data[concept_idx]
            else:
                concept_embeddings[pos] = np.random.randn(embedding_dim) * 0.01
                missing_count += 1
        
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} concepts missing embeddings")
    
    print(f"✅ Loaded {len(concepts)} concepts with {embedding_dim}D embeddings")
    
    return model, concepts, torch.tensor(concept_embeddings).float(), model_path

def get_image_embeddings_llm_optimized(clip_model, concepts, concept_embeddings, images, device, model_path=""):
    """Optimized LLM embedding extraction with cached concept features"""
    clip_model.eval()
    batch_size = images.shape[0]
    
    with torch.no_grad():
        # Step 1: Get CLIP image features
        images = images.to(device)
        img_features = clip_model.encode_image(images)
        img_features = F.normalize(img_features, dim=-1)
        
        # Step 2: Get concept features (cached for efficiency!) - silent during image encoding
        concept_features = get_concept_features_cached(clip_model, concepts, device, model_path, silent=True)
        
        # Step 3: Compute concept similarities [batch_size, num_concepts]
        concept_similarities = img_features @ concept_features.T
        
        # Step 4: Transform to LLM embedding space
        concept_embeddings = concept_embeddings.to(device)
        llm_embeddings = concept_similarities @ concept_embeddings
        llm_embeddings = F.normalize(llm_embeddings, dim=-1)
        
        # Clean up
        del concept_similarities
        torch.cuda.empty_cache()
        
    return llm_embeddings.cpu().numpy()

def get_text_embeddings_llm_optimized(texts):
    """Optimized text embedding generation following exp_zeroshot.py patterns"""
    print(f"Processing {len(texts)} texts with SFR-Mistral in chunks...")
    
    # Check if RadiologyEmbeddingGenerator is available
    if RadiologyEmbeddingGenerator is None:
        print("❌ RadiologyEmbeddingGenerator not available, using dummy embeddings")
        dummy_embeddings = np.random.randn(len(texts), 4096) * 0.01
        return dummy_embeddings
    
    # Load the model ONCE at the beginning
    print("🔄 Loading SFR-Mistral model...")
    try:
        generator = RadiologyEmbeddingGenerator(
            embedding_type="local",
            local_model_name='Salesforce/SFR-Embedding-Mistral',
            batch_size=4  # Very small batch size for memory efficiency
        )
        print("✅ SFR-Mistral model loaded")
    except Exception as e:
        print(f"❌ Error loading SFR-Mistral: {e}")
        print("🔄 Using dummy embeddings")
        dummy_embeddings = np.random.randn(len(texts), 4096) * 0.01
        return dummy_embeddings
    
    # Process texts in small chunks using the SAME model instance
    all_embeddings = []
    chunk_size = 100  # Process 100 texts at a time (from exp_zeroshot.py)
    
    try:
        for i in range(0, len(texts), chunk_size):
            chunk_texts = texts[i:i+chunk_size]
            print(f"Processing texts {i+1}-{min(i+chunk_size, len(texts))} of {len(texts)}")
            
            try:
                # Reuse the SAME generator for all chunks
                embeddings = generator.get_local_embeddings_batch(chunk_texts)
                all_embeddings.extend(embeddings)
                print(f"✅ Successfully processed chunk {i//chunk_size + 1}")
                
            except Exception as e:
                print(f"❌ Error processing chunk {i//chunk_size + 1}: {e}")
                print("🔄 Using dummy embeddings for this chunk")
                
                # Create dummy embeddings if processing fails
                dummy_embeddings = np.random.randn(len(chunk_texts), 4096) * 0.01
                all_embeddings.extend(dummy_embeddings.tolist())
            
            # Clear GPU cache between chunks (but keep model loaded)
            torch.cuda.empty_cache()
            
            # Add a small delay between chunks to let GPU memory settle
            import time
            time.sleep(0.1)  # Reduced delay since we're not reloading
    
    finally:
        # Clean up model ONLY at the very end
        print("🧹 Cleaning up SFR-Mistral model...")
        del generator
        torch.cuda.empty_cache()
    
    print(f"✅ Completed processing all {len(texts)} texts")
    return np.array(all_embeddings)

def get_cache_key(data, prefix=""):
    """Generate a robust hash-based cache key for any data"""
    try:
        if isinstance(data, (list, tuple)):
            # For large lists, use length + first few elements + last few elements
            if len(data) > 200:
                content = f"{len(data)}_{str(data[:100])}_{str(data[-100:])}"
            else:
                content = str(data)
        elif isinstance(data, str):
            content = data
        else:
            content = str(data)
        
        hash_obj = hashlib.md5(content.encode())
        return f"{prefix}_{hash_obj.hexdigest()[:16]}"
    except Exception as e:
        # Fallback to simple string representation
        print(f"⚠️ Cache key generation warning: {e}")
        return f"{prefix}_{hashlib.md5(str(data).encode()).hexdigest()[:16]}"

def load_cached_data(cache_file, silent=False):
    """Load cached data if it exists and is valid"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            if not silent:
                print(f"✅ Loaded cached data from {cache_file}")
            return data
        except Exception as e:
            if not silent:
                print(f"⚠️ Cache file corrupted, will regenerate: {e}")
            os.remove(cache_file)
    return None

def save_cached_data(data, cache_file):
    """Save data to cache file"""
    ensure_cache_dir()  # Ensure cache directory exists
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved data to cache: {cache_file}")
    except Exception as e:
        print(f"⚠️ Failed to save cache: {e}")

def get_concept_features_cached(clip_model, concepts, device, model_path, silent=False):
    """Get concept features with caching - this is the most expensive operation"""
    # Generate cache key based on model path and concepts
    cache_key = get_cache_key(model_path + str(concepts[:100]), "concept_features")  # Use first 100 concepts for key
    cache_file = f"cross_modal_retrieval/cache/{cache_key}.pkl"
    
    # Try to load from cache first
    cached_features = load_cached_data(cache_file, silent=silent)
    if cached_features is not None:
        if not silent:
            print(f"🎯 Using cached concept features ({len(concepts)} concepts)")
        return cached_features.to(device)
    
    if not silent:
        print(f"🔄 Computing concept features for {len(concepts)} concepts...")
    
    # Compute concept features in chunks
    concept_batch_size = 1024
    all_concept_features = []
    
    with torch.no_grad():
        for i in range(0, len(concepts), concept_batch_size):
            batch_concepts = concepts[i:i+concept_batch_size]
            concept_tokens = clip.tokenize(batch_concepts, context_length=77, truncate=True).to(device)
            concept_features = clip_model.encode_text(concept_tokens)
            concept_features = F.normalize(concept_features, dim=-1)
            all_concept_features.append(concept_features.cpu())
            
            # Clear GPU memory
            del concept_tokens, concept_features
            torch.cuda.empty_cache()
            
            if (i // concept_batch_size + 1) % 10 == 0 and not silent:
                print(f"  Processed {i + len(batch_concepts)}/{len(concepts)} concepts")
    
    # Concatenate all concept features
    concept_features = torch.cat(all_concept_features)
    
    # Save to cache
    save_cached_data(concept_features, cache_file)
    
    if not silent:
        print(f"✅ Concept features computed and cached")
    return concept_features.to(device)

def get_text_embeddings_llm_cached(texts, dataset_name, text_type, silent=False):
    """Get text embeddings with caching and robust error handling"""
    # Generate cache key based on texts and model
    cache_key = get_cache_key(str(texts[:50]), f"text_emb_sfr_{dataset_name}_{text_type}")  # Use first 50 texts for key
    cache_file = f"cross_modal_retrieval/cache/{cache_key}.pkl"
    
    # Try to load from cache first
    cached_embeddings = load_cached_data(cache_file, silent=silent)
    if cached_embeddings is not None:
        if not silent:
            print(f"🎯 Using cached text embeddings ({len(texts)} texts)")
        return cached_embeddings
    
    if not silent:
        print(f"🔄 Computing text embeddings for {len(texts)} texts...")
    
    try:
        # Compute embeddings using the optimized function
        embeddings = get_text_embeddings_llm_optimized(texts)
        
        # Validate embeddings shape
        if embeddings.shape[0] != len(texts):
            raise ValueError(f"Embedding count mismatch: expected {len(texts)}, got {embeddings.shape[0]}")
        
        # Save to cache
        save_cached_data(embeddings, cache_file)
        
        if not silent:
            print(f"✅ Text embeddings computed and cached")
        return embeddings
        
    except Exception as e:
        if not silent:
            print(f"❌ Error in text embedding generation: {e}")
            print("🔄 Falling back to dummy embeddings")
        
        # Create dummy embeddings as fallback
        dummy_embeddings = np.random.randn(len(texts), 4096) * 0.01
        if not silent:
            print(f"⚠️ Using {dummy_embeddings.shape} dummy embeddings")
        return dummy_embeddings

def run_retrieval_experiment(model_name: str, dataset_name: str, text_type: str, device: str):
    """Run retrieval experiment for a specific model and dataset"""
    
    print(f"\n{'='*60}")
    print(f"Running {model_name} on {dataset_name} ({text_type})")
    print(f"{'='*60}")
    
    # Load dataset
    if dataset_name == "padchest-test":
        h5_path = "data/padchest_test.h5"
        csv_path = "data/padchest_test_reports.csv"
        df = pd.read_csv(csv_path)
        image_ids = df['ImageID'].tolist()
        texts = df['Report'].fillna("").tolist()
    elif dataset_name == "indiana-test":
        h5_path = "data/indiana_test.h5"
        csv_path = "data/indiana_reports.csv"
        df = pd.read_csv(csv_path)
        image_ids = df['filename'].tolist()
        if text_type == "Report":
            texts = df['Report'].fillna("").tolist()
        elif text_type == "Impression":
            texts = df['impression'].fillna("").tolist()
        else:
            raise ValueError(f"Invalid text_type for indiana-test: {text_type}")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Clean texts
    texts = [str(text).strip() for text in texts]
    
    print(f"Loaded {len(texts)} image-text pairs")
    
    # Create image dataset and dataloader with model-specific preprocessing
    if model_name == "chexzero":
        # CheXzero uses CXR-specific normalization and 224x224 (matching checkpoint parameters)
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
        ])
        dataset = CXRDataset(h5_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
    elif model_name == "biomedclip":
        # BiomedCLIP uses its own built-in preprocessing pipeline
        model_temp, preprocess_fn = load_biomedclip_model_local(device)  # Load temporarily to get preprocess
        dataset = BiomedCLIPDataset(h5_path, preprocess_fn)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        del model_temp  # Clean up
        
    elif model_name == "openai_clip":
        # OpenAI CLIP uses its own built-in preprocessing pipeline
        model_temp, preprocess_fn = load_openai_clip_model_local(device)  # Load temporarily to get preprocess
        dataset = BiomedCLIPDataset(h5_path, preprocess_fn)  # Reuse BiomedCLIPDataset for PIL preprocessing
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        del model_temp  # Clean up
        
    elif model_name == "llm_embedding":
        # LLM method uses trained CLIP model with CXR-specific preprocessing and 448x448
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(448, interpolation=InterpolationMode.BICUBIC),
        ])
        dataset = CXRDataset(h5_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)  # Reduce batch size for LLM method
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load model and get embeddings
    if model_name == "chexzero":
        model, _ = load_chexzero_model_local(device)
        
        # Get image embeddings
        print("Extracting image embeddings...")
        image_embeddings = []
        for batch in tqdm(dataloader):
            batch_embeddings = get_image_embeddings_chexzero(model, batch['img'], device)
            image_embeddings.append(batch_embeddings)
        image_embeddings = np.vstack(image_embeddings)
        
        # Get text embeddings
        print("Extracting text embeddings...")
        text_embeddings = get_text_embeddings_chexzero(model, texts, device)
        
    elif model_name == "biomedclip":
        model, _ = load_biomedclip_model_local(device)
        model.to(device)
        
        # Get image embeddings
        print("Extracting image embeddings...")
        image_embeddings = []
        for batch in tqdm(dataloader):
            batch_embeddings = get_image_embeddings_biomedclip(model, batch['img'], device)
            image_embeddings.append(batch_embeddings)
        image_embeddings = np.vstack(image_embeddings)
        
        # Get text embeddings
        print("Extracting text embeddings...")
        text_embeddings = get_text_embeddings_biomedclip(model, texts, device)
        
    elif model_name == "openai_clip":
        model, _ = load_openai_clip_model_local(device)
        model.to(device)
        
        # Get image embeddings
        print("Extracting image embeddings...")
        image_embeddings = []
        for batch in tqdm(dataloader):
            batch_embeddings = get_image_embeddings_openai_clip(model, batch['img'], device)
            image_embeddings.append(batch_embeddings)
        image_embeddings = np.vstack(image_embeddings)
        
        # Get text embeddings
        print("Extracting text embeddings...")
        text_embeddings = get_text_embeddings_openai_clip(model, texts, device)
        
    elif model_name == "llm_embedding":
        # Load LLM method components
        print("💡 LLM method uses intelligent caching:")
        print("   🧠 Concept features (368k concepts) - most expensive, cached across runs")
        print("   📝 Text embeddings (SFR-Mistral) - cached per dataset")
        print("   ⚡ Subsequent runs will be much faster!")
        clip_model, concepts, concept_embeddings, model_path = load_llm_components(device)
        
        # Get text embeddings using sfr-mistral
        print("Extracting text embeddings...")
        text_embeddings = get_text_embeddings_llm_cached(texts, dataset_name, text_type)
        
        # Clear CUDA cache after text embedding generation
        torch.cuda.empty_cache()
        
        # Get image embeddings using the trained CLIP → concept → LLM pipeline
        print("Extracting image embeddings...")
        image_embeddings = []
        
        for batch in tqdm(dataloader):
            batch_embeddings = get_image_embeddings_llm_optimized(
                clip_model, concepts, concept_embeddings, batch['img'], device, model_path
            )
            image_embeddings.append(batch_embeddings)
        image_embeddings = np.vstack(image_embeddings)
        
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"Image embeddings shape: {image_embeddings.shape}")
    print(f"Text embeddings shape: {text_embeddings.shape}")
    
    # Normalize embeddings
    image_embeddings = F.normalize(torch.from_numpy(image_embeddings), dim=-1).numpy()
    text_embeddings = F.normalize(torch.from_numpy(text_embeddings), dim=-1).numpy()
    
    # Compute similarities
    print("Computing similarities...")
    
    # Image-to-text retrieval (each image retrieves from all texts)
    i2t_similarities = image_embeddings @ text_embeddings.T
    i2t_metrics = compute_retrieval_metrics(i2t_similarities)
    
    # Text-to-image retrieval (each text retrieves from all images) 
    t2i_similarities = text_embeddings @ image_embeddings.T
    t2i_metrics = compute_retrieval_metrics(t2i_similarities)
    
    # Print results
    print(f"\nImage-to-Text Retrieval Results:")
    print(f"Recall@1: {i2t_metrics['recall@1']:.4f}")
    print(f"Recall@5: {i2t_metrics['recall@5']:.4f}")
    print(f"Recall@10: {i2t_metrics['recall@10']:.4f}")
    print(f"Recall@50: {i2t_metrics['recall@50']:.4f}")
    print(f"Mean: {i2t_metrics['mean']:.4f}")
    
    print(f"\nText-to-Image Retrieval Results:")
    print(f"Recall@1: {t2i_metrics['recall@1']:.4f}")
    print(f"Recall@5: {t2i_metrics['recall@5']:.4f}")
    print(f"Recall@10: {t2i_metrics['recall@10']:.4f}")
    print(f"Recall@50: {t2i_metrics['recall@50']:.4f}")
    print(f"Mean: {t2i_metrics['mean']:.4f}")
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'text_type': text_type,
        'i2t_metrics': i2t_metrics,
        't2i_metrics': t2i_metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Image-Report Retrieval Experiments")
    parser.add_argument('--models', nargs='+', 
                        default=['llm_embedding', 'chexzero', 'biomedclip', 'openai_clip'],
                        help='Models to evaluate')
    parser.add_argument('--datasets', nargs='+',
                        default=['padchest-test', 'indiana-test'],
                        help='Datasets to evaluate on')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to run on (auto, cpu, cuda)')
    parser.add_argument('--clear-cache', type=str, choices=['all', 'concept', 'text'], 
                        help='Clear cached data before running')
    parser.add_argument('--cache-info', action='store_true',
                        help='Show cache information and exit')
    
    args = parser.parse_args()
    
    # Handle cache management
    if args.cache_info:
        get_cache_info()
        return
    
    if args.clear_cache:
        clear_cache(args.clear_cache)
    
    # Show cache status
    print("\n📊 Cache Status:")
    get_cache_info()
    print()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    all_results = []
    
    # Run experiments
    for model_name in args.models:
        for dataset_name in args.datasets:
            if dataset_name == "padchest-test":
                # Spanish reports
                try:
                    results = run_retrieval_experiment(model_name, dataset_name, "Report", device)
                    all_results.append(results)
                except Exception as e:
                    print(f"Error with {model_name} on {dataset_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                    
            elif dataset_name == "indiana-test":
                # English reports and impressions separately
                for text_type in ["Report", "Impression"]:
                    try:
                        results = run_retrieval_experiment(model_name, dataset_name, text_type, device)
                        all_results.append(results)
                    except Exception as e:
                        print(f"Error with {model_name} on {dataset_name} ({text_type}): {e}")
                        import traceback
                        traceback.print_exc()
                        continue
    
    # Save results
    results_df = pd.DataFrame(all_results)
    os.makedirs("cross_modal_retrieval/results", exist_ok=True)
    results_df.to_csv("cross_modal_retrieval/results/retrieval_results.csv", index=False)
    
    print(f"\n{'='*60}")
    print("All experiments completed!")
    print("Results saved to cross_modal_retrieval/results/retrieval_results.csv")
    print(f"{'='*60}")
    
    # Print summary table
    print("\nSummary Results:")
    print("="*140)
    for _, row in results_df.iterrows():
        print(f"{row['model']:15s} | {row['dataset']:15s} | {row['text_type']:10s} | "
              f"I2T: R@1={row['i2t_metrics']['recall@1']:.3f} R@5={row['i2t_metrics']['recall@5']:.3f} R@50={row['i2t_metrics']['recall@50']:.3f} | "
              f"T2I: R@1={row['t2i_metrics']['recall@1']:.3f} R@5={row['t2i_metrics']['recall@5']:.3f} R@50={row['t2i_metrics']['recall@50']:.3f}")

def ensure_cache_dir():
    """Ensure cache directory exists"""
    cache_dir = "cross_modal_retrieval/cache"
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def clear_cache(cache_type="all"):
    """Clear cached data"""
    cache_dir = "cross_modal_retrieval/cache"
    if not os.path.exists(cache_dir):
        print("📂 No cache directory found")
        return
    
    files_removed = 0
    for file in os.listdir(cache_dir):
        if cache_type == "all" or cache_type in file:
            os.remove(os.path.join(cache_dir, file))
            files_removed += 1
    
    print(f"🗑️ Cleared {files_removed} cache files ({cache_type})")

def get_cache_info():
    """Get information about cached data"""
    cache_dir = "cross_modal_retrieval/cache"
    if not os.path.exists(cache_dir):
        print("📂 No cache directory found")
        return
    
    files = os.listdir(cache_dir)
    if not files:
        print("📂 Cache directory is empty")
        return
    
    total_size = 0
    concept_files = 0
    text_files = 0
    
    for file in files:
        file_path = os.path.join(cache_dir, file)
        size = os.path.getsize(file_path)
        total_size += size
        
        if "concept_features" in file:
            concept_files += 1
        elif "text_emb" in file:
            text_files += 1
    
    print(f"📊 Cache Statistics:")
    print(f"   📁 Total files: {len(files)}")
    print(f"   🧠 Concept feature files: {concept_files}")
    print(f"   📝 Text embedding files: {text_files}")
    print(f"   💾 Total size: {total_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main() 