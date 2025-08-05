#!/usr/bin/env python3
"""
Fine-tune MedGemma-4B-IT for CXR report generation using CBM concept scores
Uses TRL SFTTrainer following the official MedGemma demo
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

import torch
from PIL import Image

from transformers import (
    AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig,
    EarlyStoppingCallback
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset

# Import metrics
try:
    from torchmetrics.text import ROUGEScore
    ROUGE_AVAILABLE = True
except ImportError:
    print("Warning: ROUGE metric not available. Install with: pip install torchmetrics")
    ROUGE_AVAILABLE = False

try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    print("Warning: CIDEr metric not available. Install with: pip install pycocoevalcap")
    CIDER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass 
class ModelArguments:
    """Arguments for model configuration"""
    model_name_or_path: str = field(
        default="google/medgemma-4b-it",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    
@dataclass
class DataArguments:
    """Arguments for data configuration"""
    train_h5_images: str = field(
        default="data/mimic.h5",
        metadata={"help": "Path to training images H5 file"}
    )
    train_h5_dataset: str = field(
        default="reports/data/mimic_train_dataset.h5",
        metadata={"help": "Path to training dataset H5 file"}
    ) 
    val_h5_images: str = field(
        default="data/chexpert.h5",
        metadata={"help": "Path to validation images H5 file"}
    )
    val_h5_dataset: str = field(
        default="reports/data/chexpert_validation_dataset.h5",
        metadata={"help": "Path to validation dataset H5 file"}
    )
    max_train_samples: Optional[int] = field(
        default=1000,  # Start with 1K samples for testing, set to None for full dataset
        metadata={"help": "Maximum number of training samples"}
    )
    max_val_samples: Optional[int] = field(
        default=50,  # Reduced for faster debugging, set to 500 for final evaluation
        metadata={"help": "Maximum number of validation samples (should be 500 for CheXpert)"}
    )

class CXRReportDataset:
    """Dataset for CXR report generation with concept scores"""
    
    def __init__(self, h5_images_path, h5_dataset_path, max_samples=None):
        self.h5_images_path = h5_images_path
        self.h5_dataset_path = h5_dataset_path
        
        # Load dataset metadata
        with h5py.File(h5_dataset_path, 'r') as f:
            self.num_samples = len(f['reports'])
            self.num_concepts = f['concept_scores'].shape[1] 
            self.concept_names = [c.decode('utf-8') for c in f['concept_names'][:]]
            
            if max_samples:
                self.num_samples = min(self.num_samples, max_samples)
            
            print(f"✅ Dataset loaded: {self.num_samples} samples, {self.num_concepts} concepts")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample: image, concept_scores, report - with validation"""
        
        # Load report and concept scores from dataset h5 file
        with h5py.File(self.h5_dataset_path, 'r') as f:
            report = f['reports'][idx].decode('utf-8')
            concept_scores = f['concept_scores'][idx]
            
            # Get the original index if available (for proper image retrieval)
            if 'original_indices' in f:
                img_idx = f['original_indices'][idx]
            else:
                img_idx = idx
        
        # Validate concept scores
        if len(concept_scores) != self.num_concepts:
            raise ValueError(f"Sample {idx}: concept_scores length {len(concept_scores)} != expected {self.num_concepts}")
        
        if not np.isfinite(concept_scores).all():
            logger.warning(f"Sample {idx}: Non-finite values in concept_scores, replacing with zeros")
            concept_scores = np.nan_to_num(concept_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Load image from CXR h5 file
        with h5py.File(self.h5_images_path, 'r') as f:
            if img_idx >= len(f['cxr']):
                logger.warning(f"Sample {idx}: img_idx {img_idx} >= dataset size {len(f['cxr'])}, using idx {idx}")
                img_idx = idx
                if img_idx >= len(f['cxr']):
                    raise ValueError(f"Sample {idx}: Cannot find valid image index")
            
            img_data = f['cxr'][img_idx]
            
            # Validate image data
            if img_data.size == 0:
                raise ValueError(f"Sample {idx}: Empty image data")
            
            # Convert to proper format for MedGemma (448x448 RGB)
            if len(img_data.shape) == 3 and img_data.shape[0] == 3:
                img_data = np.transpose(img_data, (1, 2, 0))
            elif len(img_data.shape) == 2:
                # Grayscale image, add channel dimension
                img_data = np.expand_dims(img_data, axis=-1)
            
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            
            # Ensure 448x448 resolution as requested
            image = Image.fromarray(img_data).convert('RGB')
            if image.size != (448, 448):
                image = image.resize((448, 448), Image.Resampling.LANCZOS)
        
        # Validate final data
        if report.strip() == "":
            raise ValueError(f"Sample {idx}: Empty report")
        
        if concept_scores.max() <= 0:
            logger.warning(f"Sample {idx}: All concept scores are <= 0, this may indicate a data issue")
        
        return {
            'image': image,
            'concept_scores': concept_scores.astype(np.float32),
            'report': report,
            'sample_id': idx
        }

def format_prompt_with_concepts(concept_scores, concept_names):
    """Create a prompt that includes ALL 68 concept information for comprehensive training/validation"""
    
    # Use ALL concepts instead of filtering - this ensures the model learns from complete concept information
    # Sort concepts by score (highest first) but include ALL 68 concepts
    all_indices = np.argsort(concept_scores)[::-1]  # All concepts sorted by score (descending)
    
    # Build comprehensive concept text with ALL concepts
    concept_text = "Comprehensive radiological concept analysis (all 68 concepts):\n"
    
    # Group concepts by score ranges for better organization
    high_concepts = []    # > 0.3
    medium_concepts = []  # 0.1 to 0.3
    low_concepts = []     # 0.0 to 0.1
    negative_concepts = [] # < 0.0
    
    for i in all_indices:
        score = concept_scores[i]
        concept = concept_names[i] if i < len(concept_names) else f"concept_{i}"
        
        if score > 0.3:
            high_concepts.append(f"- {concept}: {score:.3f}")
        elif score >= 0.1:
            medium_concepts.append(f"- {concept}: {score:.3f}")
        elif score >= 0.0:
            low_concepts.append(f"- {concept}: {score:.3f}")
        else:
            negative_concepts.append(f"- {concept}: {score:.3f}")
    
    # Add concepts by priority (high first, but include all)
    if high_concepts:
        concept_text += "High confidence findings (>0.3):\n" + "\n".join(high_concepts) + "\n\n"
    
    if medium_concepts:
        concept_text += "Moderate confidence findings (0.1-0.3):\n" + "\n".join(medium_concepts) + "\n\n"
    
    if low_concepts:
        concept_text += "Low confidence findings (0.0-0.1):\n" + "\n".join(low_concepts) + "\n\n"
    
    if negative_concepts:
        concept_text += "Absent/negative findings (<0.0):\n" + "\n".join(negative_concepts) + "\n\n"
    
    # Ensure we have meaningful content (should always be true with 68 concepts)
    total_concepts = len(high_concepts) + len(medium_concepts) + len(low_concepts) + len(negative_concepts)
    concept_text += f"Total concepts analyzed: {total_concepts}/68\n"
    
    prompt = f"""Based on this chest X-ray image and the comprehensive radiological concept analysis, generate a detailed radiological report with FINDINGS and IMPRESSION sections.

{concept_text}

Using the complete concept analysis above, please provide a comprehensive radiological report in the following format:
FINDINGS: [Describe the radiological findings observed in the image, referencing the relevant concepts]
IMPRESSION: [Provide clinical interpretation and conclusions based on the concept analysis]"""

    return prompt

def prepare_dataset_for_sft(dataset, processor):
    """Convert our dataset to format expected by SFTTrainer - following MedGemma demo pattern exactly"""
    
    n_threads = mp.cpu_count()  # Use all available CPU cores
    print(f"Converting {len(dataset)} samples to SFT format using {n_threads} threads...")
    
    def process_sample(i):
        """Process a single sample - this function will be called in parallel"""
        try:
            sample = dataset[i]
            
            # Create prompt with concept information
            prompt = format_prompt_with_concepts(
                sample['concept_scores'], 
                dataset.concept_names
            )
            
            # Create the training example in MedGemma chat format (exactly like demo)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": sample['report']}
                    ]
                }
            ]
            
            return {
                "messages": messages,
                "image": sample['image'],
                "sample_id": sample['sample_id'],
                "index": i  # Keep track of original order
            }
        except Exception as e:
            logger.warning(f"Failed to process sample {i}: {e}")
            return None
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor for parallel processing (better for I/O bound tasks like H5 reading)
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        # Create a list of all indices to process
        indices = list(range(len(dataset)))
        
        # Process samples in parallel with progress bar
        results = list(tqdm(
            executor.map(process_sample, indices),
            total=len(indices),
            desc=f"Converting dataset ({n_threads} threads)",
            unit="samples"
        ))
    
    # Filter out failed samples and sort by original index to maintain order
    processed_data = [result for result in results if result is not None]
    processed_data.sort(key=lambda x: x['index'])  # Maintain original order
    
    # Remove the temporary index field
    for item in processed_data:
        del item['index']
    
    processing_time = time.time() - start_time
    samples_per_second = len(processed_data) / processing_time if processing_time > 0 else 0
    
    print(f"✅ Converted {len(processed_data)} samples to SFT format")
    print(f"⚡ Processing took {processing_time:.2f}s ({samples_per_second:.1f} samples/sec with {n_threads} threads)")
    
    if len(processed_data) < len(dataset):
        failed_count = len(dataset) - len(processed_data)
        logger.warning(f"⚠️  {failed_count} samples failed to process")
    
    return HFDataset.from_list(processed_data)

def collate_fn(examples: List[Dict[str, Any]], processor):
    """Custom data collator following MedGemma demo pattern exactly"""
    texts = []
    images = []
    
    for example in examples:
        images.append([example["image"].convert("RGB")])
        texts.append(processor.apply_chat_template(
            example["messages"], 
            add_generation_prompt=False, 
            tokenize=False
        ).strip())

    # Tokenize the texts and process the images
    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

    # The labels are the input_ids, with padding and image tokens masked
    labels = batch["input_ids"].clone()

    # Mask image tokens - following MedGemma demo exactly
    image_token_id = [
        processor.tokenizer.convert_tokens_to_ids(
            processor.tokenizer.special_tokens_map["boi_token"]
        )
    ]
    # Mask tokens that are not used in the loss computation
    labels[labels == processor.tokenizer.pad_token_id] = -100
    labels[labels == image_token_id] = -100
    labels[labels == 262144] = -100

    batch["labels"] = labels
    return batch

class ReportMetrics:
    """Evaluation metrics for report generation - supports both ROUGE and CIDEr"""
    
    def __init__(self):
        # Initialize ROUGE metric (preferred)
        if ROUGE_AVAILABLE:
            self.rouge = ROUGEScore(rouge_keys=('rouge1', 'rouge2', 'rougeL'))
            self.primary_metric = "rouge"
            logger.info("Using ROUGE score as primary metric")
        elif CIDER_AVAILABLE:
            self.cider = Cider()
            self.primary_metric = "cider"
            logger.info("Using CIDEr-D score as fallback metric")
        else:
            self.primary_metric = None
            logger.warning("No evaluation metrics available")
    
    def compute(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """Compute evaluation metrics"""
        if not predictions or not references:
            logger.warning("Empty predictions or references")
            return {"score": 0.0}
        
        # Use ROUGE if available (preferred)
        if self.primary_metric == "rouge":
            try:
                # Prepare data for ROUGE
                pred_list = [pred.strip() for pred in predictions if pred.strip()]
                ref_list = [refs[0].strip() for refs in references if refs and refs[0].strip()]
                
                if len(pred_list) != len(ref_list) or len(pred_list) == 0:
                    return {"score": 0.0}
                
                # Compute ROUGE scores
                scores = self.rouge(pred_list, ref_list)
                rouge_l = float(scores['rougeL_fmeasure'].mean())
                
                return {
                    "score": rouge_l,
                    "rouge1": float(scores['rouge1_fmeasure'].mean()),
                    "rouge2": float(scores['rouge2_fmeasure'].mean()),
                    "rougeL": rouge_l
                }
            except Exception as e:
                logger.warning(f"ROUGE computation failed: {e}")
                return {"score": 0.0}
        
        # Fallback to CIDEr if ROUGE not available
        elif self.primary_metric == "cider":
            try:
                # Prepare data for CIDEr
                valid_pairs = [(i, pred.strip(), [ref.strip() for ref in refs if ref.strip()]) 
                              for i, (pred, refs) in enumerate(zip(predictions, references))
                              if pred.strip() and refs and any(ref.strip() for ref in refs)]
                
                if not valid_pairs:
                    return {"score": 0.0}
                
                res = {i: [pred] for i, pred, _ in valid_pairs}
                gts = {i: refs for i, _, refs in valid_pairs}
                
                score, _ = self.cider.compute_score(gts, res)
                return {"score": float(score), "cider_d": float(score)}
            except Exception as e:
                logger.warning(f"CIDEr computation failed: {e}")
                return {"score": 0.0}
        
        return {"score": 0.0}

class CustomSFTTrainer(SFTTrainer):
    """Custom SFTTrainer with progress bars and evaluation metrics"""
    
    def __init__(self, *args, raw_val_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_val_dataset = raw_val_dataset
        self.metrics = ReportMetrics()
        self.best_score = 0.0
        
    def training_step(self, model, inputs):
        """Training step with loss tracking"""
        model.train()
        loss = super().training_step(model, inputs)
        
        # Log training loss with progress info
        if self.state.global_step % self.args.logging_steps == 0:
            current_lr = self.get_last_lr()[0] if self.get_last_lr() else self.args.learning_rate
            progress = (self.state.global_step / self.state.max_steps) * 100 if self.state.max_steps > 0 else 0
            
            print(f"Step {self.state.global_step}/{self.state.max_steps} ({progress:.1f}%) | "
                  f"Loss: {loss:.4f} | LR: {current_lr:.2e}")
        
        return loss
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with progress bars and metrics"""
        
        # Standard evaluation first
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if not self.raw_val_dataset:
            return eval_result
        
        # Generate predictions with progress bar
        print(f"🔬 Evaluating on {len(self.raw_val_dataset)} samples with CXR images + all 68 concepts...")
        
        predictions = []
        references = []
        eval_samples = min(len(self.raw_val_dataset), 50)  # Use 50 for debugging
        
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(eval_samples), desc="Generating reports", unit="sample"):
                sample = self.raw_val_dataset[i]
                
                # Create prompt with all 68 concepts
                prompt = format_prompt_with_concepts(
                    sample['concept_scores'], 
                    self.raw_val_dataset.concept_names
                )
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }]
                
                text = self.processing_class.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                
                inputs = self.processing_class(
                    text=[text],
                    images=[[sample['image']]],
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)
                
                # Generate response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=self.processing_class.tokenizer.eos_token_id,
                )
                
                pred_text = self.processing_class.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:], 
                    skip_special_tokens=True
                ).strip()
                
                predictions.append(pred_text)
                references.append([sample['report']])
        
        # Compute metrics
        metric_result = self.metrics.compute(predictions, references)
        primary_score = metric_result.get("score", 0.0)
        
        # Add primary metric to eval results
        eval_result[f"{metric_key_prefix}_score"] = primary_score
        
        # Add specific metric scores
        if "rougeL" in metric_result:
            eval_result[f"{metric_key_prefix}_rougeL"] = metric_result["rougeL"]
            metric_name = "ROUGE-L"
        elif "cider_d" in metric_result:
            eval_result[f"{metric_key_prefix}_cider_d"] = metric_result["cider_d"]
            metric_name = "CIDEr-D"
        else:
            metric_name = "Score"
        
        print(f"📊 {metric_name}: {primary_score:.4f} (best: {self.best_score:.4f})")
        
        # Save best model
        if primary_score > self.best_score:
            self.best_score = primary_score
            best_model_path = os.path.join(self.args.output_dir, "best_model")
            self.save_model(best_model_path)
            print(f"💾 New best model saved: {primary_score:.4f}")
        
        return eval_result

def validate_dataset_pipeline(dataset, dataset_name="Dataset", num_samples_to_check=10):
    """Validate that the dataset pipeline consistently provides CXR images + concepts"""
    
    print(f"\n=== Validating {dataset_name} Pipeline ===")
    
    # Check a subset of samples
    num_to_check = min(num_samples_to_check, len(dataset))
    validation_results = {
        "has_image": 0,
        "has_concepts": 0,
        "has_report": 0,
        "prompt_includes_concepts": 0,
        "concept_count_stats": [],
        "issues": []
    }
    
    for i in range(num_to_check):
        try:
            # Get sample
            sample = dataset[i]
            
            # Check image
            if 'image' in sample and sample['image'] is not None:
                image = sample['image']
                if hasattr(image, 'size') and image.size == (448, 448):
                    validation_results["has_image"] += 1
                else:
                    validation_results["issues"].append(f"Sample {i}: Invalid image size {getattr(image, 'size', 'unknown')}")
            else:
                validation_results["issues"].append(f"Sample {i}: Missing or invalid image")
            
            # Check concept scores
            if 'concept_scores' in sample and sample['concept_scores'] is not None:
                concept_scores = sample['concept_scores']
                if len(concept_scores.shape) == 1 and concept_scores.shape[0] == dataset.num_concepts:
                    validation_results["has_concepts"] += 1
                    non_zero_count = (concept_scores > 0).sum()
                    validation_results["concept_count_stats"].append(non_zero_count)
                else:
                    validation_results["issues"].append(f"Sample {i}: Invalid concept_scores shape {concept_scores.shape}")
            else:
                validation_results["issues"].append(f"Sample {i}: Missing concept_scores")
            
            # Check report
            if 'report' in sample and sample['report'] and len(sample['report'].strip()) > 0:
                validation_results["has_report"] += 1
            else:
                validation_results["issues"].append(f"Sample {i}: Missing or empty report")
            
            # Check prompt generation
            if 'concept_scores' in sample:
                try:
                    prompt = format_prompt_with_concepts(sample['concept_scores'], dataset.concept_names)
                    # Check that prompt includes comprehensive concept analysis (all 68 concepts)
                    if "Comprehensive radiological concept analysis (all 68 concepts):" in prompt and "Total concepts analyzed: 68/68" in prompt:
                        validation_results["prompt_includes_concepts"] += 1
                    else:
                        validation_results["issues"].append(f"Sample {i}: Prompt doesn't include all 68 concepts")
                except Exception as e:
                    validation_results["issues"].append(f"Sample {i}: Prompt generation failed: {e}")
            
        except Exception as e:
            validation_results["issues"].append(f"Sample {i}: Failed to load sample: {e}")
    
    # Print results
    print(f"✅ Images: {validation_results['has_image']}/{num_to_check}")
    print(f"✅ Concept scores: {validation_results['has_concepts']}/{num_to_check}")
    print(f"✅ Reports: {validation_results['has_report']}/{num_to_check}")
    print(f"✅ Prompts with concepts: {validation_results['prompt_includes_concepts']}/{num_to_check}")
    
    if validation_results["concept_count_stats"]:
        concept_stats = np.array(validation_results["concept_count_stats"])
        print(f"📊 Non-zero concepts per sample: mean={concept_stats.mean():.1f}, min={concept_stats.min()}, max={concept_stats.max()}")
    
    # Report issues
    if validation_results["issues"]:
        print(f"⚠️  Found {len(validation_results['issues'])} issues:")
        for issue in validation_results["issues"][:5]:  # Show first 5 issues
            print(f"   - {issue}")
        if len(validation_results["issues"]) > 5:
            print(f"   ... and {len(validation_results['issues']) - 5} more issues")
    else:
        print("🎉 No issues found!")
    
    # Overall validation
    all_good = (
        validation_results["has_image"] == num_to_check and
        validation_results["has_concepts"] == num_to_check and 
        validation_results["has_report"] == num_to_check and
        validation_results["prompt_includes_concepts"] == num_to_check and
        len(validation_results["issues"]) == 0
    )
    
    if all_good:
        print(f"✅ {dataset_name} pipeline validation PASSED")
    else:
        print(f"❌ {dataset_name} pipeline validation FAILED")
    
    return all_good

def main():
    """Main training function"""
    
    # Use default arguments (following demo pattern)
    model_args = ModelArguments()
    data_args = DataArguments()
    
    print("\n=== Loading Datasets ===")
    
    # Load datasets
    train_dataset = CXRReportDataset(
        h5_images_path=data_args.train_h5_images,
        h5_dataset_path=data_args.train_h5_dataset,
        max_samples=data_args.max_train_samples
    )
    
    val_dataset = CXRReportDataset(
        h5_images_path=data_args.val_h5_images,
        h5_dataset_path=data_args.val_h5_dataset,
        max_samples=data_args.max_val_samples  # Should be 500 for CheXpert
    )
    
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)} (debugging with 50, use 500 for final evaluation)")
    
    # Note: Using reduced sample size for debugging
    if len(val_dataset) == 50:
        print(f"🔧 Debug mode: Using {len(val_dataset)} validation samples for faster iteration")
    elif len(val_dataset) != 500:
        print(f"⚠️  Warning: Expected 500 validation samples for final evaluation, got {len(val_dataset)}")
    
    # CRITICAL: Validate that every sample contains CXR images + concepts + reports
    print("\n=== Validating Data Pipeline ===")
    train_valid = validate_dataset_pipeline(train_dataset, "Training", num_samples_to_check=20)
    val_valid = validate_dataset_pipeline(val_dataset, "Validation (CheXpert)", num_samples_to_check=20)
    
    if not train_valid or not val_valid:
        raise ValueError("❌ Data pipeline validation failed! Every sample must contain CXR image + concept scores + report")
    
    print("🎉 Data pipeline validation passed - all samples contain CXR images + concepts + reports")
    
    # Verify CheXpert validation dataset specifically
    print("\n=== CheXpert Validation Dataset Verification ===")
    print(f"✅ CheXpert validation samples: {len(val_dataset)}")
    print(f"✅ Concept dimensions: {val_dataset.num_concepts} concepts per sample")
    print(f"✅ Image source: {data_args.val_h5_images}")
    print(f"✅ Dataset source: {data_args.val_h5_dataset}")
    
    # Quick check of concept score distribution
    sample_concept_scores = []
    for i in range(min(10, len(val_dataset))):
        sample = val_dataset[i]
        max_score = sample['concept_scores'].max()
        non_zero_count = (sample['concept_scores'] > 0).sum()
        sample_concept_scores.append((max_score, non_zero_count))
    
    avg_max_score = np.mean([score[0] for score in sample_concept_scores])
    avg_non_zero = np.mean([score[1] for score in sample_concept_scores])
    
    print(f"📊 CheXpert concept score stats (first 10 samples):")
    print(f"   • Average max concept score: {avg_max_score:.3f}")
    print(f"   • Average non-zero concepts per sample: {avg_non_zero:.1f}")
    print("✅ CheXpert validation will use CXR images + concept scores for evaluation")
    
    print("\n=== Loading MedGemma-4B-IT Model ===")
    
    # Check GPU capability (following MedGemma demo exactly)
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("GPU does not support bfloat16, please use a GPU that supports bfloat16.")
    
    # Model configuration following MedGemma demo exactly
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Quantization config (following MedGemma demo exactly)
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )
    
    # Load MedGemma model (correct class and model name)
    model = AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,  # google/medgemma-4b-it
        **model_kwargs
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    processor.tokenizer.padding_side = "right"  # Important for training
    
    print("✅ MedGemma-4B-IT model and processor loaded")
    
    # LoRA configuration (following MedGemma demo exactly)
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"],
    )
    
    # Convert datasets for SFTTrainer
    print("\n=== Preparing Data for SFT ===")
    train_sft_dataset = prepare_dataset_for_sft(train_dataset, processor)
    val_sft_dataset = prepare_dataset_for_sft(val_dataset, processor)
    
    print(f"✅ SFT Training samples: {len(train_sft_dataset)}")
    print(f"✅ SFT Validation samples: {len(val_sft_dataset)}")
    
    # Training configuration (following demo but adapted for our requirements)
    num_train_epochs = 3  # Following demo
    learning_rate = 1e-5  # Following demo exactly
    
    args = SFTConfig(
        output_dir="reports/results/medgemma-cxr-sft",               # Our output directory
        num_train_epochs=num_train_epochs,                          # Number of training epochs
        per_device_train_batch_size=1,                              # Very small for debugging (increase to 2 for final)
        per_device_eval_batch_size=1,                               # Very small for debugging
        gradient_accumulation_steps=4,                              # Reduced for faster debugging (effective batch size = 4)
        gradient_checkpointing=True,                                # Enable gradient checkpointing
        optim="adamw_torch_fused",                                  # Use fused AdamW optimizer
        logging_steps=50,                                           # Number of steps between logs
        save_strategy="steps",                                      # Save checkpoint every eval_steps
        eval_strategy="steps",                                      # Evaluate every eval_steps
        eval_steps=25,                                              # More frequent evaluation for debugging (use 50 for final)
        save_steps=25,                                              # More frequent saving for debugging
        learning_rate=learning_rate,                                # Learning rate from demo
        bf16=True,                                                  # Use bfloat16 precision
        max_grad_norm=0.3,                                          # Max gradient norm from demo
        warmup_ratio=0.03,                                          # Warmup ratio from demo
        lr_scheduler_type="linear",                                 # Use linear learning rate scheduler
        report_to="tensorboard",                                    # Report metrics to tensorboard
        gradient_checkpointing_kwargs={"use_reentrant": False},     # Non-reentrant checkpointing
        dataset_kwargs={"skip_prepare_dataset": True},              # Skip default dataset preparation
        remove_unused_columns=False,                                # Keep columns for data collator
        label_names=["labels"],                                     # Input keys for labels
        load_best_model_at_end=True,                                # Load best model at end
        metric_for_best_model="eval_score",                      # Use primary score as best metric
        greater_is_better=True,                                     # Higher score is better
    )
    
    # Early stopping callback (patience=5 as requested)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=5,
        early_stopping_threshold=0.001  # Minimum improvement threshold
    )
    
    # Create trainer (following demo pattern exactly)
    trainer = CustomSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_sft_dataset,
        eval_dataset=val_sft_dataset,  # Use all 500 validation samples
        peft_config=peft_config,
        processing_class=processor,
        raw_val_dataset=val_dataset,  # Pass raw dataset for CIDEr-D evaluation
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=[early_stopping],
    )
    
    print("\n=== Starting Training ===")
    print("✅ Will evaluate every 25 steps with ROUGE/CIDEr metrics on CheXpert validation")
    print("✅ Training with progress bars and loss tracking")
    print("✅ CheXpert evaluation uses CXR images + ALL 68 concept scores")
    print("✅ Will save model when score improves")
    print("✅ Early stopping with patience=5")
    print(f"🔧 Debug mode: Using {len(val_dataset)} samples")
    
    # Initial evaluation
    print("Running initial evaluation...")
    trainer.evaluate()
    
    # Train the model
    trainer.train()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_results = trainer.evaluate()
    
    print("🎉 Training completed!")
    print(f"Best score: {trainer.best_score:.4f}")
    print(f"Final score: {final_results.get('eval_score', 'N/A')}")
    
    # Save final model
    trainer.save_model("reports/results/medgemma-cxr-sft/final_model")
    processor.save_pretrained("reports/results/medgemma-cxr-sft/final_model")
    
    return trainer, final_results

if __name__ == "__main__":
    print("🚀 Starting MedGemma-4B-IT Fine-tuning with SFTTrainer")
    n_cpu_cores = mp.cpu_count()
    print("📋 Configuration (Debug Mode - Optimized for Fast Iteration):")
    print("   • Model: google/medgemma-4b-it")
    print("   • Metrics: ROUGE (preferred) or CIDEr-D (fallback)")
    print("   • Training: MIMIC dataset with CXR images + ALL 68 concept scores")
    print("   • Validation: CheXpert dataset with CXR images + ALL 68 concept scores")
    print("   • Progress bars: Training loss tracking + validation progress")
    print("   • Training samples: 1,000 (debugging) | Full: 278K+ samples")
    print("   • Validation samples: 50 (debugging) | Full: 500 samples")
    print("   • Early stopping: Yes, patience=5")
    
    try:
        trainer, results = main()
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise 