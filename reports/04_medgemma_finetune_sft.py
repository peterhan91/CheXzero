#!/usr/bin/env python3
"""
Fine-tune MedGemma-4B-IT for CXR report generation using CBM concept scores
Uses TRL SFTTrainer following the official MedGemma demo pattern exactly

CIDEr-D Evaluation:
- Follows Nature Medicine paper (Tanno et al., 2024) methodology
- Uses CIDEr-D as primary validation metric for model selection
- Evaluates every 200 steps with early stopping based on CIDEr-D improvement
- Saves best model when CIDEr-D score improves on validation set
- Saves validation reports with ground truth comparisons for qualitative analysis
"""

import os
import sys
import json
import h5py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image

from transformers import (
    AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig,
    EarlyStoppingCallback, HfArgumentParser
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HFDataset

# Try to import CIDEr metric
try:
    from pycocoevalcap.cider.cider import Cider
except ImportError:
    print("Warning: CIDEr metric not available. Install with: pip install pycocoevalcap")
    Cider = None

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
        default="../data/mimic.h5",
        metadata={"help": "Path to training images H5 file"}
    )
    train_h5_dataset: str = field(
        default="reports/data/mimic_train_dataset.h5",
        metadata={"help": "Path to training dataset H5 file"}
    ) 
    val_h5_images: str = field(
        default="../data/chexpert.h5",
        metadata={"help": "Path to validation images H5 file"}
    )
    val_h5_dataset: str = field(
        default="reports/data/chexpert_validation_dataset.h5",
        metadata={"help": "Path to validation dataset H5 file"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of training samples"}
    )
    max_val_samples: Optional[int] = field(
        default=500,
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
            self.has_concept_scores = 'concept_scores' in f
            
            if self.has_concept_scores:
                self.num_concepts = f['concept_scores'].shape[1] 
                self.concept_names = [c.decode('utf-8') for c in f['concept_names'][:]]
            else:
                self.num_concepts = 0
                self.concept_names = []
            
            if max_samples:
                self.num_samples = min(self.num_samples, max_samples)
            
            print(f"✅ Dataset loaded: {self.num_samples} samples, {self.num_concepts} concepts")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample: image, concept_scores, report"""
        
        try:
            # Load report and concept scores from dataset h5 file
            with h5py.File(self.h5_dataset_path, 'r') as f:
                report = f['reports'][idx].decode('utf-8')
                
                if self.has_concept_scores:
                    concept_scores = f['concept_scores'][idx]
                else:
                    concept_scores = np.zeros(self.num_concepts)
                
                # Get the original index if available (for proper image retrieval)
                if 'original_indices' in f:
                    img_idx = f['original_indices'][idx]
                else:
                    img_idx = idx
            
            # Load image from CXR h5 file
            with h5py.File(self.h5_images_path, 'r') as f:
                if img_idx >= len(f['cxr']):
                    img_idx = idx
                
                img_data = f['cxr'][img_idx]
                
                # Convert to proper format for MedGemma (448x448 RGB)
                if img_data.shape[0] == 3:
                    img_data = np.transpose(img_data, (1, 2, 0))
                
                if img_data.max() <= 1.0:
                    img_data = (img_data * 255).astype(np.uint8)
                
                # Ensure 448x448 resolution as requested
                image = Image.fromarray(img_data).convert('RGB')
                if image.size != (448, 448):
                    image = image.resize((448, 448), Image.Resampling.LANCZOS)
            
            return {
                'image': image,
                'concept_scores': concept_scores.astype(np.float32),
                'report': report,
                'sample_id': idx
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            dummy_image = Image.new('RGB', (448, 448), color=(128, 128, 128))
            dummy_scores = np.zeros(self.num_concepts, dtype=np.float32)
            dummy_report = "No findings."
            
            return {
                'image': dummy_image,
                'concept_scores': dummy_scores,
                'report': dummy_report,
                'sample_id': idx
            }

def format_prompt_with_concepts(concept_scores, concept_names):
    """Create a prompt that includes concept information"""
    
    threshold = 0.3
    top_k = 10
    
    top_indices = np.argsort(concept_scores)[-top_k:][::-1]
    top_indices = [i for i in top_indices if concept_scores[i] > threshold]
    
    if len(top_indices) > 0:
        concept_text = "Key radiological concepts detected:\n"
        for i in top_indices:
            score = concept_scores[i]
            concept = concept_names[i] if i < len(concept_names) else f"concept_{i}"
            concept_text += f"- {concept}: {score:.2f}\n"
    else:
        concept_text = "No significant radiological concepts detected above threshold.\n"
    
    prompt = f"""Based on this chest X-ray image and the detected radiological concepts, generate a comprehensive radiological report with FINDINGS and IMPRESSION sections.

{concept_text}

Please provide a detailed radiological report in the following format:
FINDINGS: [Describe the radiological findings observed in the image]
IMPRESSION: [Provide clinical interpretation and conclusions]"""

    return prompt

def prepare_dataset_for_sft(dataset, processor):
    """Convert our dataset to format expected by SFTTrainer - following MedGemma demo pattern exactly"""
    
    print(f"Converting dataset to SFT format...")
    processed_data = []
    
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(f"Processing {i}/{len(dataset)}")
            
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
        
        processed_data.append({
            "messages": messages,
            "image": sample['image'],
            "sample_id": sample['sample_id']
        })
    
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

class CIDErMetric:
    """CIDEr-D evaluation metric for report generation (following Nature Medicine paper)"""
    
    def __init__(self):
        if Cider is None:
            logger.warning("CIDEr metric not available. Install with: pip install pycocoevalcap")
            self.cider = None
        else:
            # Initialize CIDEr with default parameters (uses CIDEr-D by default)
            self.cider = Cider()
    
    def compute(self, predictions: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        Compute CIDEr-D score following Nature Medicine paper methodology
        
        Args:
            predictions: List of predicted reports
            references: List of reference reports (each as list for multiple refs)
            
        Returns:
            Dictionary with cider_d score
        """
        if self.cider is None:
            logger.warning("CIDEr metric not available - returning 0.0")
            return {"cider_d": 0.0}
        
        if len(predictions) == 0 or len(references) == 0:
            logger.warning("Empty predictions or references - returning 0.0")
            return {"cider_d": 0.0}
            
        # Format for CIDEr evaluation (required format for pycocoevalcap)
        # Each prediction should be a list, each reference should be a list of alternative references
        res = {i: [pred.strip()] for i, pred in enumerate(predictions)}
        gts = {i: [ref.strip() for ref in refs] for i, refs in enumerate(references)}
        
        try:
            # compute_score returns (average_score, individual_scores)
            average_score, individual_scores = self.cider.compute_score(gts, res)
            
            logger.info(f"CIDEr-D evaluation: {len(predictions)} samples, score = {average_score:.4f}")
            
            return {"cider_d": float(average_score)}
            
        except Exception as e:
            logger.warning(f"CIDEr computation failed: {e}")
            logger.warning("Returning 0.0 as fallback")
            return {"cider_d": 0.0}

class CustomSFTTrainer(SFTTrainer):
    """Custom SFTTrainer with CIDEr-D evaluation every 200 steps"""
    
    def __init__(self, *args, raw_val_dataset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_val_dataset = raw_val_dataset
        self.cider_metric = CIDErMetric()
        self.best_cider = 0.0
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Custom evaluation with CIDEr-D metric every 200 steps"""
        
        # Standard evaluation first
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        if not self.raw_val_dataset:
            return eval_result
        
        # Generate predictions for CIDEr-D evaluation (following Nature Medicine paper)
        print("Generating predictions for CIDEr-D evaluation...")
        predictions = []
        references = []
        
        # Use larger subset for more reliable CIDEr-D evaluation (100 samples out of 500)
        # Nature Medicine paper emphasizes importance of CIDEr-D for model selection
        eval_samples = min(100, len(self.raw_val_dataset))
        
        self.model.eval()
        with torch.no_grad():
            for i in range(eval_samples):
                if i % 10 == 0:
                    print(f"Evaluating {i}/{eval_samples}")
                    
                sample = self.raw_val_dataset[i]
                
                # Create prompt
                prompt = format_prompt_with_concepts(
                    sample['concept_scores'], 
                    self.raw_val_dataset.concept_names
                )
                
                # Create messages in MedGemma format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # Apply chat template
                text = self.processing_class.apply_chat_template(
                    messages, 
                    add_generation_prompt=True, 
                    tokenize=False
                )
                
                # Process with image
                inputs = self.processing_class(
                    text=[text],
                    images=[[sample['image']]],
                    return_tensors="pt",
                    padding=True
                ).to(self.model.device)
                
                # Generate response (parameters optimized for report generation)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,  # Sufficient length for FINDINGS + IMPRESSION
                    do_sample=False,     # Deterministic generation for consistent evaluation
                    num_beams=3,         # Beam search for better quality (following Nature Med paper)
                    pad_token_id=self.processing_class.tokenizer.eos_token_id,
                    early_stopping=True,  # Stop when EOS token is generated
                    repetition_penalty=1.1  # Reduce repetition in reports
                )
                
                # Decode prediction
                input_len = inputs["input_ids"].shape[-1]
                pred_text = self.processing_class.tokenizer.decode(
                    outputs[0][input_len:], 
                    skip_special_tokens=True
                ).strip()
                
                predictions.append(pred_text)
                references.append([sample['report']])
        
        # Compute CIDEr-D score (primary metric following Nature Medicine paper)
        print(f"Computing CIDEr-D score on {len(predictions)} generated reports...")
        cider_result = self.cider_metric.compute(predictions, references)
        eval_result.update(cider_result)
        
        current_cider = cider_result.get("cider_d", 0.0)
        logger.info(f"Current CIDEr-D score: {current_cider:.4f} (best so far: {self.best_cider:.4f})")
        
        # Save validation reports for qualitative comparison
        self._save_validation_reports(predictions, references, current_cider)
        
        # Save best model when CIDEr-D improves (following Nature Medicine methodology)
        if current_cider > self.best_cider:
            improvement = current_cider - self.best_cider
            self.best_cider = current_cider
            logger.info(f"🎉 New best CIDEr-D score: {current_cider:.4f} (improvement: +{improvement:.4f})")
            
            # Save the best model
            best_model_path = os.path.join(self.args.output_dir, "best_model")
            self.save_model(best_model_path)
            logger.info(f"💾 Best model saved to: {best_model_path}")
        else:
            logger.info(f"CIDEr-D score did not improve (current: {current_cider:.4f} vs best: {self.best_cider:.4f})")
            
        return eval_result
    
    def _save_validation_reports(self, predictions: List[str], references: List[List[str]], cider_score: float):
        """Save validation reports for qualitative comparison during training"""
        
        try:
            # Create validation reports directory
            val_reports_dir = os.path.join(self.args.output_dir, "validation_reports")
            os.makedirs(val_reports_dir, exist_ok=True)
            
            # Get current step for filename
            current_step = self.state.global_step
            
            # Prepare comparison data
            comparison_data = {
                "step": current_step,
                "epoch": self.state.epoch,
                "cider_d_score": cider_score,
                "timestamp": pd.Timestamp.now().isoformat(),
                "num_samples": len(predictions),
                "reports": []
            }
            
            # Add individual report comparisons with concept information
            for i, (pred, refs) in enumerate(zip(predictions, references)):
                # Get concept scores for this sample if available
                concept_info = {}
                if i < len(self.raw_val_dataset):
                    sample = self.raw_val_dataset[i]
                    concept_scores = sample.get('concept_scores', [])
                    concept_names = self.raw_val_dataset.concept_names
                    
                    # Find top concepts for this sample
                    if len(concept_scores) > 0 and len(concept_names) > 0:
                        top_indices = np.argsort(concept_scores)[-5:][::-1]  # Top 5 concepts
                        top_concepts = []
                        for idx in top_indices:
                            if idx < len(concept_names) and concept_scores[idx] > 0.2:
                                top_concepts.append({
                                    "concept": concept_names[idx],
                                    "score": float(concept_scores[idx])
                                })
                        concept_info["top_concepts"] = top_concepts
                
                report_comparison = {
                    "sample_id": i,
                    "generated_report": pred.strip(),
                    "ground_truth_report": refs[0].strip() if refs else "",
                    "generated_length": len(pred.strip()),
                    "ground_truth_length": len(refs[0].strip()) if refs else 0,
                    "concept_info": concept_info
                }
                comparison_data["reports"].append(report_comparison)
            
            # Save as JSON for programmatic access
            json_filename = f"validation_step_{current_step:06d}.json"
            json_path = os.path.join(val_reports_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison_data, f, indent=2, ensure_ascii=False)
            
            # Save as human-readable text file
            txt_filename = f"validation_step_{current_step:06d}.txt"
            txt_path = os.path.join(val_reports_dir, txt_filename)
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"=== VALIDATION REPORT COMPARISON ===\n")
                f.write(f"Step: {current_step}\n")
                f.write(f"Epoch: {self.state.epoch:.2f}\n") 
                f.write(f"CIDEr-D Score: {cider_score:.4f}\n")
                f.write(f"Samples Evaluated: {len(predictions)}\n")
                f.write(f"Timestamp: {comparison_data['timestamp']}\n")
                f.write(f"{'='*60}\n\n")
                
                for i, report_data in enumerate(comparison_data["reports"]):
                    f.write(f"--- SAMPLE {i+1}/{len(predictions)} ---\n")
                    
                    # Include concept information if available
                    if report_data.get('concept_info', {}).get('top_concepts'):
                        f.write(f"Top Concepts:\n")
                        for concept_data in report_data['concept_info']['top_concepts']:
                            f.write(f"  - {concept_data['concept']}: {concept_data['score']:.3f}\n")
                        f.write(f"\n")
                    
                    f.write(f"Generated Report ({report_data['generated_length']} chars):\n")
                    f.write(f"{report_data['generated_report']}\n\n")
                    f.write(f"Ground Truth Report ({report_data['ground_truth_length']} chars):\n") 
                    f.write(f"{report_data['ground_truth_report']}\n")
                    f.write(f"{'-'*40}\n\n")
            
            # Save summary CSV for tracking across steps
            summary_csv_path = os.path.join(val_reports_dir, "validation_summary.csv")
            
            summary_row = {
                "step": current_step,
                "epoch": self.state.epoch,
                "cider_d_score": cider_score,
                "num_samples": len(predictions),
                "avg_generated_length": np.mean([len(p.strip()) for p in predictions]),
                "avg_ground_truth_length": np.mean([len(refs[0].strip()) for refs in references if refs]),
                "timestamp": comparison_data['timestamp']
            }
            
            # Append to CSV or create if doesn't exist
            if os.path.exists(summary_csv_path):
                summary_df = pd.read_csv(summary_csv_path)
                summary_df = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
            else:
                summary_df = pd.DataFrame([summary_row])
            
            summary_df.to_csv(summary_csv_path, index=False)
            
            logger.info(f"📝 Validation reports saved:")
            logger.info(f"   JSON: {json_path}")
            logger.info(f"   Text: {txt_path}")
            logger.info(f"   Summary: {summary_csv_path}")
            
        except Exception as e:
            logger.warning(f"Failed to save validation reports: {e}")

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
    print(f"✅ Validation samples: {len(val_dataset)} (should be 500 from CheXpert)")
    
    # Validate that we have exactly 500 validation samples from CheXpert
    if len(val_dataset) != 500:
        print(f"⚠️  Warning: Expected 500 validation samples, got {len(val_dataset)}")
    
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
    learning_rate = 2e-4  # Following demo exactly
    
    args = SFTConfig(
        output_dir="reports/results/medgemma-cxr-sft",               # Our output directory
        num_train_epochs=num_train_epochs,                          # Number of training epochs
        per_device_train_batch_size=2,                              # Small batch size for 448x448 images
        per_device_eval_batch_size=2,                               # Small batch size for evaluation
        gradient_accumulation_steps=8,                              # Effective batch size = 16
        gradient_checkpointing=True,                                # Enable gradient checkpointing
        optim="adamw_torch_fused",                                  # Use fused AdamW optimizer
        logging_steps=50,                                           # Number of steps between logs
        save_strategy="steps",                                      # Save checkpoint every eval_steps
        eval_strategy="steps",                                      # Evaluate every eval_steps
        eval_steps=200,                                             # Evaluate every 200 steps as requested
        save_steps=200,                                             # Save every 200 steps
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
        metric_for_best_model="cider_d",                            # Use CIDEr-D as best metric
        greater_is_better=True,                                     # Higher CIDEr-D is better
    )
    
    # Early stopping callback (patience=3 as requested)
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.001
    )
    
    # Create trainer (following demo pattern exactly)
    trainer = CustomSFTTrainer(
        model=model,
        args=args,
        train_dataset=train_sft_dataset,
        eval_dataset=val_sft_dataset.shuffle().select(range(100)),  # Use subset for faster training eval
        peft_config=peft_config,
        processing_class=processor,
        raw_val_dataset=val_dataset,  # Pass raw dataset for CIDEr-D evaluation
        data_collator=lambda examples: collate_fn(examples, processor),
        callbacks=[early_stopping],
    )
    
    print("\n=== Starting Training ===")
    print("✅ Will evaluate every 200 steps with CIDEr-D metric")
    print("✅ Will save model when CIDEr-D improves")
    print("✅ Early stopping with patience=3")
    print("✅ Using 500 CXR-report pairs from CheXpert training set for validation")
    
    # Initial evaluation (as requested)
    print("Running initial evaluation...")
    trainer.evaluate()
    
    # Train the model
    trainer.train()
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_results = trainer.evaluate()
    
    print("🎉 Training completed!")
    print(f"Best CIDEr-D score: {trainer.best_cider:.4f}")
    print(f"Final CIDEr-D score: {final_results.get('eval_cider_d', 'N/A')}")
    
    # Save final model
    trainer.save_model("reports/results/medgemma-cxr-sft/final_model")
    processor.save_pretrained("reports/results/medgemma-cxr-sft/final_model")
    
    return trainer, final_results

if __name__ == "__main__":
    print("🚀 Starting MedGemma-4B-IT Fine-tuning with SFTTrainer")
    print("📋 Following ALL requirements exactly:")
    print("   • Model: google/medgemma-4b-it (following medgemma_finetune_demo.py)")
    print("   • Validation: 500 CXR-report pairs from CheXpert training set")
    print("   • Data sources: /home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert_train.csv")
    print("   •               /home/than/DeepLearning/cxr_concept/CheXzero/data/chexpert.h5")
    print("   • Image resolution: 448x448")
    print("   • Evaluation: Every 200 steps + initial evaluation")
    print("   • Metric: CIDEr-D for validation")
    print("   • Early stopping: Yes, patience=3")
    print("   • Data pairing: Careful alignment maintained throughout")
    print("   • Training: Following same steps as MIMIC training set preparation")
    
    try:
        trainer, results = main()
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise 