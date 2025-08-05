#!/usr/bin/env python3
"""
Fine-tune Florence-2 for medical imaging tasks using concept-aware prompting
Combines distributed training capabilities with medical dataset handling
"""

import argparse
import os
import json
import h5py
import numpy as np
import pandas as pd
from functools import partial
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import logging
from tqdm import tqdm
import multiprocessing as mp

import friendlywords as fw
import torch
import torch.distributed as dist
import torch.multiprocessing as mp_spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM, AutoProcessor, get_scheduler,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model

# Try to import medical evaluation metrics
try:
    from pycocoevalcap.cider.cider import Cider
    CIDER_AVAILABLE = True
except ImportError:
    print("Warning: CIDEr metric not available. Install with: pip install pycocoevalcap")
    CIDER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CXRReportDataset(Dataset):
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
        """Get a single sample: image, concept_scores, report"""
        
        # Load report and concept scores from dataset h5 file
        with h5py.File(self.h5_dataset_path, 'r') as f:
            report = f['reports'][idx].decode('utf-8')
            concept_scores = f['concept_scores'][idx]
            
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
            
            # Convert to proper format for Florence-2 (RGB)
            if img_data.shape[0] == 3:
                img_data = np.transpose(img_data, (1, 2, 0))
            
            if img_data.max() <= 1.0:
                img_data = (img_data * 255).astype(np.uint8)
            
            # Convert to PIL Image and ensure consistent size like medgemma
            image = Image.fromarray(img_data).convert('RGB')
            # Resize to consistent size for better training stability (following medgemma approach)
            if image.size != (448, 448):
                image = image.resize((448, 448), Image.Resampling.LANCZOS)
            
        return {
            'image': image,
            'concept_scores': concept_scores.astype(np.float32),
            'report': report,
            'sample_id': idx
        }

def format_prompt_with_concepts(concept_scores, concept_names, task_type="report_generation"):
    """
    Create a prompt that includes concept information for different tasks
    
    This follows the medgemma approach for concept-aware prompting but adapts it for Florence-2:
    - Uses <OCR_WITH_REGION> task token specific to Florence-2
    - Maintains the same concept filtering and formatting logic as medgemma
    - Provides explicit format instructions for report generation
    """
    
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
    
    if task_type == "report_generation":
        prompt = f"""Based on this chest X-ray image and the detected radiological concepts, generate a comprehensive radiological report with FINDINGS and IMPRESSION sections.

{concept_text}

Please provide a detailed radiological report in the following format:
FINDINGS: [Describe the radiological findings observed in the image]
IMPRESSION: [Provide clinical interpretation and conclusions]"""
    
    elif task_type == "vqa":
        prompt = f"""{concept_text}

Based on the chest X-ray image and the detected concepts above, answer the following question:"""
    
    else:  # general description
        prompt = f"""{concept_text}
Describe this medical image in detail, considering the following detected concepts:

{concept_text}"""

    return prompt

def setup_distributed(rank, world_size):
    """Initialize distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def collate_fn(batch, processor, device, concept_names, task_type="report_generation"):
    """Collate function for DataLoader"""
    images = []
    questions = []
    answers = []
    
    for sample in batch:
        image = sample['image']
        concept_scores = sample['concept_scores']
        report = sample['report']
        
        # Create prompt with concepts
        prompt = format_prompt_with_concepts(concept_scores, concept_names, task_type)
        
        images.append(image)
        questions.append(prompt)
        answers.append(report)
    
    # Process inputs
    inputs = processor(
        text=questions, 
        images=images, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=1024
    ).to(device)
    
    return inputs, answers

def create_data_loaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    rank,
    world_size,
    processor,
    device,
    concept_names,
    task_type="report_generation"
):
    """Create distributed data loaders"""
    
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=partial(
            collate_fn, 
            processor=processor, 
            device=device, 
            concept_names=concept_names,
            task_type=task_type
        ),
        num_workers=num_workers,
        sampler=train_sampler,
    )
    
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size//2,
        collate_fn=partial(
            collate_fn, 
            processor=processor, 
            device=device, 
            concept_names=concept_names,
            task_type=task_type
        ),
        num_workers=num_workers,
        sampler=val_sampler,
    )
    
    return train_loader, val_loader

def evaluate_model(
    rank, world_size, model, val_loader, device, train_loss, processor, 
    global_step, batch_size, max_val_samples, run_name
):
    """Evaluate model with medical-specific metrics"""
    
    if rank == 0:
        avg_train_loss = train_loss / (global_step * batch_size * world_size)
        logger.info(f"Step {global_step} - Average Training Loss: {avg_train_loss:.6f}")

    # Evaluation phase
    model.eval()
    val_loss = 0
    predictions = []
    references = []
    
    with torch.no_grad():
        val_item_count = 0
        for batch in tqdm(val_loader, desc=f"Evaluation at step {global_step}", position=rank):
            val_item_count += len(batch[1])  # batch[1] contains answers
            
            if val_item_count > max_val_samples:
                break
                
            inputs, answers = batch

            # Prepare the input and target tensors
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            
            # Prepare labels for loss calculation
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=1024,
            ).input_ids.to(device)

            # Calculate loss
            outputs = model(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss
            val_loss += loss.item()
            
            # Generate predictions for metrics
            generated_ids = model.module.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=256,
                do_sample=False,
                num_beams=3,
            )
            
            # Decode predictions
            generated_texts = processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
            # Extract only the generated part (remove input)
            for i, generated_text in enumerate(generated_texts):
                # Remove the input prompt from generated text
                # Florence-2 concatenates input + generated, so we need to extract only the new part
                if "<OCR_WITH_REGION>" in generated_text:
                    # Split by the task token and take the last part (generated response)
                    parts = generated_text.split("<OCR_WITH_REGION>")
                    if len(parts) > 1:
                        pred_text = parts[-1].strip()
                        # Further clean by removing the question part if present
                        if "Please provide a detailed radiological report" in pred_text:
                            report_start = pred_text.find("FINDINGS:")
                            if report_start != -1:
                                pred_text = pred_text[report_start:].strip()
                            else:
                                # Fallback: take text after the format instruction
                                lines = pred_text.split('\n')
                                # Look for lines that start the actual report
                                for j, line in enumerate(lines):
                                    if line.strip().startswith(('FINDINGS:', 'IMPRESSION:', 'The', 'Patient', 'Chest', 'Heart', 'Lungs')):
                                        pred_text = '\n'.join(lines[j:]).strip()
                                        break
                    else:
                        pred_text = generated_text.strip()
                else:
                    pred_text = generated_text.strip()
                
                # Clean up any remaining instruction text
                if pred_text.startswith("Based on this chest X-ray"):
                    lines = pred_text.split('\n')
                    for j, line in enumerate(lines):
                        if line.strip().startswith(('FINDINGS:', 'IMPRESSION:')):
                            pred_text = '\n'.join(lines[j:]).strip()
                            break
                
                predictions.append(pred_text)
                references.append([answers[i]])  # References should be list of lists for CIDEr

    avg_val_loss = val_loss / val_item_count if val_item_count > 0 else 0
    
    if rank == 0:
        logger.info(f"Step {global_step} - Average Validation Loss: {avg_val_loss:.6f}")

    # Calculate CIDEr score if available
    if rank == 0 and CIDER_AVAILABLE and len(predictions) > 0:
        try:
            cider_scorer = Cider()
            gts = {i: refs for i, refs in enumerate(references)}
            res = {i: [pred] for i, pred in enumerate(predictions)}
            score, scores = cider_scorer.compute_score(gts, res)
            cider_score = score
            
            logger.info(f"Step {global_step} - CIDEr Score: {cider_score:.4f}")
            
            # Save sample predictions
            save_sample_predictions(predictions[:5], references[:5], global_step, run_name)
            
        except Exception as e:
            logger.warning(f"CIDEr calculation failed: {e}")
            logger.info(f"Step {global_step} - Validation Loss: {avg_val_loss:.6f}")

    model.train()

def save_sample_predictions(predictions, references, step, run_name):
    """Save sample predictions for inspection (following medgemma evaluation approach)"""
    os.makedirs(f"./predictions/{run_name}", exist_ok=True)
    
    with open(f"./predictions/{run_name}/step_{step}_samples.json", "w") as f:
        samples = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Calculate simple metrics for each sample
            pred_words = len(pred.split()) if pred else 0
            ref_words = len(ref[0].split()) if isinstance(ref, list) and ref[0] else 0
            
            samples.append({
                "sample_id": i,
                "prediction": pred,
                "reference": ref[0] if isinstance(ref, list) else ref,
                "pred_word_count": pred_words,
                "ref_word_count": ref_words,
                "has_findings": "FINDINGS:" in pred.upper() if pred else False,
                "has_impression": "IMPRESSION:" in pred.upper() if pred else False
            })
        
        # Add summary statistics
        summary = {
            "total_samples": len(samples),
            "avg_pred_words": sum(s["pred_word_count"] for s in samples) / len(samples) if samples else 0,
            "avg_ref_words": sum(s["ref_word_count"] for s in samples) / len(samples) if samples else 0,
            "samples_with_findings": sum(s["has_findings"] for s in samples),
            "samples_with_impression": sum(s["has_impression"] for s in samples),
            "step": step
        }
        
        json.dump({"summary": summary, "samples": samples}, f, indent=2)
    
    # Also save a readable text version
    with open(f"./predictions/{run_name}/step_{step}_readable.txt", "w") as f:
        f.write(f"=== STEP {step} EVALUATION SAMPLES ===\n\n")
        for i, (pred, ref) in enumerate(zip(predictions[:3], references[:3])):  # Save first 3 for readability
            f.write(f"--- Sample {i+1} ---\n")
            f.write(f"REFERENCE:\n{ref[0] if isinstance(ref, list) else ref}\n\n")
            f.write(f"PREDICTION:\n{pred}\n\n")
            f.write("-" * 50 + "\n\n")

def train_model(
    rank, world_size, train_h5_images, train_h5_dataset, val_h5_images, val_h5_dataset,
    batch_size=4, use_lora=True, epochs=3, lr=1e-5, eval_steps=500, run_name=None, 
    max_train_samples=None, max_val_samples=500, task_type="report_generation"
):
    """Main training function"""
    
    setup_distributed(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    if run_name is None:
        run_name = fw.generate(2, separator="_")

    # Log training configuration
    if rank == 0:
        logger.info(f"Starting training run: {run_name}")
        logger.info(f"Training configuration:")
        logger.info(f"  - Train H5 images: {train_h5_images}")
        logger.info(f"  - Train H5 dataset: {train_h5_dataset}")
        logger.info(f"  - Val H5 images: {val_h5_images}")
        logger.info(f"  - Val H5 dataset: {val_h5_dataset}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Use LoRA: {use_lora}")
        logger.info(f"  - Epochs: {epochs}")
        logger.info(f"  - Learning rate: {lr}")
        logger.info(f"  - Eval steps: {eval_steps}")
        logger.info(f"  - World size: {world_size}")
        logger.info(f"  - Max train samples: {max_train_samples}")
        logger.info(f"  - Max val samples: {max_val_samples}")
        logger.info(f"  - Task type: {task_type}")

    # Load datasets
    print(f"Rank {rank}: Loading datasets...")
    train_dataset = CXRReportDataset(train_h5_images, train_h5_dataset, max_train_samples)
    val_dataset = CXRReportDataset(val_h5_images, val_h5_dataset, max_val_samples)

    # Load model and processor
    print(f"Rank {rank}: Loading Florence-2 model...")
    # Using official Microsoft model - alternatively you can use "andito/Florence-2-large-ft" for a fine-tuned version
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", trust_remote_code=True
    ).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large-ft", trust_remote_code=True
    )
    print(f"Rank {rank}: Model and processor loaded successfully.")

    if use_lora:
        print(f"Rank {rank}: Setting up LoRA...")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "linear", "Conv2d", "lm_head", "fc2"
            ],
            task_type="CAUSAL_LM",
            lora_dropout=0.1,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        if rank == 0:
            model.print_trainable_parameters()
            logger.info("LoRA configuration applied successfully")

    # Wrap model for distributed training
    model = DDP(model, device_ids=[rank])

    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, batch_size, 0, rank, world_size,
        processor, device, train_dataset.concept_names, task_type
    )

    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        for batch in tqdm(
            train_loader, 
            desc=f"Epoch {epoch + 1}/{epochs}", 
            position=rank,
            disable=rank != 0
        ):
            inputs, answers = batch

            # Prepare inputs
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs["pixel_values"].to(device)
            
            # Prepare labels
            labels = processor.tokenizer(
                text=answers,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                max_length=1024,
            ).input_ids.to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, 
                pixel_values=pixel_values, 
                labels=labels
            )
            loss = outputs.loss

            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            global_step += 1

            # Evaluation
            if global_step % eval_steps == 0:
                evaluate_model(
                    rank, world_size, model, val_loader, device, train_loss,
                    processor, global_step, batch_size, max_val_samples, run_name
                )

        # End of epoch evaluation
        evaluate_model(
            rank, world_size, model, val_loader, device, train_loss,
            processor, global_step, batch_size, max_val_samples, run_name
        )

        # Log epoch summary
        if rank == 0:
            epoch_train_loss = train_loss / len(train_loader)
            logger.info(f"Epoch {epoch + 1}/{epochs} completed - Average epoch training loss: {epoch_train_loss:.6f}")

        # Save checkpoint
        if rank == 0:
            output_dir = f"./model_checkpoints/{run_name}/epoch_{epoch + 1}"
            os.makedirs(output_dir, exist_ok=True)
            model.module.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(f"Model saved to {output_dir}")

    # Training completed
    if rank == 0:
        logger.info(f"Training completed for run: {run_name}")

    cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Florence-2 for medical imaging")
    
    # Dataset arguments
    parser.add_argument("--train-h5-images", type=str, required=True,
                       help="Path to training images H5 file")
    parser.add_argument("--train-h5-dataset", type=str, required=True,
                       help="Path to training dataset H5 file")
    parser.add_argument("--val-h5-images", type=str, required=True,
                       help="Path to validation images H5 file")
    parser.add_argument("--val-h5-dataset", type=str, required=True,
                       help="Path to validation dataset H5 file")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Batch size for training")
    parser.add_argument("--use-lora", action='store_true',
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Steps between evaluations")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Run name for wandb")
    parser.add_argument("--max-train-samples", type=int, default=None,
                       help="Maximum training samples")
    parser.add_argument("--max-val-samples", type=int, default=500,
                       help="Maximum validation samples")
    parser.add_argument("--task-type", type=str, default="report_generation",
                       choices=["report_generation", "vqa", "description"],
                       help="Type of task to fine-tune for")
    
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    print(f"Starting distributed training with {world_size} GPUs")
    
    mp_spawn.spawn(
        train_model,
        args=(
            world_size, args.train_h5_images, args.train_h5_dataset,
            args.val_h5_images, args.val_h5_dataset, args.batch_size,
            args.use_lora, args.epochs, args.lr, args.eval_steps,
            args.run_name, args.max_train_samples, args.max_val_samples,
            args.task_type
        ),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 