import os
import pprint
import argparse
from math import pi
from tqdm import tqdm
import glob
import pandas as pd

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer

from train import train_main, load_data, load_clip, preprocess_text, setup_validation
from eval import evaluate
from zero_shot import run_cxr_zero_shot, run_zero_shot

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5')
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=250)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=200)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default="pt-imp-v3.0")
    parser.add_argument('--do_validate', type=bool, default=True)
    parser.add_argument('--valid_interval', type=int, default=200)
    parser.add_argument('--val_cxr_filepath', type=str, default='data/chexpert_valid.h5')
    parser.add_argument('--val_label_path', type=str, default='data/chexpert_valid.csv')
    parser.add_argument('--val_batch_size', type=int, default=32)
    # Test dataset arguments - added for final evaluation
    parser.add_argument('--test_after_training', action='store_true', help='Test on CheXpert and PadChest after training')
    parser.add_argument('--chexpert_test_cxr', type=str, default='data/chexpert_test.h5', help='CheXpert test images')
    parser.add_argument('--chexpert_test_labels', type=str, default='data/chexpert_test.csv', help='CheXpert test labels')
    parser.add_argument('--padchest_test_cxr', type=str, default='data/padchest_test.h5', help='PadChest test images')
    parser.add_argument('--padchest_test_labels', type=str, default='data/padchest_test.csv', help='PadChest test labels')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Batch size for testing')
    # DinoV2 specific arguments
    parser.add_argument('--use_dinov2', action='store_true', help='Use DinoV2 as vision encoder')
    parser.add_argument('--dinov2_model_name', type=str, default='dinov2_vitb14', 
                        choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                        help='DinoV2 model variant to use')
    parser.add_argument('--freeze_dinov2', action='store_true', help='Freeze DinoV2 backbone weights')
    # Early stopping arguments
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait without improvement before stopping')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum change to qualify as an improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='mean_auc', 
                        choices=['mean_auc', 'loss'], help='Metric to use for early stopping')
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0):
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    model, data_loader, device, criterion, optimizer, scheduler, scaler = make(config)
    train(model, data_loader, device, criterion, optimizer, scheduler, scaler, config)

    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    # Run final testing if requested
    if config.test_after_training:
        run_final_testing(config)

    if verbose:
        print(model)
    return model

def make(config):
    pretrained = not config.random_init
    data_loader, device = load_data(
        config.cxr_filepath, config.txt_filepath, 
        batch_size=config.batch_size, 
        pretrained=pretrained, 
        use_dinov2=config.use_dinov2,
        column="impression"
    )
    model = load_clip(
        model_path=None, 
        pretrained=pretrained, 
        context_length=config.context_length,
        use_dinov2=config.use_dinov2,
        dinov2_model_name=config.dinov2_model_name,
        freeze_dinov2=config.freeze_dinov2
    )
    model.to(device)
    print('Model on Device.')

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))

    total_steps = config.epochs * len(data_loader)
    if config.lr_schedule == 'cosine':
        def lr_lambda(current_step):
            if current_step < config.warmup_steps:
                return float(current_step) / float(max(1, config.warmup_steps))
            progress = float(current_step - config.warmup_steps) / float(max(1, total_steps - config.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * pi))))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler()
    return model, data_loader, device, criterion, optimizer, scheduler, scaler

def train(model, loader, device, criterion, optimizer, scheduler, scaler, config):
    model.train()
    val_loader, y_true_val, val_labels, val_templates, _ = setup_validation(config)
    validation_enabled = val_loader is not None
    
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0
    running_loss = 0.0

    # Early stopping variables (only initialize if early stopping is enabled)
    if config.early_stopping:
        best_metric = float('-inf') if config.early_stopping_metric == 'mean_auc' else float('inf')
        epochs_without_improvement = 0
        best_epoch = 0

    optimizer.zero_grad()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        batch_count_in_epoch = 0
            
        for data in tqdm(loader):
            images = data['img'].to(device)
            texts = preprocess_text(data['txt'], model).to(device)

            with torch.cuda.amp.autocast():
                logits_per_image, logits_per_text = model(images, texts)
                labels = torch.arange(images.size(0), device=device)
                loss_img = criterion(logits_per_image, labels)
                loss_txt = criterion(logits_per_text, labels)
                loss = (loss_img + loss_txt) / 2

            scaler.scale(loss).backward()

            if (batch_ct + 1) % config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler: scheduler.step()

            example_ct += images.size(0)
            batch_ct += 1
            running_loss += loss.item()
            epoch_loss += loss.item()
            batch_count_in_epoch += 1

            if batch_ct % config.log_interval == 0:
                train_log(running_loss / config.log_interval, example_ct, epoch)
                running_loss = 0.0

            if config.do_validate and validation_enabled and (batch_ct % config.valid_interval) == 0:
                val_results_df = run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config)
                val_results_df.to_csv(os.path.join(model_save_dir, f"val_results_{batch_ct}.csv"), index=False)

            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, f"checkpoint_{batch_ct}.pt")
                save(model, model_path)
                print("Saved checkpoint to:", model_path)
        
                # Early stopping check at the end of each epoch
        if config.early_stopping and validation_enabled:
            # Run validation at the end of epoch for early stopping
            val_results_df = run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config)
            
            # Calculate metric for early stopping
            if config.early_stopping_metric == 'mean_auc':
                # Calculate mean AUC for key pathologies
                key_pathologies = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural Effusion_auc']
                available_cols = [col for col in key_pathologies if col in val_results_df.columns]
                if available_cols:
                    current_metric = val_results_df[available_cols].mean().mean()
                else:
                    # Fallback to all AUC columns
                    auc_cols = [col for col in val_results_df.columns if col.endswith('_auc')]
                    current_metric = val_results_df[auc_cols].mean().mean() if auc_cols else 0
            else:  # loss metric
                current_metric = epoch_loss / batch_count_in_epoch  # Use average epoch loss
            
            # Check for improvement
            improved = False
            if config.early_stopping_metric == 'mean_auc':
                if current_metric > best_metric + config.min_delta:
                    improved = True
            else:  # loss metric (lower is better)
                if current_metric < best_metric - config.min_delta:
                    improved = True
            
            if improved:
                best_metric = current_metric
                best_epoch = epoch
                epochs_without_improvement = 0
                # Save best model
                best_model_path = os.path.join(model_save_dir, "best_model.pt")
                save(model, best_model_path)
                print(f"New best {config.early_stopping_metric}: {current_metric:.4f} at epoch {epoch}")
            else:
                epochs_without_improvement += 1
                print(f"No improvement for {epochs_without_improvement} epochs. Best {config.early_stopping_metric}: {best_metric:.4f} at epoch {best_epoch}")
            
            # Check if we should stop
            if epochs_without_improvement >= config.patience:
                print(f"Early stopping triggered! No improvement for {config.patience} epochs.")
                print(f"Best {config.early_stopping_metric}: {best_metric:.4f} achieved at epoch {best_epoch}")
                break
        
        # Reset running loss for next epoch
        running_loss = 0.0

def train_log(loss, example_ct, epoch):
    print(f"Loss after {str(example_ct).zfill(5)} examples (Epoch {epoch}): {loss:.3f}")

def run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config):
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = val_templates[0]

    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in val_labels]
        neg_texts = [neg_template.format(c) for c in val_labels]
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)

    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation Inference"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())

    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_val = probs.cpu().numpy()
    val_results_df = evaluate(y_pred_val, y_true_val, val_labels)
    model.train()
    return val_results_df

def save(model, path):
    torch.save(model.state_dict(), path)

# ====================== TESTING FUNCTIONS - Added for final evaluation ======================

def find_best_model(config):
    """Find the model with best validation ROC-AUC from validation results."""
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    
    # First check if early stopping saved a best model
    best_model_path = os.path.join(model_save_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Using best model from early stopping: {best_model_path}")
        return best_model_path
    
    # Fall back to validation results-based selection
    val_results_pattern = os.path.join(model_save_dir, "val_results_*.csv")
    val_files = glob.glob(val_results_pattern)
    
    if not val_files:
        print("Warning: No validation results found. Using final checkpoint.")
        return os.path.join(model_save_dir, 'checkpoint.pt')
    
    best_auc = 0
    best_model_path = None
    
    # Find the model with highest average AUC on key pathologies
    key_pathologies = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural Effusion_auc']
    
    for val_file in val_files:
        # Extract batch number from filename
        batch_num = val_file.split('_')[-1].split('.')[0]
        model_path = os.path.join(model_save_dir, f"checkpoint_{batch_num}.pt")
        
        if os.path.exists(model_path):
            df = pd.read_csv(val_file)
            # Calculate mean AUC for key pathologies
            available_cols = [col for col in key_pathologies if col in df.columns]
            if available_cols:
                mean_auc = df[available_cols].mean().mean()
                print(f"Model {batch_num}: Mean AUC = {mean_auc:.4f}")
                
                if mean_auc > best_auc:
                    best_auc = mean_auc
                    best_model_path = model_path
    
    if best_model_path:
        print(f"Best model: {best_model_path} with mean AUC = {best_auc:.4f}")
        return best_model_path
    else:
        print("Using final checkpoint as fallback.")
        return os.path.join(model_save_dir, 'checkpoint.pt')

def setup_test_dataset(test_cxr_filepath, test_label_path, labels, config):
    """Setup test dataset loader and ground truth labels."""
    import zero_shot
    from torchvision.transforms import InterpolationMode
    
    print(f"Loading test labels from: {test_label_path}")
    y_true_test = zero_shot.make_true_labels(
        cxr_true_labels_path=test_label_path,
        cxr_labels=labels,
        cutlabels=True
    )
    
    # Use same resolution as validation
    input_resolution = 448 if (not config.random_init or getattr(config, 'use_dinov2', False)) else 320
    
    test_transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])
    
    print(f"Loading test CXR data from: {test_cxr_filepath}")
    test_dataset = zero_shot.CXRTestDataset(
        img_path=test_cxr_filepath,
        transform=test_transform,
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return test_loader, y_true_test

def test_model_on_dataset(model, test_loader, y_true_test, labels, templates, device, config, dataset_name):
    """Test model on a specific dataset and return results."""
    model.eval()
    context_length = getattr(model, 'context_length', config.context_length)
    pos_template, neg_template = templates[0]
    
    print(f"\n=== Testing on {dataset_name} ===")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c) for c in labels]
        neg_texts = [neg_template.format(c) for c in labels]
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing on {dataset_name}"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions and evaluate
    img_feats_cat = torch.cat(all_img_feats).to(device)
    logits_pos = img_feats_cat @ pos_features.T
    logits_neg = img_feats_cat @ neg_features.T
    probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    y_pred_test = probs.cpu().numpy()
    
    test_results_df = evaluate(y_pred_test, y_true_test, labels)
    return test_results_df

def run_final_testing(config):
    """Run testing on both CheXpert and PadChest test datasets using the best model."""
    print("\n" + "="*60)
    print("STARTING FINAL TESTING ON TEST DATASETS")
    print("="*60)
    
    # Find best model
    best_model_path = find_best_model(config)
    
    # Load the best model
    model = load_clip(
        model_path=best_model_path,
        pretrained=not config.random_init,
        context_length=config.context_length,
        use_dinov2=config.use_dinov2,
        dinov2_model_name=config.dinov2_model_name,
        freeze_dinov2=config.freeze_dinov2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results_dir = os.path.join(config.save_dir, config.model_name, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Test on CheXpert
    if os.path.exists(config.chexpert_test_cxr) and os.path.exists(config.chexpert_test_labels):
        chexpert_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema',
                          'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                          'Lung Opacity', 'No Finding','Pleural Effusion',
                          'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']
        chexpert_templates = [("{}", "no {}")]
        
        chexpert_loader, y_true_chexpert = setup_test_dataset(
            config.chexpert_test_cxr, config.chexpert_test_labels, chexpert_labels, config)
        
        chexpert_results = test_model_on_dataset(
            model, chexpert_loader, y_true_chexpert, chexpert_labels, 
            chexpert_templates, device, config, "CheXpert Test")
        
        chexpert_results.to_csv(os.path.join(results_dir, "chexpert_test_results.csv"), index=False)
        print(f"CheXpert test results saved to: {results_dir}/chexpert_test_results.csv")
        
        # Print key results
        key_cols = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural Effusion_auc']
        available_cols = [col for col in key_cols if col in chexpert_results.columns]
        if available_cols:
            mean_auc = chexpert_results[available_cols].mean().mean()
            print(f"CheXpert Mean AUC (key pathologies): {mean_auc:.4f}")
    
    # Test on PadChest  
    if os.path.exists(config.padchest_test_cxr) and os.path.exists(config.padchest_test_labels):
        # Read PadChest labels from CSV
        df_padchest = pd.read_csv(config.padchest_test_labels)
        if 'is_test' in df_padchest.columns:
            df_padchest = df_padchest[df_padchest['is_test'] == True]
        
        # Get disease labels (excluding ImageID, name, Path, is_test columns)
        exclude_cols = ['ImageID', 'name', 'Path', 'is_test']
        padchest_labels = [col.lower() for col in df_padchest.columns if col not in exclude_cols]
        padchest_templates = [("{}", "no {}")]
        
        padchest_loader, y_true_padchest = setup_test_dataset(
            config.padchest_test_cxr, config.padchest_test_labels, padchest_labels, config)
        
        padchest_results = test_model_on_dataset(
            model, padchest_loader, y_true_padchest, padchest_labels,
            padchest_templates, device, config, "PadChest Test")
        
        padchest_results.to_csv(os.path.join(results_dir, "padchest_test_results.csv"), index=False)
        print(f"PadChest test results saved to: {results_dir}/padchest_test_results.csv")
        
        # Print summary
        mean_auc = padchest_results.mean(axis=1).iloc[0] if len(padchest_results) > 0 else 0
        print(f"PadChest Mean AUC (all pathologies): {mean_auc:.4f}")
    
    print("\n" + "="*60)
    print("FINAL TESTING COMPLETED")
    print("="*60)

# ====================== END TESTING FUNCTIONS ======================

if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
