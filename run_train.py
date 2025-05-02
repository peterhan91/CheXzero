import os
import pprint
import argparse
from tqdm import tqdm

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
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5', help="Directory to load chest x-ray image data from.")
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv', help="Directory to load radiology report impressions text from.")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/", help="Directory to save the trained model.")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', action='store_true')
    parser.add_argument('--model_name', type=str, default="pt-imp")
    # --- Validation Arguments ---
    parser.add_argument('--do_validate', action='store_true', help="Perform zero-shot validation during training.")
    parser.add_argument('--valid_interval', type=int, default=1000)
    parser.add_argument('--val_cxr_filepath', type=str, default='data/chexpert_valid.h5', help="Path to validation CXR images (e.g., CheXpert val set).")
    parser.add_argument('--val_label_path', type=str, default='data/chexpert_valid.csv', help="Path to validation ground truth labels (e.g., CheXpert val labels).")
    parser.add_argument('--val_batch_size', type=int, default=32, help="Batch size for validation.") # Can be larger than training bs
    args = parser.parse_args()
    return args

def model_pipeline(config, verbose=0): 
    # make the model, data, and optimization problem
    model, data_loader, device, criterion, optimizer = make(config)

    # and use them to train the model
    train(model, data_loader, device, criterion, optimizer, config)

    # save model
    model_path = os.path.join(config.save_dir, str(config.model_name), 'checkpoint.pt')
    save(model, model_path)

    if verbose: 
        print(model)
    return model

def make(config): 
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    if config.optimizer == "adam": 
        optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == "sgd": 
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    return model, data_loader, device, criterion, optimizer

def train(model, loader, device, criterion, optimizer, config): 
    model.train()
    # --- Validation Setup ---
    val_loader, y_true_val, val_labels, val_templates, input_resolution = setup_validation(config)
    
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    if not os.path.exists(model_save_dir): 
        # Create a new folder if not exists
        os.makedirs(model_save_dir)
    
    # Run training
    total_batches = len(loader) * config.epochs
    example_ct = 0  # number of examples seen
    batch_ct = 0
    report_freq = config.log_interval
    highest_val_auc = 0 # save highest mean auc
    
    for epoch in range(config.epochs):
        running_loss = 0.0 # running loss over batch
        for data in tqdm(loader):
            # get the images
            images = data['img']

            texts = data['txt']
            texts = preprocess_text(texts, model) 
            
            # perform step for a single batch
            loss = train_batch(images, texts, model, device, criterion, optimizer)
            example_ct +=  len(images)
            batch_ct += 1
            running_loss += loss.item()

            # Report metrics every `report_freq` batch
            if (batch_ct % report_freq) == 0:
                train_log(running_loss / report_freq, example_ct, epoch)
                running_loss = 0.0
            
            # Perform validation every `valid_interval` batch
            if config.do_validate and (batch_ct % config.valid_interval) == 0:
                val_results_df = run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config)
                val_results_df.to_csv(os.path.join(model_save_dir, f"val_results_{batch_ct}.csv"), index=False)

            if (batch_ct % config.save_interval) == 0: 
                model_path = os.path.join(model_save_dir, "checkpoint_{batch_ct}.pt".format(
                    batch_ct=str(batch_ct), 
                ))
                print("Saved checkpoint to: ", model_path)
                save(model, model_path)
                
def train_batch(images, texts, model, device, criterion, optimizer):
    images, texts = images.to(device), texts.to(device)
    
    # Forward pass ➡
    logits_per_image, logits_per_text = model(images, texts)
    
    # Create labels
    batch_size = images.shape[0]
    labels = torch.arange(batch_size).to(device)
    
    # Compute loss
    loss_img = criterion(logits_per_image, labels)
    loss_txt = criterion(logits_per_text, labels)
    loss = (loss_img + loss_txt)/2 # avg. img and txt loss

    # Backward pass ⬅
    optimizer.zero_grad()
    loss.backward()
    
    # Step with optimizer
    optimizer.step()
        
    return loss

def train_log(loss, example_ct, epoch):
    loss = float(loss)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")
    
def run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config):
    model.eval() 
    context_length = model.context_length if hasattr(model, 'context_length') else config.context_length
    pos_template, neg_template = val_templates[0] 

    print(f"  Generating text embeddings for {len(val_labels)} classes using templates: ('{pos_template}', '{neg_template}')...")
    with torch.no_grad():
        pos_texts = [pos_template.format(classname) for classname in val_labels]
        neg_texts = [neg_template.format(classname) for classname in val_labels]

        pos_tokens = clip.tokenize(pos_texts, context_length=context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length=context_length).to(device)

        pos_text_features = model.encode_text(pos_tokens)
        neg_text_features = model.encode_text(neg_tokens)

        # Normalize features
        pos_text_features /= pos_text_features.norm(dim=-1, keepdim=True)
        neg_text_features /= neg_text_features.norm(dim=-1, keepdim=True)

    # 2. Get Predictions on Validation Set
    all_image_features = []
    print(f"  Encoding validation images ({len(val_loader.dataset)} samples)...")
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation Inference"):
            images = data['img'].to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            all_image_features.append(image_features.cpu()) # Move to CPU to save GPU memory

    image_features_cat = torch.cat(all_image_features).to(device) # Move back to device for matmul

    # 3. Calculate Logits using Softmax approach
    print("  Calculating logits...")
    with torch.no_grad():
        # Calculate logits for positive and negative templates
        logits_pos = image_features_cat @ pos_text_features.T
        logits_neg = image_features_cat @ neg_text_features.T
        exp_logits_pos = torch.exp(logits_pos)
        exp_logits_neg = torch.exp(logits_neg)
        probabilities = exp_logits_pos / (exp_logits_pos + exp_logits_neg)

        y_pred_val = probabilities.cpu().numpy() # Shape: (num_samples, num_classes)

    # 4. Evaluate Predictions
    print("  Evaluating predictions...")
    val_results_df = evaluate(y_pred_val, y_true_val, val_labels, label_idx_map=None)
    model.train() # Set model back to training mode
    return val_results_df # Return score and full results

def save(model, path): 
    torch.save(model.state_dict(), path)
    
if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
    

