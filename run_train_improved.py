import os
import pprint
import argparse
from math import pi
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
    parser.add_argument('--cxr_filepath', type=str, default='data/cxr.h5')
    parser.add_argument('--txt_filepath', type=str, default='data/mimic_impressions.csv')
    parser.add_argument('--batch_size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.2)
    parser.add_argument('--warmup_steps', type=int, default=250)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--grad_accum_steps', type=int, default=4)
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_dir', type=str, default="checkpoints/")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--context_length', type=int, default=77)
    parser.add_argument('--random_init', type=bool, default=True)
    parser.add_argument('--model_name', type=str, default="pt-imp-v3.0")
    parser.add_argument('--do_validate', type=bool, default=True)
    parser.add_argument('--valid_interval', type=int, default=100)
    parser.add_argument('--val_cxr_filepath', type=str, default='data/chexpert_valid.h5')
    parser.add_argument('--val_label_path', type=str, default='data/chexpert_valid.csv')
    parser.add_argument('--val_batch_size', type=int, default=32)
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

    if verbose:
        print(model)
    return model

def make(config):
    pretrained = not config.random_init
    data_loader, device = load_data(config.cxr_filepath, config.txt_filepath, batch_size=config.batch_size, pretrained=pretrained, column="impression")
    model = load_clip(model_path=None, pretrained=pretrained, context_length=config.context_length)
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
    model_save_dir = os.path.join(config.save_dir, config.model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    total_batches = len(loader) * config.epochs
    example_ct = 0
    batch_ct = 0
    running_loss = 0.0

    optimizer.zero_grad()

    for epoch in range(config.epochs):
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

            if batch_ct % config.log_interval == 0:
                train_log(running_loss / config.log_interval, example_ct, epoch)
                running_loss = 0.0

            if config.do_validate and (batch_ct % config.valid_interval) == 0:
                val_results_df = run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config)
                val_results_df.to_csv(os.path.join(model_save_dir, f"val_results_{batch_ct}.csv"), index=False)

            if (batch_ct % config.save_interval) == 0:
                model_path = os.path.join(model_save_dir, f"checkpoint_{batch_ct}.pt")
                save(model, model_path)
                print("Saved checkpoint to:", model_path)

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

if __name__ == "__main__":
    args = parse_args()
    model = model_pipeline(args)
