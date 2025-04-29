import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from PIL import Image
import h5py

import torch
from torch.utils import data
from torch import nn
import torch.optim as optim
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode

import sys
sys.path.append('../..')

import clip
from model import CLIP
from simple_tokenizer import SimpleTokenizer
import zero_shot
from eval import evaluate

class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path, txt_path, column='report', size=None, transform=None):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['cxr'][:size]
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['cxr']
            self.txt_dset = pd.read_csv(txt_path)[column]
        self.transform = transform
            
    def __len__(self):
        return len(self.txt_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img = self.img_dset[idx] # np array, (320, 320)
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)
        txt = self.txt_dset[idx] # python str
        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "

        img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, 'txt': txt }
        
        return sample

def load_data(cxr_filepath, txt_filepath, batch_size=4, column='report', pretrained=False, verbose=False): 
    if torch.cuda.is_available():  
        dev = "cuda:0" 
        cuda_available = True
        print('Using CUDA.')
    else:  
        dev = "cpu"  
        cuda_available = False
        print('Using cpu.')
    
    device = torch.device(dev)
    
    if cuda_available: 
        torch.cuda.set_device(device)

    if pretrained: 
        input_resolution = 224
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
            Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        print('Interpolation Mode: ', InterpolationMode.BICUBIC)
        print("Finished image transforms for pretrained model.")
    else: 
        input_resolution = 320
        transform = Compose([
            Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        ])
        print("Finished image transforms for clip model.")
    
    torch_dset = CXRDataset(img_path=cxr_filepath,
                        txt_path=txt_filepath, column=column, transform=transform)
    
    if verbose: 
        for i in range(len(torch_dset)):
            sample = torch_dset[i]
            plt.imshow(sample['img'][0])
            plt.show()
            print(i, sample['img'].size(), sample['txt'])
            if i == 3:
                break
    
    loader_params = {'batch_size':batch_size, 'shuffle': True, 'num_workers': 0}
    data_loader = data.DataLoader(torch_dset, **loader_params)
    return data_loader, device
    
def load_clip(model_path=None, pretrained=False, context_length=77):
    '''
    FUNCTION: load_clip
    -------------------------------
    This function loads in a model with the CLIP model 
    architecture. 
    
    args: 
        * model_path (optional) - path to model weights that the model
        will be initialized with 
        * pretrained (optional) - if True, will load the pretrained 
        CLIP model
        * context_length (optional) - length of the maximum number of 
        tokens that can be inputted into the CLIP model
    '''

    params = {
        'embed_dim':768,
        'image_resolution': 320,
        'vision_layers': 12,
        'vision_width': 768,
        'vision_patch_size': 16,
        'context_length': context_length,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 12
    }
    
    # set device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if pretrained: 
        # load clip pre-trained model
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        print("Loaded in pretrained model.")
    else: 
        model = CLIP(**params)
        print("Loaded in clip model.")
    
    # if a model_path is provided, load in weights to backbone
    if model_path != None: 
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model
    
    
def preprocess_text(texts, model):
#     if model.context_length is None: 
#         model = model.module
        
    _tokenizer = SimpleTokenizer()
    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), model.context_length, dtype=torch.long)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > model.context_length:
            tokens = tokens[:model.context_length]
            tokens[model.context_length - 1] = eot_token
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def make(config, cxr_filepath, txt_filepath, model_path=None): 
    '''
    FUNCTION: make
    ---------------------------------
    This function makes the model, the data loader, loss and optimizer. 
    
    args: 
        * config - dict, configuration of experiment
        * cxr_filepath - string, filepath to chest x-ray images
        * txt_filepath - string, filepath to corresponding text reports
        * model_path - string, filepath to previously trained model
    '''
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=config.batch_size, pretrained=config.pretrained, column=config.column)
    model = load_clip(model_path=model_path, pretrained=config.pretrained, context_length=config.context_length)
    model.to(device)
    print('Model on Device.')

    # make the optimizer 
    criterion = nn.CrossEntropyLoss().cuda()
    # todo: incorporate - torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1, verbose=False)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr)
    return model, data_loader, device, criterion, optimizer


def train_main(cxr_filepath, txt_filepath, hyperparams, output_path, model_path=None, pretrained=False): 
    '''
    args: 
        * cxr_filpath- str filepath to cxr images
        * txt_filepath- str filepath to text reports
        * hyperparams- dictionary with the following hyperparams:
        `batch_size`, `criterion`, `learning_rate`, `momentum`, `epochs`
        * output_path- str filepath to where the trained model will be saved
        * model_path- str filepath to model that will be used as baseline model for training. 
        If not provided, a model will be trained from scratch
        * pretrained- whether or not the clip model was pretrained with generic images 
    This function is the main train function for CXR-CLIP. 
    '''
    
    # unpack `hyperparams`
    batch_size = hyperparams['batch_size']
    criterion = hyperparams['criterion']
    learning_rate = hyperparams['learning_rate']
    momentum = hyperparams['momentum']
    epochs = hyperparams['epochs']
    
    # load input cxr + report data
    data_loader, device = load_data(cxr_filepath, txt_filepath, batch_size=batch_size, pretrained=pretrained)
    model = load_clip(model_path=model_path, pretrained=pretrained)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    train_clip(model, data_loader, device, criterion, optimizer, epochs, output_path)
    return model


def setup_validation(config, device):
    """Prepares validation dataloader, labels, templates, and ground truth."""
    if not config.do_validate:
        return None, None, None, None, None

    print("Setting up validation...")
    # Define the standard 14 CheXpert labels for validation
    # (Could make this configurable via args if needed)
    val_labels = ['Atelectasis','Cardiomegaly', 'Consolidation', 'Edema',
                  'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
                  'Lung Opacity', 'No Finding','Pleural Effusion',
                  'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices']

    # Define standard +/- templates for softmax evaluation
    val_templates = [("{}", "no {}")] # Using a tuple pair for softmax eval

    # Load ground truth validation labels
    print(f"Loading validation labels from: {config.val_label_path}")
    y_true_val = zero_shot.make_true_labels(
        cxr_true_labels_path=config.val_label_path,
        cxr_labels=val_labels,
        cutlabels=True # Assuming val_label_path CSV columns match val_labels
    )

    # Create validation dataset transforms (MUST match zero-shot eval transforms)
    # Check if the model uses CLIP's pretrained image resolution (224) or a custom one (e.g., 320)
    # This depends on how load_clip is configured based on 'pretrained' flag in make()
    # Let's assume it aligns with the zero_shot.py 'make' function logic
    input_resolution = 224 if not config.random_init else 320 # Adjust based on your model init
    print(f"Using validation input resolution: {input_resolution}")

    val_transform = Compose([
        # Normalize using means/stds appropriate for your data
        # These might be ImageNet stats if using CLIP's pretrained encoder,
        # or stats calculated from your CXR dataset.
        # Using example stats from zero_shot.py - VERIFY THESE ARE CORRECT FOR YOUR MODEL
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])

    # Create validation dataset
    print(f"Loading validation CXR data from: {config.val_cxr_filepath}")
    val_dataset = zero_shot.CXRTestDataset(
        img_path=config.val_cxr_filepath,
        transform=val_transform,
    )

    # Create validation dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False, # No need to shuffle for validation
        num_workers=2, # Adjust based on your system
        pin_memory=True
    )

    print("Validation setup complete.")
    return val_loader, y_true_val, val_labels, val_templates, input_resolution


def run_validation_step(model, val_loader, y_true_val, val_labels, val_templates, device, config, input_resolution):
    """Performs a zero-shot validation step and returns a key metric."""
    model.eval() # Set model to evaluation mode
    print("\nStarting zero-shot validation step...")

    # Determine context length - should match model's setting
    context_length = model.context_length if hasattr(model, 'context_length') else config.context_length

    # 1. Create Zero-Shot Classifier Weights (Text Embeddings)
    # Assuming standard softmax evaluation with +/- template pairs
    if not (isinstance(val_templates, list) and len(val_templates) > 0 and isinstance(val_templates[0], tuple)):
         print("Warning: Validation templates not in expected format (list of tuples). Using first template.")
         # Fallback or error handling needed if format is wrong
         # For now, assuming the first template is the pos/neg pair
         pos_template, neg_template = val_templates[0]
    else:
        pos_template, neg_template = val_templates[0] # Use the first pair

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

        # Calculate probabilities using softmax: P(positive) = exp(logit_pos) / (exp(logit_pos) + exp(logit_neg))
        # Add a small epsilon for numerical stability if needed, though torch.exp should handle large values
        # Prevent overflow/underflow issues with log-sum-exp trick (or direct softmax)
        # logits = torch.stack([logits_pos, logits_neg], dim=-1) # Shape: [N, num_classes, 2]
        # probabilities = torch.softmax(logits, dim=-1)[:, :, 0] # Get P(positive)
        # Safer calculation:
        exp_logits_pos = torch.exp(logits_pos)
        exp_logits_neg = torch.exp(logits_neg)
        probabilities = exp_logits_pos / (exp_logits_pos + exp_logits_neg)

        y_pred_val = probabilities.cpu().numpy() # Shape: (num_samples, num_classes)


    # 4. Evaluate Predictions
    print("  Evaluating predictions...")
    # Using the 'evaluate' function from eval.py (make sure it's imported and correct)
    # It should handle calculating various metrics including F1 and AUC
    # Pass None for label_idx_map since we assume val_labels directly map to y_true_val cols
    val_results_df = evaluate(y_pred_val, y_true_val, val_labels, label_idx_map=None)

    # --- Choose the primary metric to track ---
    # Example: Macro F1 Score or Mean AUC
    primary_metric = 'F1' # Column name in val_results_df
    average_row = 'macro avg' # Row name in val_results_df

    if average_row in val_results_df.index and primary_metric in val_results_df.columns:
         validation_score = val_results_df.loc[average_row, primary_metric]
         print(f"  Validation {average_row} {primary_metric}: {validation_score:.4f}")
    else:
        print(f"  Warning: Metric '{primary_metric}' for '{average_row}' not found in validation results. Defaulting to 0.")
        print("  Available metrics:", val_results_df)
        validation_score = 0.0 # Fallback score

    # Optionally print the full validation report
    print("  Validation Report:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(val_results_df)

    model.train() # Set model back to training mode
    return validation_score, val_results_df # Return score and full results