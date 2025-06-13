import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import h5py
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode
except ImportError:
    print("Warning: torchvision not available. Some transforms may not work.")
    # Provide fallback minimal implementation
    class Compose:
        def __init__(self, transforms):
            self.transforms = transforms
        def __call__(self, img):
            for t in self.transforms:
                img = t(img)
            return img
    
    class Normalize:
        def __init__(self, mean, std):
            self.mean = torch.tensor(mean)
            self.std = torch.tensor(std)
        def __call__(self, tensor):
            return (tensor - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
    
    class Resize:
        def __init__(self, size, interpolation=None):
            self.size = size
        def __call__(self, img):
            return torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.size, self.size), mode='bilinear'
            ).squeeze(0)
    
    class InterpolationMode:
        BICUBIC = 'bicubic'

class CXRTestDataset(Dataset):
    """Dataset class for loading CXR test data from H5 files."""
    def __init__(self, img_path, transform=None):
        super().__init__()
        self.img_dset = h5py.File(img_path, 'r')['cxr']
        self.transform = transform
        
    def __len__(self):
        return len(self.img_dset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, 3, axis=0)  # Convert to 3 channels
        img = torch.from_numpy(img)
        
        if self.transform:
            img = self.transform(img)
        
        return {'img': img}

def make_true_labels(label_path, labels, cutlabels=True):
    """Create ground truth labels array from CSV file."""
    df = pd.read_csv(label_path)
    
    # Check if this is PadChest (has ImageID column)
    if 'ImageID' in df.columns:
        # For PadChest, filter by available labels and convert to lowercase
        exclude_cols = ['ImageID', 'name', 'Path', 'is_test'] if 'is_test' in df.columns else ['ImageID', 'name', 'Path']
        available_labels = [col.lower() for col in df.columns if col not in exclude_cols]
        # Map labels to available ones
        final_labels = []
        for label in labels:
            if label.lower() in available_labels:
                final_labels.append(label.lower())
        labels = final_labels
        df.columns = [col.lower() if col not in ['ImageID', 'name', 'Path', 'is_test'] else col for col in df.columns]
    
    # For other datasets, check if labels exist in columns
    available_labels = [label for label in labels if label in df.columns or label.lower() in df.columns]
    if len(available_labels) != len(labels):
        print(f"Warning: Only {len(available_labels)}/{len(labels)} labels found in dataset")
    
    # Extract ground truth
    y_true = []
    for label in available_labels:
        if label in df.columns:
            y_true.append(df[label].values)
        elif label.lower() in df.columns:
            y_true.append(df[label.lower()].values)
    
    if not y_true:
        raise ValueError(f"No matching labels found in dataset. Available columns: {list(df.columns)}")
    
    y_true = np.column_stack(y_true)
    
    # Handle NaN values - convert to 0
    y_true = np.nan_to_num(y_true, nan=0.0)
    
    return y_true, available_labels

def evaluate_predictions(y_pred, y_true, labels):
    """Evaluate predictions and return AUC scores."""
    results = {}
    
    # Handle case where predictions and labels don't match exactly
    min_labels = min(y_pred.shape[1], len(labels))
    
    for i in range(min_labels):
        label = labels[i]
        
        # Skip if all ground truth labels are the same (no positive or negative cases)
        if len(np.unique(y_true[:, i])) < 2:
            print(f"Warning: Skipping {label} - all labels are the same value")
            continue
            
        try:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            results[f"{label}_auc"] = auc
        except Exception as e:
            print(f"Error computing AUC for {label}: {e}")
            results[f"{label}_auc"] = 0.0
    
    # Calculate mean AUC
    auc_values = [v for k, v in results.items() if k.endswith('_auc') and v > 0]
    if auc_values:
        results['mean_auc'] = np.mean(auc_values)
    else:
        results['mean_auc'] = 0.0
    
    return pd.DataFrame([results])

def setup_test_data(dataset_name, batch_size=64, input_resolution=448):
    """Setup test dataset and labels for a given dataset."""
    
    # Get the absolute path to the data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Dataset configurations (paths only)
    dataset_configs = {
        'chexpert_test': {
            'img_path': os.path.join(data_dir, 'chexpert_test.h5'),
            'label_path': os.path.join(data_dir, 'chexpert_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'padchest_test': {
            'img_path': os.path.join(data_dir, 'padchest_test.h5'),
            'label_path': os.path.join(data_dir, 'padchest_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'vindrcxr_test': {
            'img_path': os.path.join(data_dir, 'vindrcxr_test.h5'),
            'label_path': os.path.join(data_dir, 'vindrcxr_test.csv'),
            'templates': [("{}", "no {}")]
        },
        'vindrpcxr_test': {
            'img_path': os.path.join(data_dir, 'vindrpcxr_test.h5'),
            'label_path': os.path.join(data_dir, 'vindrpcxr_test.csv'),
            'templates': [("{}", "no {}")]
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
    exclude_columns = {'Study', 'Path', 'image_id', 'ImageID', 'name', 'is_test'}
    all_columns = set(df.columns)
    label_columns = all_columns - exclude_columns
    labels = sorted(list(label_columns))  # Sort for consistency
    
    # Setup transforms
    transform = Compose([
        Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944)),
        Resize(input_resolution, interpolation=InterpolationMode.BICUBIC),
    ])
    
    # Create dataset and dataloader
    dataset = CXRTestDataset(config['img_path'], transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Load ground truth
    y_true, actual_labels = make_true_labels(config['label_path'], labels)
    
    print(f"Loaded {dataset_name}:")
    print(f"  - Images: {len(dataset)}")
    print(f"  - Labels: {len(actual_labels)} ({actual_labels})")
    print(f"  - Ground truth shape: {y_true.shape}")
    
    return dataloader, y_true, actual_labels, config['templates']

def save_results(results_df, model_name, dataset_name, output_dir="benchmark/results"):
    """Save evaluation results to CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{model_name}_{dataset_name}_results.csv"
    filepath = os.path.join(output_dir, filename)
    
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to: {filepath}")
    
    # Print key metrics
    if 'mean_auc' in results_df.columns:
        mean_auc = results_df['mean_auc'].iloc[0]
        print(f"Mean AUC: {mean_auc:.4f}")
    
    # Print individual AUCs for key pathologies
    key_pathologies = ['Atelectasis_auc', 'Cardiomegaly_auc', 'Consolidation_auc', 'Edema_auc', 'Pleural effusion_auc']
    for pathology in key_pathologies:
        if pathology in results_df.columns:
            auc = results_df[pathology].iloc[0]
            print(f"{pathology}: {auc:.4f}")

def run_zero_shot_evaluation(model, dataloader, y_true, labels, templates, device, context_length=77):
    """
    Run zero-shot evaluation on a dataset using a CLIP-like model.
    
    Args:
        model: The vision-language model with encode_image and encode_text methods
        dataloader: DataLoader for the test images
        y_true: Ground truth labels
        labels: List of label names
        templates: List of (positive_template, negative_template) tuples
        device: torch device
        context_length: Maximum sequence length for text encoding
        
    Returns:
        results_df: DataFrame with evaluation results
    """
    import clip  # Import here to avoid dependency issues
    
    model.eval()
    pos_template, neg_template = templates[0]
    
    print(f"Running zero-shot evaluation on {len(dataloader.dataset)} images...")
    
    # Encode text templates
    with torch.no_grad():
        pos_texts = [pos_template.format(c.lower()) for c in labels]
        neg_texts = [neg_template.format(c.lower()) for c in labels]
        
        pos_tokens = clip.tokenize(pos_texts, context_length).to(device)
        neg_tokens = clip.tokenize(neg_texts, context_length).to(device)
        
        pos_features = model.encode_text(pos_tokens)
        neg_features = model.encode_text(neg_tokens)
        
        pos_features /= pos_features.norm(dim=-1, keepdim=True)
        neg_features /= neg_features.norm(dim=-1, keepdim=True)
    
    # Extract image features
    all_img_feats = []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Extracting image features"):
            imgs = data['img'].to(device)
            feats = model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            all_img_feats.append(feats.cpu())
    
    # Compute predictions
    img_feats_cat = torch.cat(all_img_feats).to(device)
    
    with torch.no_grad():
        logits_pos = img_feats_cat @ pos_features.T
        logits_neg = img_feats_cat @ neg_features.T
        probs = torch.exp(logits_pos) / (torch.exp(logits_pos) + torch.exp(logits_neg))
    
    y_pred = probs.cpu().numpy()
    
    # Evaluate
    results_df = evaluate_predictions(y_pred, y_true, labels)
    
    return results_df 