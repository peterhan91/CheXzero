import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision.transforms import Compose, Normalize, Resize, InterpolationMode, ToTensor  
from zero_shot import load_clip

# Assumed device setting
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for loading images
class CLIPImageDataset(Dataset):
    def __init__(self, list_of_images, preprocessing):
        self.images = list_of_images
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.preprocessing(Image.open(self.images[idx]).convert("RGB"))  # preprocess from clip.load
        return images

# get feature embeddings    
@torch.no_grad()
def get_embs(loader, model):
    all_embs = []
    for images in tqdm(loader):
        images = images.to(device)
        all_embs.append(model.encode_image(images).cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

# Function to save the probe
def save_probe(probe, dataset, base_path):
    save_path = os.path.join(base_path, dataset)
    os.makedirs(save_path, exist_ok=True)
    filename = 'linear_probe.pth'
    torch.save(probe.state_dict(), os.path.join(save_path, filename))
    return save_path

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Function to train the model
def train(model, criterion, optimizer, train_loader):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
    return model

# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Get the probabilities for the positive class (assuming outputs are logits from a binary classifier)
            probabilities = torch.sigmoid(outputs).squeeze()

            # Extend the lists to hold the true labels and the probabilities
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate the ROC-AUC score
    roc_auc = roc_auc_score(all_targets, all_probabilities)
    return roc_auc, np.array(all_targets), np.array(all_probabilities)

# Evaluation function
def run(label_csv, max_iter=1000, batch_size=32): # Added 'layer' as a parameter
    df = pd.read_csv(label_csv)
    df_test = df[df.is_test == True]
    df_train = df[df.is_test == False]
    # load in model and `torch.DataLoader`
    model = load_clip(
        model_path='/home/than/DeepLearning/CheXzero/checkpoints/CheXzero_Models/best_64_5e-05_original_22000_0.864.pt', 
        pretrained=True, 
        context_length=77
    ).to(device)

    # _, preprocess = clip.load("ViT-B/32", device='cuda', download_root='.cache')
    preprocess = Compose([
                        Resize(224, interpolation=InterpolationMode.BICUBIC),
                        ToTensor(),
                        # Normalize((101.48761, 101.48761, 101.48761), (83.43944, 83.43944, 83.43944))
                        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
                        Normalize((0.39799, 0.39799, 0.39799), (0.3272134, 0.3272134, 0.3272134))
                        # following chexzero normalization
                    ])
    train_loader = DataLoader(CLIPImageDataset(df_train["Path"].tolist(), preprocess), 
                            batch_size=batch_size)
    test_loader = DataLoader(CLIPImageDataset(df_test["Path"].tolist(), preprocess), 
                            batch_size=batch_size)
    
    train_embs = get_embs(train_loader, model)
    test_embs = get_embs(test_loader, model)

   # Get counts for each label and filter out those with fewer than 30 samples
    label_counts = df_test.iloc[:, 1:-3].apply(lambda x: x.sum(), axis=0)  # Assuming the first two columns are not labels
    valid_labels = label_counts[label_counts > 30].index.tolist()
    print(f'Found {len(valid_labels)} valid labels')
    
    # Filter the dataframe to only include valid labels
    df = df[['ImageID', 'is_test'] + valid_labels]
    target_columns = valid_labels
    target = df[target_columns].values
    is_test = df.is_test.values

    # Convert activations and labels to PyTorch tensors
    target_tensor = torch.tensor(target).float()

    # Split data into train and test
    train_activations = torch.tensor(train_embs).float()
    train_target = target_tensor[~is_test]
    test_activations = torch.tensor(test_embs).float()
    test_target = target_tensor[is_test]

    # Create Dataset and DataLoader for batching
    train_dataset = TensorDataset(train_activations, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_activations, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_activations.shape[1]
    output_dim = target.shape[1]  # Number of target labels

    # Initialize model, loss, and optimizer
    model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-18)

    # Train the model
    for _ in tqdm(range(max_iter)): # Added tqdm for progress bar
        model = train(model, criterion, optimizer, train_loader)

    # Evaluate the model
    test_auc, y_true, y_pred = evaluate_model(model, test_loader)
    
    score = {
        'test_AUC': test_auc,
    }

    return model, score, y_true, y_pred

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='padchest', help='dataset name')
    parser.add_argument('--prompt_type', type=str, default='general', help='Type of prompt')
    parser.add_argument('--save_path', type=str, default='results', help='Path where the results will be saved')
    parser.add_argument('--max_iter', type=int, default=200, help='Maximum number of iterations for training')
    parser.add_argument('--csv_path', type=str, default='csvs/padchest_test.csv', help='Path to the CSV file containing labels and metadata') # Fixed here
    parser.add_argument('--layer', type=int, default=0, help='Layer of the model to evaluate') # Added layer argument
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')

    args = parser.parse_args()

    # Setup logging
    # logging.basicConfig(filename=os.path.join(args.save_path, 'training_log.txt'), level=logging.INFO)

    scores = []
    # Load activations

    # Train and evaluate the model
    model, score, y_true, y_pred = run(
        label_csv=args.csv_path,
        max_iter=args.max_iter,
        batch_size=args.batch_size  # This could be a command line argument as well
    )
    scores.append(score)

    # Save the trained probe
    save_path = save_probe(model, args.dataset, args.save_path)
    with open(os.path.join(save_path, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)

    # Save the true and predicted labels
    np.save(os.path.join(save_path, 'y_true.npy'), y_true)
    np.save(os.path.join(save_path, 'y_pred.npy'), y_pred)

