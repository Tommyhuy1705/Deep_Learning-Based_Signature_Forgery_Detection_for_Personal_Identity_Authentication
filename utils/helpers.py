import yaml
import os
import json
import kagglehub
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from models.Triplet_Siamese_Similarity_Network import tSSN
from losses.triplet_loss import TripletLoss
import torch

def load_config(config_path):
    with open(config_path, 'r',encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config

def get_model_from_Kaggle(kaggle_handle):
    # Load token kaggle
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "r") as f:
        token = json.load(f)

    # Set environment variables for Kaggle API
    os.environ["KAGGLE_CONFIG_DIR"] = os.path.expanduser("~/.kaggle")
    os.environ["KAGGLE_USERNAME"] = token["username"]
    os.environ["KAGGLE_KEY"] = token["key"]

    model_path = kagglehub.model_download(
        handle= kaggle_handle,
    )
    print(f"Model downloaded to {model_path}")
    return model_path

def load_model(model_path,backbone,feature_dim):
    #subfolder_name = f"tSSN_{params['mode']}_margin{params['margin']}"
    #model_file = os.path.join(model_path, subfolder_name, 'tSSN.pth')
    model_file = os.path.join(model_path, 'tSSN.pth')

    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model not found at {model_file}")

    model = tSSN(backbone_name=backbone, output_dim=feature_dim)

    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {model_file}")
    return model

def train_model(model:tSSN, train_loader:DataLoader, optimizer, device, num_epochs, loss_fn:TripletLoss, early_stop=None):
    epochs_no_improve = 0
    last_loss = None
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for anchor, positive, negative in train_loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            anchor_feat, positive_feat, negative_feat = model(anchor, positive, negative)

            loss = loss_fn(anchor_feat, positive_feat, negative_feat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {avg_loss:.4f}")

        if early_stop is not None:
            if last_loss is not None and avg_loss >= last_loss:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
            else:
                epochs_no_improve = 0

            last_loss = avg_loss
            
    return model, avg_loss

def save_model(model:tSSN, dir:str, optimizer, avg_loss, model_name:str):
    # Create subfolder name according to mode and margin
    save_dir = os.path.join(dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, f'tSSN.pth')
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)

    print(f"✅ Model saved at {checkpoint_path}")

def train_model_kfold(config, loss_fn:TripletLoss, dataset, k_folds:int, batch_size:int=32):
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    acc_scores = []
    loss_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n> Fold {fold + 1}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, num_workers=4, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, num_workers=4, batch_size=batch_size, shuffle=False)

        # Each fold uses a completely new model
        device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
        model_fold = tSSN(config['model']['backbone'], config['model']['feature_dim']).to(device)
        loss_fn.to(device)
        num_epochs=config['training']['num_epochs']
        early_stop=config['training']['early_stop']

        if torch.cuda.device_count() > 1:
            model_fold = torch.nn.DataParallel(model_fold, device_ids=[0, 1])

        optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=config['training']['learning_rate'])

        # Current train fold
        model_fold, avg_train_loss = train_model(
            model=model_fold,
            train_loader=train_loader,
            optimizer=optimizer_fold,
            device=device,
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            early_stop=early_stop
        )

        # Current validation fold
        model_fold.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for anchor, positive, negative in val_loader:
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                a_feat, p_feat, n_feat = model_fold(anchor, positive, negative)

                # Calculate loss
                loss = loss_fn(a_feat, p_feat, n_feat)
                val_loss += loss.item()

                # Calculate accuracy: anchor is closer to positive than negative
                dist_ap = torch.norm(a_feat - p_feat, dim=1)
                dist_an = torch.norm(a_feat - n_feat, dim=1)
                correct += torch.sum(dist_ap < dist_an).item()
                total += anchor.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        print(f"> Fold {fold + 1} - Loss: {avg_val_loss:.10f} - Accuracy: {accuracy:.10f}%")

        loss_scores.append(avg_val_loss)
        acc_scores.append(accuracy)

    # Overall rating
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    mean_loss = np.mean(loss_scores)

    print("\n* Overall rate of the folds:")
    print(f"> Accuracy: {mean_acc:.12f} (Difference +- {std_acc:.12f})")
    print(f"> Loss: {mean_loss:.12f}")

    return mean_acc, mean_loss
