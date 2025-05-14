import yaml
import os
import json
import kagglehub
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

def load_model(model_path,backbone,feature_dim, name_model):
    # Load the model from the specified path
    model = None
    if os.path.exists(model_path):
        model = tSSN(backbone_name=backbone, output_dim=feature_dim)
            
        checkpoint = torch.load(f"{model_path}/checkpoint_epoch_100.pth", map_location='cpu') 
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model

def train_model(model, train_loader, optimizer, device, num_epochs, loss_fn):
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

    return avg_loss

def save_model(model, dir, epoch, optimizer, avg_loss):

    os.makedirs(dir, exist_ok=True)
    checkpoint_path = os.path.join(dir, f'tSSN.pth')
    model_state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)

    print(f"Checkpoint saved at {checkpoint_path}")