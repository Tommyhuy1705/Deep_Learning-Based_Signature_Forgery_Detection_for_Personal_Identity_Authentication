import yaml
import os
import json
import kagglehub
from models.siamese_network import SiameseNetwork
from models.triplet_network import TripletNetwork
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
        if name_model == "triplet":
            model = TripletNetwork(backbone, feature_dim)
        elif name_model == "siamese":
            model = SiameseNetwork(backbone, feature_dim)
            
        checkpoint = torch.load(f"{model_path}/checkpoint_epoch_50.pth", map_location='cpu') 
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Model loaded successfully")
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return model