from utils.helpers import load_config, save_model, train_model
from models.Triplet_Siamese_Similarity_Network import tSSN
from losses.triplet_loss import TripletLoss
from dataloader.tSSN_trainloader import SignatureTrainDataset

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

#seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Transform chung
transform = transforms.Compose([
    transforms.Resize((220, 150)),
    transforms.Grayscale(),  # Đảm bảo ảnh 1 kênh xám
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1 kênh -> 3 kênh
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = SignatureTrainDataset(
    org_dir=r'/kaggle/input/cedardataset/signatures/full_org',
    forg_dir=r'/kaggle/input/cedardataset/signatures/full_forg',
    transform=transform
)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Kiểm tra Triplet Dataset
print(f"Triplet Dataset - Total triplets: {len(dataset)}")

anchor, positive, negative = dataset[0]

print(f"Anchor shape: {anchor.shape}")
print(f"Positive shape: {positive.shape}")
print(f"Negative shape: {negative.shape}")
print("LOAD DATASET SUCCESSFULLY")

config = load_config(r'/kaggle/working/Deep_Learning-Based_Signature_Forgery_Detection_for_Personal_Identity_Authentication/configs/config_tSSN.yaml')
print(config)
print("LOAD CONFIG SUCCESSFULLY")

# Create model
model = tSSN(config['model']['backbone'], config['model']['feature_dim'])

loss_params = {'margin': 1, 'mode': "euclidean"} #mode : euclidean, cosine, manhattan, learnable
loss_fn = TripletLoss(margin=loss_params['margin'], mode= loss_params['mode'], input_dim=config['model']['feature_dim']) 
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
optimizer= optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

model.to(device)
loss_fn.to(device)
print("MODEL SUCCESSFULLY")

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    
avg_loss = train_model( model = model, 
                        train_loader=train_loader, 
                        optimizer= optimizer, 
                        device = device, 
                        num_epochs = config['training']['num_epochs'], 
                        loss_fn=loss_fn)       #Sẽ train ra nhiều model với loss khác nhau

save_model(model = model,
            dir = config['logging']['checkpoint_dir'],
            epoch = config['training']['num_epochs'],
            optimizer = optimizer,
            avg_loss = avg_loss,
            loss_params=  loss_params,)