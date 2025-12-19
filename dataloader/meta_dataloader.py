import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image
import os
import torchvision.transforms as transforms
# IMPORT UTILITY
try:
    from utils.helpers import ResizeWithPad
except ImportError:
    # Fallback if running locally without proper package structure
    import sys
    sys.path.append('..')
    from utils.helpers import ResizeWithPad

class SignatureEpisodeDataset(Dataset):
    """
    Dataset for Meta-Learning (Few-Shot) episodes.
    Samples support and query sets ensuring disjoint intersection to prevent data leakage.
    """
    def __init__(self, split_file_path, base_data_dir, split_name, k_shot=5, n_query_genuine=5, n_query_forgery=5, augment=False, use_full_path=False):
        
        with open(split_file_path, 'r') as f:
            self.split_data = json.load(f)[split_name] # 'meta-train' or 'meta-test'
            
        self.user_ids = list(self.split_data.keys())
        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # 1. Transform: Use ResizeWithPad to preserve Geometry
        self.base_transform = transforms.Compose([
            ResizeWithPad(target_size=(220, 150)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # ResNet expects 3 channels
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.augment_transform = transforms.Compose([
            ResizeWithPad(target_size=(220, 150)),
            transforms.Grayscale(),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.user_ids)

    def _load_images(self, paths, augment=False):
        images = []
        valid_paths = []
        transform = self.augment_transform if augment else self.base_transform
        
        for p in paths:
            # Construct full path if needed
            full_path = p if self.use_full_path else os.path.join(self.base_data_dir, p)
            try:
                img = Image.open(full_path).convert('L')
                images.append(transform(img))
                valid_paths.append(full_path)
            except Exception as e:
                # print(f"Error loading {full_path}: {e}")
                continue
                
        if not images: return None
        return torch.stack(images)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_data = self.split_data[user_id]

        # Get all available paths
        genuine_paths = user_data.get('genuine', [])[:] # Copy list
        forgery_paths = user_data.get('forgery', [])[:] # Copy list

        # --- CRITICAL FIX: DATA LEAKAGE PREVENTION ---
        required_genuine = self.k_shot + self.n_query_genuine
        
        # If not enough genuine samples, SKIP this user (Pick next)
        if len(genuine_paths) < required_genuine:
            return self.__getitem__((index + 1) % len(self))
            
        # If forgery requested but not enough, SKIP
        if self.n_query_forgery > 0 and len(forgery_paths) < self.n_query_forgery:
             return self.__getitem__((index + 1) % len(self))

        # SHUFFLE (Random sampling without replacement)
        random.shuffle(genuine_paths)
        if forgery_paths: random.shuffle(forgery_paths)

        # SPLIT DISJOINTLY
        support_paths = genuine_paths[:self.k_shot]
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query_genuine]
        
        query_forgery_paths = []
        if self.n_query_forgery > 0:
            query_forgery_paths = forgery_paths[:self.n_query_forgery]

        # Load Images
        support_imgs = self._load_images(support_paths, augment=self.augment)
        query_imgs_gen = self._load_images(query_genuine_paths, augment=False) # Query never augmented
        query_imgs_forg = self._load_images(query_forgery_paths, augment=False) if query_forgery_paths else None

        # Check for loading errors
        if support_imgs is None or query_imgs_gen is None:
             return self.__getitem__((index + 1) % len(self))
        if self.n_query_forgery > 0 and query_imgs_forg is None:
             return self.__getitem__((index + 1) % len(self))

        # Combine Query
        if query_imgs_forg is not None:
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg])
            # Labels: 1 for Genuine, 0 for Forgery
            query_labels = torch.cat([torch.ones(len(query_imgs_gen)), torch.zeros(len(query_imgs_forg))])
        else:
            query_imgs = query_imgs_gen
            query_labels = torch.ones(len(query_imgs_gen))

        return {
            'support_images': support_imgs,
            'query_images': query_imgs,
            'query_labels': query_labels.long()
        }