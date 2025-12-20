import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from PIL import Image
import os
import json
import random
import sys

# =============================================================================
# HELPER CLASS: RESIZE WITH PAD (Defined Inline to prevent Import Errors)
# =============================================================================
class ResizeWithPad:
    """
    Resizes an image to a target size while maintaining aspect ratio.
    Pads the shorter dimension with a specific fill color (default: White/255).
    """
    def __init__(self, target_size, fill=255):
        # Fix the "cannot unpack non-iterable int" error
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size
        self.fill = fill

    def __call__(self, img):
        # 1. Calculate new size maintaining aspect ratio
        original_size = img.size # (W, H)
        ratio = min(self.target_size[0] / original_size[0], self.target_size[1] / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # 2. Resize
        img = img.resize(new_size, Image.BILINEAR)
        
        # 3. Pad to reach target size
        # Calculate padding needed (Left, Top, Right, Bottom)
        delta_w = self.target_size[0] - new_size[0]
        delta_h = self.target_size[1] - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # 4. Apply Padding
        return pad(img, padding, fill=self.fill, padding_mode='constant')

# =============================================================================
# DATASET CLASS
# =============================================================================
class SignatureEpisodeDataset(Dataset):
    """
    Signature Verification Episode Dataset (Self-Contained Fixed Version).
    """

    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
        # 1. Metadata Parsing
        try:
            with open(split_file_path, 'r') as f:
                raw_data = json.load(f)
                if isinstance(raw_data, dict) and split_name in raw_data:
                    self.users_data = raw_data[split_name]
                    self.user_ids = list(self.users_data.keys())
                elif isinstance(raw_data, list):
                    self.users_data = raw_data
                    self.user_ids = range(len(self.users_data))
                else:
                    self.users_data = raw_data
                    self.user_ids = list(self.users_data.keys())
        except FileNotFoundError:
            raise FileNotFoundError(f"[Critical] Split file not found: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # ---------------------------------------------------------------------
        # TRANSFORM PIPELINE (With Inline ResizeWithPad)
        # ---------------------------------------------------------------------
        # Logic: 
        # 1. Resize & Pad with White (255) to match paper background.
        # 2. Convert to Tensor (0.0 - 1.0).
        # 3. Invert (1.0 - x): White paper (1.0) becomes Black (0.0). Ink (0.0) becomes Feature (1.0).
        # ---------------------------------------------------------------------

        self.transform = transforms.Compose([
            ResizeWithPad(224, fill=255), # <--- Fix applied here
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224, fill=255),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255), # Fill white on rotate
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        # 1. Retrieve User Info
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        # 2. Key Retrieval
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        if not genuine_paths:
            raise ValueError(f"User {user_id} has no genuine signatures.")

        # 3. Sampling
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            genuine_paths = genuine_paths * ((required_genuine // len(genuine_paths)) + 1)
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        query_forg_paths = []
        if self.n_query_forgery > 0 and forgery_paths:
            if len(forgery_paths) < self.n_query_forgery:
                forgery_paths = forgery_paths * ((self.n_query_forgery // len(forgery_paths)) + 1)
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # 4. Loading
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
        
        # Error Check
        if support_imgs is None:
            raise FileNotFoundError(f"[Load Error] Could not load Support Set for {user_id}. Check 'ResizeWithPad' logic or File Paths.")
        
        if query_imgs_gen is None:
             raise FileNotFoundError(f"[Load Error] Could not load Genuine Queries for {user_id}.")

        # 5. Combine
        if query_imgs_forg is not None:
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg], dim=0)
            labels_gen = torch.ones(len(query_imgs_gen), dtype=torch.float32)
            labels_forg = torch.zeros(len(query_imgs_forg), dtype=torch.float32)
            query_labels = torch.cat([labels_gen, labels_forg], dim=0)
        else:
            query_imgs = query_imgs_gen
            query_labels = torch.ones(len(query_imgs_gen), dtype=torch.float32)

        return {
            'support_images': support_imgs,
            'query_images': query_imgs,
            'query_labels': query_labels,
            'user_id': str(user_id)
        }

    def _load_batch(self, paths, augment=False):
        images = []
        for path in paths:
            if self.use_full_path or os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.base_data_dir, path)
            
            try:
                img = Image.open(full_path).convert('RGB')
                if augment and hasattr(self, 'augment_transform'):
                    tensor = self.augment_transform(img)
                else:
                    tensor = self.transform(img)
                images.append(tensor)
            except Exception as e:
                # Print explicit error to help debugging if it happens again
                print(f"[Warning] Image load failed: {full_path}. Error: {e}")
                continue
                
        if len(images) == 0: return None
        return torch.stack(images)