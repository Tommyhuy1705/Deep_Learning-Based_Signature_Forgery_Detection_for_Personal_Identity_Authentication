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
# CLASS: IMAGE PREPROCESSING UTILITY
# =============================================================================
class ResizeWithPad:
    """
    Resizes an image to a target resolution while maintaining its original aspect ratio.
    Padding is applied symmetrically to fit the target dimensions.

    Attributes:
        target_size (tuple): The target output resolution (width, height).
        fill (int): Pixel intensity used for padding (default: 255 for white background).
    """
    def __init__(self, target_size, fill=255):
        # Robust handling for single integer input
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        elif isinstance(target_size, (list, tuple)):
            self.target_size = tuple(target_size)
        else:
            raise ValueError("target_size must be an integer or a tuple.")
        self.fill = fill

    def __call__(self, img):
        original_size = img.size  # (Width, Height)
        target_w, target_h = self.target_size
        
        # Determine scaling factor
        ratio = min(target_w / original_size[0], target_h / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # Resize with Bilinear interpolation
        img = img.resize(new_size, Image.BILINEAR)
        
        # Calculate padding
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # Apply padding
        return pad(img, padding, fill=self.fill, padding_mode='constant')


# =============================================================================
# CLASS: SIGNATURE EPISODE DATASET
# =============================================================================
class SignatureEpisodeDataset(Dataset):
    """
    Signature Verification Dataset for Metric-Based Meta-Learning.

    Methodology:
    - Implements Episodic Sampling (N-way K-shot) for few-shot learning.
    - Applies Photometric Inversion (1.0 - x) to convert sparse black-ink signatures
      into high-activation features suitable for CNNs.
    - Incorporates robust error handling to skip corrupted samples or missing files 
      during runtime, ensuring pipeline stability.

    Attributes:
        split_file_path (str): Path to the JSON manifest.
        base_data_dir (str): Root directory for image files.
        split_name (str): Partition key ('meta-train' or 'meta-test').
        k_shot (int): Number of support samples per class.
    """

    def __init__(self, split_file_path, base_data_dir, split_name='train', 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 target_size=(224, 224), augment=False, use_full_path=False):
        
        # Load JSON Manifest
        try:
            with open(split_file_path, 'r') as f:
                raw_data = json.load(f)
                # Adaptive parsing for different JSON schemas
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

        # Inference Transformation Pipeline (Deterministic)
        self.transform = transforms.Compose([
            ResizeWithPad(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Training Transformation Pipeline (Stochastic)
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(target_size),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        """
        Retrieves a data episode (Support Set + Query Set) for a specific user.
        Includes recursive fallback logic to handle missing or corrupted data.
        """
        # 1. Retrieve User Metadata
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        # 2. Extract File Paths
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        # --- Robust Skip Logic ---
        # If essential data is missing, recursively try the next user index.
        if not genuine_paths:
            return self.__getitem__((index + 1) % len(self))
        
        if self.n_query_forgery > 0 and not forgery_paths:
            return self.__getitem__((index + 1) % len(self))
        # -------------------------

        # 3. Sampling Strategy (With Replacement)
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            factor = (required_genuine // len(genuine_paths)) + 1
            genuine_paths = genuine_paths * factor
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        query_forg_paths = []
        if self.n_query_forgery > 0:
            if len(forgery_paths) < self.n_query_forgery:
                factor = (self.n_query_forgery // len(forgery_paths)) + 1
                forgery_paths = forgery_paths * factor
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # 4. Load Images
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = None
        
        if self.n_query_forgery > 0:
            query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
            # If loading forgeries fails, skip user
            if query_imgs_forg is None:
                return self.__getitem__((index + 1) % len(self))

        # If loading genuine images fails, skip user
        if support_imgs is None or query_imgs_gen is None:
            return self.__getitem__((index + 1) % len(self))

        # 5. Aggregate Tensors
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
        """
        Loads and processes a batch of images from disk.
        """
        images = []
        for path in paths:
            # Resolve path (Absolute vs Relative)
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
                # Log warning but attempt to proceed
                # print(f"[Warning] Failed to load image: {full_path}. Error: {e}")
                continue
                
        if len(images) == 0: return None
        return torch.stack(images)