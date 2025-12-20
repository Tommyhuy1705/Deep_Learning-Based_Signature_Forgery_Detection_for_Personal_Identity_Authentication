import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import random
import sys

# Attempt to import custom utility for aspect-ratio preserving resize
try:
    from utils.helpers import ResizeWithPad
except ImportError:
    # Fallback or local path adjustment if necessary
    sys.path.append('..')
    from utils.helpers import ResizeWithPad

class SignatureEpisodeDataset(Dataset):
    """
    Signature Verification Episode Dataset for Few-Shot Learning.

    Description:
    This dataset class implements the episodic sampling protocol required for
    metric-based meta-learning. It constructs N-way K-shot tasks by sampling
    support and query sets from user-specific signature distributions.

    Key Features:
    1. Robust Key Retrieval: Handles case-sensitivity inconsistencies in JSON schema 
       (e.g., 'Forged' vs 'forged').
    2. Photometric Inversion: Inverts pixel intensities (1.0 - x) to align 
       sparse signature strokes with CNN activation patterns.
    3. Stochastic Augmentation: Applies affine transformations to support sets.

    Attributes:
        split_file_path (str): Path to the JSON manifest.
        base_data_dir (str): Root directory of raw images.
        k_shot (int): Support samples per user.
    """

    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
        # Load dataset metadata
        try:
            with open(split_file_path, 'r') as f:
                full_data = json.load(f)
                # Handle nested dict (BHSig) or flat list (CEDAR)
                if isinstance(full_data, dict) and split_name in full_data:
                    self.users_data = full_data[split_name]
                    self.user_ids = list(self.users_data.keys())
                elif isinstance(full_data, list):
                    self.users_data = full_data
                    self.user_ids = range(len(self.users_data))
                else:
                    # Fallback: Maybe the JSON is just the dict itself (legacy support)
                    self.users_data = full_data
                    self.user_ids = list(self.users_data.keys())
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file not found at: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # ---------------------------------------------------------------------
        # DATA TRANSFORMATION PIPELINE (With Photometric Inversion)
        # ---------------------------------------------------------------------
        
        # 1. Standard Transform (Validation/Inference)
        self.transform = transforms.Compose([
            ResizeWithPad(224),
            transforms.ToTensor(),
            # Invert: Ink (0) -> (1) for high activation
            transforms.Lambda(lambda x: 1.0 - x), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # 2. Augmentation Transform (Training)
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                # Invert
                transforms.Lambda(lambda x: 1.0 - x),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        """
        Constructs a single episode (Task).
        """
        # Retrieve user metadata
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        # ---------------------------------------------------------------------
        # ROBUST KEY RETRIEVAL (CASE-INSENSITIVE)
        # ---------------------------------------------------------------------
        # Check for 'genuine', 'Genuine', etc.
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged')

        if genuine_paths is None:
            raise KeyError(f"Dataset manifest missing 'genuine' key for user {user_id}")
        
        # Handle cases where forgery might not exist (though rare in BHSig/CEDAR)
        if forgery_paths is None:
            forgery_paths = []

        # ---------------------------------------------------------------------
        # SAMPLING STRATEGY
        # ---------------------------------------------------------------------
        
        # A. Genuine Samples
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            factor = (required_genuine // len(genuine_paths)) + 1
            genuine_paths = (genuine_paths * factor)
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        # B. Forgery Samples
        query_forg_paths = []
        if self.n_query_forgery > 0 and len(forgery_paths) > 0:
            if len(forgery_paths) < self.n_query_forgery:
                factor = (self.n_query_forgery // len(forgery_paths)) + 1
                forgery_paths = (forgery_paths * factor)
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # ---------------------------------------------------------------------
        # LOADING
        # ---------------------------------------------------------------------
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
        
        # Integrity check
        if support_imgs is None or query_imgs_gen is None:
             return self.__getitem__((index + 1) % len(self))

        # Combine
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
                continue
                
        if len(images) == 0: return None
        return torch.stack(images)