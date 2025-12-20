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
    1. Disjoint Sampling: Ensures strict separation between support and query samples 
       within an episode to prevent data leakage.
    2. Photometric Inversion: Inverts pixel intensities (1.0 - x) to align 
       sparse signature strokes (black ink) with the high-activation patterns 
       expected by CNNs pre-trained on ImageNet.
    3. Stochastic Augmentation: Applies affine transformations to the support set 
       to improve intra-class variance robustness.

    Attributes:
        split_file_path (str): Path to the JSON manifest containing user splits.
        base_data_dir (str): Root directory of the raw image data.
        split_name (str): The specific partition to load (e.g., 'meta-train', 'meta-test').
        k_shot (int): Number of support samples per user (reference signatures).
        n_query_genuine (int): Number of positive query samples (genuine signatures).
        n_query_forgery (int): Number of negative query samples (skilled forgeries).
    """

    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
        # Load dataset metadata from JSON manifest
        try:
            with open(split_file_path, 'r') as f:
                full_data = json.load(f)
                # Handle both structure types: nested by split_name or flat list (for cross-domain)
                if isinstance(full_data, dict) and split_name in full_data:
                    self.users_data = full_data[split_name] # Dict structure
                    self.user_ids = list(self.users_data.keys())
                elif isinstance(full_data, list):
                    self.users_data = full_data # List structure (e.g., CEDAR)
                    self.user_ids = range(len(self.users_data))
                else:
                    raise ValueError(f"Invalid JSON structure or split name '{split_name}' not found.")
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file not found at: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # ---------------------------------------------------------------------
        # DATA TRANSFORMATION PIPELINE
        # ---------------------------------------------------------------------
        # Scientific Note:
        # Standard CNNs (ResNet) respond maximally to high pixel values (white).
        # Signatures are typically black ink (0) on white paper (1).
        # We apply an inversion transform (x' = 1 - x) to convert ink to 
        # high-intensity features against a low-intensity background.
        # ---------------------------------------------------------------------

        # 1. Standard Transform (Validation/Inference)
        self.transform = transforms.Compose([
            ResizeWithPad(224),     # Preserves aspect ratio
            transforms.ToTensor(),  # Converts [0, 255] to [0.0, 1.0]
            
            # --- CRITICAL: Photometric Inversion ---
            transforms.Lambda(lambda x: 1.0 - x), 
            # ---------------------------------------
            
            # ImageNet Normalization
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # 2. Augmentation Transform (Training Support Set)
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224),
                # Geometric perturbations: Rotation, Translation, Scaling
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                
                # --- CRITICAL: Photometric Inversion ---
                transforms.Lambda(lambda x: 1.0 - x),
                # ---------------------------------------
                
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        """Returns the total number of available users (episodes)."""
        return len(self.user_ids)

    def __getitem__(self, index):
        """
        Constructs a single episode (Task) for the specified user index.

        Returns:
            dict: Contains 'support_images', 'query_images', 'query_labels', and 'user_id'.
        """
        # Retrieve user metadata
        # Handle difference between Dict (BHSig) and List (CEDAR) structures
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
            genuine_paths = user_info['genuine']
            forgery_paths = user_info['forged']
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]
            genuine_paths = user_info['genuine']
            forgery_paths = user_info['forged']

        # ---------------------------------------------------------------------
        # SAMPLING STRATEGY
        # ---------------------------------------------------------------------
        
        # A. Genuine Samples (Support + Query)
        required_genuine = self.k_shot + self.n_query_genuine
        
        # Sampling with replacement if insufficient data
        if len(genuine_paths) < required_genuine:
            # Replicate samples to meet requirement
            factor = (required_genuine // len(genuine_paths)) + 1
            genuine_paths = (genuine_paths * factor)
        
        # Randomize selection
        # Note: We create a local copy to avoid modifying the original list in memory
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        
        # Partition into Support and Query (Disjoint sets)
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        # B. Forgery Samples (Query only)
        query_forg_paths = []
        if self.n_query_forgery > 0:
            if len(forgery_paths) < self.n_query_forgery:
                factor = (self.n_query_forgery // len(forgery_paths)) + 1
                forgery_paths = (forgery_paths * factor)
            
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # ---------------------------------------------------------------------
        # IMAGE LOADING & TENSOR CONSTRUCTION
        # ---------------------------------------------------------------------
        
        # Load Support Set (Apply augmentation if enabled)
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        
        # Load Query Set (No augmentation for reliable evaluation)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
        
        # Error Handling: If loading failed (e.g., corrupt files), try next user
        if support_imgs is None or query_imgs_gen is None:
             return self.__getitem__((index + 1) % len(self))

        # Concatenate Genuine and Forgery Queries
        if query_imgs_forg is not None:
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg], dim=0)
            # Label Assignment: 1 for Genuine, 0 for Forgery
            labels_gen = torch.ones(len(query_imgs_gen), dtype=torch.float32)
            labels_forg = torch.zeros(len(query_imgs_forg), dtype=torch.float32)
            query_labels = torch.cat([labels_gen, labels_forg], dim=0)
        else:
            query_imgs = query_imgs_gen
            query_labels = torch.ones(len(query_imgs_gen), dtype=torch.float32)

        return {
            'support_images': support_imgs,  # Shape: [K, 3, 224, 224]
            'query_images': query_imgs,      # Shape: [Q, 3, 224, 224]
            'query_labels': query_labels,    # Shape: [Q]
            'user_id': str(user_id)
        }

    def _load_batch(self, paths, augment=False):
        """
        Helper function to load and transform a batch of images.
        """
        images = []
        for path in paths:
            # Resolve Absolute/Relative Paths
            if self.use_full_path or os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.base_data_dir, path)
            
            try:
                # Load Image -> Convert to RGB (ensure 3 channels)
                img = Image.open(full_path).convert('RGB')
                
                # Apply appropriate transform pipeline
                if augment and hasattr(self, 'augment_transform'):
                    tensor = self.augment_transform(img)
                else:
                    tensor = self.transform(img)
                
                images.append(tensor)
            except Exception as e:
                # Log error but don't crash immediately (allow retry logic in __getitem__)
                # print(f"[Warning] Failed to load image {full_path}: {e}")
                continue
                
        if len(images) == 0:
            return None
            
        return torch.stack(images)