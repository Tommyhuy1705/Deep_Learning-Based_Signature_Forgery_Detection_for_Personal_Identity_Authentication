import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import json
import random
import sys

# Attempt to import custom utility for aspect-ratio preserving resize.
# If running in a nested directory structure, append parent path.
try:
    from utils.helpers import ResizeWithPad
except ImportError:
    sys.path.append('..')
    from utils.helpers import ResizeWithPad

class SignatureEpisodeDataset(Dataset):
    """
    Signature Verification Episode Dataset for Few-Shot Learning.

    Description:
    This dataset class implements the episodic sampling protocol required for
    metric-based meta-learning algorithms (e.g., Prototypical Networks, Siamese Networks).
    It constructs N-way K-shot tasks by sampling support and query sets from 
    user-specific signature distributions.

    Key Features:
    1. **Robust Schema Parsing**: Adaptively handles variations in JSON metadata structures 
       (Dictionary vs. List) and key capitalization (e.g., 'Genuine' vs. 'genuine').
    2. **Photometric Inversion**: Applies an intensity inversion transform ($I' = 1 - I$) 
       to align sparse signature strokes (typically black ink) with the high-activation 
       patterns expected by CNNs pre-trained on ImageNet.
    3. **Stochastic Sampling with Replacement**: Ensures robust episode generation even 
       when the number of available samples is less than the required K-shot + N-query.

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
        
        # ---------------------------------------------------------------------
        # 1. METADATA LOADING & PARSING
        # ---------------------------------------------------------------------
        try:
            with open(split_file_path, 'r') as f:
                raw_data = json.load(f)
                
                # Handling different JSON schemas (BHSig Dict vs. CEDAR List)
                if isinstance(raw_data, dict) and split_name in raw_data:
                    self.users_data = raw_data[split_name] # Dict: {uid: data}
                    self.user_ids = list(self.users_data.keys())
                elif isinstance(raw_data, list):
                    self.users_data = raw_data             # List: [{id: uid, ...}]
                    self.user_ids = range(len(self.users_data))
                else:
                    # Fallback for flat dictionary structures
                    self.users_data = raw_data
                    self.user_ids = list(self.users_data.keys())
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"[Critical] Split file not found at: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # ---------------------------------------------------------------------
        # 2. DATA TRANSFORMATION PIPELINE
        # ---------------------------------------------------------------------
        # Scientific Rationale:
        # Standard CNNs respond maximally to high pixel values. Since signatures 
        # are black (0) on white (1), we invert the tensor to make ink distinct.
        # ---------------------------------------------------------------------

        # A. Validation/Inference Transform (Deterministic)
        self.transform = transforms.Compose([
            ResizeWithPad(224),
            transforms.ToTensor(),
            # Inversion: Black Ink (0) -> White Feature (1)
            transforms.Lambda(lambda x: 1.0 - x), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # B. Training Transform (Stochastic Augmentation)
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224),
                # Affine perturbations to simulate intra-class variability
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
                transforms.ToTensor(),
                # Inversion must apply to augmented images as well
                transforms.Lambda(lambda x: 1.0 - x),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        """Returns the total number of available users/episodes."""
        return len(self.user_ids)

    def __getitem__(self, index):
        """
        Constructs a single Few-Shot Episode (Task).
        
        Returns:
            dict: {
                'support_images': Tensor [K, C, H, W],
                'query_images':   Tensor [Q, C, H, W],
                'query_labels':   Tensor [Q],
                'user_id':        str
            }
        """
        # ---------------------------------------------------------------------
        # 1. USER DATA RETRIEVAL
        # ---------------------------------------------------------------------
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        # Robust Key Retrieval (Case-Insensitive)
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        # Integrity Check
        if not genuine_paths:
            raise ValueError(f"[Data Error] User {user_id} contains no genuine signatures.")

        # ---------------------------------------------------------------------
        # 2. SAMPLING STRATEGY (WITH REPLACEMENT)
        # ---------------------------------------------------------------------
        # Objective: Partition data into Support and Query sets.
        # If insufficient data, replicate samples to meet requirements.
        
        # A. Genuine Samples
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            factor = (required_genuine // len(genuine_paths)) + 1
            genuine_paths = (genuine_paths * factor)
        
        # Shuffle to randomize selection
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        
        # Split Disjoint Sets
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        # B. Forgery Samples (Query Only)
        query_forg_paths = []
        if self.n_query_forgery > 0 and len(forgery_paths) > 0:
            if len(forgery_paths) < self.n_query_forgery:
                factor = (self.n_query_forgery // len(forgery_paths)) + 1
                forgery_paths = (forgery_paths * factor)
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # ---------------------------------------------------------------------
        # 3. IMAGE LOADING & BATCH CONSTRUCTION
        # ---------------------------------------------------------------------
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
        
        # --- CRITICAL ERROR HANDLING (NO RECURSION) ---
        # Explicitly raise error if files are missing to prevent silent failure loop.
        if support_imgs is None:
            raise FileNotFoundError(f"[IO Error] Failed to load SUPPORT set for User {user_id}. check paths!")
            
        if query_imgs_gen is None:
             raise FileNotFoundError(f"[IO Error] Failed to load GENUINE queries for User {user_id}.")

        # 4. TENSOR AGGREGATION
        if query_imgs_forg is not None:
            # Concatenate Genuine and Forgery queries
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg], dim=0)
            # Labels: 1 = Genuine, 0 = Forged
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
        Helper function to load and preprocess a batch of images from disk.
        """
        images = []
        for path in paths:
            # Resolve Path (Absolute vs Relative)
            if self.use_full_path or os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.base_data_dir, path)
            
            try:
                # Load Image -> Convert to RGB (3 Channels)
                img = Image.open(full_path).convert('RGB')
                
                # Apply Transform
                if augment and hasattr(self, 'augment_transform'):
                    tensor = self.augment_transform(img)
                else:
                    tensor = self.transform(img)
                images.append(tensor)
            except Exception as e:
                # Log warning but continue processing other images in batch
                print(f"[Warning] Image load failed: {full_path}. Error: {e}")
                continue
                
        if len(images) == 0:
            return None # Trigger error in __getitem__
            
        return torch.stack(images)