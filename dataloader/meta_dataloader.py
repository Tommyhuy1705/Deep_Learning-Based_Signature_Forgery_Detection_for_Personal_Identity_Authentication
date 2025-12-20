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
# MODULE: IMAGE PREPROCESSING UTILITIES
# =============================================================================
class ResizeWithPad:
    """
    Resizes an image to a target resolution while preserving aspect ratio.
    
    Mathematical Formulation:
    Let I be the input image with dimensions (W, H).
    Let T = (W_t, H_t) be the target dimensions.
    The scaling factor s is calculated as: s = min(W_t/W, H_t/H).
    The new dimensions are (W', H') = (s*W, s*H).
    Padding P is applied symmetrically such that the final output matches T.

    Attributes:
        target_size (tuple or int): Desired output size. If int, square (size, size) is assumed.
        fill (int): Pixel intensity for padding (default: 255 for white background).
    """
    def __init__(self, target_size, fill=255):
        # Robust handling for integer inputs to prevent TypeError during unpacking
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        elif isinstance(target_size, (list, tuple)):
            self.target_size = tuple(target_size)
        else:
            raise ValueError("target_size must be an int or a tuple (width, height)")
            
        self.fill = fill

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input image in RGB format.
        Returns:
            PIL.Image: Resized and padded image.
        """
        original_size = img.size  # (Width, Height)
        target_w, target_h = self.target_size
        
        # 1. Compute scaling factor
        ratio = min(target_w / original_size[0], target_h / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # 2. Resize with Bilinear Interpolation
        img = img.resize(new_size, Image.BILINEAR)
        
        # 3. Compute Padding (Centering)
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # 4. Apply Padding
        return pad(img, padding, fill=self.fill, padding_mode='constant')


# =============================================================================
# MODULE: FEW-SHOT EPISODIC DATALOADER
# =============================================================================
class SignatureEpisodeDataset(Dataset):
    """
    Signature Verification Dataset for Metric-Based Meta-Learning.

    Description:
    This dataset implements an episodic sampling protocol suitable for N-way K-shot 
    classification. It is designed to handle heterogenous data sources (BHSig, CEDAR)
    and incorporates domain-specific preprocessing techniques.

    Key Features:
    1. **Photometric Inversion**: Inverts pixel intensities ($I' = 1 - I$) to transform 
       sparse black ink signatures into high-activation features on a zero-background.
       This is critical for leveraging ImageNet-pretrained weights.
    2. **Stochastic Sampling with Replacement**: Addresses data scarcity by oversampling 
       when the class support size is insufficient.
    3. **Robust Schema Parsing**: Adaptively handles variations in JSON metadata structures.

    Attributes:
        split_file_path (str): Path to the JSON manifest.
        base_data_dir (str): Root directory for raw images.
        split_name (str): Partition key (e.g., 'meta-train').
        k_shot (int): Support set size per user.
    """

    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
        # ---------------------------------------------------------------------
        # 1. METADATA LOADING
        # ---------------------------------------------------------------------
        try:
            with open(split_file_path, 'r') as f:
                raw_data = json.load(f)
                
                # Parsing logic for different JSON schemas
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
            raise FileNotFoundError(f"[Critical] Split manifest not found: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # ---------------------------------------------------------------------
        # 2. TRANSFORMATION PIPELINE
        # ---------------------------------------------------------------------
        # Rationale: We use fill=255 (White) for padding. The subsequent Lambda 
        # transform (1.0 - x) converts White (1.0) to Black (0.0), ensuring the 
        # padded background has zero activation, focusing the model on the ink.
        # ---------------------------------------------------------------------

        # A. Inference Transform (Deterministic)
        self.transform = transforms.Compose([
            ResizeWithPad(224, fill=255),          # Resize & Pad White
            transforms.ToTensor(),                 # [0, 255] -> [0.0, 1.0]
            transforms.Lambda(lambda x: 1.0 - x),  # Invert: Ink=1.0, Bg=0.0
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        # B. Training Transform (Stochastic Augmentation)
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224, fill=255),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: 1.0 - x), # Invert
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        """
        Retrieves a data episode. Throws explicit errors if data is missing/corrupt
        to prevent silent failures in downstream metric calculation.
        """
        # 1. Retrieve User Info
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        # 2. Path Retrieval (Case-Insensitive)
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        # Strict Validation
        if not genuine_paths:
            raise ValueError(f"[Data Error] User {user_id}: No GENUINE signatures found in metadata.")

        # 3. Sampling Strategy
        # A. Genuine (Support + Query)
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            # Sampling with replacement
            factor = (required_genuine // len(genuine_paths)) + 1
            genuine_paths = genuine_paths * factor
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        # B. Forgery (Query)
        query_forg_paths = []
        if self.n_query_forgery > 0:
            if not forgery_paths:
                # If forgeries are requested but none exist, raise error immediately
                raise ValueError(f"[Data Error] User {user_id}: Requested {self.n_query_forgery} forgeries but metadata is empty.")
            
            if len(forgery_paths) < self.n_query_forgery:
                factor = (self.n_query_forgery // len(forgery_paths)) + 1
                forgery_paths = forgery_paths * factor
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # 4. Image Loading
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        
        # Load forgeries only if required
        query_imgs_forg = None
        if self.n_query_forgery > 0:
            query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
            if query_imgs_forg is None:
                 raise FileNotFoundError(f"[IO Error] User {user_id}: Failed to load FORGERY images. Check paths: {query_forg_paths[0]}")

        # Check integrity
        if support_imgs is None:
            raise FileNotFoundError(f"[IO Error] User {user_id}: Failed to load SUPPORT images.")
        if query_imgs_gen is None:
            raise FileNotFoundError(f"[IO Error] User {user_id}: Failed to load GENUINE QUERY images.")

        # 5. Tensor Aggregation
        if query_imgs_forg is not None:
            query_imgs = torch.cat([query_imgs_gen, query_imgs_forg], dim=0)
            # Label Construction: 1.0 (Genuine), 0.0 (Forgery)
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
        Loads and transforms a batch of images.
        """
        images = []
        for path in paths:
            # Resolve paths
            if self.use_full_path or os.path.isabs(path):
                full_path = path
            else:
                full_path = os.path.join(self.base_data_dir, path)
            
            try:
                # Load -> RGB -> Transform
                img = Image.open(full_path).convert('RGB')
                if augment and hasattr(self, 'augment_transform'):
                    tensor = self.augment_transform(img)
                else:
                    tensor = self.transform(img)
                images.append(tensor)
            except Exception as e:
                # Log error but attempt to continue with other images in batch
                # If all fail, the method returns None and triggers the Dataset error
                print(f"[Warning] Failed to load image: {full_path}. Error: {e}")
                continue
                
        if len(images) == 0:
            return None
        return torch.stack(images)