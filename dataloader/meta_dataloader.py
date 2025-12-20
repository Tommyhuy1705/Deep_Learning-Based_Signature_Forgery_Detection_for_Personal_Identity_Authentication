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
    def __init__(self, target_size, fill=255):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        elif isinstance(target_size, (list, tuple)):
            self.target_size = tuple(target_size)
        else:
            raise ValueError("target_size must be an int or a tuple")
        self.fill = fill

    def __call__(self, img):
        original_size = img.size
        target_w, target_h = self.target_size
        ratio = min(target_w / original_size[0], target_h / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        img = img.resize(new_size, Image.BILINEAR)
        delta_w = target_w - new_size[0]
        delta_h = target_h - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        return pad(img, padding, fill=self.fill, padding_mode='constant')

# =============================================================================
# MODULE: FEW-SHOT EPISODIC DATALOADER (ROBUST VERSION)
# =============================================================================
class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
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
            raise FileNotFoundError(f"[Critical] Split manifest not found: {split_file_path}")

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        # Inference Transform
        self.transform = transforms.Compose([
            ResizeWithPad(224, fill=255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentation Transform
        if self.augment:
            self.augment_transform = transforms.Compose([
                ResizeWithPad(224, fill=255),
                transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=255),
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

        # 2. Path Retrieval
        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        # --- ROBUST SKIP LOGIC ---
        # Nếu user này bị lỗi dữ liệu (thiếu ảnh thật hoặc thiếu ảnh giả),
        # ta sẽ tự động nhảy sang user tiếp theo (index + 1)
        if not genuine_paths:
            # print(f"[Skip] User {user_id}: Missing Genuine images. Trying next user...")
            return self.__getitem__((index + 1) % len(self))

        if self.n_query_forgery > 0 and not forgery_paths:
            # print(f"[Skip] User {user_id}: Missing Forgery images. Trying next user...")
            return self.__getitem__((index + 1) % len(self))
        # -------------------------

        # 3. Sampling
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            genuine_paths = genuine_paths * ((required_genuine // len(genuine_paths)) + 1)
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        query_forg_paths = []
        if self.n_query_forgery > 0:
            if len(forgery_paths) < self.n_query_forgery:
                forgery_paths = forgery_paths * ((self.n_query_forgery // len(forgery_paths)) + 1)
            query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # 4. Loading
        support_imgs = self._load_batch(support_paths, augment=self.augment)
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False)
        query_imgs_forg = None
        
        if self.n_query_forgery > 0:
            query_imgs_forg = self._load_batch(query_forg_paths, augment=False)
            # Nếu load ảnh giả thất bại, cũng skip luôn user này
            if query_imgs_forg is None:
                return self.__getitem__((index + 1) % len(self))

        # Nếu load ảnh thật thất bại, skip user
        if support_imgs is None or query_imgs_gen is None:
            return self.__getitem__((index + 1) % len(self))

        # 5. Tensor Aggregation
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
            except Exception:
                continue 
        if len(images) == 0: return None
        return torch.stack(images)