import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
from PIL import Image
import os
import json
import random
import sys

# --- Class ResizeWithPad (Giữ nguyên) ---
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

# --- Main Dataset Class (NO RECURSION - DEBUG MODE) ---
class SignatureEpisodeDataset(Dataset):
    def __init__(self, split_file_path, base_data_dir, split_name, 
                 k_shot=5, n_query_genuine=5, n_query_forgery=5, 
                 augment=False, use_full_path=False):
        
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

        self.base_data_dir = base_data_dir
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.use_full_path = use_full_path

        self.transform = transforms.Compose([
            ResizeWithPad(224, fill=255),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
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
        if isinstance(self.users_data, list):
            user_info = self.users_data[index]
            user_id = user_info.get('id', str(index))
        else:
            user_id = self.user_ids[index]
            user_info = self.users_data[user_id]

        genuine_paths = user_info.get('genuine') or user_info.get('Genuine')
        forgery_paths = user_info.get('forged') or user_info.get('Forged') or []

        # 1. Check Metadata
        if not genuine_paths:
            raise ValueError(f"User {user_id}: Danh sách ảnh Genuine trống trong JSON!")

        # 2. Sampling
        required_genuine = self.k_shot + self.n_query_genuine
        if len(genuine_paths) < required_genuine:
            genuine_paths = genuine_paths * ((required_genuine // len(genuine_paths)) + 1)
        
        genuine_paths_shuffled = random.sample(genuine_paths, len(genuine_paths))
        support_paths = genuine_paths_shuffled[:self.k_shot]
        query_gen_paths = genuine_paths_shuffled[self.k_shot : self.k_shot + self.n_query_genuine]

        query_forg_paths = []
        if self.n_query_forgery > 0:
            if not forgery_paths:
                # Nếu config yêu cầu ảnh giả mà user không có -> Báo lỗi ngay để biết
                print(f"[CẢNH BÁO] User {user_id} thiếu ảnh Forged, bỏ qua phần query này.")
            elif len(forgery_paths) < self.n_query_forgery:
                forgery_paths = forgery_paths * ((self.n_query_forgery // len(forgery_paths)) + 1)
                query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)
            else:
                query_forg_paths = random.sample(forgery_paths, self.n_query_forgery)

        # 3. Loading (BÁO LỖI NẾU KHÔNG TÌM THẤY FILE)
        support_imgs = self._load_batch(support_paths, augment=self.augment, user_id=user_id, set_type="Support")
        query_imgs_gen = self._load_batch(query_gen_paths, augment=False, user_id=user_id, set_type="Query Gen")
        
        query_imgs_forg = None
        if query_forg_paths:
            query_imgs_forg = self._load_batch(query_forg_paths, augment=False, user_id=user_id, set_type="Query Forg")

        # 4. Aggregation
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

    def _load_batch(self, paths, augment=False, user_id="Unknown", set_type="Unknown"):
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
                # --- ĐÂY LÀ CHỖ QUAN TRỌNG NHẤT ---
                # Thay vì bỏ qua, in ra đường dẫn bị lỗi và dừng chương trình
                raise FileNotFoundError(f"LỖI LOAD ẢNH [User: {user_id} | Set: {set_type}]:\n"
                                        f"Không thể mở file: {full_path}\n"
                                        f"Lỗi chi tiết: {str(e)}")
                
        if len(images) == 0:
             raise ValueError(f"Batch rỗng cho User {user_id} ({set_type})")
        return torch.stack(images)