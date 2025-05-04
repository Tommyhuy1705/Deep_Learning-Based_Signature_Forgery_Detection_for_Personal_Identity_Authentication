import random
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

class SiameseSignatureDataset(Dataset):
    def __init__(self, org_dir, forg_dir, transform=None):
        self.org_images = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir) if f.endswith('.png')])
        self.forg_images = sorted([os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if f.endswith('.png')])
        self.transform = transform
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        pairs = []
        # Tạo positive pairs (cùng người, chữ ký thật với thật)
        for i in range(len(self.org_images)):
            # Lấy 2 chữ ký khác nhau của cùng 1 người (dựa vào tên file)
            base_name = os.path.basename(self.org_images[i]).split('_')[1]
            matching = [img for img in self.org_images if f"_{base_name}_" in img]
            if len(matching) > 1:
                pairs.append((random.choice(matching), self.org_images[i], 1))  # Label 1 = similar

        # Tạo negative pairs (chữ ký thật với giả hoặc khác người)
        for i in range(len(self.org_images)):
            # Chữ ký thật với giả của cùng người
            base_name = os.path.basename(self.org_images[i]).split('_')[1]
            forg_match = [img for img in self.forg_images if f"_{base_name}_" in img]
            if forg_match:
                pairs.append((self.org_images[i], random.choice(forg_match), 0))  # Label 0 = dissimilar

            # Chữ ký của 2 người khác nhau
            if i < len(self.org_images) - 1:
                pairs.append((self.org_images[i], self.org_images[i+1], 0))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]

        img1 = Image.open(img1_path).convert('L')
        img2 = Image.open(img2_path).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), torch.tensor(label, dtype=torch.float32)