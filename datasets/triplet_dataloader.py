from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import numpy as np


class TripletSignatureDataset(Dataset):
    def __init__(self, org_dir, forg_dir, transform=None):
        self.org_images = sorted([os.path.join(org_dir, f) for f in os.listdir(org_dir) if f.endswith('.png')])
        self.forg_images = sorted([os.path.join(forg_dir, f) for f in os.listdir(forg_dir) if f.endswith('.png')])
        self.transform = transform
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []

        # Tạo triplets với anchor là chữ ký thật
        for anchor in self.org_images:
            # Lấy base name (ví dụ: "10" từ "original_10_1.png")
            base_name = os.path.basename(anchor).split('_')[1]

            # Positive: chữ ký thật khác của cùng người
            positives = [img for img in self.org_images
                        if f"_{base_name}_" in img and img != anchor]

            # Negative: chữ ký giả của cùng người hoặc chữ ký thật của người khác
            forg_negatives = [img for img in self.forg_images if f"_{base_name}_" in img]
            other_negatives = [img for img in self.org_images if f"_{base_name}_" not in img]
            negatives = forg_negatives + other_negatives

            if positives and negatives:
                positive = random.choice(positives)
                negative = random.choice(negatives)
                triplets.append((anchor, positive, negative))

        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]

        anchor = Image.open(anchor_path).convert('L')
        positive = Image.open(positive_path).convert('L')
        negative = Image.open(negative_path).convert('L')

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return (anchor, positive, negative)