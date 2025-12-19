from torch.utils.data import Dataset
from PIL import Image
import random
import os
import re
import torchvision.transforms as transforms
try:
    from utils.helpers import ResizeWithPad
except ImportError:
    import sys
    sys.path.append('..')
    from utils.helpers import ResizeWithPad

class SignaturePretrainDataset(Dataset):
    """
    Dataset for Pre-training with Hard Negative Mining.
    """
    def __init__(self, org_dir, forg_dir, transform=None, user_list=None):
        """
        Args:
            user_list (list, optional): List of user IDs (strings) to include. 
                                        Used to restrict training to Background Set only.
        """
        self.org_images = []
        self.forg_images = []
        
        # Recursively find all images (handles CEDAR/BHSig structures)
        for root, _, files in os.walk(org_dir):
            for file in files:
                if file.lower().endswith(('.png','.tif','.jpg','.jpeg')):
                     self.org_images.append(os.path.join(root, file))
        
        for root, _, files in os.walk(forg_dir):
            for file in files:
                 if file.lower().endswith(('.png','.tif','.jpg','.jpeg')):
                     self.forg_images.append(os.path.join(root, file))

        # Filter by user_list if provided (CRITICAL for Splitting)
        if user_list is not None:
            self.org_images = [p for p in self.org_images if self._get_user_id(os.path.basename(p)) in user_list]
            self.forg_images = [p for p in self.forg_images if self._get_user_id(os.path.basename(p)) in user_list]

        if transform is None:
            self.transform = transforms.Compose([
                ResizeWithPad((220, 150)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform
            
        self.triplets = self._create_triplets()
        print(f"Generated {len(self.triplets)} triplets (Hard Mining enabled).")

    def _get_user_id(self, filename):
        # Support both CEDAR (_1_) and BHSig (B-S-1-...)
        # Try BHSig first
        match = re.search(r'([BH])-[SFG]-(\d+)-', filename, re.IGNORECASE)
        if match: return f"{match.group(1)}-{int(match.group(2))}"
        
        # Try CEDAR
        match = re.search(r'_(\d+)_', filename)
        if match: return str(int(match.group(1)))
        
        return None

    def _create_triplets(self):
        triplets = []
        user_genuine_map = {}

        # Map User -> List of Genuine Images
        for img_path in self.org_images:
            uid = self._get_user_id(os.path.basename(img_path))
            if uid:
                if uid not in user_genuine_map: user_genuine_map[uid] = []
                user_genuine_map[uid].append(img_path)
        
        all_user_ids = list(user_genuine_map.keys())

        for anchor_path in self.org_images:
            anchor_uid = self._get_user_id(os.path.basename(anchor_path))
            if not anchor_uid: continue

            # Positive: Another genuine signature of same user
            positives = [p for p in user_genuine_map.get(anchor_uid, []) if p != anchor_path]
            if not positives: continue
            positive_path = random.choice(positives)

            # --- HARD NEGATIVE MINING LOGIC ---
            # 1. Skilled Forgery (Hard Negative): Forgery of THIS user
            # Find forgeries that match this user ID
            current_forgeries = [f for f in self.forg_images if self._get_user_id(os.path.basename(f)) == anchor_uid]
            
            # 2. Random Negative (Easy Negative): Genuine signature of OTHER user
            other_uid = random.choice([u for u in all_user_ids if u != anchor_uid])
            random_negatives = user_genuine_map.get(other_uid, [])

            # Probability: 70% pick Hard Negative (if available), 30% pick Easy
            negative_path = None
            if current_forgeries and (random.random() < 0.7 or not random_negatives):
                negative_path = random.choice(current_forgeries)
            elif random_negatives:
                negative_path = random.choice(random_negatives)
            
            if negative_path:
                triplets.append((anchor_path, positive_path, negative_path))

        return triplets

    def __getitem__(self, idx):
        anchor, pos, neg = self.triplets[idx]
        a_img = self.transform(Image.open(anchor).convert('L'))
        p_img = self.transform(Image.open(pos).convert('L'))
        n_img = self.transform(Image.open(neg).convert('L'))
        return a_img, p_img, n_img
    
    def __len__(self): return len(self.triplets)