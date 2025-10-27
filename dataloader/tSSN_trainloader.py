from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import os
import re
import numpy as np

class SignaturePretrainDataset(Dataset):
    """
    A PyTorch Dataset class for creating (Anchor, Positive, Negative) triplets
    for pre-training the signature feature extractor using standard Triplet Loss.

    This dataloader samples triplets based on user identity derived from filenames.
    """
    def __init__(self, org_dir, forg_dir, transform=None):
        """
        Initializes the SignaturePretrainDataset.

        Args:
            org_dir (str): Path to the directory containing genuine signature images.
                           Filenames are expected to follow a pattern like 'original_USERID_SAMPLENO.png'.
            forg_dir (str): Path to the directory containing forged signature images.
                            Filenames are expected to follow a pattern like 'forgeries_USERID_SAMPLENO.png'.
            transform (callable, optional): torchvision transforms to be applied to the images.
        """
        # Ensure directories exist
        if not os.path.isdir(org_dir):
            raise FileNotFoundError(f"Genuine signatures directory not found: {org_dir}")
        if not os.path.isdir(forg_dir):
            raise FileNotFoundError(f"Forged signatures directory not found: {forg_dir}")

        # Load image paths, filtering for common image extensions
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')
        self.org_images = sorted([
            os.path.join(org_dir, f) for f in os.listdir(org_dir)
            if f.lower().endswith(supported_extensions)
        ])
        self.forg_images = sorted([
            os.path.join(forg_dir, f) for f in os.listdir(forg_dir)
            if f.lower().endswith(supported_extensions)
        ])

        if not self.org_images:
            print(f"Warning: No genuine images found in {org_dir}")
        if not self.forg_images:
            print(f"Warning: No forged images found in {forg_dir}")


        self.transform = transform
        self.triplets = self._create_triplets()

        if not self.triplets:
            print("Warning: No triplets could be generated. Check image paths and filename conventions.")
        else:
             print(f"Generated {len(self.triplets)} triplets for pre-training.")

    def _get_user_id_from_filename(self, filename):
        """Extracts user ID from filename using regex (e.g., '_10_' -> 10)."""
        # Attempt to find patterns like _USERID_
        match = re.search(r'_(\d+)_', filename)
        if match:
            return int(match.group(1))
        else:
            # Add fallback patterns if needed, e.g., for BHSig filenames B-S-011-G-01.tif
            match = re.search(r'-(\d+)-', filename)
            if match:
                 return int(match.group(1))
        # print(f"Warning: Could not extract user ID from filename: {filename}")
        return None # Return None if no ID found

    def _create_triplets(self):
        """
        Generates a list of (anchor, positive, negative) path triplets.

        Logic:
        - Anchor: A genuine signature.
        - Positive: Another genuine signature from the same user as the anchor.
        - Negative: EITHER a forged signature of the anchor's user OR a genuine signature from a different user.
        """
        triplets = []
        user_genuine_map = {} # Cache genuine images per user

        # Group genuine images by user ID
        for img_path in self.org_images:
            filename = os.path.basename(img_path)
            user_id = self._get_user_id_from_filename(filename)
            if user_id is not None:
                if user_id not in user_genuine_map:
                    user_genuine_map[user_id] = []
                user_genuine_map[user_id].append(img_path)

        all_user_ids = list(user_genuine_map.keys())
        if not all_user_ids:
             print("Error: No user IDs could be extracted from genuine filenames.")
             return []


        # Iterate through each genuine image as an anchor
        for anchor_path in self.org_images:
            anchor_filename = os.path.basename(anchor_path)
            anchor_user_id = self._get_user_id_from_filename(anchor_filename)

            if anchor_user_id is None:
                continue # Skip if user ID couldn't be extracted

            # --- Find Positive Sample ---
            # Another genuine signature from the same user, excluding the anchor itself
            possible_positives = [
                img for img in user_genuine_map.get(anchor_user_id, [])
                if img != anchor_path
            ]
            if not possible_positives:
                continue # Skip if no other genuine sample exists for this user

            positive_path = random.choice(possible_positives)

            # --- Find Negative Sample ---
            possible_negatives = []
            # Option 1: Forged signature of the same user
            forged_negatives = [
                f_img for f_img in self.forg_images
                if self._get_user_id_from_filename(os.path.basename(f_img)) == anchor_user_id
            ]
            possible_negatives.extend(forged_negatives)

            # Option 2: Genuine signature from a different user
            other_user_ids = [uid for uid in all_user_ids if uid != anchor_user_id]
            if other_user_ids:
                 other_user_id = random.choice(other_user_ids)
                 genuine_negatives = user_genuine_map.get(other_user_id, [])
                 possible_negatives.extend(genuine_negatives)


            if not possible_negatives:
                continue # Skip if no negative sample can be found

            negative_path = random.choice(possible_negatives)

            triplets.append((anchor_path, positive_path, negative_path))

        return triplets

    def __len__(self):
        """Returns the total number of triplets generated."""
        return len(self.triplets)

    def __getitem__(self, idx):
        """
        Retrieves a triplet of images (anchor, positive, negative) at the given index.

        Args:
            idx (int): The index of the triplet.

        Returns:
            tuple: A tuple containing the transformed anchor, positive, and negative image tensors.
                   Returns None if any image fails to load.
        """
        anchor_path, positive_path, negative_path = self.triplets[idx]

        try:
            # Load images and convert to grayscale ('L')
            anchor_img = Image.open(anchor_path).convert('L')
            positive_img = Image.open(positive_path).convert('L')
            negative_img = Image.open(negative_path).convert('L')

            # Apply transformations if provided
            if self.transform:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)

            return anchor_img, positive_img, negative_img

        except FileNotFoundError as e:
            print(f"Error: Image file not found in triplet at index {idx}: {e}. Returning None.")
            return None # Or handle more gracefully, e.g., skip in DataLoader collate_fn
        except Exception as e:
            print(f"Error loading images for triplet at index {idx}: {e}. Returning None.")
            return None