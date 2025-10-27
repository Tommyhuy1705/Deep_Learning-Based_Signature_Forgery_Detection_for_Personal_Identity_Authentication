import torch
from torch.utils.data import Dataset
import random
import json
from PIL import Image
import os
from torchvision import transforms

class SignatureEpisodeDataset(Dataset):
    """
    A PyTorch Dataset class for creating few-shot learning episodes for signature verification.

    Each item retrieved from this dataset represents one "episode" or "task",
    simulating the scenario of adapting to a new user with limited samples.
    An episode consists of a support set (known genuine samples) and a query set
    (samples to be classified, including both genuine and forged).
    """
    def __init__(self, split_file_path, base_data_dir, split_name, k_shot=10, n_query_genuine=15, n_query_forgery=15, augment=False, use_full_path=False):
        """
        Initializes the SignatureEpisodeDataset.

        Args:
            split_file_path (str): Path to the JSON file containing user IDs and image paths for the split (e.g., meta-train, meta-test).
            base_data_dir (str): Root directory of the dataset (e.g., '/kaggle/input/cedar-dataset/signatures'). Used only if use_full_path is False.
            split_name (str): The name of the split to load ('meta-train' or 'meta-test').
            k_shot (int): Number of genuine samples in the support set.
            n_query_genuine (int): Number of genuine samples in the query set.
            n_query_forgery (int): Number of forgery samples in the query set.
            augment (bool): Whether to apply data augmentation (RandomAffine) to the images. Should be True for training, False for evaluation.
            use_full_path (bool): If True, treats paths in the JSON file as absolute paths. If False, joins paths with base_data_dir.
        """
        try:
            with open(split_file_path, 'r') as f:
                split_content = json.load(f)[split_name]
        except FileNotFoundError:
            raise FileNotFoundError(f"Split file not found at: {split_file_path}")
        except KeyError:
            raise KeyError(f"Split name '{split_name}' not found in the JSON file.")

        self.data = {}
        if use_full_path:
            # Assumes JSON contains absolute paths
            self.data = split_content
        else:
            # Constructs absolute paths by joining base_data_dir
            if base_data_dir is None:
                raise ValueError("base_data_dir must be provided when use_full_path is False")
            for user_id, paths in split_content.items():
                # Make sure user_id is stored as string consistently
                str_user_id = str(user_id)
                self.data[str_user_id] = {
                    'genuine': [os.path.join(base_data_dir, p) for p in paths['genuine']],
                    'forgery': [os.path.join(base_data_dir, p) for p in paths['forgery']]
                }

        # Store user IDs as strings for consistent dictionary access
        self.user_ids = list(self.data.keys())
        self.k_shot = k_shot
        self.n_query_genuine = n_query_genuine
        self.n_query_forgery = n_query_forgery
        self.augment = augment
        self.length = len(self.user_ids) # Number of users/episodes in this split

        # Define image transformations
        self.base_transform = transforms.Compose([
            transforms.Resize((220, 150)),
            transforms.Grayscale(), # Ensure single channel first
            transforms.ToTensor(), # Converts to [0, 1] range and CxHxW format
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # Convert grayscale to 3 channels for ResNet
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize to [-1, 1] range
        ])

        # Augmentation transform includes random affine transformations
        self.augment_transform = transforms.Compose([
            transforms.Resize((220, 150)),
            transforms.Grayscale(),
            # Slight random rotations, translations, and scaling
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        print(f"Initialized SignatureEpisodeDataset for '{split_name}' split.")
        print(f"  Number of users/episodes: {self.length}")
        print(f"  k_shot: {self.k_shot}, n_query_genuine: {self.n_query_genuine}, n_query_forgery: {self.n_query_forgery}")
        print(f"  Augmentation: {'Enabled' if self.augment else 'Disabled'}")
        print(f"  Using full paths: {use_full_path}")


    def __len__(self):
        """Returns the number of episodes (users) in the dataset."""
        return self.length

    def __getitem__(self, index):
        """
        Generates a single few-shot learning episode.

        Args:
            index (int): The index of the user (episode) to retrieve.

        Returns:
            dict: A dictionary containing the support set images, query set images,
                  query set labels, and the user ID for the episode. Returns None if
                  critical errors occur (e.g., unable to load sufficient images).
        """
        user_id = self.user_ids[index]
        user_data = self.data[user_id]

        genuine_paths = user_data.get('genuine', [])
        forgery_paths = user_data.get('forgery', [])

        # --- Handle cases with insufficient samples using resampling ---
        required_genuine = self.k_shot + self.n_query_genuine
        if not genuine_paths: # No genuine samples available for this user
             print(f"Warning: No genuine samples found for user {user_id}. Skipping episode by returning None.")
             return self.__getitem__((index + 1) % len(self)) # Simplest fix: try next user


        if len(genuine_paths) < required_genuine:
            # Use random.choices for sampling with replacement
            genuine_paths = random.choices(genuine_paths, k=required_genuine)

        if not forgery_paths: # Handle case with no forgery samples if needed
            if self.n_query_forgery > 0 and not forgery_paths:
                print(f"Warning: No forgery samples found for user {user_id}, but {self.n_query_forgery} were requested. Trying next user.")
                return self.__getitem__((index + 1) % len(self))
            elif self.n_query_forgery > 0 and len(forgery_paths) < self.n_query_forgery:
                 forgery_paths = random.choices(forgery_paths, k=self.n_query_forgery)
            elif self.n_query_forgery == 0: # If 0 forgeries requested, ensure list is empty
                 forgery_paths = []


        # Shuffle paths before splitting to ensure randomness
        random.shuffle(genuine_paths)
        if forgery_paths: random.shuffle(forgery_paths)

        # Create Support Set (k_shot genuine samples)
        support_paths = genuine_paths[:self.k_shot]

        # Create Query Set (n_query genuine + n_query forgery samples)
        query_genuine_paths = genuine_paths[self.k_shot : self.k_shot + self.n_query_genuine]
        query_forgery_paths = forgery_paths[:self.n_query_forgery] # Slicing handles empty list correctly

        query_paths = query_genuine_paths + query_forgery_paths
        # Ensure labels match the actual number of samples selected
        query_labels = [1] * len(query_genuine_paths) + [0] * len(query_forgery_paths)

        # Select the appropriate transform based on the 'augment' flag
        transform_to_apply = self.augment_transform if self.augment else self.base_transform

        # Helper function to load and transform images, with error handling
        def _load_images(paths):
            images = []
            valid_paths = []
            for p in paths:
                try:
                    img = Image.open(p).convert('L')
                    images.append(img)
                    valid_paths.append(p)
                except FileNotFoundError:
                    print(f"Error: Image file not found at {p}. Skipping.")
                except Exception as e:
                    print(f"Error loading image {p}: {e}. Skipping.")

            if not images:
                return None

            # Apply transformations
            try:
                transformed_images = [transform_to_apply(img) for img in images]
                return torch.stack(transformed_images)
            except Exception as e:
                 print(f"Error applying transforms to images (paths: {valid_paths}): {e}")
                 return None


        support_images = _load_images(support_paths)
        query_images = _load_images(query_paths)

        # Handle potential loading failures
        if support_images is None or query_images is None or len(query_images) != len(query_labels):
            print(f"Critical error loading images for user {user_id}. Attempting next user.")
            return self.__getitem__((index + 1) % len(self))


        return {
            'support_images': support_images,    # Tensor shape: [k_shot, C, H, W]
            'query_images': query_images,        # Tensor shape: [n_query_genuine + n_query_forgery, C, H, W]
            'query_labels': torch.tensor(query_labels, dtype=torch.long), # Tensor shape: [n_query_genuine + n_query_forgery]
            'user_id': user_id                   # User identifier (string)
        }