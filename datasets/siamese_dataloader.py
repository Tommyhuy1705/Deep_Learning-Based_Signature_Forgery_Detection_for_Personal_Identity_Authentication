import os
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import BatchSampler
from torchvision import transforms
import torch
import yaml

class SiameseDataset(Dataset):
    def __init__(self, original_data_path, forgeries_data_path, transform=None):
        """
        Dataset cho Siamese Network, tạo các cặp ảnh (genuine-genuine hoặc genuine-forgery)
        
        Args:
            original_data_path: Đường dẫn đến thư mục chứa chữ ký thật
            forgeries_data_path: Đường dẫn đến thư mục chứa chữ ký giả
            transform: Các biến đổi áp dụng lên ảnh
        """
        self.original_data_path = original_data_path
        self.forgeries_data_path = forgeries_data_path
        self.transform = transform
        
        # Lấy danh sách người dùng (folder con)
        self.users = os.listdir(self.original_data_path)
        
        # Tạo tất cả các cặp và nhãn tương ứng
        self.pairs = []
        self.labels = []
        self._create_pairs()
        
        print(f"Tổng số cặp ảnh: {len(self.pairs)}")
        print(f"Số cặp positive (1): {self.labels.count(1)}")
        print(f"Số cặp negative (0): {self.labels.count(0)}")   
    
    def _create_pairs(self):
        """Tạo các cặp positive và negative"""
        for user in self.users:
            original_user_path = os.path.join(self.original_data_path, user)
            forgery_user_path = os.path.join(self.forgeries_data_path, user)
            
            if not os.path.exists(original_user_path) or not os.path.exists(forgery_user_path):
                #print(f"[WARN] Không tìm thấy thư mục giả mạo cho {user}, bỏ qua...")
                continue
                
            original_images = [f for f in os.listdir(original_user_path) if os.path.isfile(os.path.join(original_user_path, f))]
            forgery_images = [f for f in os.listdir(forgery_user_path) if os.path.isfile(os.path.join(forgery_user_path, f))]
            
            # print(f"[INFO] User: {user} - Thật: {len(original_images)} ảnh, Giả: {len(forgery_images)} ảnh")

            # # Nếu số ảnh không đủ thì skip
            # if len(original_images) < 2 or len(forgery_images) == 0:
            #     print(f"[WARN] Không đủ ảnh để tạo cặp cho {user}")
            #     continue

            # Cặp positive (2 ảnh thật)
            for i in range(len(original_images)):
                for j in range(i + 1, len(original_images)):
                    img1 = os.path.join(original_user_path, original_images[i])
                    img2 = os.path.join(original_user_path, original_images[j])
                    self.pairs.append((img1, img2))
                    self.labels.append(1)  # ảnh thật cùng người
            
            # Cặp negative (1 ảnh thật, 1 ảnh giả)
            for i in range(len(original_images)):
                # Chọn ngẫu nhiên một ảnh giả của cùng người dùng
                for j in range(min(len(forgery_images), len(original_images))):
                    img1 = os.path.join(original_user_path, original_images[i])
                    img2 = os.path.join(forgery_user_path, forgery_images[j])
                    self.pairs.append((img1, img2))
                    self.labels.append(0)  # 1 ảnh giả, 1 ảnh thật
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Đọc ảnh
        img1 = Image.open(img1_path).convert("L")  # Chuyển sang ảnh xám
        img2 = Image.open(img2_path).convert("L")
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return (img1, img2), torch.tensor(label, dtype=torch.float32)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - tạo các batch gồm số lượng bằng nhau cặp positive và cặp negative
    từ dataset
    """
    
    def __init__(self, labels, n_classes, n_samples):
        """
        Args:
            labels: list hoặc numpy array chứa nhãn của các cặp ảnh
            n_classes: số lượng class mỗi batch (ở đây là 2: positive và negative)
            n_samples: số lượng mẫu mỗi class trong mỗi batch
        """
        self.labels = np.array(labels)
        self.n_classes = n_classes  # Ở đây là 2 (positive và negative)
        self.n_samples = n_samples
        self.batch_size = n_classes * n_samples
        
        # Phân chia chỉ số theo nhãn
        self.label_indices = {}
        self.label_indices[0] = np.where(self.labels == 0)[0].tolist()
        self.label_indices[1] = np.where(self.labels == 1)[0].tolist()
        
        self.used_indices = {0: [], 1: []}
        self.count = 0
        
        # Tính số lượng batch
        self.n_batches = min([len(indices) // self.n_samples 
                            for indices in self.label_indices.values()])
    
    def __iter__(self):
        """Tạo và trả về các batch cân bằng"""
        for _ in range(self.n_batches):
            batch = []
            for label in self.label_indices.keys():
                # Nếu đã dùng hết indices, shuffle lại
                if len(self.used_indices[label]) + self.n_samples > len(self.label_indices[label]):
                    self.used_indices[label] = []
                    random.shuffle(self.label_indices[label])
                
                # Lấy các indices cho batch hiện tại
                indices = self.label_indices[label][len(self.used_indices[label]):len(self.used_indices[label]) + self.n_samples]
                batch.extend(indices)
                self.used_indices[label].extend(indices)
            
            # Trộn các indices trong batch
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.n_batches


def get_transforms(img_size):
    """
    Tạo các biến đổi cho ảnh
    
    Args:
        img_size: tuple (width, height) - kích thước ảnh sau khi resize
    """
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Chuẩn hóa cho ảnh xám
    ])


def load_config(config_path):
    """
    Load cấu hình từ file YAML
    Args:
        config_path: Đường dẫn đến file cấu hình
    
    Returns:
        dict: Dictionary chứa cấu hình
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def create_dataloaders(config):
    """
    Tạo dataloader cho train và validation từ file config
    
    Args:
        config_path: Đường dẫn đến file cấu hình
    
    Returns:
        train_loader, val_loader: DataLoader cho train và validation
    """
    # Load cấu hình từ file
    #config = load_config(config_path)
    
    # Lấy thông số từ config
    original_data_path = config['dataset']['original_data_path']
    forgeries_data_path = config['dataset']['forgeries_data_path']
    batch_size = config['training']['batch_size']
    img_size = tuple(config['dataset']['input_size']) # (width, height)
    
    # Tạo transforms cho ảnh
    transform = get_transforms(img_size)
    
    # Tạo dataset chung cho cả train và validation
    full_dataset = SiameseDataset(
        original_data_path=original_data_path,
        forgeries_data_path=forgeries_data_path,
        transform=transform
    )
    
    # Tính số lượng mẫu cho train và validation (80-20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Chia dataset ngẫu nhiên
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Đặt seed để có kết quả reproducible
    )
    
    print(f"Số lượng mẫu tập train: {len(train_dataset)}")
    print(f"Số lượng mẫu tập validation: {len(val_dataset)}")
    
    # Tính số lượng mẫu mỗi class trong batch
    samples_per_class = batch_size // 2
    
    # Tạo danh sách nhãn cho train và validation
    train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
    val_labels = [full_dataset.labels[i] for i in val_dataset.indices]
    
    # Tạo balanced sampler
    train_sampler = BalancedBatchSampler(
        labels=train_labels,
        n_classes=2,  # positive (1) và negative (0)
        n_samples=samples_per_class
    )
    
    val_sampler = BalancedBatchSampler(
        labels=val_labels,
        n_classes=2,
        n_samples=samples_per_class
    )
    
    # Tạo dataloader
    train_loader = DataLoader(
    train_dataset,
    batch_sampler=train_sampler,
    num_workers=4,
    pin_memory=True
)

# Validation DataLoader
    val_loader = DataLoader(
    val_dataset,
    batch_sampler=val_sampler,
    num_workers=4,
    pin_memory=True
)
    return train_loader, val_loader