import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from models.Triplet_Siamese_Similarity_Network import tSSN
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, precision_recall_curve
)

from losses.triplet_loss import DistanceNet

# Hàm tính khoảng cách dựa trên metric được chọn
def calculate_distance(anchor_feat, test_feat, metric='euclidean',device='cuda'):
    distance_net = DistanceNet(input_dim=512).to(device)
    if metric == 'euclidean':
        return F.pairwise_distance(anchor_feat, test_feat)
    elif metric == 'cosine':
        anchor_feat = F.normalize(anchor_feat, p=2, dim=1)
        test_feat = F.normalize(test_feat, p=2, dim=1)
        return 1 - torch.sum(anchor_feat * test_feat, dim=1)
    elif metric == 'manhattan':
        return torch.sum(torch.abs(anchor_feat - test_feat), dim=1)
    elif metric == 'learnable':
        return distance_net(anchor_feat, test_feat)
    else:
        raise ValueError(f"Metric không được hỗ trợ: {metric}")

# Hàm đánh giá mô hình
def evaluate_model(model:tSSN, metric, dataloader:DataLoader, device):
    model.eval()
    distances = []
    labels = []

    with torch.no_grad():
        for (img1, img2), label in tqdm(dataloader, desc=f'Evaluating with {metric}'):
            img1, img2 = img1.to(device), img2.to(device)
            emb1 = model.feature_extractor(img1)
            emb2 = model.feature_extractor(img2)
            dist = calculate_distance(emb1, emb2, metric)
            distances.extend(dist.cpu().numpy())
            labels.extend(label.numpy())

    distances = np.array(distances)
    labels = np.array(labels)

    # Tìm ngưỡng tối ưu bằng ROC curve
    fpr, tpr, thresholds = roc_curve(labels, -distances)  # -distances vì nhỏ hơn nghĩa là giống hơn
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = -thresholds[optimal_idx]  # Chuyển lại dấu

    # Tính các chỉ số hiệu suất
    predictions = (distances <= optimal_threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = auc(fpr, tpr)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': optimal_threshold
    }