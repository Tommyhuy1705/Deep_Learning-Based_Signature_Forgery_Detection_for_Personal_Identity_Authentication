import torch
import torch.nn as nn
import torch.nn.functional as F

def tripletLoss(anchor, positive, negative, margin=1.0):
    pos_dist = torch.norm(anchor - positive, p=2, dim=1)  # Euclidean distance: ||a - p||
    neg_dist = torch.norm(anchor - negative, p=2, dim=1)  # Euclidean distance: ||a - n||

    # Công thức Loss: max(0, d(a, p) - d(a, n) + margin)
    loss = F.relu(pos_dist - neg_dist + margin)

    # Trung bình loss trên batch
    return loss.mean()
