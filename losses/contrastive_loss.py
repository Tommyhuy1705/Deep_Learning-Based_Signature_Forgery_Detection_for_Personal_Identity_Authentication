
import torch

def contrastiveLoss(y_true, dist, margin=1.0):
    # Loss cho cặp giống nhau: y_true = 0 -> (dist)^2
    positive_pair_loss = (1 - y_true) * torch.pow(dist, 2)

    # Loss cho cặp không giống nhau: y_true = 1 -> max(0, margin - dist)^2
    negative_pair_loss = y_true * torch.pow(torch.clamp(margin - dist, min=0.0), 2)

    # Tính tổng Loss và trung bình
    loss = torch.mean(positive_pair_loss + negative_pair_loss)
    return loss
