import torch
import torch.nn.functional as F

def euclidean(output1, output2):
    return F.pairwise_distance(output1, output2, keepdim= True)

# def cosin_distance(output1, output2):
#     return F

def contrastiveLoss(output1, output2, label, margin):
    label = label.float()
    label = label.type_as(output1)

    distance = euclidean(output1, output2)

    #Contrastive loss
    loss = torch.mean(
        label * torch.pow(distance, 2) +
        (1 - label) * torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    )
    return loss