from torch.nn import functional as F
import torch

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return torch.mean(losses)
    
    #Cosine Distance
    #cos_distance(x, y) = 1 − x * y / ∥x∥ * ∥y∥


    def forward(self, anchor, positive, negative):
        # Normalize vectors
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)

        # Cosine distance = 1 - cosine similarity
        dist_ap = 1 - torch.sum(anchor * positive, dim=1)
        dist_an = 1 - torch.sum(anchor * negative, dim=1)

        losses = F.relu(dist_ap - dist_an + self.margin)
        return torch.mean(losses)
    
    # Manhattan Distance
    #L1(x,y)=∑ ∣xi −yi∣
    def forward(self, anchor, positive, negative):
        dist_ap = torch.sum(torch.abs(anchor - positive), dim=1)
        dist_an = torch.sum(torch.abs(anchor - negative), dim=1)
        losses = F.relu(dist_ap - dist_an + self.margin)
        return torch.mean(losses)
    
#Learnable Distance --> Use Distance Net --> Connect Triplet Loss với Distance Net

class DistanceNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super(DistanceNet, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, x1, x2):
        # Nối 2 vector đầu vào
        x = torch.cat([x1, x2], dim=1)
        return self.model(x).squeeze(1)  # đầu ra là khoảng cách (scalar)
    
class TripletLossLearnable(torch.nn.Module):
    def __init__(self, distance_net, margin=1.0):
        super(TripletLossLearnable, self).__init__()
        self.distance_net = distance_net
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = self.distance_net(anchor, positive)  # distance anchor-positive
        d_an = self.distance_net(anchor, negative)  # distance anchor-negative

        losses = F.relu(d_ap - d_an + self.margin)
        return torch.mean(losses)