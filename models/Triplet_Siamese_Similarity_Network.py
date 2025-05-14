import torch
import torch.nn as nn
from models.feature_extractor import ResNetFeatureExtractor


class tSSN(nn.Module):
    def __init__(self,backbone_name, output_dim):
        super(tSSN, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(backbone_name = backbone_name,output_dim = output_dim)

    def forward(self, anchor, positive, negative):
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)
        return anchor_feat, positive_feat, negative_feat
