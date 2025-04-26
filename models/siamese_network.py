import torch
import torch.nn as nn
from feature_extractor import ResNetFeatureExtractor

class SiameseNetwork(nn.Module):
    def __init__(self,backbone_name, output_dim):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = ResNetFeatureExtractor(backbone_name = backbone_name,output_dim = output_dim)

    def forward(self, img1, img2):
        img1_feat = self.feature_extractor(img1)
        img2_feat = self.feature_extractor(img2)
        return img1_feat, img2_feat
