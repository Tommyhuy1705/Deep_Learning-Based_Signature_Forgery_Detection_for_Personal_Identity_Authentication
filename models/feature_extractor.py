import torch.nn as nn
import torchvision.models as models
import torch

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet18', output_dim=512, pretrained=True):
        super().__init__()

        assert backbone_name in ['resnet18', 'resnet34'], "Only resnet18 and resnet34 are supported"

        # Load ResNet
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        else:  # Resnet34
            resnet = models.resnet34(pretrained=pretrained)

        # Delete Fully Connected layer since only feature vector is needed
        resnet.fc = nn.Identity()

        self.backbone = resnet
        if output_dim != 512:
            self.fc = nn.Linear(512, output_dim)  # Because resnet18 and resnet34 both output 512, adjust if different output is desired

    def forward(self, x):
        x = self.backbone(x)  # Output shape: (Batch, 512)
        return x