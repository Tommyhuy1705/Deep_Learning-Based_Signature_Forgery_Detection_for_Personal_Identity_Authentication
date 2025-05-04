import torch.nn as nn
import torchvision.models as models
import torch

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, backbone_name='resnet18', output_dim=512, pretrained=True):
        super().__init__()

        assert backbone_name in ['resnet18', 'resnet34'], "Chỉ hỗ trợ resnet18 và resnet34"

        # Load ResNet
        if backbone_name == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
        else:  # resnet34
            resnet = models.resnet34(pretrained=pretrained)

        # Xóa lớp Fully Connected do chỉ cần feature vector
        resnet.fc = nn.Identity()

        self.backbone = resnet
        if output_dim != 512:
            self.fc = nn.Linear(512, output_dim)  # vì resnet18 và resnet34 đều out 512 chỉnh nếu mong muốn đầu ra khác

    def forward(self, x):
        x = self.backbone(x)  # Output shape: (Batch, 512)
        return x
