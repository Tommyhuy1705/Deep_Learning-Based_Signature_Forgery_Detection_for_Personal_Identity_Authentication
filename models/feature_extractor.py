import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    """
    A feature extractor module based on a pre-trained ResNet model (ResNet-34).

    This module loads a specified ResNet backbone (defaulting to ResNet-34),
    removes its final classification layer (fc), and outputs the feature vector
    from the penultimate layer.
    """
    def __init__(self, backbone_name='resnet34', output_dim=512, pretrained=True):
        """
        Initializes the ResNetFeatureExtractor.

        Args:
            backbone_name (str): The name of the ResNet backbone to use.
                                 Currently supports 'resnet18' and 'resnet34'. Defaults to 'resnet34'.
            output_dim (int): The desired dimension of the output feature vector.
                              If different from the backbone's natural output (512 for ResNet18/34),
                              a linear layer will be added. Defaults to 512.
            pretrained (bool): Whether to load pre-trained ImageNet weights. Defaults to True.
        """
        super().__init__()

        # Ensure supported backbone is selected
        if backbone_name not in ['resnet18', 'resnet34']:
             raise ValueError("Unsupported backbone_name. Choose 'resnet18' or 'resnet34'.")

        # Load the specified ResNet model
        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet18(weights=weights)
            original_dim = 512
        else: # resnet34
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet34(weights=weights)
            original_dim = 512

        # Remove the final fully connected layer (classification layer)
        # We only need the features before classification.
        resnet.fc = nn.Identity()

        self.backbone = resnet
        self.output_dim = output_dim

        # Add an optional linear layer if the desired output dimension
        # differs from the backbone's original feature dimension.
        if self.output_dim != original_dim:
            self.fc = nn.Linear(original_dim, self.output_dim)
            print(f"Added final linear layer to map features from {original_dim} to {self.output_dim} dimensions.")
        else:
            # If dimensions match, use Identity to avoid unnecessary layer
            self.fc = nn.Identity()


    def forward(self, x):
        """
        Performs the forward pass to extract features.

        Args:
            x (torch.Tensor): Input image tensor. Shape: (batch_size, 3, H, W).

        Returns:
            torch.Tensor: Output feature vector. Shape: (batch_size, output_dim).
        """
        # Pass input through the ResNet backbone (excluding the original fc layer)
        features = self.backbone(x) # Shape: (batch_size, original_dim)

        # Apply the final linear layer (or Identity if dimensions match)
        output_features = self.fc(features) # Shape: (batch_size, output_dim)

        return output_features