import torch
import torch.nn as nn
from models.feature_extractor import ResNetFeatureExtractor

class tSSN(nn.Module):
    """
    A basic Triplet Siamese Similarity Network structure.

    This class wraps the feature extractor and provides a convenient forward
    method that takes anchor, positive, and negative inputs simultaneously,
    primarily used during the pre-training phase with standard Triplet Loss.

    In the meta-learning phase, the feature_extractor is used directly,
    and the metric generation is handled separately by the MetricGenerator.
    """
    def __init__(self, backbone_name='resnet34', output_dim=512, pretrained=True):
        """
        Initializes the tSSN model.

        Args:
            backbone_name (str): The name of the ResNet backbone for the feature extractor.
                                 Defaults to 'resnet34'.
            output_dim (int): The output dimension of the feature extractor. Defaults to 512.
            pretrained (bool): Whether the feature extractor should use pre-trained weights.
                               Defaults to True.
        """
        super(tSSN, self).__init__()
        # Instantiate the shared feature extractor
        self.feature_extractor = ResNetFeatureExtractor(
            backbone_name=backbone_name,
            output_dim=output_dim,
            pretrained=pretrained
        )

    def forward(self, anchor, positive, negative):
        """
        Processes a triplet of images through the shared feature extractor.

        Args:
            anchor (torch.Tensor): Batch of anchor images.
            positive (torch.Tensor): Batch of positive images.
            negative (torch.Tensor): Batch of negative images.

        Returns:
            tuple: A tuple containing the feature embeddings for anchor, positive,
                   and negative images, respectively.
        """
        # Apply the *same* feature extractor to all three inputs (Siamese architecture)
        anchor_feat = self.feature_extractor(anchor)
        positive_feat = self.feature_extractor(positive)
        negative_feat = self.feature_extractor(negative)

        return anchor_feat, positive_feat, negative_feat