import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Adaptive Diagonal Metric Generator (Feature Weighting Network).

    Research Context:
    In few-shot signature verification, learning a full Mahalanobis matrix (W) 
    often leads to overfitting due to the high dimensionality (d=512) relative 
    to the scarcity of support samples (k=8). 
    
    Proposed Solution:
    This module implements a 'Diagonal Metric Learning' approach. Instead of 
    modeling correlations between all feature dimensions, it estimates a 
    feature-wise importance vector (omega). This assumes that the underlying 
    manifold can be locally approximated by re-scaling the axes of the 
    hyperspace.

    Mathematical Formulation:
        Let mu be the prototype of the support set.
        The weight vector is computed as: omega = sigma(f(mu))
        The distance metric becomes: d(x, mu)^2 = sum(omega_i * (x_i - mu_i)^2)

    Attributes:
        embedding_dim (int): Dimensionality of the input feature space (ResNet34 output).
        hidden_dim (int): Dimensionality of the latent representation in the generator.
    """

    def __init__(self, embedding_dim=512, hidden_dim=256, dropout=0.3):
        """
        Initializes the Feature Weighting Network.

        Args:
            embedding_dim (int): Size of the feature vector (default: 512).
            hidden_dim (int): Size of the hidden layer (default: 256).
            dropout (float): Dropout rate for regularization (default: 0.3).
        """
        super(MetricGenerator, self).__init__()
        
        # Multi-Layer Perceptron (MLP) for inferring feature importance
        self.inference_network = nn.Sequential(
            # Projection to latent space
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            # Regularization to prevent co-adaptation of neurons
            nn.Dropout(dropout),
            
            # Projection back to feature space
            nn.Linear(hidden_dim, embedding_dim),
            
            # Sigmoid activation ensures weights are in range (0, 1),
            # representing the relative importance or reliability of each feature.
            nn.Sigmoid() 
        )

    def forward(self, prototype):
        """
        Forward pass to generate the adaptive metric weights.

        Args:
            prototype (Tensor): The centroid of the support set. 
                                Shape: [Batch_Size, embedding_dim]

        Returns:
            feature_weights (Tensor): The estimated importance vector (omega).
                                      Shape: [Batch_Size, embedding_dim]
        """
        # Estimate channel-wise importance
        feature_weights = self.inference_network(prototype)
        return feature_weights