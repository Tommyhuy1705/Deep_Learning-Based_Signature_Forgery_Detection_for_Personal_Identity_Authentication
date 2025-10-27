import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Standard Triplet Loss (for Pre-training)
# =============================================================================

class TripletLoss(nn.Module):
    """
    Implements the standard Triplet Loss function with fixed distance metrics.

    This loss encourages the distance between an anchor and a positive sample
    to be smaller than the distance between the anchor and a negative sample
    by at least a margin.

    L(A, P, N) = max( d(A, P) - d(A, N) + margin, 0 )
    """
    def __init__(self, margin=1.0, mode='euclidean'):
        """
        Initializes the TripletLoss.

        Args:
            margin (float): The margin value. Defaults to 1.0.
            mode (str): The distance metric to use. Options: 'euclidean', 'cosine', 'manhattan'.
                        Defaults to 'euclidean'.
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.mode = mode.lower()

        if self.mode not in ['euclidean', 'cosine', 'manhattan']:
            raise ValueError(f"Unsupported distance mode: {mode}. Choose from 'euclidean', 'cosine', 'manhattan'.")

    def forward(self, anchor, positive, negative):
        """
        Calculates the Triplet Loss for a batch of triplets.

        Args:
            anchor (torch.Tensor): Embeddings for anchor samples. Shape: (batch_size, embedding_dim).
            positive (torch.Tensor): Embeddings for positive samples. Shape: (batch_size, embedding_dim).
            negative (torch.Tensor): Embeddings for negative samples. Shape: (batch_size, embedding_dim).

        Returns:
            torch.Tensor: The mean Triplet Loss for the batch.
        """
        if self.mode == 'euclidean':
            distance_positive = F.pairwise_distance(anchor, positive, p=2)
            distance_negative = F.pairwise_distance(anchor, negative, p=2)
        elif self.mode == 'cosine':
            # Cosine similarity -> Cosine distance = 1 - similarity
            distance_positive = 1.0 - F.cosine_similarity(anchor, positive, dim=1)
            distance_negative = 1.0 - F.cosine_similarity(anchor, negative, dim=1)
        elif self.mode == 'manhattan':
            distance_positive = F.pairwise_distance(anchor, positive, p=1) # L1 distance
            distance_negative = F.pairwise_distance(anchor, negative, p=1)

        # Calculate the triplet loss: max(d(a,p) - d(a,n) + margin, 0)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        # Return the mean loss over the batch
        return torch.mean(losses)

# =============================================================================
# Mahalanobis Distance (for Meta-Learning)
# =============================================================================

def pairwise_mahalanobis_distance(x1, x2, W):
    """
    Computes the pairwise Mahalanobis distance matrix between two sets of embeddings.

    The Mahalanobis distance is defined as d(x, y)^2 = (x - y)^T * W * (x - y),
    where W is a positive semi-definite matrix learned by the meta-learner.

    Args:
        x1 (torch.Tensor): First set of embeddings. Shape: (N, embedding_dim).
        x2 (torch.Tensor): Second set of embeddings. Shape: (M, embedding_dim).
        W (torch.Tensor): The learned Mahalanobis matrix. Shape: (embedding_dim, embedding_dim).
                           It should be symmetric and positive semi-definite.

    Returns:
        torch.Tensor: The pairwise Mahalanobis distance matrix. Shape: (N, M).
                      Note: Returns the squared distance, consistent with the definition.
                      If non-squared distance is needed, take sqrt after calling this.
                      However, for triplet loss comparisons, squared distance is sufficient and often preferred.
    """
    # Ensure W is on the same device as the embeddings
    W = W.to(x1.device)

    # Calculate the difference matrix: diff[i, j, k] = x1[i, k] - x2[j, k]
    # x1: (N, D) -> unsqueeze(1) -> (N, 1, D)
    # x2: (M, D) -> unsqueeze(0) -> (1, M, D)
    # diff: (N, M, D)
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)

    # Efficiently compute (x1_i - x2_j)^T * W * (x1_i - x2_j) for all pairs (i, j)
    # using Einstein summation convention.
    # 'ijk,kl,ijl->ij' means:
    #   ijk: diff tensor (N, M, D)
    #   kl: W matrix (D, D)
    #   ijl: diff tensor (N, M, D) - implicitly transposed in the operation
    #   -> ij: Resulting distance matrix (N, M)
    try:
        dist_mat = torch.einsum('ijk,kl,ijl->ij', diff, W, diff)
    except RuntimeError as e:
        print(f"Error during einsum calculation in pairwise_mahalanobis_distance: {e}")
        # Add fallback or re-raise depending on desired robustness
        # Fallback to Euclidean might hide issues with W matrix.
        # Let's re-raise for now to make potential issues visible.
        raise e

    # Add a small epsilon for numerical stability if needed, especially before sqrt
    # dist_mat = dist_mat + 1e-6

    return dist_mat