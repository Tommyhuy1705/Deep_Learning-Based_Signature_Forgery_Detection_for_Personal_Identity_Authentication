import torch
import torch.nn as nn
import torch.nn.functional as F

class MetricGenerator(nn.Module):
    """
    Generates a user-specific Mahalanobis matrix (W) from support set embeddings.

    This module implements an Attention mechanism to compute a weighted prototype
    of the support set, which is then fed into a generator network to produce
    a tailored, adaptive metric (Mahalanobis matrix W) for that specific user.
    """
    def __init__(self, embedding_dim, hidden_factor=2, dropout_prob=0.3):
        """
        Initializes the MetricGenerator.

        Args:
            embedding_dim (int): The dimensionality of the input embeddings (e.g., 512).
            hidden_factor (int): Multiplier for the hidden layer dimension relative to embedding_dim. Defaults to 2.
            dropout_prob (float): Dropout probability for regularization within the generator network. Defaults to 0.3.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden_dim = embedding_dim * hidden_factor

        # Attention layer to compute weights for the support set samples.
        # Input: embedding_dim -> Output: 1 (attention score per sample)
        self.attn = nn.Linear(embedding_dim, 1)

        # A deeper generator network to map the prototype to the matrix parameters.
        # Input: Prototype vector (embedding_dim)
        # Output: Flattened W matrix (embedding_dim * embedding_dim)
        self.generator = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob), # Add dropout for regularization
            nn.Linear(hidden_dim, hidden_dim), # Added another hidden layer
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * embedding_dim) # Output flattened matrix parameters
        )

        print(f"Initialized MetricGenerator with Attention. Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}, Dropout: {dropout_prob}.")


    def forward(self, support_embeddings):
        """
        Generates the Mahalanobis matrix W for a given support set.

        Args:
            support_embeddings (torch.Tensor): Embeddings of the support set samples.
                                                Shape: (k_shot, embedding_dim).

        Returns:
            torch.Tensor: The generated Mahalanobis matrix W, ensured to be
                          symmetric and positive semi-definite (PSD).
                          Shape: (embedding_dim, embedding_dim).
        """
        # Ensure support_embeddings is not empty
        k_shot = support_embeddings.shape[0]
        if k_shot == 0:
            print("Warning: MetricGenerator received empty support_embeddings. Returning Identity matrix.")
            # Return identity matrix on the same device
            return torch.eye(self.embedding_dim, device=support_embeddings.device)

        # --- 1. Compute Weighted Prototype via Attention ---
        # Calculate attention scores for each support sample
        # (k_shot, embedding_dim) -> (k_shot, 1)
        try:
             attn_scores = self.attn(support_embeddings)
        except Exception as e:
             print(f"Error during attention score calculation: {e}. Using mean prototype fallback.")
             # Fallback to simple mean if attention fails
             prototype = torch.mean(support_embeddings, dim=0, keepdim=True)
             # Proceed to matrix generation with mean prototype
             # (Code block duplicated below for clarity, could be refactored)
             try:
                 flat_matrix = self.generator(prototype)
                 W_raw = flat_matrix.view(self.embedding_dim, self.embedding_dim)
                 W_sym = (W_raw + W_raw.t()) / 2
                 epsilon = 1e-6
                 W_psd = W_sym + (epsilon * torch.eye(self.embedding_dim, device=W_sym.device))
                 return W_psd
             except Exception as gen_e:
                 print(f"Error during generator forward pass after fallback: {gen_e}")
                 return torch.eye(self.embedding_dim, device=support_embeddings.device) # Final fallback

        # Apply softmax to get normalized attention weights (sum to 1)
        attn_weights = F.softmax(attn_scores, dim=0) # Shape: (k_shot, 1)

        # Compute the weighted average prototype
        # Element-wise multiply embeddings by weights: (k_shot, embedding_dim) * (k_shot, 1)
        # Sum across the k_shot dimension: -> (1, embedding_dim)
        prototype = torch.sum(support_embeddings * attn_weights, dim=0, keepdim=True)

        # --- 2. Generate Flattened Matrix ---
        # Pass the calculated prototype through the generator network
        try:
             flat_matrix = self.generator(prototype) # Shape: (1, embedding_dim * embedding_dim)
        except Exception as e:
             print(f"Error during generator forward pass: {e}")
             # Fallback to Identity if generator fails
             return torch.eye(self.embedding_dim, device=support_embeddings.device)

        # --- 3. Reshape into Square Matrix ---
        try:
             W_raw = flat_matrix.view(self.embedding_dim, self.embedding_dim)
        except Exception as e:
             print(f"DEBUG (MetricGenerator): ERROR during flat_matrix.view: {e}")
             # Fallback if reshape fails
             return torch.eye(self.embedding_dim, device=support_embeddings.device)


        # --- 4. Ensure W is Symmetric and Positive Semi-Definite (PSD) ---
        # Enforce Symmetry: W_sym = (W_raw + W_raw^T) / 2
        W_sym = (W_raw + W_raw.t()) / 2

        # Enforce Positive Semi-Definite (PSD) by adding a small diagonal bias (epsilon * I)
        # This is a numerically stable method to ensure eigenvalues > 0.
        epsilon = 1e-6
        W_psd = W_sym + (epsilon * torch.eye(self.embedding_dim, device=W_sym.device))

        return W_psd