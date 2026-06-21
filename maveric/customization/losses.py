"""
Loss functions for model customization.

This module contains loss functions used during CLIP model fine-tuning,
including contrastive learning losses for caption-based training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

    This loss is used in caption-based training mode where each image
    is paired with its caption. The loss encourages:
    - High similarity between matching (image, caption) pairs
    - Low similarity between non-matching pairs in the batch

    This is the same loss function used in CLIP's original training.

    Args:
        temperature: Temperature parameter for scaling logits (default: 0.07)

    Shape:
        - image_embeds: (batch_size, embedding_dim) - Normalized image embeddings
        - text_embeds: (batch_size, embedding_dim) - Normalized text embeddings
        - Output: Scalar loss value

    Example:
        >>> loss_fn = InfoNCELoss(temperature=0.07)
        >>> image_embeds = F.normalize(torch.randn(32, 512), dim=-1)
        >>> text_embeds = F.normalize(torch.randn(32, 512), dim=-1)
        >>> loss = loss_fn(image_embeds, text_embeds)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE loss.

        Args:
            image_embeds: Normalized image embeddings (batch_size, embedding_dim)
            text_embeds: Normalized text embeddings (batch_size, embedding_dim)

        Returns:
            Scalar loss value (averaged over batch)
        """
        batch_size = image_embeds.shape[0]

        # Compute similarity matrix: (batch_size, batch_size)
        # logits[i, j] = similarity between image[i] and text[j]
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature

        # Create labels: diagonal elements are positive pairs
        # labels[i] = i means image[i] matches with text[i]
        labels = torch.arange(batch_size, device=image_embeds.device)

        # Compute cross-entropy loss in both directions
        # Image-to-text: For each image, find its matching text
        loss_i2t = F.cross_entropy(logits, labels)

        # Text-to-image: For each text, find its matching image
        loss_t2i = F.cross_entropy(logits.T, labels)

        # Average both directions (symmetric loss)
        loss = (loss_i2t + loss_t2i) / 2.0

        return loss
