"""
Condition Encoder Module for Simplified Anomaly Synthesis Pipeline

This module implements a simple visual+text concat approach:
- I_ref → CLIP Visual Encoder → Linear → P_a (visual condition)
- prompt → CLIP Text Encoder → Linear → P_b (text condition)
- P_c = concat(P_a, P_b) → U-Net cross attention condition

No RFA (Region-Focused Attention), no IP-Adapter, no I_mask dependency.
Similar to anomalydiffusion-master's cross attention approach.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from PIL import Image


class VisualProjection(nn.Module):
    """
    Projects CLIP visual embeddings to U-Net cross attention dimension.

    Input: CLIP image_embeds [batch, hidden_size] (e.g., [batch, 1024] for ViT-H/14)
    Output: P_a [batch, num_visual_tokens, cross_attention_dim]
    """

    def __init__(
        self,
        clip_embed_dim: int = 1024,  # CLIP projection_dim (ViT-H/14: 1024)
        cross_attention_dim: int = 768,  # SD 1.5 cross attention dim
        num_visual_tokens: int = 4,  # Number of visual tokens to generate
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.cross_attention_dim = cross_attention_dim

        # Project CLIP embedding to multiple visual tokens
        self.proj = nn.Linear(clip_embed_dim, num_visual_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, clip_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip_embeds: [batch, clip_embed_dim]

        Returns:
            P_a: [batch, num_visual_tokens, cross_attention_dim]
        """
        batch_size = clip_embeds.shape[0]

        # Project and reshape
        x = self.proj(clip_embeds)  # [batch, num_visual_tokens * cross_attention_dim]
        x = x.view(batch_size, self.num_visual_tokens, self.cross_attention_dim)
        x = self.norm(x)

        return x


class TextProjection(nn.Module):
    """
    Projects CLIP text embeddings to U-Net cross attention dimension.

    Input: CLIP text hidden states [batch, seq_len, hidden_size]
    Output: P_b [batch, seq_len, cross_attention_dim]

    Note: For SD 1.5, text encoder output is already 768-dim, so this is optional.
    If input_dim == output_dim, this acts as identity with optional layer norm.
    """

    def __init__(
        self,
        text_embed_dim: int = 768,  # CLIP text encoder hidden size
        cross_attention_dim: int = 768,  # SD 1.5 cross attention dim
        use_projection: bool = True,  # Whether to use linear projection
    ):
        super().__init__()
        self.use_projection = use_projection

        if use_projection and text_embed_dim != cross_attention_dim:
            self.proj = nn.Linear(text_embed_dim, cross_attention_dim)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeds: [batch, seq_len, text_embed_dim]

        Returns:
            P_b: [batch, seq_len, cross_attention_dim]
        """
        x = self.proj(text_embeds)
        x = self.norm(x)
        return x


class ConditionEncoder(nn.Module):
    """
    Main condition encoder that combines visual and text conditions.

    Pipeline:
    1. I_ref → CLIP Visual Encoder → VisualProjection → P_a [batch, N_v, dim]
    2. prompt → CLIP Text Encoder → TextProjection → P_b [batch, N_t, dim]
    3. P_c = concat(P_a, P_b, dim=1) → [batch, N_v + N_t, dim]

    P_c is used directly as encoder_hidden_states for U-Net cross attention.
    No split, same W_k and W_v are used for all tokens (like anomalydiffusion).
    """

    def __init__(
        self,
        clip_visual_dim: int = 1024,  # CLIP ViT-H/14 projection_dim
        clip_text_dim: int = 768,  # CLIP text encoder hidden size
        cross_attention_dim: int = 768,  # SD 1.5 cross attention dim
        num_visual_tokens: int = 4,  # Number of visual tokens
    ):
        super().__init__()

        self.visual_proj = VisualProjection(
            clip_embed_dim=clip_visual_dim,
            cross_attention_dim=cross_attention_dim,
            num_visual_tokens=num_visual_tokens,
        )

        self.text_proj = TextProjection(
            text_embed_dim=clip_text_dim,
            cross_attention_dim=cross_attention_dim,
            use_projection=(clip_text_dim != cross_attention_dim),
        )

        self.num_visual_tokens = num_visual_tokens
        self.cross_attention_dim = cross_attention_dim

    def forward(
        self,
        clip_image_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Combine visual and text conditions into a single condition tensor.

        Args:
            clip_image_embeds: [batch, clip_visual_dim] - CLIP image embeddings
            text_embeds: [batch, seq_len, clip_text_dim] - CLIP text encoder output

        Returns:
            P_c: [batch, num_visual_tokens + seq_len, cross_attention_dim]
        """
        # Visual projection: [batch, clip_visual_dim] -> [batch, N_v, dim]
        P_a = self.visual_proj(clip_image_embeds)

        # Text projection: [batch, seq_len, clip_text_dim] -> [batch, seq_len, dim]
        P_b = self.text_proj(text_embeds)

        # Concatenate along sequence dimension
        # P_c: [batch, N_v + seq_len, dim]
        P_c = torch.cat([P_a, P_b], dim=1)

        return P_c

    def get_uncond_embeds(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Generate unconditional embeddings (zeros) for classifier-free guidance.

        Args:
            batch_size: Batch size
            seq_len: Text sequence length (typically 77)
            device: Target device
            dtype: Target dtype

        Returns:
            uncond_P_c: [batch, num_visual_tokens + seq_len, cross_attention_dim]
        """
        total_len = self.num_visual_tokens + seq_len
        return torch.zeros(
            batch_size, total_len, self.cross_attention_dim,
            device=device, dtype=dtype
        )


def create_condition_encoder(
    image_encoder_config,
    text_encoder_config,
    unet_config,
    num_visual_tokens: int = 4,
) -> ConditionEncoder:
    """
    Factory function to create ConditionEncoder from model configs.

    Args:
        image_encoder_config: CLIP image encoder config (has projection_dim)
        text_encoder_config: CLIP text encoder config (has hidden_size)
        unet_config: U-Net config (has cross_attention_dim)
        num_visual_tokens: Number of visual tokens to generate

    Returns:
        ConditionEncoder instance
    """
    return ConditionEncoder(
        clip_visual_dim=image_encoder_config.projection_dim,
        clip_text_dim=text_encoder_config.hidden_size,
        cross_attention_dim=unet_config.cross_attention_dim,
        num_visual_tokens=num_visual_tokens,
    )
