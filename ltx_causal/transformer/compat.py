"""
Weight-compatible module implementations from ltx_core.

These are exact re-implementations of modules from ltx_core to ensure
state_dict key compatibility when loading original LTX-2 checkpoints.

The module hierarchy and naming must match exactly to produce identical
state_dict keys.

Source of truth:
    ltx_core/model/transformer/gelu_approx.py     → GELUApprox
    ltx_core/model/transformer/timestep_embedding.py → Timesteps, TimestepEmbedding, PixArtAlphaCombinedTimestepSizeEmbeddings
    ltx_core/model/transformer/adaln.py            → AdaLayerNormSingle
    ltx_core/model/transformer/text_projection.py  → PixArtAlphaTextProjection
    ltx_core/model/transformer/feed_forward.py     → FeedForward
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ============================================================================
# GELUApprox (from gelu_approx.py)
# ============================================================================

class GELUApprox(nn.Module):
    """GELU activation with linear projection.

    Creates state_dict key: `proj.weight`, `proj.bias`
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(self.proj(x), approximate="tanh")


# ============================================================================
# FeedForward (from feed_forward.py)
# ============================================================================

class FeedForward(nn.Module):
    """Feed-forward network using GELUApprox for weight compatibility.

    State dict keys:
        net.0.proj.weight  (GELUApprox linear)
        net.0.proj.bias
        net.2.weight       (output linear)
        net.2.bias

    This matches the original ltx_core FeedForward exactly.
    """

    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Identity(),
            nn.Linear(inner_dim, dim_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# Timestep Embedding (from timestep_embedding.py)
# ============================================================================

def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    """Sinusoidal timestep embeddings (DDPM-style)."""
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(
        start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
    )
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    emb = scale * emb

    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Module):
    """Sinusoidal timestep projection."""

    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        scale: int = 1,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Module):
    """MLP timestep embedding.

    State dict keys: linear_1.weight, linear_1.bias, linear_2.weight, linear_2.bias
    """

    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim: Optional[int] = None,
        sample_proj_bias: bool = True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = nn.SiLU()
        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim

        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None

    def forward(
        self, sample: torch.Tensor, condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """Combined timestep + size embeddings for PixArt-Alpha.

    State dict keys:
        time_proj.*          (Timesteps - no learnable params)
        timestep_embedder.*  (TimestepEmbedding with linear_1, linear_2)
    """

    def __init__(
        self,
        embedding_dim: int,
        size_emb_dim: int,
    ):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=embedding_dim,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype,
    ) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(
            timesteps_proj.to(dtype=hidden_dtype)
        )
        return timesteps_emb


# ============================================================================
# AdaLayerNormSingle (from adaln.py)
# ============================================================================

class AdaLayerNormSingle(nn.Module):
    """Adaptive Layer Norm Single (adaLN-single) from PixArt-Alpha.

    State dict keys:
        emb.time_proj.*              (Timesteps)
        emb.timestep_embedder.*      (TimestepEmbedding)
        linear.weight                (output projection)
        linear.bias

    Args:
        embedding_dim: Hidden dimension
        embedding_coefficient: Number of output scale/shift/gate values.
            Default 6 for per-block (shift/scale/gate for MSA + shift/scale/gate for MLP).
            Use 4 for cross-attention scale/shift, 1 for cross-attention gate.
    """

    def __init__(self, embedding_dim: int, embedding_coefficient: int = 6):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
        )

        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            embedding_dim,
            embedding_coefficient * embedding_dim,
            bias=True,
        )

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            timestep: Flattened timestep values [N]
            hidden_dtype: Target dtype

        Returns:
            (scaled_timestep, embedded_timestep):
                scaled_timestep: [N, coefficient * embedding_dim]
                embedded_timestep: [N, embedding_dim]
        """
        embedded_timestep = self.emb(timestep, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


# ============================================================================
# PixArtAlphaTextProjection (from text_projection.py)
# ============================================================================

class PixArtAlphaTextProjection(nn.Module):
    """Text caption projection for PixArt-Alpha.

    State dict keys:
        linear_1.weight, linear_1.bias
        linear_2.weight, linear_2.bias

    Args:
        in_features: Input caption dimension (3840 for Gemma)
        hidden_size: Hidden/output dimension
        out_features: Output dimension (defaults to hidden_size)
        act_fn: Activation function type
    """

    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        out_features: Optional[int] = None,
        act_fn: str = "gelu_tanh",
    ):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(
            in_features=in_features, out_features=hidden_size, bias=True
        )
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(
            in_features=hidden_size, out_features=out_features, bias=True
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
