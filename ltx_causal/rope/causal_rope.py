"""
Causal RoPE: Rotary Position Embedding with temporal offset for causal training.
This module implements:
- causal_precompute_freqs_cis: Precompute RoPE frequencies with frame offset
- apply_interleaved_rotary_emb: Apply interleaved rotary embedding
- Supports both video (3D positions) and audio (1D positions)

Positions are in PHYSICAL coordinates (seconds / pixels), matching the
original LTX-2 pipeline.
"""

import math
from enum import Enum
from typing import Tuple, Optional, List, Union

import torch
from einops import rearrange


class CausalRopeType(Enum):
    """RoPE type compatible with LTX-2."""
    INTERLEAVED = "interleaved"
    SPLIT = "split"


# ============================================================================
# LTX-2 Physical Coordinate Constants
# ============================================================================
# Positions must be in PHYSICAL space (seconds / pixels), NOT latent indices.
# Using latent indices causes ~33x spatial error and ~3-5x temporal error,
# leading to gradient norms of 1e+15.

_VIDEO_TEMPORAL_SCALE = 8        # VAE temporal downsampling factor
_VIDEO_SPATIAL_SCALE = 32        # VAE spatial downsampling factor
_VIDEO_FPS = 24                  # Video frames per second
_AUDIO_DOWNSAMPLE_FACTOR = 4     # Audio latent temporal downsample factor
_AUDIO_HOP_LENGTH = 160          # Mel spectrogram hop length
_AUDIO_SAMPLE_RATE = 16000       # Audio mel processing sample rate


# ============================================================================
# Frequency Grid Generation
# ============================================================================

def generate_freq_grid(
    theta: float,
    max_pos_count: int,
    inner_dim: int,
    device: Union[torch.device, str] = "cuda",
) -> torch.Tensor:
    """
    Generate frequency grid for RoPE.

    Compatible with LTX-2's generate_freq_grid_pytorch.

    Args:
        theta: Base frequency (typically 10000)
        max_pos_count: Number of position dimensions
        inner_dim: Inner dimension of attention
        device: Target device

    Returns:
        Frequency indices tensor
    """
    start = 1
    end = theta
    n_elem = 2 * max_pos_count

    indices = theta ** (
        torch.linspace(
            math.log(start, theta),
            math.log(end, theta),
            inner_dim // n_elem,
            dtype=torch.float32,
            device=device,
        )
    )
    indices = indices * math.pi / 2

    return indices


# ============================================================================
# Standard RoPE Application
# ============================================================================

def apply_interleaved_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply interleaved rotary embedding (pairs adjacent dimensions)."""
    t_dup = rearrange(input_tensor, "... (d r) -> ... d r", r=2)
    t1, t2 = t_dup.unbind(dim=-1)
    t_dup = torch.stack((-t2, t1), dim=-1)
    input_tensor_rot = rearrange(t_dup, "... d r -> ... (d r)")

    out = input_tensor * cos_freqs + input_tensor_rot * sin_freqs
    return out


def apply_split_rotary_emb(
    input_tensor: torch.Tensor,
    cos_freqs: torch.Tensor,
    sin_freqs: torch.Tensor,
) -> torch.Tensor:
    """Apply split rotary embedding (LLaMA-style, first/second half).

    Handles auto-reshape when input is 3D [B, T, D] but cos is 4D [B, H, T, D//2].
    """
    needs_reshape = False
    if input_tensor.ndim != 4 and cos_freqs.ndim == 4:
        b, h, t, _ = cos_freqs.shape
        input_tensor = input_tensor.reshape(b, t, h, -1).swapaxes(1, 2)
        needs_reshape = True

    split_input = rearrange(input_tensor, "... (d r) -> ... d r", d=2)
    first_half_input = split_input[..., :1, :]
    second_half_input = split_input[..., 1:, :]

    output = split_input * cos_freqs.unsqueeze(-2)
    first_half_output = output[..., :1, :]
    second_half_output = output[..., 1:, :]

    first_half_output.addcmul_(-sin_freqs.unsqueeze(-2), second_half_input)
    second_half_output.addcmul_(sin_freqs.unsqueeze(-2), first_half_input)

    output = rearrange(output, "... d r -> ... (d r)")
    if needs_reshape:
        output = output.swapaxes(1, 2).reshape(b, t, -1)

    return output


# ============================================================================
# Causal RoPE with Frame Offset
# ============================================================================

def causal_precompute_freqs_cis(
    grid_sizes: torch.Tensor,
    dim: int,
    theta: float = 10000.0,
    max_pos: Optional[List[int]] = None,
    start_frame: int = 0,
    rope_type: CausalRopeType = CausalRopeType.SPLIT,
    device: Union[torch.device, str] = "cuda",
    dtype: torch.dtype = torch.float32,
    is_audio: bool = False,
    rope_extrapolation: str = "off",
    rope_train_max_seconds: float = 8.0,
    num_attention_heads: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute RoPE frequencies with causal frame offset.

    CRITICAL: Positions are in PHYSICAL coordinates (seconds / pixels), matching
    the original LTX-2 pipeline (ltx-core). Using bare latent indices causes
    gradient norm explosion (~1e15).

    For video 3D:
        - Temporal: pixel frames with causal fix → seconds (/ fps=24)
        - Spatial: pixel midpoints (latent * 32 + 16)
    For audio 1D (is_audio=True):
        - Temporal: mel frames with causal fix → seconds (* hop/sr)
    For video temporal 1D (is_audio=False):
        - Same as video temporal axis

    Long-horizon extrapolation:
        When ``rope_extrapolation == "ntk"`` and the maximum temporal position
        of this call exceeds ``rope_train_max_seconds``, the RoPE base
        frequency ``theta`` is scaled following the NTK-aware formula
        (bloc97): ``theta_eff = theta * r ** (dim / (dim - 2))`` with
        ``r = max(1, t_max_seconds / rope_train_max_seconds)``.  This
        preserves low-frequency components while stretching high frequencies
        so that previously unseen absolute positions land inside the trained
        frequency band.

    Args:
        grid_sizes: Position grid sizes [B, num_dims] or [B, num_dims, 2]
                   For video: [B, 3] with (F, H, W)
                   For audio: [B, 1] with (F,)
        dim: Inner dimension for RoPE
        theta: Base frequency
        max_pos: Maximum positions per dimension (default [20, 2048, 2048])
        start_frame: Starting frame offset for causal generation
        rope_type: INTERLEAVED or SPLIT
        device: Target device
        dtype: Output dtype
        is_audio: If True and num_pos_dims==1, use audio timing conversion.
                  If False and num_pos_dims==1, use video temporal conversion.
        rope_extrapolation: "off" (default) | "ntk".
        rope_train_max_seconds: Longest clip length (in seconds) seen during
                  training; only consulted when ``rope_extrapolation == "ntk"``.

    Returns:
        (cos_freq, sin_freq): Tuple of frequency tensors
    """
    if max_pos is None:
        max_pos = [20, 2048, 2048]

    # Handle 2D grid_sizes (start/end format) by taking mean
    if grid_sizes.ndim == 3:
        grid_sizes = (grid_sizes[..., 0] + grid_sizes[..., 1]) / 2.0

    num_pos_dims = grid_sizes.shape[1]

    # ---------------- NTK-aware theta scaling ----------------
    # The temporal axis of ``grid_sizes[:, 0]`` is F (latent frames). Convert
    # the farthest absolute latent frame visited by this call into seconds,
    # using the same physical-coordinate formula as the per-branch code
    # below.  We take a slight over-estimate (``+ 1`` implicit via end frame
    # convention) — safe, because NTK is monotone in ``r``.
    theta_eff = float(theta)
    if rope_extrapolation == "ntk":
        try:
            max_latent_f_local = int(grid_sizes[:, 0].max().item())
        except RuntimeError:
            max_latent_f_local = 0
        max_latent_abs = int(start_frame) + max_latent_f_local
        if max_latent_abs > 0:
            if is_audio:
                mel_end = max(
                    0.0,
                    max_latent_abs * _AUDIO_DOWNSAMPLE_FACTOR
                    + 1 - _AUDIO_DOWNSAMPLE_FACTOR,
                )
                t_max_seconds = mel_end * _AUDIO_HOP_LENGTH / _AUDIO_SAMPLE_RATE
            else:
                pixel_end = max(
                    0.0,
                    max_latent_abs * _VIDEO_TEMPORAL_SCALE
                    + 1 - _VIDEO_TEMPORAL_SCALE,
                )
                t_max_seconds = pixel_end / _VIDEO_FPS
            train_max = max(float(rope_train_max_seconds), 1e-6)
            ratio = max(1.0, t_max_seconds / train_max)
            if ratio > 1.0:
                # bloc97's NTK-aware scaling: preserves low-freq components
                # exactly, stretches high-freq so that positions beyond the
                # train range still fall inside the trained frequency band.
                exponent = float(dim) / max(float(dim - 2), 1.0)
                theta_eff = float(theta) * (ratio ** exponent)
    elif rope_extrapolation not in ("off", "ntk"):
        raise ValueError(
            f"rope_extrapolation must be 'off' or 'ntk', got {rope_extrapolation!r}"
        )

    # Generate frequency indices using the (possibly scaled) theta
    indices = generate_freq_grid(theta_eff, num_pos_dims, dim, device)

    # Build position indices with frame offset
    all_freqs = []

    for batch_idx in range(grid_sizes.shape[0]):
        sizes = grid_sizes[batch_idx].tolist()
        seq_len = int(math.prod(sizes))

        if num_pos_dims == 3:
            # Video: 3D positions (F, H, W) in PHYSICAL coordinates
            f, h, w = [int(s) for s in sizes]

            # Temporal: latent frame → pixel bounds → midpoint in seconds
            # Matches get_pixel_coords() + causal_fix + /fps in ltx-core
            latent_t = torch.arange(start_frame, start_frame + f, device=device, dtype=torch.float32)
            pixel_t_start = (latent_t * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
            pixel_t_end = ((latent_t + 1) * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
            t_seconds = (pixel_t_start + pixel_t_end) / 2.0 / _VIDEO_FPS

            # Spatial: latent coord → pixel bounds → midpoint
            # Matches get_pixel_coords() (no causal fix for spatial)
            h_pixels = torch.arange(h, device=device, dtype=torch.float32) * _VIDEO_SPATIAL_SCALE + _VIDEO_SPATIAL_SCALE / 2.0
            w_pixels = torch.arange(w, device=device, dtype=torch.float32) * _VIDEO_SPATIAL_SCALE + _VIDEO_SPATIAL_SCALE / 2.0

            # Normalize by max_pos (matching get_fractional_positions)
            t_frac = t_seconds / max_pos[0]
            h_frac = h_pixels / max_pos[1]
            w_frac = w_pixels / max_pos[2]

            # Build meshgrid: [F, H, W, 3]
            grid_t, grid_h, grid_w = torch.meshgrid(t_frac, h_frac, w_frac, indexing='ij')
            fractional_positions = torch.stack([grid_t, grid_h, grid_w], dim=-1)
            fractional_positions = fractional_positions.reshape(seq_len, num_pos_dims)

            # Compute frequencies matching original generate_freqs:
            #   freqs = (indices * (frac * 2 - 1)).transpose(-1, -2).flatten(2)
            # This interleaves: [freq0_t, freq0_h, freq0_w, freq1_t, freq1_h, freq1_w, ...]
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
            freqs = freqs.transpose(-1, -2).flatten(1)

        elif num_pos_dims == 1:
            # 1D temporal positions
            f = int(sizes[0])
            latent_t = torch.arange(start_frame, start_frame + f, device=device, dtype=torch.float32)

            if is_audio:
                # Audio: latent frame → mel frame with causal fix → seconds
                # Matches AudioPatchifier._get_audio_latent_time_in_sec()
                mel_start = (latent_t * _AUDIO_DOWNSAMPLE_FACTOR + 1 - _AUDIO_DOWNSAMPLE_FACTOR).clamp(min=0)
                mel_end = ((latent_t + 1) * _AUDIO_DOWNSAMPLE_FACTOR + 1 - _AUDIO_DOWNSAMPLE_FACTOR).clamp(min=0)
                t_seconds = (mel_start + mel_end) / 2.0 * _AUDIO_HOP_LENGTH / _AUDIO_SAMPLE_RATE
            else:
                # Video temporal: latent frame → pixel frame with causal fix → seconds
                # Same conversion as the temporal axis of the 3D video case
                pixel_t_start = (latent_t * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
                pixel_t_end = ((latent_t + 1) * _VIDEO_TEMPORAL_SCALE + 1 - _VIDEO_TEMPORAL_SCALE).clamp(min=0)
                t_seconds = (pixel_t_start + pixel_t_end) / 2.0 / _VIDEO_FPS

            t_frac = t_seconds / max_pos[0]

            # For 1D: fractional_positions = [seq_len, 1]
            fractional_positions = t_frac.unsqueeze(-1)
            freqs = (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
            freqs = freqs.transpose(-1, -2).flatten(1)

        else:
            raise ValueError(f"Unsupported num_pos_dims: {num_pos_dims}")

        all_freqs.append(freqs)

    # Stack batches
    freqs = torch.stack(all_freqs, dim=0)  # [B, seq_len, D_freq * num_pos_dims]

    if rope_type == CausalRopeType.SPLIT:
        # SPLIT mode: output (B, H, T, D//2) matching LTX-core split_freqs_cis
        expected_freqs = dim // 2
        current_freqs = freqs.shape[-1]
        pad_size = expected_freqs - current_freqs

        cos_freq = freqs.cos()
        sin_freq = freqs.sin()

        if pad_size > 0:
            cos_padding = torch.ones_like(cos_freq[..., :pad_size])
            sin_padding = torch.zeros_like(sin_freq[..., :pad_size])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

        # Reshape: [B, T, D//2] → [B, T, H, D//2//H] → [B, H, T, D//2//H]
        b, t = cos_freq.shape[0], cos_freq.shape[1]
        cos_freq = cos_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)
        sin_freq = sin_freq.reshape(b, t, num_attention_heads, -1).swapaxes(1, 2)

    else:
        # INTERLEAVED mode: output (B, T, dim) with repeat_interleave
        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)

        # Pad size matches original: dim % (2 * num_pos_dims)
        n_elem = 2 * num_pos_dims
        pad_size = dim % n_elem
        if pad_size != 0:
            cos_padding = torch.ones_like(cos_freq[..., :pad_size])
            sin_padding = torch.zeros_like(sin_freq[..., :pad_size])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

    return cos_freq.to(dtype), sin_freq.to(dtype)
