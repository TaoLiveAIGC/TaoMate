"""
Configuration classes for causal LTX-2 generation.

This module defines configuration dataclasses for:
- CausalMaskConfig: Parameters for mask construction
- CausalGenerationConfig: Full generation configuration
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# ============================================================================
# Time Alignment Constants
# ============================================================================

VIDEO_FPS = 24  # Original video frame rate
AUDIO_SAMPLE_RATE = 16000  # Audio sample rate
MEL_HOP_LENGTH = 160  # Mel spectrogram hop length

VIDEO_VAE_TEMPORAL_COMPRESSION = 8  # VAE temporal compression factor
AUDIO_VAE_TEMPORAL_COMPRESSION = 4  # Audio VAE temporal compression

# Latent frame rates
VIDEO_LATENT_FPS = VIDEO_FPS / VIDEO_VAE_TEMPORAL_COMPRESSION  # 3 fps
AUDIO_LATENT_FPS = AUDIO_SAMPLE_RATE / MEL_HOP_LENGTH / AUDIO_VAE_TEMPORAL_COMPRESSION  # 25 fps

# Alignment ratio: audio frames per video frame
ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS  # ~8.33


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class CausalMaskConfig:
    """Configuration for causal mask construction."""

    # Time alignment
    video_latent_fps: float = VIDEO_LATENT_FPS
    audio_latent_fps: float = AUDIO_LATENT_FPS

    # Video tokens per frame (for 512x768 resolution with patch_size=1: 16*24 = 384)
    video_frame_seqlen: int = 384

    # Audio tokens per frame (typically 1 for LTX-2)
    audio_frame_seqlen: int = 1

    # Block size for causal generation (3 video frames = 1 second = 25 audio frames)
    num_frame_per_block: int = 3

    # Video frames in Block 0. 0 = legacy Global Prefix (1-3-3-3).
    # >0 = OmniForcing layout (e.g. 4 → 4-3-3-3).
    num_frame_per_block_first: int = 0

    # Number of learnable sink tokens prepended to audio sequence.
    # Sinks are part of Block 0, visible to all audio via causal masking.
    # They use identity RoPE (cos=1, sin=0) and are NOT supervised.
    num_audio_sink_tokens: int = 0

    # Whether to use FlexAttention for causal masks. When False, generates
    # dense boolean masks for standard SDPA (OmniForcing alignment).
    use_flex_attention: bool = True

    # Per-head gated attention (LTX-2 checkpoint feature).
    # When True, each attention layer has a ``to_gate_logits`` linear that
    # produces per-head gates: ``2 * sigmoid(gate)`` so zero-init → identity.
    apply_gated_attention: bool = False

    def get_audio_block_size(self) -> int:
        """Audio frames per video block: exactly 25 (1 second = 3 video : 25 audio)."""
        return 25


@dataclass
class CausalGenerationConfig:
    """Full configuration for causal generation."""

    # Mask configuration
    mask_config: CausalMaskConfig = field(default_factory=CausalMaskConfig)

    # Model dimensions (LTX-2 19B defaults)
    num_layers: int = 48
    video_num_heads: int = 32
    video_head_dim: int = 128
    audio_num_heads: int = 32
    audio_head_dim: int = 64

    # Video shape parameters
    video_height: int = 512
    video_width: int = 768
    num_frames: int = 121  # Default video frames

    # Training parameters
    batch_size: int = 1
    dtype: torch.dtype = torch.bfloat16

    @property
    def num_video_latent_frames(self) -> int:
        """Compute number of video latent frames."""
        return 1 + (self.num_frames - 1) // VIDEO_VAE_TEMPORAL_COMPRESSION

    @property
    def num_audio_latent_frames(self) -> int:
        """Compute number of audio latent frames."""
        video_duration_sec = self.num_frames / VIDEO_FPS
        return int(round(video_duration_sec * AUDIO_LATENT_FPS))

    @property
    def video_frame_seqlen(self) -> int:
        """Tokens per video latent frame.

        LTX-2 uses patch_size=1 (no spatial grouping), so each spatial
        position in the latent grid is one token:
            tokens_per_frame = (H / vae_spatial_compression) * (W / vae_spatial_compression)
        For 512x768: (512/32) * (768/32) = 16 * 24 = 384
        """
        h_tokens = self.video_height // 32  # 512/32 = 16
        w_tokens = self.video_width // 32   # 768/32 = 24
        return h_tokens * w_tokens  # 384


# ============================================================================
# Utility Functions
# ============================================================================

def get_video_token_time(video_latent_frame: int) -> float:
    """Get timestamp (in seconds) for a video latent frame."""
    return video_latent_frame / VIDEO_LATENT_FPS


def get_audio_token_time(audio_latent_frame: int) -> float:
    """Get timestamp (in seconds) for an audio latent frame."""
    return audio_latent_frame / AUDIO_LATENT_FPS


def get_audio_range_for_video_frame(video_frame: int) -> tuple[int, int]:
    """
    Get audio frame range [start, end) that corresponds to a video frame.

    Args:
        video_frame: Video latent frame index

    Returns:
        (audio_start, audio_end): Audio frame range
    """
    video_time_start = video_frame / VIDEO_LATENT_FPS
    video_time_end = (video_frame + 1) / VIDEO_LATENT_FPS

    audio_start = round(video_time_start * AUDIO_LATENT_FPS)
    audio_end = round(video_time_end * AUDIO_LATENT_FPS)

    return audio_start, audio_end


def compute_num_blocks(
    total_video_latent_frames: int,
    num_frame_per_block: int = 3,
) -> int:
    """Compute number of generation blocks (including Global Prefix).

    Block 0 is the Global Prefix (V_0 + A_0, 1 video frame + 1 audio frame).
    Blocks 1..N are standard blocks of num_frame_per_block each.
    """
    import math
    if total_video_latent_frames <= 1:
        return 1  # Only Global Prefix
    remaining = total_video_latent_frames - 1
    return 1 + math.ceil(remaining / num_frame_per_block)
