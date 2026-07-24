"""
Causal attention modules for LTX-2.

This module provides:
- AVCausalMaskBuilder: Builds causal masks for audio-video joint attention
- CausalLTXAttention: Causal attention with Flexattention support
- Flexattention utilities for efficient block-wise causal attention
"""

from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    compute_av_blocks,
    compute_aligned_audio_frames,
    compute_causal_log_scales,
    verify_causal_masks,
)
from ltx_causal.attention.flex_attention_utils import (
    get_flex_attention,
    pad_to_multiple,
    unpad_from_multiple,
)

__all__ = [
    "AVCausalMaskBuilder",
    "compute_av_blocks",
    "compute_aligned_audio_frames",
    "compute_causal_log_scales",
    "verify_causal_masks",
    "get_flex_attention",
    "pad_to_multiple",
    "unpad_from_multiple",
]
