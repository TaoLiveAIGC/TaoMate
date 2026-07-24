"""
Flexattention utilities for efficient block-wise causal attention.

This module provides:
- Compiled flex_attention wrapper with max-autotune
- Padding utilities for 128-byte alignment (required by flexattention)
- Fallback to standard attention when flexattention unavailable
"""

import math
from typing import Optional, Callable, Union, Tuple, TYPE_CHECKING

import torch
import torch.nn.functional as F

# Try to import flexattention
try:
    from torch.nn.attention.flex_attention import (
        flex_attention as _flex_attention,
        create_block_mask,
        BlockMask,
    )
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    _flex_attention = None
    create_block_mask = None
    # Use TYPE_CHECKING to avoid runtime issues while keeping type hints
    if TYPE_CHECKING:
        from torch.nn.attention.flex_attention import BlockMask


# ============================================================================
# Compiled Flexattention
# ============================================================================

# Compile flex_attention only when explicitly enabled.  The compiled path can be
# faster after warmup, but it also starts Inductor compile_worker processes and
# causes latency jitter in the interactive service on some machines.
_COMPILE_FLEX_ATTENTION = True
if FLEX_ATTENTION_AVAILABLE and _COMPILE_FLEX_ATTENTION:
    flex_attention = torch.compile(
        _flex_attention,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
    )
elif FLEX_ATTENTION_AVAILABLE:
    flex_attention = _flex_attention
else:
    flex_attention = None


def get_flex_attention():
    """Get the compiled flex_attention function."""
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError(
            "Flexattention is not available. "
            "Please install PyTorch 2.2+ with CUDA support."
        )
    return flex_attention


# ============================================================================
# Padding Utilities
# ============================================================================

def compute_padding_size(length: int, multiple: int = 128) -> int:
    """Compute padding needed to reach next multiple."""
    if length % multiple == 0:
        return 0
    return multiple - (length % multiple)


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 128,
    dim: int = 1,
    value: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """
    Pad tensor to multiple of given size along specified dimension.

    Args:
        tensor: Input tensor
        multiple: Target multiple (default 128 for flexattention)
        dim: Dimension to pad
        value: Padding value

    Returns:
        (padded_tensor, pad_size): Padded tensor and amount of padding added
    """
    current_size = tensor.shape[dim]
    pad_size = compute_padding_size(current_size, multiple)

    if pad_size == 0:
        return tensor, 0

    # Build padding specification for F.pad
    # F.pad pads from last dimension backwards
    ndim = tensor.ndim
    pad_spec = [0] * (2 * ndim)

    # Convert dim to positive index
    if dim < 0:
        dim = ndim + dim

    # F.pad expects: (left_last, right_last, left_second_last, right_second_last, ...)
    # We want to pad on the right side of the target dimension
    pad_idx = 2 * (ndim - 1 - dim) + 1  # Right side of target dim
    pad_spec[pad_idx] = pad_size

    padded = F.pad(tensor, pad_spec, mode='constant', value=value)
    return padded, pad_size


def unpad_from_multiple(
    tensor: torch.Tensor,
    pad_size: int,
    dim: int = 1,
) -> torch.Tensor:
    """
    Remove padding added by pad_to_multiple.

    Args:
        tensor: Padded tensor
        pad_size: Amount of padding to remove
        dim: Dimension that was padded

    Returns:
        Unpadded tensor
    """
    if pad_size == 0:
        return tensor

    # Build slice to remove padding
    slices = [slice(None)] * tensor.ndim

    # Convert dim to positive index
    if dim < 0:
        dim = tensor.ndim + dim

    slices[dim] = slice(None, -pad_size)
    return tensor[tuple(slices)]


def pad_qkv_for_flex_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    multiple: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Pad Q, K, V tensors for flexattention.

    Expected input shape: [B, seq_len, num_heads, head_dim]
    Padding is applied to seq_len dimension (dim=1).

    Args:
        q, k, v: Query, key, value tensors
        multiple: Padding multiple

    Returns:
        (padded_q, padded_k, padded_v, pad_size)
    """
    q_padded, pad_size = pad_to_multiple(q, multiple, dim=1, value=0.0)
    k_padded, _ = pad_to_multiple(k, multiple, dim=1, value=0.0)
    v_padded, _ = pad_to_multiple(v, multiple, dim=1, value=0.0)
    return q_padded, k_padded, v_padded, pad_size


# ============================================================================
# Attention Functions
# ============================================================================

def flex_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: Optional["BlockMask"] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Forward pass using flex_attention with automatic padding.

    Args:
        q: Query tensor [B, seq_len, num_heads, head_dim]
        k: Key tensor [B, seq_len, num_heads, head_dim]
        v: Value tensor [B, seq_len, num_heads, head_dim]
        block_mask: Optional BlockMask for causal attention
        scale: Optional attention scale (default: 1/sqrt(head_dim))

    Returns:
        Output tensor [B, seq_len, num_heads, head_dim]
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("Flexattention not available")

    original_seq_len = q.shape[1]

    # Pad to multiple of 128
    q_padded, k_padded, v_padded, pad_size = pad_qkv_for_flex_attention(q, k, v)

    # Transpose for flex_attention: [B, H, L, D]
    q_t = q_padded.transpose(1, 2)
    k_t = k_padded.transpose(1, 2)
    v_t = v_padded.transpose(1, 2)

    # Apply flex_attention
    out = flex_attention(
        q_t, k_t, v_t,
        block_mask=block_mask,
        scale=scale,
    )

    # Transpose back: [B, L, H, D]
    out = out.transpose(1, 2)

    # Remove padding
    if pad_size > 0:
        out = out[:, :original_seq_len]

    return out


def standard_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Standard scaled dot-product attention (fallback).

    Args:
        q: Query tensor [B, seq_len, num_heads, head_dim]
        k: Key tensor [B, kv_len, num_heads, head_dim]
        v: Value tensor [B, kv_len, num_heads, head_dim]
        mask: Optional attention mask [B, 1, seq_len, kv_len] or [seq_len, kv_len]
        scale: Optional attention scale

    Returns:
        Output tensor [B, seq_len, num_heads, head_dim]
    """
    # Transpose for SDPA: [B, H, L, D]
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    # Process mask
    attn_mask = None
    if mask is not None:
        if mask.dtype == torch.bool:
            # Convert boolean mask to float mask
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(~mask, float("-inf"))
        else:
            attn_mask = mask

        # Expand dimensions if needed
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

    # Apply SDPA
    out = F.scaled_dot_product_attention(
        q_t, k_t, v_t,
        attn_mask=attn_mask,
        scale=scale,
        dropout_p=0.0,
        is_causal=False,
    )

    # Transpose back: [B, L, H, D]
    return out.transpose(1, 2)


# ============================================================================
# Block Mask Utilities
# ============================================================================

def create_causal_block_mask(
    mask_fn: Callable[[int, int, int, int], bool],
    batch_size: Optional[int],
    num_heads: Optional[int],
    q_len: int,
    kv_len: int,
    device: Union[torch.device, str] = "cuda",
    compile_mask: bool = True,
) -> "BlockMask":
    """
    Create a BlockMask for causal attention patterns.

    Args:
        mask_fn: Function (b, h, q_idx, kv_idx) -> bool indicating if attention is allowed
        batch_size: Batch size (None for batch-independent mask)
        num_heads: Number of heads (None for head-independent mask)
        q_len: Query sequence length
        kv_len: Key/value sequence length
        device: Target device
        compile_mask: Whether to compile the mask function

    Returns:
        BlockMask object
    """
    if not FLEX_ATTENTION_AVAILABLE:
        raise RuntimeError("Flexattention not available")

    # Pad lengths to multiple of 128
    q_len_padded = math.ceil(q_len / 128) * 128
    kv_len_padded = math.ceil(kv_len / 128) * 128

    return create_block_mask(
        mask_fn,
        B=batch_size,
        H=num_heads,
        Q_LEN=q_len_padded,
        KV_LEN=kv_len_padded,
        device=device,
        _compile=compile_mask,
    )
