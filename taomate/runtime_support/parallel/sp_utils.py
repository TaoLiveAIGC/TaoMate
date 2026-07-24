"""
Sequence-axis utilities for Ulysses Sequence Parallel.

These helpers operate on the *outer* tensor layout used between transformer
blocks: ``[B, S, ...]``. They are intended to be called at the model entry
(slice the full sequence into the local SP shard) and exit (gather back to
the full sequence). When SP is disabled they are no-ops, so call sites can
unconditionally invoke them.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .mesh import get_sp_group, get_sp_rank, get_sp_world_size


# ---------------------------------------------------------------------------
# Sequence sharding (entry / exit of the model)
# ---------------------------------------------------------------------------
def split_sequence(
    x: torch.Tensor,
    dim: int = 1,
    sp_rank: Optional[int] = None,
    sp_size: Optional[int] = None,
) -> torch.Tensor:
    """Take the local SP shard of a full sequence tensor along ``dim``.

    Pads on the right to make ``S`` divisible by ``sp_size`` if necessary.
    The padded length is **not** stored here; callers that need to unpad
    should rely on ``pad_sequence_to_multiple`` upstream and remember the
    original length themselves.
    """
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return x
    if sp_rank is None:
        sp_rank = get_sp_rank()

    S = x.shape[dim]
    if S % sp_size != 0:
        pad = sp_size - (S % sp_size)
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        zeros = x.new_zeros(pad_shape)
        x = torch.cat([x, zeros], dim=dim)
        S = x.shape[dim]

    chunk = S // sp_size
    return x.narrow(dim, sp_rank * chunk, chunk).contiguous()


class _GatherSequence(torch.autograd.Function):
    """All-gather along ``dim`` for autograd; backward is a per-rank slice."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, sp_size: int, sp_rank: int, dim: int):
        ctx.group = group
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        ctx.dim = dim
        ctx.local_size = x.shape[dim]
        if sp_size <= 1:
            return x
        # all_gather expects identical shapes on all ranks.
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(sp_size)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.sp_size <= 1:
            return grad_output, None, None, None, None
        # Slice out our local shard from the full gradient.
        start = ctx.sp_rank * ctx.local_size
        grad_input = grad_output.narrow(ctx.dim, start, ctx.local_size).contiguous()
        return grad_input, None, None, None, None


def gather_sequence(
    x: torch.Tensor,
    dim: int = 1,
    group=None,
    sp_size: Optional[int] = None,
    sp_rank: Optional[int] = None,
) -> torch.Tensor:
    """All-gather ``x`` along ``dim`` over the SP group (autograd-friendly).

    No-op when SP is disabled.
    """
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return x
    if group is None:
        group = get_sp_group()
    if sp_rank is None:
        sp_rank = get_sp_rank()
    return _GatherSequence.apply(x, group, sp_size, sp_rank, dim)


# ---------------------------------------------------------------------------
# Length alignment
# ---------------------------------------------------------------------------
def pad_sequence_to_multiple(
    x: torch.Tensor,
    multiple: int,
    dim: int = 1,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, int, int]:
    """Right-pad ``x`` along ``dim`` so its length is a multiple of ``multiple``.

    Returns ``(padded_tensor, original_length, padded_length)``.
    If already aligned, returns the input unchanged with matching metadata.
    """
    S = x.shape[dim]
    if multiple <= 1 or S % multiple == 0:
        return x, S, S
    pad = multiple - (S % multiple)
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    padding = x.new_full(pad_shape, pad_value)
    out = torch.cat([x, padding], dim=dim)
    return out, S, S + pad


def unpad_sequence(
    x: torch.Tensor,
    original_length: int,
    dim: int = 1,
) -> torch.Tensor:
    """Trim ``x`` along ``dim`` back to ``original_length``.

    Inverse of ``pad_sequence_to_multiple``; safe when no padding was applied.
    """
    if x.shape[dim] == original_length:
        return x
    return x.narrow(dim, 0, original_length).contiguous()
