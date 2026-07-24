"""
Autograd-friendly all-to-all primitives for Ulysses Sequence Parallel.

Tensor layout convention used by TaoMate attention:
    Seq-sharded form  : [B, S_local,  H,        D_h]   (S_local = S / sp)
    Head-sharded form : [B, S,        H_local,  D_h]   (H_local = H / sp)

Two ops:
    SeqAllToAllHead :  Seq-sharded  -> Head-sharded   (used before SDPA)
    HeadAllToAllSeq :  Head-sharded -> Seq-sharded    (used after  SDPA)

Both ops are pure communication: forward = all-to-all in one direction,
backward = all-to-all in the opposite direction (Ulysses ops are
self-adjoint up to a transpose). Implemented as ``torch.autograd.Function``
so they integrate with FSDP and activation checkpointing without extra glue.

Caller is responsible for ensuring:
- ``S`` is divisible by ``sp_size`` (use ``pad_sequence_to_multiple`` upstream)
- ``H`` is divisible by ``sp_size`` (asserted at runtime)
- All ranks in the SP group call with identical shapes
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

try:
    import torch.distributed._functional_collectives as funcol
except Exception:  # pragma: no cover - depends on the PyTorch build.
    funcol = None

from .mesh import get_sp_group, get_sp_world_size


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _all_to_all_single(input_: torch.Tensor, group) -> torch.Tensor:
    """Wrapper around ``dist.all_to_all_single`` that handles non-contiguous
    inputs and returns a fresh contiguous tensor of identical shape."""
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    output = torch.empty_like(input_)
    dist.all_to_all_single(output, input_, group=group)
    return output


class _AsyncAllToAll:
    """Small inference-only handle for non-blocking Ulysses all-to-all."""

    def __init__(
        self,
        *,
        input_: torch.Tensor,
        output: torch.Tensor,
        work,
        sp_size: int,
        direction: str,
        B: int,
        S_local: int,
        S: int,
        H_local: int,
        D_h: int,
        wait_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        self.input_ = input_
        self.output = output
        self.work = work
        self.wait_fn = wait_fn
        self.sp_size = int(sp_size)
        self.direction = direction
        self.B = int(B)
        self.S_local = int(S_local)
        self.S = int(S)
        self.H_local = int(H_local)
        self.D_h = int(D_h)

    def wait(self) -> torch.Tensor:
        if self.wait_fn is not None:
            self.output = self.wait_fn(self.output)
        elif self.work is not None:
            self.work.wait()
        if self.direction == "seq_to_head":
            x = self.output.permute(1, 0, 2, 3, 4).contiguous()
            return x.reshape(
                self.B, self.sp_size * self.S_local, self.H_local, self.D_h,
            )
        if self.direction == "head_to_seq":
            x = self.output.permute(1, 2, 0, 3, 4).contiguous()
            return x.reshape(
                self.B, self.S_local, self.sp_size * self.H_local, self.D_h,
            )
        return self.output


class _AsyncAllToAllMany:
    """Inference-only handle for a packed non-blocking all-to-all."""

    def __init__(self, handle: _AsyncAllToAll, batch_sizes) -> None:
        self.handle = handle
        self.batch_sizes = tuple(int(x) for x in batch_sizes)

    def wait(self):
        out = self.handle.wait()
        return tuple(out.split(self.batch_sizes, dim=0))


def _all_to_all_single_async(input_: torch.Tensor, group):
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    if funcol is not None:
        # Prefer functional collectives for inference async Ulysses: the
        # returned tensor carries the pending collective and can be waited only
        # when attention consumes Q/K/V, matching the V-first overlap schedule.
        output = funcol.all_to_all_single(input_, None, None, group)
        return output, None, funcol.wait_tensor
    output = torch.empty_like(input_)
    work = dist.all_to_all_single(output, input_, group=group, async_op=True)
    return output, work, None


def _seq_to_head(x: torch.Tensor, sp_size: int, group) -> torch.Tensor:
    """[B, S_local, H, D_h]  -> [B, S, H_local, D_h]

    Algorithm:
    1. Split H into ``sp_size`` chunks: [B, S_local, sp, H_local, D_h]
    2. Move ``sp`` to dim 0 and make contiguous: [sp, B, S_local, H_local, D_h]
    3. all_to_all_single (scatters the ``sp`` chunks across ranks)
    4. Each rank now holds the full S, restructure to [B, S, H_local, D_h]
    """
    B, S_local, H, D_h = x.shape
    assert H % sp_size == 0, f"H ({H}) must be divisible by sp_size ({sp_size})"
    H_local = H // sp_size

    # [B, S_local, sp, H_local, D_h] -> [sp, B, S_local, H_local, D_h]
    x = x.reshape(B, S_local, sp_size, H_local, D_h)
    x = x.permute(2, 0, 1, 3, 4).contiguous()

    # all-to-all on dim 0 (the sp dimension)
    x = _all_to_all_single(x, group)

    # x is [sp, B, S_local, H_local, D_h] but the sp dim now indexes
    # source-rank chunks, which together form the full sequence. Move it
    # next to S_local and merge.
    # -> [B, sp, S_local, H_local, D_h] -> [B, sp * S_local, H_local, D_h]
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    x = x.reshape(B, sp_size * S_local, H_local, D_h)
    return x


def _head_to_seq(x: torch.Tensor, sp_size: int, group) -> torch.Tensor:
    """[B, S, H_local, D_h]  -> [B, S_local, H, D_h]

    Inverse of ``_seq_to_head``.
    """
    B, S, H_local, D_h = x.shape
    assert S % sp_size == 0, f"S ({S}) must be divisible by sp_size ({sp_size})"
    S_local = S // sp_size

    # [B, sp, S_local, H_local, D_h] -> [sp, B, S_local, H_local, D_h]
    x = x.reshape(B, sp_size, S_local, H_local, D_h)
    x = x.permute(1, 0, 2, 3, 4).contiguous()

    x = _all_to_all_single(x, group)

    # [sp, B, S_local, H_local, D_h] -> [B, S_local, sp, H_local, D_h]
    # -> [B, S_local, sp * H_local, D_h]
    x = x.permute(1, 2, 0, 3, 4).contiguous()
    x = x.reshape(B, S_local, sp_size * H_local, D_h)
    return x


def _seq_to_head_async(x: torch.Tensor, sp_size: int, group) -> _AsyncAllToAll:
    B, S_local, H, D_h = x.shape
    assert H % sp_size == 0, f"H ({H}) must be divisible by sp_size ({sp_size})"
    H_local = H // sp_size
    x = x.reshape(B, S_local, sp_size, H_local, D_h)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    output, work, wait_fn = _all_to_all_single_async(x, group)
    return _AsyncAllToAll(
        input_=x,
        output=output,
        work=work,
        wait_fn=wait_fn,
        sp_size=sp_size,
        direction="seq_to_head",
        B=B,
        S_local=S_local,
        S=sp_size * S_local,
        H_local=H_local,
        D_h=D_h,
    )


def _head_to_seq_async(x: torch.Tensor, sp_size: int, group) -> _AsyncAllToAll:
    B, S, H_local, D_h = x.shape
    assert S % sp_size == 0, f"S ({S}) must be divisible by sp_size ({sp_size})"
    S_local = S // sp_size
    x = x.reshape(B, sp_size, S_local, H_local, D_h)
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    output, work, wait_fn = _all_to_all_single_async(x, group)
    return _AsyncAllToAll(
        input_=x,
        output=output,
        work=work,
        wait_fn=wait_fn,
        sp_size=sp_size,
        direction="head_to_seq",
        B=B,
        S_local=S_local,
        S=S,
        H_local=H_local,
        D_h=D_h,
    )


# ---------------------------------------------------------------------------
# Autograd functions
# ---------------------------------------------------------------------------
class SeqAllToAllHead(torch.autograd.Function):
    """Seq-sharded -> Head-sharded all-to-all. Forward pre-attention."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, sp_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.sp_size = sp_size
        if sp_size <= 1:
            return x
        return _seq_to_head(x, sp_size, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sp_size = ctx.sp_size
        if sp_size <= 1:
            return grad_output, None, None
        # backward of Seq->Head is Head->Seq on the gradient
        grad_input = _head_to_seq(grad_output.contiguous(), sp_size, ctx.group)
        return grad_input, None, None


class HeadAllToAllSeq(torch.autograd.Function):
    """Head-sharded -> Seq-sharded all-to-all. Used post-attention."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, sp_size: int) -> torch.Tensor:
        ctx.group = group
        ctx.sp_size = sp_size
        if sp_size <= 1:
            return x
        return _head_to_seq(x, sp_size, group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        sp_size = ctx.sp_size
        if sp_size <= 1:
            return grad_output, None, None
        grad_input = _seq_to_head(grad_output.contiguous(), sp_size, ctx.group)
        return grad_input, None, None


# ---------------------------------------------------------------------------
# Functional wrappers (preferred call site)
# ---------------------------------------------------------------------------
def seq_all_to_all_head(
    x: torch.Tensor,
    group=None,
    sp_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert ``[B, S/sp, H, D_h]`` -> ``[B, S, H/sp, D_h]``.

    When SP is disabled or ``sp_size <= 1`` this is a pass-through.
    """
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return x
    if group is None:
        group = get_sp_group()
    return SeqAllToAllHead.apply(x, group, sp_size)


def seq_all_to_all_head_many(
    tensors,
    group=None,
    sp_size: Optional[int] = None,
):
    """Fused inference-only ``seq_all_to_all_head`` for same-shaped tensors.

    For KV-cache inference Q/K/V often have identical ``[B, S_local, H, D]``
    shapes.  Concatenating them on the batch axis and running one all-to-all
    is mathematically identical to launching separate collectives, but avoids
    repeated NCCL latency.  This intentionally bypasses the autograd wrapper
    only when gradients are disabled; training keeps the original path.
    """
    tensors = tuple(tensors)
    if not tensors:
        return tuple()
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1 or len(tensors) == 1:
        return tensors
    if torch.is_grad_enabled():
        return tuple(seq_all_to_all_head(t, group=group, sp_size=sp_size) for t in tensors)
    if group is None:
        group = get_sp_group()
    base_shape = tuple(tensors[0].shape)
    if any(tuple(t.shape) != base_shape for t in tensors):
        return tuple(seq_all_to_all_head(t, group=group, sp_size=sp_size) for t in tensors)
    batch_sizes = [int(t.shape[0]) for t in tensors]
    packed = torch.cat(tensors, dim=0)
    out = _seq_to_head(packed, sp_size, group)
    return tuple(out.split(batch_sizes, dim=0))


def seq_all_to_all_head_many_async(
    tensors,
    group=None,
    sp_size: Optional[int] = None,
):
    """Packed async ``seq_all_to_all_head`` for same-shaped tensors.

    This is the non-blocking counterpart to ``seq_all_to_all_head_many``.  It is
    inference-only and returns a handle whose ``wait()`` method yields the
    unpacked tensor tuple.
    """
    tensors = tuple(tensors)
    if not tensors:
        raise ValueError("seq_all_to_all_head_many_async requires at least one tensor")
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        class _ImmediateMany:
            def __init__(self, values):
                self.values = values
            def wait(self):
                return self.values
        return _ImmediateMany(tensors)
    if len(tensors) == 1:
        return seq_all_to_all_head_async(tensors[0], group=group, sp_size=sp_size)
    if torch.is_grad_enabled():
        raise RuntimeError("seq_all_to_all_head_many_async is inference-only")
    if group is None:
        group = get_sp_group()
    base_shape = tuple(tensors[0].shape)
    if any(tuple(t.shape) != base_shape for t in tensors):
        raise ValueError("seq_all_to_all_head_many_async requires same-shaped tensors")
    batch_sizes = [int(t.shape[0]) for t in tensors]
    packed = torch.cat(tensors, dim=0)
    return _AsyncAllToAllMany(
        _seq_to_head_async(packed, sp_size, group),
        batch_sizes,
    )


def head_all_to_all_seq(
    x: torch.Tensor,
    group=None,
    sp_size: Optional[int] = None,
) -> torch.Tensor:
    """Convert ``[B, S, H/sp, D_h]`` -> ``[B, S/sp, H, D_h]``.

    When SP is disabled or ``sp_size <= 1`` this is a pass-through.
    """
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return x
    if group is None:
        group = get_sp_group()
    return HeadAllToAllSeq.apply(x, group, sp_size)


def seq_all_to_all_head_async(
    x: torch.Tensor,
    group=None,
    sp_size: Optional[int] = None,
) -> _AsyncAllToAll:
    """Inference-only non-blocking variant of ``seq_all_to_all_head``.

    The returned handle must be resolved with ``.wait()`` before the tensor is
    consumed.  It intentionally does not implement autograd; call sites should
    use this only under no-grad/inference-mode paths.
    """
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return _AsyncAllToAll(
            input_=x,
            output=x,
            work=None,
            wait_fn=None,
            sp_size=1,
            direction="identity",
            B=x.shape[0],
            S_local=x.shape[1],
            S=x.shape[1],
            H_local=x.shape[2],
            D_h=x.shape[3],
        )
    if group is None:
        group = get_sp_group()
    return _seq_to_head_async(x, sp_size, group)


def head_all_to_all_seq_async(
    x: torch.Tensor,
    group=None,
    sp_size: Optional[int] = None,
) -> _AsyncAllToAll:
    """Inference-only non-blocking variant of ``head_all_to_all_seq``."""
    if sp_size is None:
        sp_size = get_sp_world_size()
    if sp_size <= 1:
        return _AsyncAllToAll(
            input_=x,
            output=x,
            work=None,
            wait_fn=None,
            sp_size=1,
            direction="identity",
            B=x.shape[0],
            S_local=x.shape[1],
            S=x.shape[1],
            H_local=x.shape[2],
            D_h=x.shape[3],
        )
    if group is None:
        group = get_sp_group()
    return _head_to_seq_async(x, sp_size, group)
