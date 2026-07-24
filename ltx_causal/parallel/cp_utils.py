"""
Ulysses-style Context Parallel (CP) primitives.

This module enables splitting a single training sample across multiple
GPUs along the sequence dimension for the attention path, and along the
head dimension for the QKV/attention kernel — i.e. DeepSpeed Ulysses.

Design goals
------------
1. **Bit-identical fallback when CP is disabled.**
   When ``cp_size == 1`` (the default) every helper here is an identity
   no-op.  Calling any of these helpers from existing training code adds
   *zero* collectives and *zero* tensor copies.

2. **Single source of truth for CP state.**
   The CP process group, world-mesh handle, local rank and size are all
   stored as module-level globals.  Call :func:`init_cp` exactly once
   per process *after* ``torch.distributed`` has been initialised.

3. **Autograd-safe collectives.**
   :class:`_SeqAllToAll` is a custom :class:`torch.autograd.Function`
   whose backward is itself an all-to-all in the opposite direction —
   matching the gradient routing required by Ulysses.

4. **Padding-aware shard helpers.**
   Causal sequences are not always divisible by ``cp_size``; the
   ``pad_seq_to_multiple`` / ``unpad_seq`` pair lets callers pad on the
   way in and trim on the way out without ever touching the attention
   kernel's notion of length.

The public API is re-exported through :mod:`ltx_causal.parallel`.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# CP process group spanning the ranks that share a single training sample.
_CP_GROUP: Optional[dist.ProcessGroup] = None
# Number of ranks in the CP group.  ``1`` means CP is disabled.
_CP_SIZE: int = 1
# Local rank within the CP group, 0 ≤ rank < cp_size.
_CP_RANK: int = 0
# Optional handle to the 2D DeviceMesh used to derive _CP_GROUP.  Stored so
# downstream code (e.g. FSDP) can also build its own subgroup from the
# same mesh and stay consistent.
_WORLD_MESH = None

# Re-entrant counter for :func:`cp_disabled`.  When > 0 every CP helper
# (``is_cp_enabled``, the all-to-all wrappers, the splitting/gathering
# utilities) behaves as if CP was disabled, *without* tearing down the
# global CP state.  Used by code paths that must run on the full
# sequence on every rank — chiefly KV-cache inference, where each rank
# accumulates its own cache and CP would be both incorrect and wasteful.
_CP_BYPASS_DEPTH: int = 0


# ---------------------------------------------------------------------------
# Initialisation / accessors
# ---------------------------------------------------------------------------

def init_cp(
    cp_size: int,
    world_mesh=None,
    cp_group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Configure the global CP state.

    Args:
        cp_size: Desired CP world size.  Must divide ``WORLD_SIZE``.  A
            value of ``1`` disables CP entirely (no collectives, all
            helpers become identity).
        world_mesh: Optional :class:`torch.distributed.DeviceMesh` whose
            ``"cp"`` dimension defines the CP group.  Preferred path.
        cp_group: Optional pre-built CP ProcessGroup.  Used as a fallback
            when ``world_mesh`` is unavailable.

    Notes:
        * Safe to call with ``cp_size=1`` even when ``dist`` is not
          initialised — this is the default and is a no-op.
        * Idempotent: calling it again with the same ``cp_size`` resets
          the cached group only when the new ``world_mesh`` differs.
    """
    global _CP_GROUP, _CP_SIZE, _CP_RANK, _WORLD_MESH

    if cp_size <= 1:
        _CP_GROUP = None
        _CP_SIZE = 1
        _CP_RANK = 0
        _WORLD_MESH = None
        return

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "init_cp(cp_size>1) requires torch.distributed to be initialised. "
            "Call launch_distributed_job() (or dist.init_process_group) first."
        )

    world_size = dist.get_world_size()
    if world_size % cp_size != 0:
        raise ValueError(
            f"cp_size={cp_size} must divide WORLD_SIZE={world_size}."
        )

    # Resolve the CP subgroup.
    resolved_group: Optional[dist.ProcessGroup] = None
    if world_mesh is not None:
        # Expect a 2D mesh with names ("fsdp", "cp"). The CP slice gives the
        # ProcessGroup containing all ranks that hold the same FSDP shard
        # but split the sequence.
        try:
            resolved_group = world_mesh["cp"].get_group()
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Failed to extract CP subgroup from world_mesh: {exc}. "
                "Expected a 2D DeviceMesh with names=('fsdp', 'cp')."
            ) from exc
    elif cp_group is not None:
        resolved_group = cp_group
    else:
        # Build a fresh CP group.  Lay out CP contiguously across ranks
        # so each CP-group spans `cp_size` consecutive global ranks.
        # FSDP groups are then formed by the orthogonal stride.
        global_rank = dist.get_rank()
        cp_block = global_rank // cp_size
        ranks_in_group = list(range(cp_block * cp_size, (cp_block + 1) * cp_size))
        # Every CP group must be created collectively across the whole world.
        for block_idx in range(world_size // cp_size):
            ranks = list(range(block_idx * cp_size, (block_idx + 1) * cp_size))
            grp = dist.new_group(ranks=ranks)
            if global_rank in ranks:
                resolved_group = grp

        assert resolved_group is not None
        _ = ranks_in_group  # silence unused warning

    _CP_GROUP = resolved_group
    _CP_SIZE = cp_size
    _CP_RANK = dist.get_rank(group=_CP_GROUP)
    _WORLD_MESH = world_mesh


def is_cp_enabled() -> bool:
    """True iff CP is active (cp_size > 1, dist initialised, not bypassed).

    When inside a :func:`cp_disabled` context this returns ``False`` even
    if the process-wide CP state is configured for cp_size>1.  All
    helpers downstream key off this function so wrapping a code block
    in ``with cp_disabled():`` makes it behave as a single-CP path
    (no all-to-all, no sequence splitting), which is exactly what KV-cache
    inference needs.
    """
    if _CP_BYPASS_DEPTH > 0:
        return False
    return _CP_SIZE > 1 and _CP_GROUP is not None


@contextmanager
def cp_disabled():
    """Context manager that temporarily disables CP within its scope.

    Re-entrant: nested ``cp_disabled()`` blocks behave correctly.
    Use for code paths that must process the full (un-sharded) sequence
    on every rank, e.g. ``forward_inference`` where each rank manages
    its own KV cache and CP slicing would corrupt the cache layout.

    Example::

        with cp_disabled():
            # is_cp_enabled() is False here; attention runs on full seq
            out = model.forward_inference(...)
    """
    global _CP_BYPASS_DEPTH
    _CP_BYPASS_DEPTH += 1
    try:
        yield
    finally:
        _CP_BYPASS_DEPTH -= 1


def get_cp_size() -> int:
    """CP world size; ``1`` when disabled or inside ``cp_disabled()``."""
    if _CP_BYPASS_DEPTH > 0:
        return 1
    return _CP_SIZE


def get_cp_rank() -> int:
    """CP local rank within the CP group; ``0`` when disabled or bypassed."""
    if _CP_BYPASS_DEPTH > 0:
        return 0
    return _CP_RANK


def get_cp_group() -> Optional[dist.ProcessGroup]:
    """The CP ProcessGroup, or ``None`` when disabled or bypassed."""
    if _CP_BYPASS_DEPTH > 0:
        return None
    return _CP_GROUP


def get_cp_mesh():
    """The 2D world mesh handle (when CP was initialised from a mesh)."""
    return _WORLD_MESH


# ---------------------------------------------------------------------------
# Autograd-safe all-to-all
# ---------------------------------------------------------------------------

class _SeqAllToAll(torch.autograd.Function):
    """All-to-all that swaps the sequence and head dimensions.

    Forward:  ``[B, L_local, H, D] -> [B, L_full, H_local, D]``
              (when ``forward_seq_to_head=True``).
    Backward: same all-to-all in the opposite direction so gradients
              follow the forward layout exactly.

    Implementation detail: we always work with a contiguous shape of
    ``[cp_size, ..., L_chunk, H_chunk, D]`` so :func:`dist.all_to_all_single`
    can be used directly on the flat buffer.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        x: torch.Tensor,
        cp_size: int,
        cp_group: dist.ProcessGroup,
        seq_dim: int,
        head_dim: int,
        forward_seq_to_head: bool,
    ) -> torch.Tensor:
        ctx.cp_size = cp_size
        ctx.cp_group = cp_group
        ctx.seq_dim = seq_dim
        ctx.head_dim = head_dim
        ctx.forward_seq_to_head = forward_seq_to_head
        return _all_to_all_swap(
            x,
            cp_size=cp_size,
            cp_group=cp_group,
            seq_dim=seq_dim,
            head_dim=head_dim,
            seq_to_head_direction=forward_seq_to_head,
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        # Reverse direction in backward.
        grad_in = _all_to_all_swap(
            grad_output.contiguous(),
            cp_size=ctx.cp_size,
            cp_group=ctx.cp_group,
            seq_dim=ctx.seq_dim,
            head_dim=ctx.head_dim,
            seq_to_head_direction=not ctx.forward_seq_to_head,
        )
        return grad_in, None, None, None, None, None


def _all_to_all_swap(
    x: torch.Tensor,
    cp_size: int,
    cp_group: dist.ProcessGroup,
    seq_dim: int,
    head_dim: int,
    seq_to_head_direction: bool,
) -> torch.Tensor:
    """Run an all-to-all that swaps the seq and head dims.

    Layout invariant
    ----------------
    - ``seq_to_head_direction=True``  : split ``head_dim`` across ranks,
      gather ``seq_dim``  → output is full-sequence, head-shard.
    - ``seq_to_head_direction=False`` : split ``seq_dim`` across ranks,
      gather ``head_dim`` → output is seq-shard, full-head.

    The implementation packs the to-be-scattered dim onto the leading
    axis, calls ``all_to_all_single``, then permutes back.
    """
    if cp_size == 1:
        return x

    if seq_to_head_direction:
        # Split heads across ranks (chunk along head_dim into cp_size slices),
        # then move that chunk axis to dim=0 for the all-to-all.
        assert x.size(head_dim) % cp_size == 0, (
            f"head dim ({x.size(head_dim)}) must be divisible by cp_size ({cp_size})"
        )
        # Reshape: insert a new chunk axis right before head_dim.
        new_shape = list(x.shape)
        new_shape[head_dim] = new_shape[head_dim] // cp_size
        new_shape.insert(head_dim, cp_size)
        x_split = x.contiguous().view(*new_shape)
        # Move the chunk axis (currently at position head_dim) to dim 0.
        x_split = x_split.movedim(head_dim, 0).contiguous()
        # all_to_all_single along dim 0.
        out = torch.empty_like(x_split)
        dist.all_to_all_single(out, x_split, group=cp_group)
        # Now dim 0 indexes "incoming chunk from rank r" — these are
        # consecutive seq slices.  Move the chunk axis back, but this time
        # to the seq position so chunks concatenate along the sequence.
        # After the swap, head_dim has shrunk by cp_size (head shard) and
        # seq_dim grows by cp_size.
        out = out.movedim(0, seq_dim).contiguous()
        # Collapse chunk axis (currently at seq_dim) into seq_dim (next axis).
        gather_shape = list(out.shape)
        # gather_shape[seq_dim] == cp_size, gather_shape[seq_dim + 1] == L_local
        gather_shape[seq_dim + 1] = gather_shape[seq_dim] * gather_shape[seq_dim + 1]
        gather_shape.pop(seq_dim)
        return out.view(*gather_shape)
    else:
        # head_to_seq: chunk along seq_dim, gather along head_dim.
        assert x.size(seq_dim) % cp_size == 0, (
            f"seq dim ({x.size(seq_dim)}) must be divisible by cp_size ({cp_size})"
        )
        new_shape = list(x.shape)
        new_shape[seq_dim] = new_shape[seq_dim] // cp_size
        new_shape.insert(seq_dim, cp_size)
        x_split = x.contiguous().view(*new_shape)
        x_split = x_split.movedim(seq_dim, 0).contiguous()
        out = torch.empty_like(x_split)
        dist.all_to_all_single(out, x_split, group=cp_group)
        # Place the chunk axis next to head_dim and collapse — head dim grows.
        out = out.movedim(0, head_dim).contiguous()
        gather_shape = list(out.shape)
        gather_shape[head_dim + 1] = gather_shape[head_dim] * gather_shape[head_dim + 1]
        gather_shape.pop(head_dim)
        return out.view(*gather_shape)


# ---------------------------------------------------------------------------
# Public seq_to_head / head_to_seq
# ---------------------------------------------------------------------------

def seq_to_head(
    x: torch.Tensor,
    seq_dim: int = 1,
    head_dim: int = 2,
) -> torch.Tensor:
    """Convert a seq-sharded, full-head tensor into a full-seq, head-sharded one.

    Default layout is ``[B, L, H, D]`` (i.e. ``seq_dim=1``, ``head_dim=2``).

    No-op when CP is disabled.
    """
    if not is_cp_enabled():
        return x
    return _SeqAllToAll.apply(
        x, _CP_SIZE, _CP_GROUP, seq_dim, head_dim, True
    )


def head_to_seq(
    x: torch.Tensor,
    seq_dim: int = 1,
    head_dim: int = 2,
) -> torch.Tensor:
    """Inverse of :func:`seq_to_head`.

    Converts ``[B, L_full, H_local, D]`` back to ``[B, L_local, H, D]``.
    No-op when CP is disabled.
    """
    if not is_cp_enabled():
        return x
    return _SeqAllToAll.apply(
        x, _CP_SIZE, _CP_GROUP, seq_dim, head_dim, False
    )


# ---------------------------------------------------------------------------
# Sequence shard / gather helpers (used at model entry / exit boundaries)
# ---------------------------------------------------------------------------

def split_seq(x: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Take this rank's slice along ``dim`` from a full-sequence tensor.

    Caller must ensure ``x.size(dim) % cp_size == 0`` (use
    :func:`pad_seq_to_multiple` if not).  No-op when CP is disabled.
    """
    if not is_cp_enabled():
        return x
    cp_size = _CP_SIZE
    rank = _CP_RANK
    L = x.size(dim)
    if L % cp_size != 0:
        raise ValueError(
            f"split_seq: dim={dim} has length {L}, not divisible by cp_size={cp_size}. "
            "Pad with pad_seq_to_multiple() before calling."
        )
    chunk = L // cp_size
    # narrow is a view, no copy.
    return x.narrow(dim, rank * chunk, chunk).contiguous()


def gather_seq(
    x: torch.Tensor,
    dim: int = 1,
    original_len: Optional[int] = None,
) -> torch.Tensor:
    """All-gather ``x`` across the CP group along ``dim``.

    When ``original_len`` is provided, the gathered tensor is trimmed to
    that length on ``dim`` (used to undo padding).  No-op when CP is
    disabled.
    """
    if not is_cp_enabled():
        if original_len is not None and original_len != x.size(dim):
            return x.narrow(dim, 0, original_len).contiguous()
        return x

    cp_size = _CP_SIZE
    cp_group = _CP_GROUP
    x = x.contiguous()
    gather_list = [torch.empty_like(x) for _ in range(cp_size)]
    dist.all_gather(gather_list, x, group=cp_group)
    out = torch.cat(gather_list, dim=dim)
    if original_len is not None and original_len != out.size(dim):
        out = out.narrow(dim, 0, original_len).contiguous()
    return out


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------

def pad_seq_to_multiple(
    x: torch.Tensor,
    multiple: int,
    dim: int = 1,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, int]:
    """Right-pad ``x`` along ``dim`` so the size is divisible by ``multiple``.

    Returns ``(padded, original_len)``.  When already divisible (or
    ``multiple <= 1``) this is a cheap no-op that returns ``x`` itself.
    """
    if multiple <= 1:
        return x, x.size(dim)

    L = x.size(dim)
    rem = L % multiple
    if rem == 0:
        return x, L

    pad_len = multiple - rem
    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad_tensor = x.new_full(pad_shape, pad_value)
    return torch.cat([x, pad_tensor], dim=dim), L


def unpad_seq(x: torch.Tensor, original_len: int, dim: int = 1) -> torch.Tensor:
    """Inverse of :func:`pad_seq_to_multiple` (just narrows back)."""
    if x.size(dim) == original_len:
        return x
    return x.narrow(dim, 0, original_len).contiguous()


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------

def cp_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """SUM all-reduce within the CP group; no-op when CP is disabled.

    Used to aggregate per-shard losses (or any per-shard scalar) into a
    single full-sequence value before backward.
    """
    if not is_cp_enabled():
        return x
    x = x.contiguous()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=_CP_GROUP)
    return x


def cp_broadcast(x: torch.Tensor, src_rank: int = 0) -> torch.Tensor:
    """Broadcast ``x`` from ``src_rank`` (CP-local) to every CP rank.

    Used to synchronise random decisions (e.g. timestep sampling) across
    all CP ranks that process the same logical sample.  No-op when CP is
    disabled.
    """
    if not is_cp_enabled():
        return x
    x = x.contiguous()
    dist.broadcast(x, src=_global_rank_of(src_rank), group=_CP_GROUP)
    return x


def _global_rank_of(cp_local_rank: int) -> int:
    """Translate a CP-local rank into the corresponding global rank."""
    if not is_cp_enabled():
        return cp_local_rank
    # ``dist.get_global_rank`` exists in torch >= 2.0.
    if hasattr(dist, "get_global_rank"):
        return dist.get_global_rank(_CP_GROUP, cp_local_rank)
    # Fallback: fetch via group membership (slow path, only used once).
    if "WORLD_SIZE" in os.environ:
        ws = int(os.environ["WORLD_SIZE"])
    else:
        ws = dist.get_world_size()
    for r in range(ws):
        try:
            if dist.get_group_rank(_CP_GROUP, r) == cp_local_rank:
                return r
        except (ValueError, RuntimeError):
            continue
    raise RuntimeError(
        f"Could not resolve global rank for cp_local_rank={cp_local_rank}"
    )
