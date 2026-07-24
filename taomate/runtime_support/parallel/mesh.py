"""
Device mesh management for Ulysses Sequence Parallel.

The 2D mesh shape is (dp_size, sp_size). FSDP keeps using a flat 1D mesh
covering the full world (so cross-node sharding remains intact); SP only
uses the `sp` sub-process-group for all-to-all communication.

When sp_size == 1, the SP path is a no-op and every helper here returns
sentinel values (None group / rank 0 / size 1).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
except ImportError:  # pragma: no cover - older torch fallback
    init_device_mesh = None
    DeviceMesh = None  # type: ignore


# Module-level singletons. Populated by `init_sp_mesh()` and read by helpers.
_DEVICE_MESH: Optional["DeviceMesh"] = None
_SP_SIZE: int = 1
_DP_SIZE: int = 1


def init_sp_mesh(
    world_size: int,
    sp_size: int,
    device_type: str = "cuda",
) -> Tuple[Optional["DeviceMesh"], int, int]:
    """Create a 2D device mesh ``(dp, sp)`` and cache it as module state.

    Parameters
    ----------
    world_size: int
        Total number of ranks (matches ``dist.get_world_size()`` when
        distributed is initialized).
    sp_size: int
        Sequence-parallel group size. ``1`` disables SP entirely.
    device_type: str
        Device kind passed to ``init_device_mesh``.

    Returns
    -------
    (mesh, dp_size, sp_size)
        ``mesh`` is ``None`` when SP is disabled or distributed is not
        initialized; in that case ``(None, world_size, 1)`` is returned so
        the caller can still treat the run as pure DP.
    """
    global _DEVICE_MESH, _SP_SIZE, _DP_SIZE

    if sp_size <= 1:
        _DEVICE_MESH = None
        _SP_SIZE = 1
        _DP_SIZE = max(world_size, 1)
        return None, _DP_SIZE, 1

    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sp_size ({sp_size})."
        )
    if init_device_mesh is None:
        raise RuntimeError(
            "torch.distributed.device_mesh.init_device_mesh is unavailable; "
            "please upgrade PyTorch (>=2.2) to use Sequence Parallel."
        )
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "init_sp_mesh requires torch.distributed to be initialized first."
        )

    dp_size = world_size // sp_size
    mesh = init_device_mesh(device_type, (dp_size, sp_size), mesh_dim_names=("dp", "sp"))

    _DEVICE_MESH = mesh
    _SP_SIZE = sp_size
    _DP_SIZE = dp_size
    return mesh, dp_size, sp_size


def get_device_mesh() -> Optional["DeviceMesh"]:
    """Return the cached 2D ``(dp, sp)`` device mesh, or ``None`` when SP is off."""
    return _DEVICE_MESH


def is_sp_enabled() -> bool:
    """Return True iff SP is active (sp_size > 1 and a mesh has been initialized)."""
    return _DEVICE_MESH is not None and _SP_SIZE > 1


# ---------------------------------------------------------------------------
# Sequence-Parallel sub-group accessors
# ---------------------------------------------------------------------------
def get_sp_group():
    """Return the SP process group, or ``None`` when SP is disabled."""
    if _DEVICE_MESH is None:
        return None
    return _DEVICE_MESH["sp"].get_group()


def get_sp_world_size() -> int:
    return _SP_SIZE


def get_sp_rank() -> int:
    if _DEVICE_MESH is None:
        return 0
    return int(_DEVICE_MESH["sp"].get_local_rank())


# ---------------------------------------------------------------------------
# Data-Parallel sub-group accessors (DataLoader sharding key)
# ---------------------------------------------------------------------------
def get_dp_group():
    if _DEVICE_MESH is None:
        # In pure DP runs (sp_size==1) the "DP group" is the global group.
        if dist.is_available() and dist.is_initialized():
            return dist.group.WORLD
        return None
    return _DEVICE_MESH["dp"].get_group()


def get_dp_world_size() -> int:
    if _DEVICE_MESH is None:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1
    return _DP_SIZE


def get_dp_rank() -> int:
    if _DEVICE_MESH is None:
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
    return int(_DEVICE_MESH["dp"].get_local_rank())
