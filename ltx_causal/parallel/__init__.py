"""
Parallelism utilities for causal LTX models.

Currently provides Ulysses-style Context Parallel (CP) primitives via
:mod:`ltx_causal.parallel.cp_utils`. The defaults are full no-ops so
that ``cp_size == 1`` paths remain bit-identical to the original
single-rank training behaviour.
"""

from ltx_causal.parallel.cp_utils import (
    init_cp,
    is_cp_enabled,
    cp_disabled,
    get_cp_size,
    get_cp_rank,
    get_cp_group,
    get_cp_mesh,
    seq_to_head,
    head_to_seq,
    split_seq,
    gather_seq,
    cp_all_reduce_sum,
    cp_broadcast,
    pad_seq_to_multiple,
    unpad_seq,
)

__all__ = [
    "init_cp",
    "is_cp_enabled",
    "cp_disabled",
    "get_cp_size",
    "get_cp_rank",
    "get_cp_group",
    "get_cp_mesh",
    "seq_to_head",
    "head_to_seq",
    "split_seq",
    "gather_seq",
    "cp_all_reduce_sum",
    "cp_broadcast",
    "pad_seq_to_multiple",
    "unpad_seq",
]
