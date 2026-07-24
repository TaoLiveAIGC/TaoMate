"""
Ulysses Sequence Parallel infrastructure for Stage 3 distillation.

Public API:
- init_sp_mesh / get_sp_group / get_sp_rank / get_sp_world_size / get_dp_rank / get_dp_world_size
- is_sp_enabled
- SeqAllToAllHead, HeadAllToAllSeq           (autograd-friendly all-to-all)
- seq_all_to_all_head, head_all_to_all_seq   (functional wrappers)
- seq_all_to_all_head_many                   (inference-only fused wrapper)
- seq_all_to_all_head_many_async             (inference-only fused async wrapper)
- seq_all_to_all_head_async, head_all_to_all_seq_async (inference-only async wrappers)
- split_sequence, gather_sequence            (entry/exit helpers)
- pad_sequence_to_multiple, unpad_sequence   (length alignment helpers)

Design notes:
- FSDP mesh remains a flat 1D mesh covering the full world; SP only uses a sub-process-group.
- When sp_size == 1 every helper is a no-op so existing call sites are unaffected.
"""

from .mesh import (
    init_sp_mesh,
    get_device_mesh,
    get_sp_group,
    get_sp_rank,
    get_sp_world_size,
    get_dp_group,
    get_dp_rank,
    get_dp_world_size,
    is_sp_enabled,
)
from .all_to_all import (
    SeqAllToAllHead,
    HeadAllToAllSeq,
    seq_all_to_all_head,
    seq_all_to_all_head_many,
    seq_all_to_all_head_many_async,
    head_all_to_all_seq,
    seq_all_to_all_head_async,
    head_all_to_all_seq_async,
)
from .sp_utils import (
    split_sequence,
    gather_sequence,
    pad_sequence_to_multiple,
    unpad_sequence,
)

__all__ = [
    "init_sp_mesh",
    "get_device_mesh",
    "get_sp_group",
    "get_sp_rank",
    "get_sp_world_size",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "is_sp_enabled",
    "SeqAllToAllHead",
    "HeadAllToAllSeq",
    "seq_all_to_all_head",
    "seq_all_to_all_head_many",
    "seq_all_to_all_head_many_async",
    "head_all_to_all_seq",
    "seq_all_to_all_head_async",
    "head_all_to_all_seq_async",
    "split_sequence",
    "gather_sequence",
    "pad_sequence_to_multiple",
    "unpad_sequence",
]
