#!/usr/bin/env python3
"""Resident windowed runtime for TaoMate interactive generation."""

import argparse
import copy
import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchaudio

from ltx_core.types import Audio
from ltx_causal.attention.mask_builder import (
    AVBlock,
    compute_aligned_audio_frames,
    compute_av_blocks,
)
from taomate.runtime_support.models.text_encoder import create_text_encoder_wrapper
from taomate.runtime_support.models.vae import create_vae_wrappers
from taomate.runtime_support.learned_memory import LearnedMemoryState
# Model runtime
from taomate.inference.model_runtime import (
    load_model_generator,
    compute_denoising_sigmas,
    add_noise,
    KVCacheCausalPipeline,
    _encode_image_to_latent,
    _load_or_encode_audio_condition,
    _resolve_cond_inputs,
    _resolve_prefix_ctx_sigma,
    VIDEO_FPS,
)
try:
    from ltx_causal.transformer.kv_cache import dump_inference_profile
    try:
        from ltx_causal.transformer.kv_cache import (
            is_inference_profile_enabled,
        )
    except ImportError:
        from ltx_causal.transformer.kv_cache import (
            is_profiling_enabled as is_inference_profile_enabled,
        )
except ImportError:
    def is_inference_profile_enabled() -> bool:
        return False

    def dump_inference_profile(*args, **kwargs) -> None:
        return None

# ─── Constants ──────────────────────────────────────────────────────────────
FRAMES_PER_WINDOW = 121  # 5 blocks: first block 4 frames + 4 blocks * 3 frames = 16 latent frames → (16-1)*8+1 = 121 pixel frames
LATENT_FRAMES_PER_WINDOW = 16  # = (FRAMES_PER_WINDOW - 1) // 8 + 1
LATENT_FRAMES_PER_GEN = 15  # 5 standard blocks × 3 frames (no first-block special) for windows with prefix
BLOCKS_PER_WINDOW = 5  # first block (4 latent frames) + 4 standard blocks (3 each)
NUM_WINDOWS = 12  # 12 * 5s = 60s
NUM_FRAME_PER_BLOCK = 3
NUM_FRAME_PER_BLOCK_FIRST = 4  # OmniForcing style
AUDIO_FRAMES_PER_BLOCK = 25  # audio frames per standard block


def _validate_audio_prefix_alignment(
    prefix_video_latent: Optional[torch.Tensor],
    prefix_audio_latent: Optional[torch.Tensor],
    *,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
    label: str,
) -> None:
    """Reject A/V prefixes whose block-aligned latent lengths disagree."""
    if prefix_audio_latent is None:
        return
    if prefix_video_latent is None:
        raise ValueError(f"{label}: audio prefix exists without a video prefix")
    video_frames = int(prefix_video_latent.shape[1])
    expected_audio_frames = compute_aligned_audio_frames(
        video_frames,
        num_frame_per_block,
        num_frame_per_block_first,
    )
    actual_audio_frames = int(prefix_audio_latent.shape[1])
    if actual_audio_frames != expected_audio_frames:
        raise ValueError(
            f"{label}: prefix A/V alignment mismatch: {video_frames} video latent "
            f"frames require {expected_audio_frames} audio latent frames, got "
            f"{actual_audio_frames}"
        )


def _block_callback_requests_stop(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(
            result.get("stop")
            or result.get("stop_after_block")
            or result.get("stop_after_current_block")
        )
    return bool(result)


def _read_json_if_exists(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _generated_block_count_from_latents(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor],
    *,
    is_bootstrap_window: bool,
) -> int:
    if audio_latent is not None and int(audio_latent.shape[1]) > 0:
        return max(1, int(audio_latent.shape[1]) // AUDIO_FRAMES_PER_BLOCK)
    frames = int(video_latent.shape[1])
    if frames <= 0:
        return 0
    if is_bootstrap_window:
        if frames <= NUM_FRAME_PER_BLOCK_FIRST:
            return 1
        remaining = max(0, frames - NUM_FRAME_PER_BLOCK_FIRST)
        return 1 + (remaining + NUM_FRAME_PER_BLOCK - 1) // NUM_FRAME_PER_BLOCK
    return (frames + NUM_FRAME_PER_BLOCK - 1) // NUM_FRAME_PER_BLOCK


class _FirstStreamProfiler:
    """Low-overhead timing for the path until *_stream0000.mp4 is published."""

    def __init__(self, *, enabled: bool, device: str) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.start = time.perf_counter()
        self.last = self.start
        self.first_stream_done = False
        self._lock = threading.Lock()

    def mark(self, stage: str, **fields: Any) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        with self._lock:
            delta = now - self.last
            total = now - self.start
            self.last = now
            suffix = ""
            if fields:
                clean_fields = [
                    f"{key}={value}"
                    for key, value in fields.items()
                    if value is not None
                ]
                if clean_fields:
                    suffix = " " + " ".join(clean_fields)
            print(
                f"\n[Profile:first_stream] +{delta:.3f}s "
                f"total={total:.3f}s {stage}{suffix}",
                flush=True,
            )

    def should_profile_stream(self, stream_idx: int) -> bool:
        return self.enabled and int(stream_idx) == 0 and not self.first_stream_done

    def finish_first_stream(self) -> None:
        self.first_stream_done = True


@dataclass
class _StagedLatent:
    tensor: torch.Tensor
    ready_event: Optional[Any] = None
    source_ref: Optional[torch.Tensor] = None


def _maybe_to_device_dtype(
    value: Optional[torch.Tensor],
    *,
    device: str,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if not torch.is_tensor(value):
        return None
    return value.to(device=device, dtype=dtype).contiguous()


def _restore_prefix_renorm_anchor(
    kv_pipeline: KVCacheCausalPipeline,
    sink_video: Optional[torch.Tensor],
) -> bool:
    """Restore block-0 video statistics before an interactive continuation."""
    if (
        not bool(getattr(kv_pipeline, "prefix_renorm", False))
        or sink_video is None
        or not torch.is_tensor(sink_video)
        or sink_video.numel() == 0
    ):
        return False
    mean = sink_video.mean(dim=(1, 3, 4), keepdim=True)
    std = sink_video.std(dim=(1, 3, 4), keepdim=True).clamp_min(1e-6)
    kv_pipeline._anchor_v = (mean.detach(), std.detach())
    return True


def _maybe_to_cpu(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if value is None or not torch.is_tensor(value):
        return None
    return value.detach().to("cpu").contiguous()


def _maybe_reference_stats_to_cpu(
    value: Optional[Tuple[torch.Tensor, ...]],
) -> Optional[Tuple[torch.Tensor, ...]]:
    if not isinstance(value, tuple):
        return None
    return tuple(
        t.detach().to("cpu").contiguous() if torch.is_tensor(t) else t
        for t in value
    )


_INTERACTIVE_PREFIX_STATE_CACHE: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
_INTERACTIVE_PREFIX_STATE_CACHE_MAX = 8
_INTERACTIVE_PREFIX_KV_CACHE: "OrderedDict[Tuple[str, str, str], Dict[str, Any]]" = OrderedDict()
# A retained TP-sharded model KV is much larger than the portable latent state.
# Keep only the two most recently active conversations on GPU; older sessions
# recover through the global-position clean-prefix fallback.
_INTERACTIVE_PREFIX_KV_CACHE_MAX = 2


def _move_learned_memory_state_to_device(
    state: Any,
    *,
    device: str,
) -> Any:
    """Restore every tensor owned by a serialized learned-memory state.

    ``torch.load(..., map_location='cpu')`` is intentional for portable
    conversation state, but the state mixes EMA tensors, frozen anchors and
    color-stat tuples.  Moving only the latent prefix leaves these fields split
    across CPU and CUDA and fails before ``with_conditional_memory`` gets a
    chance to cast its output.
    """
    if not isinstance(state, LearnedMemoryState):
        return state
    tensor_fields = (
        "video",
        "audio",
        "video_anchor",
        "audio_anchor",
        "color_proto",
        "color_anchor_proto",
    )
    tuple_fields = (
        "color_stats",
        "color_anchor_stats",
    )
    for field in tensor_fields:
        value = getattr(state, field, None)
        if torch.is_tensor(value):
            setattr(state, field, value.to(device=device).contiguous())
    for field in tuple_fields:
        value = getattr(state, field, None)
        if isinstance(value, tuple):
            setattr(
                state,
                field,
                tuple(
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item) else item
                    for item in value
                ),
            )
    return state


def _clone_kv_cache(kv_cache: Any) -> Any:
    """Clone a cached prefix KV container without copying GPU tensor payloads.

    ``forward_inference`` appends generated K/V with ``torch.cat`` and assigns a
    new ``LayerKVCache`` back into the per-request ``KVCache.layers`` list.  It
    does not mutate the cached prefix tensors in-place, so a shallow structural
    clone is enough and avoids a multi-GB GPU deepcopy before every interactive
    continuation.
    """
    if kv_cache is None:
        return None
    try:
        from ltx_causal.transformer.kv_cache import KVCache, LayerKVCache
    except Exception:
        return kv_cache
    if not isinstance(kv_cache, KVCache):
        return kv_cache
    cloned = KVCache(
        layers=[
            LayerKVCache(
                video_self_k=layer.video_self_k,
                video_self_v=layer.video_self_v,
                audio_self_k=layer.audio_self_k,
                audio_self_v=layer.audio_self_v,
                a2v_k=layer.a2v_k,
                a2v_v=layer.a2v_v,
                v2a_k=layer.v2a_k,
                v2a_v=layer.v2a_v,
            )
            for layer in kv_cache.layers
        ]
    )
    segments = getattr(kv_cache, "_interactive_segments", None)
    if segments is not None:
        cloned._interactive_segments = [dict(s) for s in segments]
    return cloned


def _kv_audio_token_count(
    block: Any,
    *,
    include_audio_sinks: bool,
    num_audio_sink_tokens: int,
) -> int:
    audio_frames = max(
        0,
        int(getattr(block, "audio_end", 0)) - int(getattr(block, "audio_start", 0)),
    )
    if include_audio_sinks:
        audio_frames += max(0, int(num_audio_sink_tokens))
    return audio_frames


def _make_interactive_kv_segment(
    *,
    kind: str,
    block: Any,
    video_hw: Tuple[int, int],
    include_audio_sinks: bool,
    num_audio_sink_tokens: int,
    prefix_slot: Optional[int] = None,
    window_slot: Optional[int] = None,
    source_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    h, w = int(video_hw[0]), int(video_hw[1])
    video_frames = max(
        0,
        int(getattr(block, "video_end", 0)) - int(getattr(block, "video_start", 0)),
    )
    segment: Dict[str, Any] = {
        "kind": str(kind),
        "block_idx": int(getattr(block, "block_idx", -1)),
        "video_tokens": int(video_frames * h * w),
        "audio_tokens": _kv_audio_token_count(
            block,
            include_audio_sinks=include_audio_sinks,
            num_audio_sink_tokens=num_audio_sink_tokens,
        ),
        "include_audio_sinks": bool(include_audio_sinks),
    }
    if prefix_slot is not None:
        segment["prefix_slot"] = int(prefix_slot)
    if window_slot is not None:
        segment["window_slot"] = int(window_slot)
    if source_prompt is not None:
        segment["source_prompt"] = str(source_prompt)
    return segment


def _attach_interactive_kv_segments(
    kv_cache: Any,
    segments: Optional[List[Dict[str, Any]]],
) -> Any:
    if kv_cache is not None and segments is not None:
        try:
            kv_cache._interactive_segments = [dict(s) for s in segments]
        except Exception:
            pass
    return kv_cache


def _clone_interactive_kv_segments(kv_cache: Any) -> List[Dict[str, Any]]:
    segments = getattr(kv_cache, "_interactive_segments", None)
    if not segments:
        return []
    return [dict(s) for s in segments]


def _kv_segment_offsets(
    segments: List[Dict[str, Any]],
) -> List[Tuple[int, int, int, int]]:
    video_pos = 0
    audio_pos = 0
    offsets: List[Tuple[int, int, int, int]] = []
    for segment in segments:
        video_tokens = max(0, int(segment.get("video_tokens") or 0))
        audio_tokens = max(0, int(segment.get("audio_tokens") or 0))
        offsets.append(
            (
                video_pos,
                video_pos + video_tokens,
                audio_pos,
                audio_pos + audio_tokens,
            )
        )
        video_pos += video_tokens
        audio_pos += audio_tokens
    return offsets


def _slice_kv_tensor(
    tensor: Optional[torch.Tensor],
    spans: List[Tuple[int, int]],
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    pieces = [tensor[:, start:end] for start, end in spans if end > start]
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=1)


def _slice_kv_cache_by_segment_indices(
    kv_cache: Any,
    source_segments: List[Dict[str, Any]],
    segment_indices: List[int],
) -> Optional[Any]:
    if kv_cache is None or not segment_indices:
        return None
    try:
        from ltx_causal.transformer.kv_cache import KVCache, LayerKVCache
    except Exception:
        return None
    if not isinstance(kv_cache, KVCache):
        return None
    offsets = _kv_segment_offsets(source_segments)
    if any(idx < 0 or idx >= len(offsets) for idx in segment_indices):
        return None
    video_spans = [(offsets[idx][0], offsets[idx][1]) for idx in segment_indices]
    audio_spans = [(offsets[idx][2], offsets[idx][3]) for idx in segment_indices]
    if kv_cache.layers:
        first = kv_cache.layers[0]
        if first.video_self_k is not None:
            video_len = int(first.video_self_k.shape[1])
            if any(end > video_len for _, end in video_spans):
                return None
        if first.audio_self_k is not None:
            audio_len = int(first.audio_self_k.shape[1])
            if any(end > audio_len for _, end in audio_spans):
                return None

    new_layers = []
    for layer in kv_cache.layers:
        new_layers.append(
            LayerKVCache(
                video_self_k=_slice_kv_tensor(layer.video_self_k, video_spans),
                video_self_v=_slice_kv_tensor(layer.video_self_v, video_spans),
                audio_self_k=_slice_kv_tensor(layer.audio_self_k, audio_spans),
                audio_self_v=_slice_kv_tensor(layer.audio_self_v, audio_spans),
                a2v_k=_slice_kv_tensor(layer.a2v_k, audio_spans),
                a2v_v=_slice_kv_tensor(layer.a2v_v, audio_spans),
                v2a_k=_slice_kv_tensor(layer.v2a_k, video_spans),
                v2a_v=_slice_kv_tensor(layer.v2a_v, video_spans),
            )
        )
    return KVCache(layers=new_layers)


def _concat_optional_kv_tensor(
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if left is None:
        return right
    if right is None:
        return left
    return torch.cat((left, right), dim=1)


def _concat_kv_caches(left: Any, right: Any) -> Optional[Any]:
    """Append disjoint clean-KV segments without replaying the transformer."""
    if left is None:
        return _clone_kv_cache(right)
    if right is None:
        return _clone_kv_cache(left)
    try:
        from ltx_causal.transformer.kv_cache import KVCache, LayerKVCache
    except Exception:
        return None
    if not isinstance(left, KVCache) or not isinstance(right, KVCache):
        return None
    if len(left.layers) != len(right.layers):
        raise RuntimeError(
            "Cannot append KV caches with different layer counts: "
            f"{len(left.layers)} vs {len(right.layers)}"
        )
    layers = []
    for lhs, rhs in zip(left.layers, right.layers):
        layers.append(
            LayerKVCache(
                video_self_k=_concat_optional_kv_tensor(lhs.video_self_k, rhs.video_self_k),
                video_self_v=_concat_optional_kv_tensor(lhs.video_self_v, rhs.video_self_v),
                audio_self_k=_concat_optional_kv_tensor(lhs.audio_self_k, rhs.audio_self_k),
                audio_self_v=_concat_optional_kv_tensor(lhs.audio_self_v, rhs.audio_self_v),
                a2v_k=_concat_optional_kv_tensor(lhs.a2v_k, rhs.a2v_k),
                a2v_v=_concat_optional_kv_tensor(lhs.a2v_v, rhs.a2v_v),
                v2a_k=_concat_optional_kv_tensor(lhs.v2a_k, rhs.v2a_k),
                v2a_v=_concat_optional_kv_tensor(lhs.v2a_v, rhs.v2a_v),
            )
        )
    return KVCache(layers=layers)


def _reset_speculative_kv_journal(
    journal: Optional[Dict[str, Any]],
    kv_cache: Any,
) -> None:
    if journal is None or kv_cache is None:
        return
    journal["kv_cache"] = _clone_kv_cache(kv_cache)
    journal["segments"] = _clone_interactive_kv_segments(kv_cache)
    _attach_interactive_kv_segments(journal["kv_cache"], journal["segments"])


def _append_latest_clean_kv_to_journal(
    journal: Optional[Dict[str, Any]],
    kv_cache: Any,
    cache_segments: List[Dict[str, Any]],
) -> None:
    """Journal one clean block before the active cache is pruned.

    The journal is not passed to attention. It only preserves rollback points
    for asynchronous ASR, while generation continues to use sink + recent KV.
    """
    if journal is None or kv_cache is None or not cache_segments:
        return
    latest_segment = dict(cache_segments[-1])
    block_idx = int(latest_segment.get("block_idx", -1))
    existing_segments = list(journal.get("segments") or [])
    if any(int(segment.get("block_idx", -2)) == block_idx for segment in existing_segments):
        return
    latest_cache = _slice_kv_cache_by_segment_indices(
        kv_cache,
        cache_segments,
        [len(cache_segments) - 1],
    )
    if latest_cache is None:
        raise RuntimeError(f"Unable to journal clean KV for block {block_idx}")
    merged = _concat_kv_caches(journal.get("kv_cache"), latest_cache)
    if merged is None:
        raise RuntimeError(f"Unable to append clean KV journal block {block_idx}")
    existing_segments.append(latest_segment)
    journal["kv_cache"] = _attach_interactive_kv_segments(merged, existing_segments)
    journal["segments"] = existing_segments


def _commit_speculative_kv_journal(
    journal: Optional[Dict[str, Any]],
    *,
    absolute_block_limit: int,
    recent_blocks: int,
) -> Optional[Any]:
    """Commit only blocks strictly before ``absolute_block_limit``."""
    if journal is None:
        return None
    kv_cache = journal.get("kv_cache")
    segments = list(journal.get("segments") or [])
    if kv_cache is None or not segments:
        return None
    keep_indices = [
        idx
        for idx, segment in enumerate(segments)
        if int(segment.get("block_idx", -1)) < int(absolute_block_limit)
    ]
    selected = _slice_kv_cache_by_segment_indices(
        kv_cache,
        segments,
        keep_indices,
    )
    if selected is None:
        return None
    selected_segments = [segments[idx] for idx in keep_indices]
    _attach_interactive_kv_segments(selected, selected_segments)
    return _prune_clean_kv_to_sink_recent(
        selected,
        recent_blocks=recent_blocks,
    )


def _prune_clean_kv_to_sink_recent(
    kv_cache: Any,
    *,
    recent_blocks: int,
) -> Optional[Any]:
    """Keep block 0 plus the newest committed non-sink blocks.

    Segment metadata is recorded only after a sigma=0 clean commit. Slicing by
    those segments therefore never promotes a denoising-step cache entry into
    long-lived context.
    """
    if kv_cache is None:
        return None
    segments = _clone_interactive_kv_segments(kv_cache)
    if not segments:
        return None

    sink_idx = next(
        (
            idx
            for idx, segment in enumerate(segments)
            if int(segment.get("block_idx", -1)) == 0
        ),
        None,
    )
    non_sink_indices = [
        idx for idx in range(len(segments)) if idx != sink_idx
    ]
    keep_recent = max(0, int(recent_blocks))
    recent_indices = (
        non_sink_indices[-keep_recent:] if keep_recent > 0 else []
    )
    keep_indices = (
        ([sink_idx] if sink_idx is not None else []) + recent_indices
    )
    # Preserve chronological physical order even if metadata arrived through a
    # cache reuse path rather than direct generation.
    keep_indices = sorted(set(keep_indices))
    if keep_indices == list(range(len(segments))):
        return kv_cache
    pruned = _slice_kv_cache_by_segment_indices(
        kv_cache,
        segments,
        keep_indices,
    )
    if pruned is None:
        return None
    return _attach_interactive_kv_segments(
        pruned,
        [segments[idx] for idx in keep_indices],
    )


def _clone_learned_memory_state(
    state: Optional[LearnedMemoryState],
) -> Optional[LearnedMemoryState]:
    return copy.deepcopy(state) if state is not None else None


def _generated_block_slices(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor],
    *,
    first_global_block_idx: int,
    block_count: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]]:
    records: List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]] = []
    video_pos = 0
    audio_pos = 0
    for offset in range(max(0, int(block_count))):
        block_idx = int(first_global_block_idx) + offset
        block = _global_block_at_index(
            block_idx=block_idx,
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
        )
        video_end = video_pos + int(block.video_frames)
        audio_end = audio_pos + int(block.audio_frames)
        block_video = video_latent[:, video_pos:video_end]
        block_audio = (
            audio_latent[:, audio_pos:audio_end]
            if audio_latent is not None else None
        )
        if int(block_video.shape[1]) != int(block.video_frames):
            raise RuntimeError(
                f"Generated video latent ended inside block {block_idx}: "
                f"expected {block.video_frames}, got {block_video.shape[1]}"
            )
        if block_audio is not None and int(block_audio.shape[1]) != int(block.audio_frames):
            raise RuntimeError(
                f"Generated audio latent ended inside block {block_idx}: "
                f"expected {block.audio_frames}, got {block_audio.shape[1]}"
            )
        records.append((block_idx, block_video, block_audio))
        video_pos = video_end
        audio_pos = audio_end
    if video_pos != int(video_latent.shape[1]):
        raise RuntimeError(
            "Generated video latent has trailing frames outside complete blocks: "
            f"consumed={video_pos}, total={video_latent.shape[1]}"
        )
    if audio_latent is not None and audio_pos != int(audio_latent.shape[1]):
        raise RuntimeError(
            "Generated audio latent has trailing frames outside complete blocks: "
            f"consumed={audio_pos}, total={audio_latent.shape[1]}"
        )
    return records


def _trim_generated_latents_to_blocks(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor],
    *,
    first_global_block_idx: int,
    generated_block_count: int,
    keep_block_count: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]],
]:
    records = _generated_block_slices(
        video_latent,
        audio_latent,
        first_global_block_idx=first_global_block_idx,
        block_count=generated_block_count,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    keep = max(0, min(int(keep_block_count), len(records)))
    kept_records = records[:keep]
    kept_video_frames = sum(int(record[1].shape[1]) for record in kept_records)
    kept_audio_frames = sum(
        int(record[2].shape[1]) if record[2] is not None else 0
        for record in kept_records
    )
    trimmed_video = video_latent[:, :kept_video_frames].contiguous()
    trimmed_audio = (
        audio_latent[:, :kept_audio_frames].contiguous()
        if audio_latent is not None else None
    )
    return trimmed_video, trimmed_audio, records


def _replay_learned_memory_blocks(
    state: Optional[LearnedMemoryState],
    records: List[Tuple[int, torch.Tensor, Optional[torch.Tensor]]],
) -> Optional[LearnedMemoryState]:
    if state is None:
        return None
    for block_idx, block_video, block_audio in records:
        if int(block_idx) == 0:
            state.set_reference(block_video, block_audio)
        state.update(block_video, block_audio)
    return state


def _rebuild_committed_interactive_prefix(
    *,
    initial_prefix_video: Optional[torch.Tensor],
    initial_prefix_audio: Optional[torch.Tensor],
    initial_decode_prefix_video: Optional[torch.Tensor],
    initial_prefix_source_prompts: Optional[List[str]],
    initial_sink_video: Optional[torch.Tensor],
    initial_sink_audio: Optional[torch.Tensor],
    initial_sink_source_prompt: Optional[str],
    request_start_block_offset: int,
    generated_records: List[Dict[str, Any]],
    committed_generated_blocks: int,
    recent_blocks: int,
    decode_context_latents: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> Dict[str, Any]:
    """Rebuild portable prefix state from the same ASR commit point as KV."""
    committed_records = list(generated_records[:max(0, int(committed_generated_blocks))])
    absolute_limit = int(request_start_block_offset) + len(committed_records)
    recent_candidates: List[Dict[str, Any]] = []

    sink_video = initial_sink_video
    sink_audio = initial_sink_audio
    sink_prompt = initial_sink_source_prompt
    if initial_prefix_video is not None:
        sink_frames = int(num_frame_per_block_first)
        if sink_video is None:
            sink_video = initial_prefix_video[:, :sink_frames]
        sink_audio_frames = compute_aligned_audio_frames(
            sink_frames,
            num_frame_per_block,
            num_frame_per_block_first,
        )
        if sink_audio is None and initial_prefix_audio is not None:
            sink_audio = initial_prefix_audio[:, :sink_audio_frames]
        source_prompts = list(initial_prefix_source_prompts or [])
        if sink_prompt is None and source_prompts:
            sink_prompt = source_prompts[0]

        recent_video = initial_prefix_video[:, sink_frames:]
        if int(recent_video.shape[1]) % int(num_frame_per_block) != 0:
            raise RuntimeError(
                "Initial compact prefix does not contain complete recent blocks: "
                f"frames={recent_video.shape[1]}"
            )
        initial_recent_count = int(recent_video.shape[1]) // int(num_frame_per_block)
        recent_audio = (
            initial_prefix_audio[:, sink_audio_frames:]
            if initial_prefix_audio is not None else None
        )
        recent_prompts = source_prompts[-initial_recent_count:] if initial_recent_count else []
        first_recent_idx = max(1, int(request_start_block_offset) - initial_recent_count)
        for offset in range(initial_recent_count):
            v_start = offset * int(num_frame_per_block)
            v_end = v_start + int(num_frame_per_block)
            a_start = offset * AUDIO_FRAMES_PER_BLOCK
            a_end = a_start + AUDIO_FRAMES_PER_BLOCK
            recent_candidates.append(
                {
                    "block_idx": first_recent_idx + offset,
                    "video": recent_video[:, v_start:v_end],
                    "audio": (
                        recent_audio[:, a_start:a_end]
                        if recent_audio is not None else None
                    ),
                    "prompt": (
                        recent_prompts[offset]
                        if offset < len(recent_prompts) else sink_prompt or ""
                    ),
                }
            )

    for record in committed_records:
        if int(record["block_idx"]) == 0:
            sink_video = record["video"]
            sink_audio = record.get("audio")
            sink_prompt = str(record.get("prompt") or "")
        else:
            recent_candidates.append(record)

    if sink_video is None:
        return {
            "prefix_video_latent": initial_prefix_video,
            "prefix_audio_latent": initial_prefix_audio,
            "decode_prefix_video_latent": initial_decode_prefix_video,
            "prefix_source_prompts": initial_prefix_source_prompts,
            "sink_video_latent": initial_sink_video,
            "sink_audio_latent": initial_sink_audio,
            "sink_source_prompt": initial_sink_source_prompt,
        }

    recent_candidates = [
        record
        for record in recent_candidates
        if 0 < int(record["block_idx"]) < absolute_limit
    ]
    recent_candidates.sort(key=lambda record: int(record["block_idx"]))
    keep_recent = max(0, int(recent_blocks))
    if keep_recent > 0:
        recent_candidates = recent_candidates[-keep_recent:]
    else:
        recent_candidates = []

    prefix_video = torch.cat(
        [sink_video, *(record["video"] for record in recent_candidates)],
        dim=1,
    ).contiguous()
    prefix_audio = None
    if sink_audio is not None and all(record.get("audio") is not None for record in recent_candidates):
        prefix_audio = torch.cat(
            [sink_audio, *(record["audio"] for record in recent_candidates)],
            dim=1,
        ).contiguous()
    prefix_prompts = [
        sink_prompt or "",
        *(str(record.get("prompt") or sink_prompt or "") for record in recent_candidates),
    ]

    accepted_video = (
        torch.cat([record["video"] for record in committed_records], dim=1)
        if committed_records else None
    )
    decode_prefix = initial_decode_prefix_video
    if accepted_video is not None:
        decode_prefix = (
            accepted_video
            if decode_prefix is None
            else torch.cat([decode_prefix, accepted_video], dim=1)
        )
    max_decode = max(0, int(decode_context_latents))
    if decode_prefix is not None and max_decode > 0:
        decode_prefix = decode_prefix[:, -max_decode:].contiguous()
    elif max_decode <= 0:
        decode_prefix = None

    return {
        "prefix_video_latent": prefix_video,
        "prefix_audio_latent": prefix_audio,
        "decode_prefix_video_latent": decode_prefix,
        "prefix_source_prompts": prefix_prompts,
        "sink_video_latent": sink_video,
        "sink_audio_latent": sink_audio,
        "sink_source_prompt": sink_prompt,
    }


def _global_generation_blocks(
    *,
    global_window_index: int,
    block_count: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> List[AVBlock]:
    """Return globally positioned blocks for one continuation chunk."""
    first_block_idx = max(0, int(global_window_index)) * int(block_count)
    return _global_generation_blocks_from_offset(
        first_block_idx=first_block_idx,
        block_count=block_count,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )


def _global_generation_blocks_from_offset(
    *,
    first_block_idx: int,
    block_count: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> List[AVBlock]:
    """Return globally positioned blocks from an exact absolute offset.

    Interactive ASR can stop a request after any block, so a request/window
    count is not sufficient to recover the next RoPE position.  Keeping the
    absolute generated-block count avoids silently jumping by five blocks on a
    short turn.
    """
    first_block_idx = max(0, int(first_block_idx))
    end_block_idx = first_block_idx + int(block_count)
    if first_block_idx <= 0:
        raise ValueError(
            "Cross-chunk KV continuation requires a non-bootstrap window"
        )
    total_video_frames = (
        int(num_frame_per_block_first)
        + (end_block_idx - 1) * int(num_frame_per_block)
    )
    blocks = compute_av_blocks(
        total_video_latent_frames=total_video_frames,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    selected = blocks[first_block_idx:end_block_idx]
    if len(selected) != int(block_count):
        raise RuntimeError(
            "Failed to construct continuous global block positions: "
            f"first_block={first_block_idx}, expected={block_count}, "
            f"got={len(selected)}"
        )
    return selected


def _global_block_at_index(
    *,
    block_idx: int,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
) -> AVBlock:
    """Return one globally positioned block from its absolute block index."""
    block_idx = int(block_idx)
    if block_idx < 0:
        raise ValueError(f"block_idx must be >= 0, got {block_idx}")
    total_video_frames = (
        int(num_frame_per_block_first)
        + block_idx * int(num_frame_per_block)
    )
    blocks = compute_av_blocks(
        total_video_latent_frames=total_video_frames,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    if block_idx >= len(blocks):
        raise RuntimeError(
            "Failed to reconstruct global prefix block: "
            f"block_idx={block_idx}, blocks={len(blocks)}"
        )
    return blocks[block_idx]


def _refresh_retained_modal_kv(
    *,
    kv_pipeline: KVCacheCausalPipeline,
    persistent_kv_cache: Any,
    conditional_dict: Dict[str, Any],
    prefix_video_latent: torch.Tensor,
    prefix_audio_latent: Optional[torch.Tensor],
    next_video_shape: Tuple[int, ...],
    next_audio_shape: Optional[Tuple[int, ...]],
    num_frame_per_block: int,
    num_frame_per_block_first: int,
    num_audio_sink_tokens: int,
    learned_memory_state: Optional[LearnedMemoryState],
) -> Any:
    """Keep persistent video self-KV while rebuilding prompt-sensitive KV.

    The benchmark prompt changes every five seconds and includes both the
    already-spoken prefix and the next utterance. Reusing audio/cross-modal KV
    from the previous prompt mixes incompatible conditioning states. This
    helper re-prefills the exact retained latent segments with the current
    prompt and their original global positions, then replaces every cache field
    except video self-attention K/V.
    """
    try:
        from ltx_causal.transformer.kv_cache import KVCache
    except Exception as exc:
        raise RuntimeError("Selective KV rebuild requires KVCache support") from exc
    if not isinstance(persistent_kv_cache, KVCache):
        raise TypeError(
            "Selective KV rebuild expected KVCache, got "
            f"{type(persistent_kv_cache).__name__}"
        )
    segments = _clone_interactive_kv_segments(persistent_kv_cache)
    if not segments:
        raise RuntimeError("Selective KV rebuild requires committed segment metadata")

    F_prefix = int(prefix_video_latent.shape[1])
    F_total = F_prefix + int(next_video_shape[1])
    local_blocks = compute_av_blocks(
        total_video_latent_frames=F_total,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    local_prefix_blocks = [
        block for block in local_blocks if int(block.video_end) <= F_prefix
    ]
    if not local_prefix_blocks:
        raise RuntimeError("Selective KV rebuild received an empty latent prefix")

    has_sink = int(segments[0].get("block_idx", -1)) == 0
    warm_count = len(segments) - (1 if has_sink else 0)
    warm_candidates = local_prefix_blocks[1:] if has_sink else local_prefix_blocks
    if warm_count > len(warm_candidates):
        raise ValueError(
            "Selective KV rebuild does not have enough retained prefix latents: "
            f"segments={len(segments)}, warm_needed={warm_count}, "
            f"warm_available={len(warm_candidates)}. Reduce max_prefix_blocks "
            "or retain a larger latent prefix."
        )
    selected_local_blocks = (
        [local_prefix_blocks[0], *warm_candidates[-warm_count:]]
        if has_sink and warm_count > 0
        else [local_prefix_blocks[0]]
        if has_sink
        else warm_candidates[-warm_count:]
        if warm_count > 0
        else []
    )
    if len(selected_local_blocks) != len(segments):
        raise RuntimeError(
            "Selective KV latent/segment layout mismatch: "
            f"local={len(selected_local_blocks)}, segments={len(segments)}"
        )

    device_t = next(kv_pipeline.generator.parameters()).device
    dtype_t = next(kv_pipeline.generator.parameters()).dtype
    B = int(next_video_shape[0])
    video_hw = (int(next_video_shape[-2]), int(next_video_shape[-1]))
    F_audio_prefix = (
        int(prefix_audio_latent.shape[1])
        if prefix_audio_latent is not None else 0
    )
    _validate_audio_prefix_alignment(
        prefix_video_latent,
        prefix_audio_latent,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
        label="retained cross-chunk KV refresh",
    )

    rebuilt_cache = None
    with torch.no_grad():
        for local_block, segment in zip(selected_local_blocks, segments):
            global_block = _global_block_at_index(
                block_idx=int(segment.get("block_idx", -1)),
                num_frame_per_block=num_frame_per_block,
                num_frame_per_block_first=num_frame_per_block_first,
            )
            local_video_frames = int(local_block.video_end) - int(local_block.video_start)
            global_video_frames = int(global_block.video_end) - int(global_block.video_start)
            local_audio_frames = int(local_block.audio_end) - int(local_block.audio_start)
            global_audio_frames = int(global_block.audio_end) - int(global_block.audio_start)
            include_audio_sinks = bool(segment.get("include_audio_sinks", False))
            expected_video_tokens = local_video_frames * video_hw[0] * video_hw[1]
            expected_audio_tokens = local_audio_frames + (
                int(num_audio_sink_tokens) if include_audio_sinks else 0
            )
            if (
                local_video_frames != global_video_frames
                or local_audio_frames != global_audio_frames
                or int(segment.get("video_tokens", -1)) != expected_video_tokens
                or int(segment.get("audio_tokens", -1)) != expected_audio_tokens
            ):
                raise RuntimeError(
                    "Selective KV segment shape mismatch for global block "
                    f"{global_block.block_idx}: segment={segment}, "
                    f"local_video_frames={local_video_frames}, "
                    f"local_audio_frames={local_audio_frames}"
                )

            vb = prefix_video_latent[
                :, int(local_block.video_start):int(local_block.video_end)
            ].to(device=device_t, dtype=dtype_t)
            ab = None
            if prefix_audio_latent is not None:
                audio_start = int(local_block.audio_start)
                audio_end = int(local_block.audio_end)
                if audio_end > F_audio_prefix:
                    raise RuntimeError(
                        "Selective KV audio prefix slice exceeds available latent: "
                        f"end={audio_end}, available={F_audio_prefix}"
                    )
                ab = prefix_audio_latent[:, audio_start:audio_end].to(
                    device=device_t, dtype=dtype_t,
                )
            vs = torch.zeros((B, vb.shape[1]), device=device_t, dtype=dtype_t)
            audio_ts = (
                torch.zeros((B, ab.shape[1]), device=device_t, dtype=dtype_t)
                if ab is not None else None
            )
            block_conditional_dict = conditional_dict
            if learned_memory_state is not None:
                block_conditional_dict = learned_memory_state.with_conditional_memory(
                    block_conditional_dict,
                    device=device_t,
                    dtype=dtype_t,
                )
            _, _, rebuilt_cache = kv_pipeline.generator.model.forward_inference(
                video_latent=vb,
                audio_latent=ab,
                timesteps=vs,
                audio_timesteps=audio_ts,
                video_context=block_conditional_dict["video_context"],
                audio_context=block_conditional_dict["audio_context"],
                video_context_mask=block_conditional_dict.get("video_context_mask"),
                audio_context_mask=block_conditional_dict.get("audio_context_mask"),
                learned_memory_video=block_conditional_dict.get("learned_memory_video"),
                learned_memory_audio=block_conditional_dict.get("learned_memory_audio"),
                learned_memory_color=block_conditional_dict.get("learned_memory_color"),
                kv_cache=rebuilt_cache,
                video_start_frame=int(global_block.video_start),
                audio_start_frame=int(global_block.audio_start),
                include_audio_sinks=include_audio_sinks,
                pyramid_policy=kv_pipeline.pyramid_policy,
                kv_cache_only=True,
            )

    if rebuilt_cache is None or len(rebuilt_cache.layers) != len(persistent_kv_cache.layers):
        raise RuntimeError("Selective KV rebuild produced an invalid cache")
    for persistent_layer, rebuilt_layer in zip(
        persistent_kv_cache.layers, rebuilt_cache.layers,
    ):
        persistent_layer.audio_self_k = rebuilt_layer.audio_self_k
        persistent_layer.audio_self_v = rebuilt_layer.audio_self_v
        persistent_layer.a2v_k = rebuilt_layer.a2v_k
        persistent_layer.a2v_v = rebuilt_layer.a2v_v
        persistent_layer.v2a_k = rebuilt_layer.v2a_k
        persistent_layer.v2a_v = rebuilt_layer.v2a_v
    _attach_interactive_kv_segments(persistent_kv_cache, segments)
    return persistent_kv_cache


def _interactive_prefix_state_cache_key(
    path: str,
    *,
    device: str,
    dtype: torch.dtype,
) -> Tuple[str, str, str]:
    return (os.path.abspath(path), str(device), str(dtype))


def _invalidate_interactive_prefix_state_cache(path: Optional[str]) -> None:
    if not path:
        return
    absolute_path = os.path.abspath(path)
    stale_keys = [
        cache_key
        for cache_key in _INTERACTIVE_PREFIX_STATE_CACHE
        if cache_key[0] == absolute_path
    ]
    for cache_key in stale_keys:
        _INTERACTIVE_PREFIX_STATE_CACHE.pop(cache_key, None)


def _put_interactive_prefix_state_cache(
    path: Optional[str],
    payload: Dict[str, Any],
    *,
    device: str,
    dtype: torch.dtype,
) -> None:
    if not path:
        return
    state = dict(payload)
    for key in (
        "prefix_video_latent",
        "prefix_audio_latent",
        "decode_prefix_video_latent",
        "sink_video_latent",
        "sink_audio_latent",
    ):
        state[key] = _maybe_to_device_dtype(state.get(key), device=device, dtype=dtype)
    state["learned_memory_state"] = _move_learned_memory_state_to_device(
        state.get("learned_memory_state"),
        device=device,
    )
    cache_key = _interactive_prefix_state_cache_key(path, device=device, dtype=dtype)
    _INTERACTIVE_PREFIX_STATE_CACHE[cache_key] = state
    _INTERACTIVE_PREFIX_STATE_CACHE.move_to_end(cache_key)
    while len(_INTERACTIVE_PREFIX_STATE_CACHE) > _INTERACTIVE_PREFIX_STATE_CACHE_MAX:
        _INTERACTIVE_PREFIX_STATE_CACHE.popitem(last=False)


def _notify_avatar_worker_interactive_ready(
    *,
    prefix_state_out: Optional[str],
    output_dir: Optional[str],
    window_count: int,
) -> None:
    """Tell the interactive-avatar server that continuation state is usable.

    The persistent worker may still be finishing stream chunk muxing, but the
    next-turn prefix state is already cached/scheduled.  This lets the web
    service update conversation metadata and unblock the UI without changing
    generation math or video quality.
    """
    status_path = os.environ.get("AVATAR_WORKER_STATUS_FILE", "").strip()
    if not status_path:
        return
    try:
        status: Dict[str, Any] = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    status.update(loaded)
            except json.JSONDecodeError:
                pass
        status.update(
            {
                "status": "running",
                "phase": "interactive_prefix_state_ready",
                "time": time.time(),
                "prefix_state_out": prefix_state_out,
                "output_dir": output_dir,
                "window_count": int(window_count),
            }
        )
        tmp_path = f"{status_path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, status_path)
    except Exception as exc:
        print(f" [interactive ready status err: {exc}]", end="", flush=True)


def _put_interactive_prefix_kv_cache(
    path: Optional[str],
    *,
    kv_cache: Any,
    prefix_source_prompts: Optional[List[str]],
    prefix_video_shape: Optional[Tuple[int, ...]],
    prefix_audio_shape: Optional[Tuple[int, ...]],
    window_count: int,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Store a GPU KV cache for the exact next-prefix layout in this process."""
    if not path or kv_cache is None or not prefix_source_prompts:
        return
    cache_key = _interactive_prefix_state_cache_key(path, device=device, dtype=dtype)
    _INTERACTIVE_PREFIX_KV_CACHE[cache_key] = {
        "kv_cache": kv_cache,
        "prefix_source_prompts": list(prefix_source_prompts),
        "prefix_video_shape": tuple(prefix_video_shape) if prefix_video_shape else None,
        "prefix_audio_shape": tuple(prefix_audio_shape) if prefix_audio_shape else None,
        "window_count": int(window_count),
        "updated_at": time.time(),
    }
    _INTERACTIVE_PREFIX_KV_CACHE.move_to_end(cache_key)
    while len(_INTERACTIVE_PREFIX_KV_CACHE) > _INTERACTIVE_PREFIX_KV_CACHE_MAX:
        _INTERACTIVE_PREFIX_KV_CACHE.popitem(last=False)


def _get_interactive_prefix_kv_cache(
    path: Optional[str],
    *,
    prefix_source_prompts: Optional[List[str]],
    prefix_video_shape: Optional[Tuple[int, ...]],
    prefix_audio_shape: Optional[Tuple[int, ...]],
    window_count: int,
    device: str,
    dtype: torch.dtype,
) -> Optional[Any]:
    """Return a session KV cache only when the canonical prefix layout matches."""
    if not path or not prefix_source_prompts:
        return None
    cache_key = _interactive_prefix_state_cache_key(path, device=device, dtype=dtype)
    cached = _INTERACTIVE_PREFIX_KV_CACHE.get(cache_key)
    if not cached:
        return None
    cached_window_count = int(cached.get("window_count") or -1)
    if cached_window_count >= 0 and cached_window_count != int(window_count):
        return None
    if list(cached.get("prefix_source_prompts") or []) != list(prefix_source_prompts):
        return None
    if tuple(cached.get("prefix_video_shape") or ()) != tuple(prefix_video_shape or ()):
        return None
    cached_audio_shape = cached.get("prefix_audio_shape")
    if cached_audio_shape is None:
        if prefix_audio_shape is not None:
            return None
    elif tuple(cached_audio_shape) != tuple(prefix_audio_shape or ()):
        return None
    _INTERACTIVE_PREFIX_KV_CACHE.move_to_end(cache_key)
    return _clone_kv_cache(cached.get("kv_cache"))


def _interactive_selected_prefix_prompt_indices(
    prefix_source_prompts: Optional[List[str]],
    *,
    block0_sink_enabled: bool,
    max_prefix_blocks: Optional[int],
) -> List[int]:
    if not prefix_source_prompts:
        return []
    count = len(prefix_source_prompts)
    if max_prefix_blocks is None:
        return list(range(count))
    max_warm_prefix = max(0, int(max_prefix_blocks))
    if block0_sink_enabled and count > 0:
        warm_indices = list(range(1, count))
        if max_warm_prefix > 0:
            warm_indices = warm_indices[-max_warm_prefix:]
        else:
            warm_indices = []
        return [0, *warm_indices]
    if max_warm_prefix <= 0:
        return []
    return list(range(max(0, count - max_warm_prefix), count))


def _interactive_selected_prefix_prompts_for_kv_config(
    prefix_source_prompts: Optional[List[str]],
    *,
    block0_sink_enabled: bool,
    max_prefix_blocks: Optional[int],
) -> Optional[List[str]]:
    if not prefix_source_prompts:
        return None
    indices = _interactive_selected_prefix_prompt_indices(
        prefix_source_prompts,
        block0_sink_enabled=block0_sink_enabled,
        max_prefix_blocks=max_prefix_blocks,
    )
    return [prefix_source_prompts[i] for i in indices]


def _interactive_selected_prefix_prompts_for_kv(
    prefix_source_prompts: Optional[List[str]],
    kv_pipeline: KVCacheCausalPipeline,
) -> Optional[List[str]]:
    return _interactive_selected_prefix_prompts_for_kv_config(
        prefix_source_prompts,
        block0_sink_enabled=bool(getattr(kv_pipeline, "block0_sink_enabled", False)),
        max_prefix_blocks=getattr(kv_pipeline, "_max_prefix_blocks", None),
    )


def _can_prepare_interactive_prefix_kv(kv_pipeline: KVCacheCausalPipeline) -> bool:
    if not bool(getattr(kv_pipeline, "block0_sink_enabled", False)):
        return False
    # Cached prefix K/V must represent the exact clean prefix layout.  Runtime
    # context noise intentionally perturbs prefix prefill, so keep that path
    # uncached.
    context_noise = float(getattr(kv_pipeline, "context_noise", 0.0) or 0.0)
    context_noise_max = float(getattr(kv_pipeline, "context_noise_max", 0.0) or 0.0)
    return context_noise == 0.0 and context_noise_max == 0.0


def _load_interactive_prefix_state(
    path: Optional[str],
    *,
    device: str,
    dtype: torch.dtype,
) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    cache_key = _interactive_prefix_state_cache_key(path, device=device, dtype=dtype)
    cached = _INTERACTIVE_PREFIX_STATE_CACHE.get(cache_key)
    if cached is not None:
        _INTERACTIVE_PREFIX_STATE_CACHE.move_to_end(cache_key)
        return dict(cached)
    if not os.path.exists(path):
        return None
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"interactive prefix state must be a dict: {path}")

    state = dict(raw)
    for key in (
        "prefix_video_latent",
        "prefix_audio_latent",
        "decode_prefix_video_latent",
        "sink_video_latent",
        "sink_audio_latent",
    ):
        state[key] = _maybe_to_device_dtype(state.get(key), device=device, dtype=dtype)
    state["learned_memory_state"] = _move_learned_memory_state_to_device(
        state.get("learned_memory_state"),
        device=device,
    )
    _put_interactive_prefix_state_cache(path, state, device=device, dtype=dtype)
    return state


def _save_interactive_prefix_state(
    path: Optional[str],
    payload: Dict[str, Any],
) -> bool:
    if not path:
        return False
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    _invalidate_interactive_prefix_state_cache(path)
    return True


def _build_interactive_prefix_state_payload(
    *,
    case_id: str,
    window_count: int,
    prefix_video_latent: Optional[torch.Tensor],
    prefix_audio_latent: Optional[torch.Tensor],
    decode_prefix_video_latent: Optional[torch.Tensor],
    prefix_source_prompts: Optional[List[str]],
    sink_video_latent: Optional[torch.Tensor],
    sink_audio_latent: Optional[torch.Tensor],
    sink_source_prompt: Optional[str],
    learned_memory_state: Any,
    cpu: bool,
    block_count: Optional[int] = None,
) -> Dict[str, Any]:
    def _tensor(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if cpu:
            return _maybe_to_cpu(value)
        if value is None or not torch.is_tensor(value):
            return None
        return value.detach().contiguous()

    return {
        "version": 2,
        "case_id": case_id,
        "updated_at": time.time(),
        "window_count": int(window_count),
        "block_count": (
            int(block_count)
            if block_count is not None
            else int(window_count) * BLOCKS_PER_WINDOW
        ),
        "prefix_video_latent": _tensor(prefix_video_latent),
        "prefix_audio_latent": _tensor(prefix_audio_latent),
        "decode_prefix_video_latent": _tensor(decode_prefix_video_latent),
        "prefix_source_prompts": (
            list(prefix_source_prompts) if prefix_source_prompts is not None else None
        ),
        "sink_video_latent": _tensor(sink_video_latent),
        "sink_audio_latent": _tensor(sink_audio_latent),
        "sink_source_prompt": sink_source_prompt,
        "learned_memory_state": learned_memory_state,
    }
KVStats = Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor]]


def compute_latent_shapes(
    num_frames: int,
    video_height: int,
    video_width: int,
    batch_size: int = 1,
    latent_channels: int = 128,
    vae_temporal_compression: int = 8,
    vae_spatial_compression: int = 32,
    video_fps: float = 24.0,
    audio_sample_rate: int = 16000,
    audio_hop_length: int = 160,
    audio_latent_downsample: int = 4,
) -> Tuple[list, list]:
    """Compute LTX video/audio latent shapes without importing training code."""
    if (num_frames - 1) % vae_temporal_compression != 0:
        raise ValueError(
            f"num_frames must be 1 + {vae_temporal_compression}*k, got {num_frames}"
        )
    latent_frames = 1 + (num_frames - 1) // vae_temporal_compression
    latent_h = video_height // vae_spatial_compression
    latent_w = video_width // vae_spatial_compression
    video_duration = float(num_frames) / float(video_fps)
    audio_latent_fps = (
        float(audio_sample_rate)
        / float(audio_hop_length)
        / float(audio_latent_downsample)
    )
    audio_frames = round(video_duration * audio_latent_fps)
    return (
        [batch_size, latent_frames, latent_channels, latent_h, latent_w],
        [batch_size, audio_frames, latent_channels],
    )


def _center_crop_video_pixels(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Center-frame decoded RGB pixels without changing generation latents.

    When the requested output exceeds the decoded canvas in one dimension, crop
    the source to the target aspect ratio first and resize that crop. This keeps
    portrait delivery on the well-tested landscape generation canvas while the
    latent layout, RoPE positions, and persistent KV cache stay unchanged.
    """
    target_height = int(target_height or 0)
    target_width = int(target_width or 0)
    if target_height <= 0 and target_width <= 0:
        return video
    if video.ndim != 4:
        raise ValueError(f"Expected video [F,H,W,C] for crop, got {tuple(video.shape)}")
    height = int(video.shape[1])
    width = int(video.shape[2])
    target_height = target_height or height
    target_width = target_width or width
    if target_height == height and target_width == width:
        return video

    if target_height <= height and target_width <= width:
        crop_height = target_height
        crop_width = target_width
    else:
        source_aspect = float(width) / float(height)
        target_aspect = float(target_width) / float(target_height)
        if source_aspect > target_aspect:
            crop_height = height
            crop_width = max(1, min(width, int(round(height * target_aspect))))
        else:
            crop_width = width
            crop_height = max(1, min(height, int(round(width / target_aspect))))

    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    cropped = video[:, top:top + crop_height, left:left + crop_width, :].contiguous()
    if crop_height == target_height and crop_width == target_width:
        return cropped

    resized = F.interpolate(
        cropped.permute(0, 3, 1, 2),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.permute(0, 2, 3, 1).contiguous()


def _tensor_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _tensor_tree_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_cpu(v) for v in value)
    return value


def _is_cuda_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and ("cuda" in msg or "cublas" in msg or "gpu" in msg)


def _tensor_tree_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _tensor_tree_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_tensor_tree_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_tensor_tree_to_device(v, device) for v in value)
    return value


def _validate_video_file(path: str, min_frames: int = 1) -> None:
    """Fail fast if ffmpeg/torchvision left a non-playable mp4 behind."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames,nb_frames",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[-1000:]}")

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    frame_count = int(streams[0].get("nb_read_frames") or streams[0].get("nb_frames") or 0)
    if frame_count != min_frames:
        raise RuntimeError(f"ffprobe found {frame_count} frames in {path}, expected {min_frames}")


def _replace_with_valid_video(tmp_path: str, out_path: str, min_frames: int) -> None:
    _validate_video_file(tmp_path, min_frames=min_frames)
    os.replace(tmp_path, out_path)


def _audio_to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Return channel-first mono audio without changing sample timing."""
    if audio.ndim == 1:
        return audio.unsqueeze(0).contiguous()
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [C,N] or [N,C], got {tuple(audio.shape)}")
    if audio.shape[0] <= 8:
        return audio.mean(dim=0, keepdim=True).contiguous()
    if audio.shape[1] <= 8:
        return audio.mean(dim=1, keepdim=False).unsqueeze(0).contiguous()
    raise ValueError(f"Cannot infer audio channel axis from shape {tuple(audio.shape)}")


def _write_valid_mp4(
    write_video_fn,
    out_path: str,
    video: torch.Tensor,
    *,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    validate: bool = True,
) -> None:
    tmp_path = f"{out_path}.tmp.{os.getpid()}.mp4"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    def _publish_tmp() -> None:
        if validate:
            _replace_with_valid_video(tmp_path, out_path, min_frames=video.shape[0])
        else:
            os.replace(tmp_path, out_path)

    wrote = False
    if audio_array is not None and audio_fps is not None:
        audio_array = _audio_to_mono(audio_array)
        try:
            write_video_fn(
                tmp_path,
                video,
                fps=fps,
                audio_array=audio_array,
                audio_fps=int(audio_fps),
                audio_codec="aac",
            )
            _publish_tmp()
            wrote = True
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    if not wrote:
        write_video_fn(tmp_path, video, fps=fps)
        _publish_tmp()


def _write_fragmented_mp4_ffmpeg_raw(
    out_path: str,
    video: torch.Tensor,
    *,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    validate: bool = True,
) -> None:
    """Write a browser-MSE-friendly fMP4 preview chunk.

    The normal final-video writer is left unchanged. This path is only for
    low-latency realtime preview chunks: each chunk is a tiny fragmented MP4
    that can still be opened directly, but can also be appended into one
    MediaSource timeline by the web UI.
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video shape [F,H,W,3], got {tuple(video.shape)}")
    frames, height, width, _ = video.shape
    tmp_path = f"{out_path}.tmp.{os.getpid()}.mp4"
    audio_pipe_read: Optional[int] = None
    audio_pipe_write: Optional[int] = None
    audio_writer: Optional[threading.Thread] = None
    audio_errors: List[BaseException] = []
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
    ]
    audio_pcm_bytes: Optional[bytes] = None
    if audio_array is not None and audio_fps is not None:
        audio = _audio_to_mono(audio_array.detach().float().cpu())
        audio_pcm = audio.transpose(0, 1).contiguous()
        audio_pcm = (audio_pcm.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
        channels = int(audio_pcm.shape[1])
        audio_pcm_bytes = audio_pcm.numpy().tobytes()
        audio_pipe_read, audio_pipe_write = os.pipe()
        os.set_inheritable(audio_pipe_read, True)
        cmd.extend(
            [
                "-f",
                "s16le",
                "-ar",
                str(int(audio_fps)),
                "-ac",
                str(channels),
                "-i",
                f"pipe:{audio_pipe_read}",
                "-shortest",
            ]
        )

    cmd.extend(
        [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-threads",
            "2",
            "-profile:v",
            "baseline",
            "-level",
            "3.1",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
    )
    if audio_array is not None and audio_fps is not None:
        cmd.extend(["-c:a", "aac", "-b:a", "128k", "-ac:a", "1"])
    else:
        cmd.append("-an")
    cmd.extend(
        [
            "-movflags",
            "+frag_keyframe+empty_moov+default_base_moof",
            tmp_path,
        ]
    )

    pass_fds = (audio_pipe_read,) if audio_pipe_read is not None else ()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        pass_fds=pass_fds,
    )
    if audio_pipe_read is not None:
        os.close(audio_pipe_read)
        audio_pipe_read = None
    if audio_pipe_write is not None and audio_pcm_bytes is not None:
        def _write_audio_pipe(fd: int, payload: bytes) -> None:
            try:
                with os.fdopen(fd, "wb", closefd=True) as f:
                    f.write(payload)
            except BaseException as exc:
                audio_errors.append(exc)

        audio_writer = threading.Thread(
            target=_write_audio_pipe,
            args=(audio_pipe_write, audio_pcm_bytes),
            name="realtime-ffmpeg-audio-pipe",
            daemon=True,
        )
        audio_pipe_write = None
        audio_writer.start()
    assert proc.stdin is not None
    try:
        frames_per_write = 8
        for start in range(0, frames, frames_per_write):
            chunk = video[start:start + frames_per_write].detach().to(
                device="cpu",
                dtype=torch.uint8,
            ).contiguous().clone()
            chunk.untyped_storage()._write_file(proc.stdin, False, False, 1)
            del chunk
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        returncode = proc.wait()
        if audio_writer is not None:
            audio_writer.join(timeout=2.0)
    except Exception:
        proc.kill()
        proc.wait()
        raise
    finally:
        if audio_pipe_read is not None:
            os.close(audio_pipe_read)
        if audio_pipe_write is not None:
            os.close(audio_pipe_write)
    if returncode != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(stderr[-2000:])
    if audio_errors:
        raise RuntimeError(f"ffmpeg audio pipe failed: {audio_errors[0]}")
    if validate:
        _replace_with_valid_video(tmp_path, out_path, min_frames=frames)
    else:
        os.replace(tmp_path, out_path)


def _write_fragmented_mp4_preview(
    write_video_fn,
    out_path: str,
    video: torch.Tensor,
    *,
    fps: int,
    audio_array: Optional[torch.Tensor] = None,
    audio_fps: Optional[int] = None,
    validate: bool = True,
) -> None:
    _write_fragmented_mp4_ffmpeg_raw(
        out_path,
        video,
        fps=fps,
        audio_array=audio_array,
        audio_fps=audio_fps,
        validate=validate,
    )


def _write_preview_frame_jpeg(out_path: str, pixels: torch.Tensor) -> None:
    """Publish the first decoded RGB frame as a lightweight visual cue."""
    if pixels.ndim != 4 or int(pixels.shape[0]) <= 0 or int(pixels.shape[-1]) != 3:
        raise ValueError(f"expected pixels [F,H,W,3], got {tuple(pixels.shape)}")
    frame_hwc = pixels[0].detach().to(device="cpu", dtype=torch.uint8).contiguous()
    tmp_path = f"{out_path}.tmp.{os.getpid()}.jpg"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    try:
        from torchvision.io import write_jpeg

        frame_chw = frame_hwc.permute(2, 0, 1).contiguous()
        write_jpeg(frame_chw, tmp_path, quality=90)
        os.replace(tmp_path, out_path)
        return
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    frame = frame_hwc.unsqueeze(0)
    _, height, width, _ = frame.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{int(width)}x{int(height)}",
        "-i",
        "-",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        tmp_path,
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _, stderr_bytes = proc.communicate(input=frame.numpy().tobytes())
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        returncode = proc.returncode
    except Exception:
        proc.kill()
        proc.wait()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    if returncode != 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(stderr[-2000:])
    os.replace(tmp_path, out_path)


def _module_first_device(module: torch.nn.Module) -> Optional[str]:
    for param in module.parameters(recurse=True):
        return str(param.device)
    for buffer in module.buffers(recurse=True):
        return str(buffer.device)
    return None


def _ensure_module_device(module: torch.nn.Module, device: str) -> None:
    target = str(device)
    current = _module_first_device(module)
    if current != target:
        module.to(target)


def _decode_video_latents_to_pixels(
    video_vae: Any,
    tiling_config: Any,
    video_latent: torch.Tensor,
    *,
    device: str,
    ensure_device: bool = True,
) -> torch.Tensor:
    if ensure_device:
        _ensure_module_device(video_vae.decoder, device)
    latent = video_latent.to(device).permute(0, 2, 1, 3, 4)
    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for chunk in video_vae.decoder.decode_video(latent, tiling_config):
            chunks.append(chunk.cpu())
    del latent
    pixels = torch.cat(chunks, dim=0)
    del chunks
    return pixels


def _decode_realtime_block_pixels(
    video_vae: Any,
    tiling_config: Any,
    block_video_latent: torch.Tensor,
    *,
    context_video_latent: Optional[torch.Tensor],
    device: str,
    decode_device: Optional[str] = None,
    ensure_device: bool = True,
    stage_on_cpu: bool = True,
) -> torch.Tensor:
    """Decode a completed causal block with optional VAE context.

    This is output-only. It never writes back to the generation latent, so the
    final model result remains governed by the normal full/window decode path.
    """
    block_latent = (
        block_video_latent.detach().to("cpu")
        if stage_on_cpu else block_video_latent.detach()
    )
    target_device = str(decode_device or device)
    if context_video_latent is not None and context_video_latent.shape[1] > 0:
        full_latent = torch.cat([context_video_latent, block_latent], dim=1)
        full_pixels = _decode_video_latents_to_pixels(
            video_vae,
            tiling_config,
            full_latent,
            device=target_device,
            ensure_device=ensure_device,
        )
        prefix_pixel_frames = (context_video_latent.shape[1] - 1) * 8 + 1
        pixels = full_pixels[prefix_pixel_frames:].contiguous()
        del full_pixels, full_latent
        return pixels
    return _decode_video_latents_to_pixels(
        video_vae,
        tiling_config,
        block_latent,
        device=target_device,
        ensure_device=ensure_device,
    )


def _decode_realtime_block_audio(
    audio_vae: Any,
    block_audio_latent: Optional[torch.Tensor],
    *,
    device: str,
    decode_device: Optional[str] = None,
    ensure_device: bool = True,
) -> Optional[torch.Tensor]:
    if block_audio_latent is None:
        return None
    try:
        target_device = str(decode_device or device)
        if ensure_device:
            _ensure_module_device(audio_vae, target_device)
        audio_latent_gpu = block_audio_latent.detach().to(target_device)
        with torch.no_grad():
            waveform = audio_vae.decode_to_waveform(audio_latent_gpu)
        audio = _audio_to_mono(waveform[0].float().cpu())
        del audio_latent_gpu, waveform
        return audio
    except Exception as exc:
        print(f" [realtime audio err: {exc}]", end="", flush=True)
        return None


class _RealtimeBlockStreamer:
    """Decode and publish preview MP4s without changing generation latents."""

    def __init__(
        self,
        *,
        write_video_fn: Any,
        stream_dir: str,
        case_id: str,
        segment_index: int,
        stream_index_offset: Optional[int] = None,
        video_vae: Any,
        audio_vae: Any,
        tiling_config: Any,
        initial_video_prefix: Optional[torch.Tensor],
        max_context_latents: int,
        queue_size: int,
        validate_streams: bool,
        fragmented_mp4: bool,
        stream_workers: int,
        blocks_per_chunk: int,
        first_chunk_blocks: int,
        output_video_height: int,
        output_video_width: int,
        device: str,
        decode_device: str,
        audio_decode_device: str,
        fast_preview_frame: bool,
        write_preview_frame: bool,
        keep_cuda_latents: bool,
        write_asr_audio_sidecar: bool = False,
        profiler: Optional[_FirstStreamProfiler] = None,
    ) -> None:
        self._write_video_fn = write_video_fn
        self._stream_dir = stream_dir
        self._case_id = case_id
        self._segment_index = int(segment_index)
        self._stream_index_offset = (
            None if stream_index_offset is None else int(stream_index_offset)
        )
        self._video_vae = video_vae
        self._audio_vae = audio_vae
        self._tiling_config = tiling_config
        self._max_context_latents = max(0, int(max_context_latents))
        self._validate_streams = bool(validate_streams)
        self._fragmented_mp4 = bool(fragmented_mp4)
        self._stream_workers = max(1, int(stream_workers))
        self._blocks_per_chunk = max(1, int(blocks_per_chunk))
        self._first_chunk_blocks = max(1, int(first_chunk_blocks))
        self._output_video_height = int(output_video_height or 0)
        self._output_video_width = int(output_video_width or 0)
        self._device = device
        self._decode_device = decode_device
        self._audio_decode_device = audio_decode_device or decode_device
        self._keep_cuda_latents = (
            bool(keep_cuda_latents)
            and str(self._decode_device).startswith("cuda")
        )
        self._stage_streams: Dict[str, torch.cuda.Stream] = {}
        self._video_prefix_items: List[_StagedLatent] = []
        if initial_video_prefix is not None:
            prefix_device = self._decode_device if self._keep_cuda_latents else "cpu"
            self._video_prefix_items.append(
                _StagedLatent(
                    initial_video_prefix.detach().to(
                        prefix_device,
                        non_blocking=self._keep_cuda_latents,
                    ).contiguous()
                )
            )
        self._fast_preview_frame = bool(fast_preview_frame)
        self._write_preview_frame = bool(write_preview_frame)
        self._write_asr_audio_sidecar = bool(write_asr_audio_sidecar)
        self._profiler = profiler
        self._errors: List[str] = []
        self._next_publish_block = 0
        self._ready_streams: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        self._publish_condition = threading.Condition()
        self._pending_chunk_index = 0
        self._pending_video_blocks: List[_StagedLatent] = []
        self._pending_audio_blocks: List[Optional[_StagedLatent]] = []
        self._pending_context: Optional[List[_StagedLatent]] = None
        # The realtime path decodes every causal block. Moving large VAEs with
        # .to(device) for every block is usually a no-op, but it still walks the
        # full module tree. Pin them once for this streamer instead.
        _ensure_module_device(self._video_vae.decoder, self._decode_device)
        if self._audio_vae is not None:
            _ensure_module_device(self._audio_vae, self._audio_decode_device)
        self._queue: "queue.Queue[Optional[Tuple[int, List[_StagedLatent], Optional[List[_StagedLatent]], Optional[List[_StagedLatent]], float]]]"
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._threads: List[threading.Thread] = []
        for worker_idx in range(self._stream_workers):
            thread = threading.Thread(
                target=self._run,
                name=f"realtime-block-stream-{case_id}-{worker_idx}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _stream_idx(self, block_index: int) -> int:
        if self._stream_index_offset is not None:
            return int(self._stream_index_offset) + int(block_index)
        return self._segment_index * 100 + int(block_index)

    def _get_stage_stream(self, device: str) -> torch.cuda.Stream:
        key = str(torch.device(device))
        stream = self._stage_streams.get(key)
        if stream is None:
            with torch.cuda.device(torch.device(device)):
                stream = torch.cuda.Stream()
            self._stage_streams[key] = stream
        return stream

    def _stage_latent(self, latent: torch.Tensor, target_device: str) -> _StagedLatent:
        detached = latent.detach()
        if not self._keep_cuda_latents or not str(target_device).startswith("cuda"):
            return _StagedLatent(detached.to("cpu").contiguous())

        target = torch.device(target_device)
        source_event: Optional[torch.cuda.Event] = None
        if detached.is_cuda:
            with torch.cuda.device(detached.device):
                source_event = torch.cuda.Event()
                torch.cuda.current_stream(detached.device).record_event(source_event)

        stream = self._get_stage_stream(str(target))
        with torch.cuda.device(target):
            with torch.cuda.stream(stream):
                if source_event is not None:
                    stream.wait_event(source_event)
                staged = detached.to(target, non_blocking=True)
                if not staged.is_contiguous():
                    staged = staged.contiguous()
                ready_event = torch.cuda.Event()
                stream.record_event(ready_event)
        source_ref = detached if detached.is_cuda and detached.device != staged.device else None
        return _StagedLatent(staged, ready_event, source_ref)

    @staticmethod
    def _wait_staged_latent(item: _StagedLatent) -> None:
        if item.ready_event is None or not item.tensor.is_cuda:
            return
        with torch.cuda.device(item.tensor.device):
            torch.cuda.current_stream(item.tensor.device).wait_event(item.ready_event)

    @staticmethod
    def _tail_staged_latents(
        items: List[_StagedLatent],
        max_frames: int,
    ) -> List[_StagedLatent]:
        if max_frames <= 0 or not items:
            return list(items)
        total_frames = sum(int(item.tensor.shape[1]) for item in items)
        if total_frames <= max_frames:
            return list(items)
        skip_frames = total_frames - max_frames
        tail: List[_StagedLatent] = []
        for item in items:
            frame_count = int(item.tensor.shape[1])
            if skip_frames >= frame_count:
                skip_frames -= frame_count
                continue
            tensor = item.tensor
            if skip_frames > 0:
                tensor = tensor[:, skip_frames:]
                skip_frames = 0
            tail.append(_StagedLatent(tensor, item.ready_event, item.source_ref))
        return tail

    def _concat_staged_latents(
        self,
        items: Optional[List[_StagedLatent]],
        *,
        max_frames: int = 0,
    ) -> Optional[torch.Tensor]:
        if not items:
            return None
        for item in items:
            self._wait_staged_latent(item)
        tensors = [item.tensor for item in items]
        if len(tensors) == 1:
            output = tensors[0]
        else:
            output = torch.cat(tensors, dim=1)
        if max_frames > 0 and output.shape[1] > max_frames:
            output = output[:, -max_frames:].contiguous()
        elif not output.is_contiguous():
            output = output.contiguous()
        return output

    def submit(
        self,
        block_index: int,
        block_video: torch.Tensor,
        block_audio: Optional[torch.Tensor],
    ) -> None:
        stream_idx = self._stream_idx(block_index)
        profile_this = (
            self._profiler is not None
            and self._profiler.should_profile_stream(stream_idx)
        )
        if profile_this:
            self._profiler.mark(
                "streamer_submit_first_begin",
                video_shape=tuple(block_video.shape),
                audio_shape=tuple(block_audio.shape) if block_audio is not None else None,
            )
        context_video_staged = list(self._video_prefix_items)
        block_video_staged = self._stage_latent(block_video, self._decode_device)
        if profile_this:
            self._profiler.mark(
                "streamer_submit_first_video_staged",
                staging="cuda" if self._keep_cuda_latents else "cpu",
                device=str(block_video_staged.tensor.device),
                async_event=block_video_staged.ready_event is not None,
            )
        if block_audio is not None:
            block_audio_staged = self._stage_latent(block_audio, self._audio_decode_device)
        else:
            block_audio_staged = None
        if profile_this:
            self._profiler.mark(
                "streamer_submit_first_audio_staged",
                staging="cuda" if (
                    self._keep_cuda_latents
                    and block_audio_staged is not None
                    and str(block_audio_staged.tensor.device).startswith("cuda")
                ) else "cpu",
                device=str(block_audio_staged.tensor.device) if block_audio_staged is not None else None,
                async_event=(
                    block_audio_staged.ready_event is not None
                    if block_audio_staged is not None else None
                ),
            )
        self._video_prefix_items.append(block_video_staged)
        if self._max_context_latents > 0:
            self._video_prefix_items = self._tail_staged_latents(
                self._video_prefix_items,
                self._max_context_latents,
            )
        if not self._pending_video_blocks:
            self._pending_context = context_video_staged or None
        self._pending_video_blocks.append(block_video_staged)
        self._pending_audio_blocks.append(block_audio_staged)
        enqueued = False
        target_blocks = (
            self._first_chunk_blocks
            if self._pending_chunk_index == 0 else self._blocks_per_chunk
        )
        if len(self._pending_video_blocks) >= target_blocks:
            self._enqueue_pending_chunk()
            enqueued = True
        if profile_this and enqueued:
            self._profiler.mark("streamer_submit_first_enqueued")
        elif profile_this:
            self._profiler.mark(
                "streamer_submit_first_buffered",
                buffered_blocks=len(self._pending_video_blocks),
                blocks_per_chunk=target_blocks,
            )

    def close(self) -> None:
        self._enqueue_pending_chunk()
        for _ in self._threads:
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._errors:
            print(f" [realtime stream errors={len(self._errors)}]", end="", flush=True)

    def flush(self) -> None:
        """Start any buffered final chunk without waiting for writer threads.

        ``close()`` still performs the required join.  This method lets the
        main generation thread overlap post-generation bookkeeping, such as
        preparing the next interactive prefix K/V, with VAE decode and mp4 mux.
        """
        self._enqueue_pending_chunk()

    def _enqueue_pending_chunk(self) -> None:
        if not self._pending_video_blocks:
            return
        video_chunk = list(self._pending_video_blocks)
        audio_chunk: Optional[List[_StagedLatent]] = None
        if self._pending_audio_blocks and all(
            audio is not None for audio in self._pending_audio_blocks
        ):
            audio_chunk = [
                audio for audio in self._pending_audio_blocks if audio is not None
            ]
        context_video = self._pending_context
        stream_index = int(self._pending_chunk_index)
        self._pending_chunk_index += 1
        self._pending_video_blocks = []
        self._pending_audio_blocks = []
        self._pending_context = None
        self._queue.put((
            stream_index,
            video_chunk,
            audio_chunk,
            context_video,
            time.perf_counter(),
        ))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                block_index, block_video_items, block_audio_items, context_video_items, submit_ts = item
                self._write_one(
                    block_index,
                    block_video_items,
                    block_audio_items,
                    context_video_items,
                    submit_ts,
                )
            except Exception as exc:
                self._errors.append(str(exc))
                print(f" [realtime stream err: {exc}]", end="", flush=True)
                try:
                    block_index = int(item[0]) if item is not None else -1
                    if block_index >= 0:
                        self._mark_stream_failed(block_index)
                except Exception:
                    pass
            finally:
                self._queue.task_done()

    def _publish_ready_stream(
        self,
        block_index: int,
        ready_path: str,
        out_path: str,
    ) -> None:
        with self._publish_condition:
            self._ready_streams[int(block_index)] = (ready_path, out_path)
            self._publish_contiguous_locked()
            while int(block_index) >= self._next_publish_block:
                self._publish_condition.wait(timeout=0.1)

    def _mark_stream_failed(self, block_index: int) -> None:
        with self._publish_condition:
            self._ready_streams[int(block_index)] = (None, None)
            self._publish_contiguous_locked()

    def _publish_contiguous_locked(self) -> None:
        while self._next_publish_block in self._ready_streams:
            ready_path, out_path = self._ready_streams.pop(self._next_publish_block)
            if ready_path and out_path:
                os.replace(ready_path, out_path)
            self._next_publish_block += 1
            self._publish_condition.notify_all()

    def _write_one(
        self,
        block_index: int,
        block_video_items: List[_StagedLatent],
        block_audio_items: Optional[List[_StagedLatent]],
        context_video_items: Optional[List[_StagedLatent]],
        submit_ts: float,
    ) -> None:
        stream_idx = self._stream_idx(block_index)
        out_path = os.path.join(
            self._stream_dir,
            f"{self._case_id}_stream{stream_idx:04d}.mp4",
        )
        ready_path = (
            f"{out_path}.ready.{os.getpid()}.{threading.get_ident()}"
            if self._stream_workers > 1 else out_path
        )
        if self._stream_workers > 1 and os.path.exists(ready_path):
            os.remove(ready_path)
        profile_this = (
            self._profiler is not None
            and self._profiler.should_profile_stream(stream_idx)
        )
        profile_all = self._profiler is not None and self._profiler.enabled
        t0 = time.perf_counter() if profile_all else 0.0
        if profile_this:
            self._profiler.mark("streamer_write_first_begin")
        block_video_cpu = self._concat_staged_latents(block_video_items)
        if block_video_cpu is None:
            raise RuntimeError("Realtime streamer received an empty video chunk")
        block_audio_cpu = self._concat_staged_latents(block_audio_items)
        context_video_cpu = self._concat_staged_latents(
            context_video_items,
            max_frames=self._max_context_latents,
        )
        audio_wave: Optional[torch.Tensor] = None
        audio_errors: List[BaseException] = []
        audio_thread: Optional[threading.Thread] = None

        def _decode_audio_worker() -> None:
            nonlocal audio_wave
            try:
                audio_wave = _decode_realtime_block_audio(
                    self._audio_vae,
                    block_audio_cpu,
                    device=self._device,
                    decode_device=self._audio_decode_device,
                    ensure_device=False,
                )
                if self._write_asr_audio_sidecar and audio_wave is not None:
                    audio_rate = int(self._audio_vae.vocoder.output_sampling_rate)
                    asr_audio = audio_wave
                    if asr_audio.ndim == 1:
                        asr_audio = asr_audio.unsqueeze(0)
                    elif asr_audio.ndim > 2:
                        asr_audio = asr_audio.reshape(-1, asr_audio.shape[-1])
                    if asr_audio.shape[0] > 1:
                        asr_audio = asr_audio.mean(dim=0, keepdim=True)
                    if audio_rate != 16000:
                        asr_audio = torchaudio.functional.resample(
                            asr_audio,
                            audio_rate,
                            16000,
                        )
                    asr_path = os.path.join(
                        self._stream_dir,
                        f"{self._case_id}_stream{stream_idx:04d}.asr.wav",
                    )
                    asr_ready_path = (
                        f"{asr_path}.ready.{os.getpid()}.{threading.get_ident()}"
                    )
                    torchaudio.save(
                        asr_ready_path,
                        asr_audio.clamp(-1.0, 1.0),
                        16000,
                        format="wav",
                        encoding="PCM_S",
                        bits_per_sample=16,
                    )
                    os.replace(asr_ready_path, asr_path)
            except BaseException as exc:
                audio_errors.append(exc)

        if self._audio_vae is not None and block_audio_cpu is not None:
            audio_thread = threading.Thread(
                target=_decode_audio_worker,
                name=f"realtime-audio-decode-{self._case_id}-{stream_idx:04d}",
                daemon=True,
            )
            audio_thread.start()
        preview_published = False
        if stream_idx == 0 and self._write_preview_frame and self._fast_preview_frame:
            preview_path = os.path.join(
                self._stream_dir,
                f"{self._case_id}_preview0000.jpg",
            )
            try:
                preview_pixels = _decode_realtime_block_pixels(
                    self._video_vae,
                    self._tiling_config,
                    block_video_cpu[:, :1].contiguous(),
                    context_video_latent=context_video_cpu,
                    device=self._device,
                    decode_device=self._decode_device,
                    ensure_device=False,
                    stage_on_cpu=not self._keep_cuda_latents,
                )
                preview_pixels = _center_crop_video_pixels(
                    preview_pixels,
                    self._output_video_height,
                    self._output_video_width,
                )
                _write_preview_frame_jpeg(preview_path, preview_pixels)
                preview_published = True
                if profile_this:
                    self._profiler.mark(
                        "first_preview_frame_published",
                        file=os.path.basename(preview_path),
                        mode="single_latent_with_context" if context_video_cpu is not None else "single_latent",
                    )
            except Exception as exc:
                print(f" [fast preview frame err: {exc}]", end="", flush=True)
        pixels = _decode_realtime_block_pixels(
            self._video_vae,
            self._tiling_config,
            block_video_cpu,
            context_video_latent=context_video_cpu,
            device=self._device,
            decode_device=self._decode_device,
            ensure_device=False,
            stage_on_cpu=not self._keep_cuda_latents,
        )
        pixels = _center_crop_video_pixels(
            pixels,
            self._output_video_height,
            self._output_video_width,
        )
        t_video = time.perf_counter() if profile_all else 0.0
        if profile_this:
            self._profiler.mark("streamer_write_first_video_decoded", frames=pixels.shape[0])
        if stream_idx == 0 and self._write_preview_frame and not preview_published:
            preview_path = os.path.join(
                self._stream_dir,
                f"{self._case_id}_preview0000.jpg",
            )
            try:
                _write_preview_frame_jpeg(preview_path, pixels)
                if profile_this:
                    self._profiler.mark(
                        "first_preview_frame_published",
                        file=os.path.basename(preview_path),
                    )
            except Exception as exc:
                print(f" [preview frame err: {exc}]", end="", flush=True)
        audio_t0 = time.perf_counter() if profile_all else 0.0
        if audio_thread is not None:
            audio_thread.join()
        if audio_errors:
            print(f" [realtime audio thread err: {audio_errors[0]}]", end="", flush=True)
        t_audio = time.perf_counter() if profile_all else 0.0
        if profile_this:
            self._profiler.mark(
                "streamer_write_first_audio_decoded",
                samples=audio_wave.shape[-1] if audio_wave is not None else None,
            )
        audio_rate = (
            int(self._audio_vae.vocoder.output_sampling_rate)
            if audio_wave is not None else None
        )
        if profile_this:
            self._profiler.mark("streamer_write_first_mp4_begin")
        try:
            if self._fragmented_mp4:
                _write_fragmented_mp4_preview(
                    self._write_video_fn,
                    ready_path,
                    pixels,
                    fps=VIDEO_FPS,
                    audio_array=audio_wave,
                    audio_fps=audio_rate,
                    validate=self._validate_streams,
                )
            else:
                _write_valid_mp4(
                    self._write_video_fn,
                    ready_path,
                    pixels,
                    fps=VIDEO_FPS,
                    audio_array=audio_wave,
                    audio_fps=audio_rate,
                    validate=self._validate_streams,
                )
        except Exception as exc:
            if audio_wave is None:
                raise
            print(f" [realtime audio mux err: {exc}]", end="", flush=True)
            if self._fragmented_mp4:
                _write_fragmented_mp4_preview(
                    self._write_video_fn,
                    ready_path,
                    pixels,
                    fps=VIDEO_FPS,
                    validate=self._validate_streams,
                )
            else:
                _write_valid_mp4(
                    self._write_video_fn,
                    ready_path,
                    pixels,
                    fps=VIDEO_FPS,
                    validate=self._validate_streams,
                )
        if self._stream_workers > 1:
            self._publish_ready_stream(int(block_index), ready_path, out_path)
        print(f" rt={os.path.basename(out_path)}", end="", flush=True)
        if profile_all:
            t_done = time.perf_counter()
            print(
                f"\n[Profile:rt_stream] stream={stream_idx:04d} "
                f"queue_wait={t0 - submit_ts:.3f}s "
                f"video_decode={t_video - t0:.3f}s "
                f"audio_decode={t_audio - audio_t0:.3f}s "
                f"mp4_write={t_done - t_audio:.3f}s "
                f"total={t_done - t0:.3f}s "
                f"frames={pixels.shape[0]}",
                flush=True,
            )
        if profile_this:
            self._profiler.mark(
                "first_stream_file_published",
                file=os.path.basename(out_path),
            )
            self._profiler.finish_first_stream()


def _write_video_ffmpeg_raw(out_path: str, video: torch.Tensor, fps: int) -> None:
    """Write an RGB uint8 video tensor through ffmpeg without NumPy."""
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected video shape [F,H,W,3], got {tuple(video.shape)}")

    frames, height, width, _ = video.shape
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        frames_per_write = 8
        for start in range(0, frames, frames_per_write):
            chunk = video[start:start + frames_per_write].detach().to(
                device="cpu",
                dtype=torch.uint8,
            ).contiguous().clone()
            # Avoid bytes(storage): on large 1min videos it converts byte-by-byte
            # in Python and can stall the save stage for minutes.
            # The clone is required because a contiguous view still shares the
            # full-video storage, and _write_file writes the whole storage.
            chunk.untyped_storage()._write_file(proc.stdin, False, False, 1)
            del chunk
        proc.stdin.close()
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr is not None else ""
        returncode = proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    if returncode != 0:
        raise RuntimeError(stderr[-2000:])
    if frames == 0:
        raise RuntimeError("ffmpeg wrote zero-frame video")
    _validate_video_file(out_path, min_frames=frames)


def _normalize_case_segments(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return internal generation chunks from legacy segments or one prompt.

    Interactive jobs use a single complete LTX prompt for the whole reply.  The
    5-second units below are execution chunks only; they do not imply separate
    prompt windows.
    """
    if "segments" in case:
        return case["segments"]

    prompts = case.get("prompt")
    base_seed = int(case.get("seed", 0))
    if isinstance(prompts, list):
        return [
            {"segment_id": i, "prompt": prompt, "seed": base_seed}
            for i, prompt in enumerate(prompts)
        ]
    if isinstance(prompts, str):
        repeat_count = max(
            1,
            int(
                case.get("prompt_repeat_count")
                or case.get("generation_chunk_count")
                or 1
            ),
        )
        return [
            {
                "segment_id": i,
                "chunk_index": i,
                "prompt_window_id": int(case.get("prompt_window_id") or 0),
                "prompt": prompts,
                "seed": base_seed,
                "is_prompt_repeat": bool(i > 0),
            }
            for i in range(repeat_count)
        ]
    raise ValueError("Each 1min case must contain `segments` or `prompt`.")


_SPEAKER_SAYS_RE = re.compile(r'((Speaker[_ ]?\d+)\s+says:\s*")([^"]*)(")')
_SPEECH_OVERLAP_PUNCTUATION = set(
    "，。！？!?；;：:、,.…—-~～（）()《》<>【】[]“”\"'‘’ \n\r\t"
)


def _speaker_says_texts(prompt: str) -> List[str]:
    return [match.group(3) for match in _SPEAKER_SAYS_RE.finditer(str(prompt or ""))]


def _log_model_prompt_debug(
    *,
    output_dir: str,
    case_id: str,
    window_idx: int,
    segment: Dict[str, Any],
) -> None:
    prompt = str(segment.get("prompt", ""))
    says = _speaker_says_texts(prompt)
    payload = {
        "case_id": case_id,
        "window_idx": int(window_idx),
        "segment_id": segment.get("segment_id"),
        "seed": segment.get("seed"),
        "speaker_says": says,
        "prompt_chars": len(prompt),
        "prompt": prompt,
    }
    print(
        "\n[ModelPrompt] "
        f"case={case_id} window={window_idx} segment_id={segment.get('segment_id')} "
        f"seed={segment.get('seed')} prompt_chars={len(prompt)}",
        flush=True,
    )
    for idx, text in enumerate(says):
        print(f"[ModelPrompt] Speaker_{idx + 1}_says={text}", flush=True)
    print(f"[ModelPrompt] prompt_begin\n{prompt}\n[ModelPrompt] prompt_end", flush=True)
    try:
        os.makedirs(output_dir, exist_ok=True)
        debug_path = os.path.join(output_dir, "model_prompts_debug.jsonl")
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[ModelPrompt][WARN] failed to write prompt debug log: {exc}", flush=True)


def _normalized_speech_chars_with_offsets(text: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    offsets: List[int] = []
    for idx, ch in enumerate(str(text or "")):
        if ch in _SPEECH_OVERLAP_PUNCTUATION:
            continue
        chars.append(ch)
        offsets.append(idx)
    return "".join(chars), offsets


def _strip_spoken_overlap(current: str, history: str) -> Tuple[str, int]:
    """Remove history-tail overlap from a segment-level speech string.

    The service may add sentence punctuation to the previous display segment
    while the next cumulative prompt joins the same text without that boundary
    punctuation. Match overlap in a punctuation-insensitive space, but return a
    slice of the original current text so the visible speech is otherwise
    untouched.
    """
    cur = current.strip()
    hist = history.strip()
    cur_norm, cur_offsets = _normalized_speech_chars_with_offsets(cur)
    hist_norm, _ = _normalized_speech_chars_with_offsets(hist)
    max_len = min(len(cur_norm), len(hist_norm))
    for n in range(max_len, 0, -1):
        if hist_norm.endswith(cur_norm[:n]):
            cut = cur_offsets[n - 1] + 1
            return cur[cut:].lstrip(), n
    return cur, 0


def _dedupe_segment_speech(
    segments: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Drop repeated leading speech from overlapped 5s prompt segments.

    Long-video benchmark prompts often include the previous spoken sentence at
    the start of the next 5s segment for context. That is useful for text
    planning, but the audio/AV generator treats it as fresh speech and reads it
    again. We only remove exact prefix overlap against the same speaker's
    already-emitted speech, leaving visual/action text untouched.
    """
    speaker_history: Dict[str, str] = {}
    out: List[Dict[str, Any]] = []
    removed_total = 0

    for seg in segments:
        seg_out = dict(seg)
        prompt = str(seg_out.get("prompt", ""))

        def repl(match: re.Match) -> str:
            nonlocal removed_total
            speaker = match.group(2)
            speech = match.group(3)
            history = speaker_history.get(speaker, "")
            deduped, removed = _strip_spoken_overlap(speech, history)
            if removed > 0:
                removed_total += removed
            speaker_history[speaker] = history + deduped
            return f"{match.group(1)}{deduped}{match.group(4)}"

        seg_out["prompt"] = _SPEAKER_SAYS_RE.sub(repl, prompt)
        out.append(seg_out)

    return out, removed_total


def _case_condition_task(case: Dict[str, Any], segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve case-level image/audio metadata, falling back to segment 0."""
    cond_task = dict(case)
    first_segment = segments[0] if segments else {}
    for key in (
        "conditioning_mode",
        "first_frame_path",
        "end_frame_path",
        "image",
        "images",
        "audio_path",
        "wav_path",
        "audio_latent_path",
        "audio_start_time",
    ):
        if key not in cond_task and isinstance(first_segment, dict) and key in first_segment:
            cond_task[key] = first_segment[key]
    return cond_task


def _read_training_prefix_blocks_from_config(config_path: str) -> Optional[int]:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*self_forcing_prefix_blocks\s*:\s*([0-9]+)\s*$", line)
                if m:
                    return max(0, int(m.group(1)))
    except OSError:
        return None
    return None


def _infer_training_prefix_blocks(model_ckpt: str, fallback: int = 2) -> int:
    """Read self_forcing_prefix_blocks from the training config.

    Normal training runs keep config.yaml beside checkpoint_NNNNNN/. Some copied
    inference checkpoints keep only checkpoint folders in a later run directory;
    in that case search sibling run configs under the run root and use
    the newest available config instead of silently falling back.
    """
    ckpt = os.path.abspath(model_ckpt)
    run_dir = os.path.dirname(os.path.dirname(ckpt))
    config_path = os.path.join(run_dir, "config.yaml")
    value = _read_training_prefix_blocks_from_config(config_path)
    if value is not None:
        return value

    exp_dir = os.path.dirname(run_dir)
    candidates: List[str] = []
    try:
        for name in os.listdir(exp_dir):
            sibling_config = os.path.join(exp_dir, name, "config.yaml")
            if os.path.isfile(sibling_config):
                candidates.append(sibling_config)
    except OSError:
        candidates = []
    candidates.sort(key=lambda p: (os.path.getmtime(p), p), reverse=True)
    for candidate in candidates:
        value = _read_training_prefix_blocks_from_config(candidate)
        if value is not None:
            print(f"[Config] Inferred self_forcing_prefix_blocks={value} from {candidate}")
            return value

    return int(fallback)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TaoMate interactive inference")
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--original_ckpt", required=True)
    parser.add_argument("--gemma_path", required=True)
    parser.add_argument("--benchmark_json", required=True)
    parser.add_argument("--prompt_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--video_width", type=int, default=864)
    parser.add_argument("--output_video_height", type=int, default=0)
    parser.add_argument("--output_video_width", type=int, default=0)
    parser.add_argument("--interactive_prefix_state_in")
    parser.add_argument("--interactive_prefix_state_out")
    parser.add_argument("--interactive_stop_control_path", required=True)
    args = parser.parse_args(argv)

    fixed = {
        "denoising_steps": [1000, 757, 522, 0],
        "num_inference_steps": 40,
        "num_frame_per_block": NUM_FRAME_PER_BLOCK,
        "num_frame_per_block_first": NUM_FRAME_PER_BLOCK_FIRST,
        "num_audio_sink_tokens": 0,
        "causal_rope_type": "split",
        "conditioning_mode": "t2v",
        "first_frame_path": None,
        "end_frame_path": None,
        "audio_path": None,
        "audio_latent_path": None,
        "audio_start_time": 0.0,
        "max_prefix_blocks": 2,
        "context_noise": 0.0,
        "context_noise_max": 0.0,
        "context_noise_schedule": "constant",
        "prefix_renorm": True,
        "prefix_renorm_alpha": 1.0,
        "torch_inference_mode": True,
        "realtime_max_open_streamers": 2,
        "realtime_stream_queue_size": 8,
        "realtime_stream_workers": 4,
        "realtime_stream_blocks_per_chunk": 1,
        "realtime_first_chunk_blocks": 1,
        "realtime_decode_context_latents": 4,
        "profile_first_stream": False,
        "resident_prompt_cache_size": 64,
        "prompt_encode_device": "cuda:2",
        "prompt_cache_wait_timeout": 900.0,
        "prompt_cache_wait_poll_interval": 0.05,
        "log_model_prompts": True,
        "interactive_dynamic_stop": True,
        "interactive_dynamic_stop_max_windows": 3,
        "device": "cuda",
        "dtype": "bfloat16",
        "max_cases": None,
        "seed_offset": 0,
        "learned_memory": True,
        "learned_memory_mode": "memory_kv_side_branch",
        "learned_memory_layer_interval": 4,
        "learned_memory_video_dim": 512,
        "learned_memory_audio_dim": 256,
        "learned_memory_heads": 8,
        "learned_memory_video_downsample": 4,
        "learned_memory_audio_tokens": 64,
        "learned_memory_video_beta": 0.15,
        "learned_memory_audio_beta": 0.10,
        "learned_memory_video_anchor_tether": 0.20,
        "learned_memory_audio_anchor_tether": 0.10,
        "learned_memory_identity_anchor": True,
        "learned_memory_identity_anchor_scale": 1.0,
        "learned_memory_ref_video_anchor": True,
        "learned_memory_drift_gate": True,
        "learned_memory_drift_gate_threshold": 0.05,
        "learned_memory_drift_gate_temperature": 0.10,
        "learned_memory_drift_gate_min": 0.10,
        "learned_memory_drift_gate_apply_to_color": True,
        "learned_memory_color_alpha": 0.04,
        "learned_memory_color_proto_alpha": 0.015,
        "learned_memory_color_update_beta": 0.03,
        "learned_memory_color_anchor_tether": 0.60,
        "learned_memory_color_proto_grid": 4,
        "learned_memory_color_drift_threshold": 2.0,
        "learned_memory_color_max_correction": 0.35,
        "learned_memory_color_film": True,
        "learned_memory_color_film_hidden_dim": 256,
    }
    for name, value in fixed.items():
        setattr(args, name, value)
    return args



def generate_window_kvcache(
    kv_pipeline: KVCacheCausalPipeline,
    conditional_dict: Dict[str, Any],
    prefix_conditional_dicts: Optional[List[Dict[str, Any]]],
    video_shape: Tuple[int, ...],
    audio_shape: Optional[Tuple[int, ...]],
    prefix_video_latent: Optional[torch.Tensor],
    prefix_audio_latent: Optional[torch.Tensor],
    seed: int,
    device: str,
    dtype: torch.dtype,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
    num_audio_sink_tokens: int,
    conditioning_mode: str = "t2v",
    first_frame_latent: Optional[torch.Tensor] = None,
    end_frame_latent: Optional[torch.Tensor] = None,
    audio_condition_latent: Optional[torch.Tensor] = None,
    learned_memory_state: Optional[LearnedMemoryState] = None,
    prefix_kv_cache: Optional[Any] = None,
    global_window_index: int = 0,
    global_block_offset: Optional[int] = None,
    block_callback: Optional[Any] = None,
    speculative_kv_journal: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Generate one 5-second window (5 blocks = 16 latent frames) using KV-cache.

    For Window 1 (no prefix): uses KVCacheCausalPipeline.generate() directly,
    which handles Block 0 correctly (empty KV cache + causal forward_inference).
    For Window N (with prefix): builds KV cache from prefix blocks, then
    denoises new blocks with the cached prefix — same fast path as
    run_model_inference.sh --runtime kv_cache.

    Args:
        kv_pipeline: KVCacheCausalPipeline instance
        conditional_dict: Current-window text embeddings used for generation
        prefix_conditional_dicts: Optional per-prefix-block text embeddings used
            only when pre-filling clean prefix KV blocks
        video_shape: [B, F_latent, C, H, W] for this window's generation
        audio_shape: [B, F_audio, C_audio] or None
        prefix_video_latent: [B, F_prefix, C, H, W] clean latent from previous window
        prefix_audio_latent: [B, F_prefix_audio, C_audio] or None
        seed: Random seed
        device: Device string
        dtype: Data type
        num_frame_per_block: Frames per standard block (3)
        num_frame_per_block_first: Frames in first block (4, OmniForcing)
        num_audio_sink_tokens: Audio sink tokens count
        conditioning_mode: t2v/i2v/ii2v/ta2v/tia2v for this window
        first_frame_latent: VAE-encoded first-frame latent for i2v/ii2v
        end_frame_latent: Optional VAE-encoded end-frame latent for ii2v
    Returns:
        (video_latent, audio_latent) - clean latents for the generated window
    """
    inference_context = torch.inference_mode
    if prefix_video_latent is None:
        # Window 1: no prefix, generate all blocks from scratch.
        # KVCacheCausalPipeline.generate() handles Block 0 correctly:
        #   - empty KV cache + causal forward_inference (OmniForcing style)
        with inference_context():
            video_out, audio_out = kv_pipeline.generate(
                video_shape=video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                seed=seed,
                conditioning_mode=conditioning_mode,
                first_frame_latent=first_frame_latent,
                end_frame_latent=end_frame_latent,
                audio_condition_latent=audio_condition_latent,
                block_callback=block_callback,
                learned_memory_state=learned_memory_state,
            )
            return video_out, audio_out

    if prefix_kv_cache is None:
        raise RuntimeError("Continuation generation requires retained clean KV")

    # Window N: has prefix from previous window.
    # Strategy: build a total buffer (prefix + gen), pre-fill prefix,
    # build KV cache from prefix blocks, denoise new blocks via forward_inference.
    B = video_shape[0]
    F_gen = video_shape[1]
    F_prefix = prefix_video_latent.shape[1]
    F_total = F_prefix + F_gen

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device_t = next(kv_pipeline.generator.parameters()).device
    dtype_t = next(kv_pipeline.generator.parameters()).dtype
    video_hw = (int(video_shape[-2]), int(video_shape[-1]))

    # Build blocks for total sequence (prefix + generation)
    blocks = compute_av_blocks(
        total_video_latent_frames=F_total,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )

    # Determine which blocks are prefix (already filled) vs generate (to denoise)
    prefix_block_count = 0
    for blk in blocks:
        if blk.video_end <= F_prefix:
            prefix_block_count += 1
        else:
            break
    prefix_blocks_list = blocks[:prefix_block_count]
    gen_blocks = blocks[prefix_block_count:]
    if prefix_conditional_dicts is not None and len(prefix_conditional_dicts) != len(prefix_blocks_list):
        raise ValueError(
            f"Expected {len(prefix_blocks_list)} prefix conditional dicts, "
            f"got {len(prefix_conditional_dicts)}"
        )
    prefix_conditional_by_block_idx = {}
    if prefix_conditional_dicts is not None:
        prefix_conditional_by_block_idx = {
            pb.block_idx: cond
            for pb, cond in zip(prefix_blocks_list, prefix_conditional_dicts)
        }
    if global_block_offset is not None:
        runtime_gen_blocks = _global_generation_blocks_from_offset(
            first_block_idx=global_block_offset,
            block_count=len(gen_blocks),
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
        )
    else:
        runtime_gen_blocks = _global_generation_blocks(
            global_window_index=global_window_index,
            block_count=len(gen_blocks),
            num_frame_per_block=num_frame_per_block,
            num_frame_per_block_first=num_frame_per_block_first,
        )
    if kv_pipeline.profile_callback is not None:
        kv_pipeline._profile(
            "window_kv_prefix_plan",
            F_prefix=F_prefix,
            F_gen=F_gen,
            prefix_blocks=len(prefix_blocks_list),
            gen_blocks=len(gen_blocks),
            reuse_window_kv=True,
            global_window_index=int(global_window_index),
            global_block_offset=(
                None if global_block_offset is None else int(global_block_offset)
            ),
            max_prefix_blocks=(
                "none"
                if kv_pipeline._max_prefix_blocks is None
                else int(kv_pipeline._max_prefix_blocks)
            ),
        )

    # Audio alignment
    F_audio_total = compute_aligned_audio_frames(
        F_total, num_frame_per_block, num_frame_per_block_first,
    )
    F_audio_prefix = prefix_audio_latent.shape[1] if prefix_audio_latent is not None else 0
    _validate_audio_prefix_alignment(
        prefix_video_latent,
        prefix_audio_latent,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
        label="KV-cache generation",
    )
    audio_channels = audio_shape[2] if audio_shape is not None else None
    cond_mode_eff = (conditioning_mode or "t2v").lower()
    audio_conditioned = cond_mode_eff in ("ta2v", "tia2v")
    if audio_conditioned:
        if audio_shape is None or audio_condition_latent is None:
            raise ValueError(
                f"conditioning_mode={cond_mode_eff!r} requires audio_condition_latent"
            )
        if tuple(audio_condition_latent.shape) != tuple(audio_shape):
            raise ValueError(
                "Window audio condition shape mismatch: "
                f"got {tuple(audio_condition_latent.shape)}, expected {tuple(audio_shape)}"
            )

    def _audio_condition_for_block(block: Any) -> Optional[torch.Tensor]:
        if not audio_conditioned:
            return None
        start = int(block.audio_start) - F_audio_prefix
        end = int(block.audio_end) - F_audio_prefix
        if start < 0 or end > int(audio_condition_latent.shape[1]):
            raise ValueError(
                f"Generated block {block.block_idx} audio range [{start}, {end}) "
                "falls outside the window audio condition."
            )
        return audio_condition_latent[:, start:end].to(
            device=device_t, dtype=dtype_t,
        )

    # Full buffers
    video = torch.zeros((B, F_total, *video_shape[2:]), device=device_t, dtype=dtype_t)
    video[:, :F_prefix] = prefix_video_latent.to(device=device_t, dtype=dtype_t)

    audio = None
    if audio_shape is not None:
        audio = torch.zeros((B, F_audio_total, audio_channels), device=device_t, dtype=dtype_t)
        if prefix_audio_latent is not None:
            audio[:, :F_audio_prefix] = prefix_audio_latent.to(device=device_t, dtype=dtype_t)

    with inference_context():
        initial_prefix_blocks = list(prefix_blocks_list)
        max_warm_prefix = kv_pipeline._max_prefix_blocks
        if max_warm_prefix is not None:
            if kv_pipeline.block0_sink_enabled and initial_prefix_blocks:
                sink_block = initial_prefix_blocks[0]
                warm_candidates = [
                    b for b in initial_prefix_blocks
                    if b.block_idx != sink_block.block_idx
                ]
                warm_prefix = (
                    warm_candidates[-max_warm_prefix:]
                    if max_warm_prefix > 0 else []
                )
                initial_prefix_blocks = [sink_block, *warm_prefix]
            else:
                initial_prefix_blocks = (
                    initial_prefix_blocks[-max_warm_prefix:]
                    if max_warm_prefix > 0 else []
                )

        kv_cache = prefix_kv_cache
        cross_chunk_recent = 2
        pruned_cache = _prune_clean_kv_to_sink_recent(
            kv_cache,
            recent_blocks=cross_chunk_recent,
        )
        if pruned_cache is None:
            raise RuntimeError(
                "Unable to initialize retained clean KV from segment metadata"
            )
        kv_cache = pruned_cache
        cache_segments = _clone_interactive_kv_segments(kv_cache)
        stop_video_end: Optional[int] = None
        stop_audio_end: Optional[int] = None
        reused_process_prefix_kv = kv_cache is not None
        if reused_process_prefix_kv:
            initial_prefix_blocks = []
        window_size = max(1, len(initial_prefix_blocks))
        if kv_pipeline.profile_callback is not None:
            kv_pipeline._profile(
                "window_kv_reuse_prefix_selected",
                prefix_blocks=len(initial_prefix_blocks),
                block_ids=",".join(str(b.block_idx) for b in initial_prefix_blocks),
                process_prefix_kv_reused=bool(reused_process_prefix_kv),
            )
            if reused_process_prefix_kv:
                kv_pipeline._profile(
                    "window_prefix_kv_cache_reused",
                    prefix_blocks=len(prefix_blocks_list),
                    committed_segments=len(cache_segments),
                )

        # Prefix cache
        for k, pb in enumerate(initial_prefix_blocks):
            vb = video[:, pb.video_start:pb.video_end]
            ab = audio[:, pb.audio_start:pb.audio_end] if audio is not None else None
            pb_conditional_dict = prefix_conditional_by_block_idx.get(pb.block_idx, conditional_dict)
            if learned_memory_state is not None:
                pb_conditional_dict = learned_memory_state.with_conditional_memory(
                    pb_conditional_dict, device=device_t, dtype=dtype_t,
                )

            sigma_k = _resolve_prefix_ctx_sigma(
                k=k, window=window_size,
                base=kv_pipeline.context_noise,
                s_max=kv_pipeline.context_noise_max,
                schedule=kv_pipeline.context_noise_schedule,
            )
            if sigma_k > 0.0 and int(pb.block_idx) != 0:
                c_sigma = torch.tensor(sigma_k, device=device_t, dtype=dtype_t)
                v_expand = c_sigma.expand(B, vb.shape[1])
                vb = add_noise(vb, torch.randn_like(vb), v_expand)
                if ab is not None and ab.shape[1] > 0 and not audio_conditioned:
                    a_expand = c_sigma.expand(B, ab.shape[1])
                    ab = add_noise(ab, torch.randn_like(ab), a_expand)

            vs = torch.zeros((B, vb.shape[1]), device=device_t, dtype=dtype_t)
            a_s = (
                torch.zeros((B, ab.shape[1]), device=device_t, dtype=dtype_t)
                if ab is not None else None
            )
            kv_pipeline._profile_sync(device_t)
            _t_prefill = (
                time.perf_counter()
                if kv_pipeline.profile_callback is not None else None
            )
            _, _, kv_cache = kv_pipeline.generator.model.forward_inference(
                video_latent=vb, audio_latent=ab,
                timesteps=vs, audio_timesteps=a_s,
                video_context=pb_conditional_dict["video_context"],
                audio_context=pb_conditional_dict["audio_context"],
                video_context_mask=pb_conditional_dict.get("video_context_mask"),
                audio_context_mask=pb_conditional_dict.get("audio_context_mask"),
                learned_memory_video=pb_conditional_dict.get("learned_memory_video"),
                learned_memory_audio=pb_conditional_dict.get("learned_memory_audio"),
                learned_memory_color=pb_conditional_dict.get("learned_memory_color"),
                kv_cache=kv_cache,
                video_start_frame=pb.video_start,
                audio_start_frame=pb.audio_start,
                include_audio_sinks=(pb.block_idx == 0 and num_audio_sink_tokens > 0),
                pyramid_policy=kv_pipeline.pyramid_policy,
                kv_cache_only=True,
            )
            if learned_memory_state is not None and int(pb.block_idx) == 0:
                learned_memory_state.set_reference(vb, ab)
            if _t_prefill is not None:
                kv_pipeline._profile_sync(device_t)
                kv_pipeline._profile(
                    "window_prefix_prefill_block_done",
                    block=pb.block_idx,
                    video_frames=vb.shape[1],
                    audio_frames=ab.shape[1] if ab is not None else 0,
                    elapsed=f"{time.perf_counter() - _t_prefill:.3f}s",
                )
            cache_segments.append(
                _make_interactive_kv_segment(
                    kind="prefix",
                    block=pb,
                    video_hw=video_hw,
                    include_audio_sinks=(
                        pb.block_idx == 0 and num_audio_sink_tokens > 0
                    ),
                    num_audio_sink_tokens=num_audio_sink_tokens,
                    prefix_slot=pb.block_idx,
                )
            )
        if kv_cache is None:
            kv_cache = kv_pipeline.generator.init_kv_cache()

        for gen_block_i, (block, runtime_block) in enumerate(
            zip(gen_blocks, runtime_gen_blocks)
        ):
            if (
                int(block.video_frames) != int(runtime_block.video_frames)
                or int(block.audio_frames) != int(runtime_block.audio_frames)
            ):
                raise RuntimeError(
                    "Local output block and global-RoPE runtime block have "
                    "different shapes: "
                    f"local={block.block_idx} "
                    f"({block.video_frames}v/{block.audio_frames}a), "
                    f"runtime={runtime_block.block_idx} "
                    f"({runtime_block.video_frames}v/{runtime_block.audio_frames}a)"
                )
            is_final_gen_block = gen_block_i == len(gen_blocks) - 1
            pre_block_cache_layers = list(kv_cache.layers)
            if kv_pipeline.profile_callback is not None and gen_block_i == 0:
                kv_pipeline._profile(
                    "window_first_gen_block_begin",
                    block=block.block_idx,
                    global_block=runtime_block.block_idx,
                    global_video_start=runtime_block.video_start,
                    prefix_blocks=len(initial_prefix_blocks),
                )
            block_conditional_dict = (
                learned_memory_state.with_conditional_memory(
                    conditional_dict, device=device_t, dtype=dtype_t,
                )
                if learned_memory_state is not None else conditional_dict
            )
            cur_video, cur_audio = kv_pipeline._denoise_block_with_kv(
                block=runtime_block,
                B=B,
                video_tail_shape=video_shape[2:],
                audio_channels=audio_channels,
                conditional_dict=block_conditional_dict,
                kv_cache=kv_cache,
                device=device_t,
                dtype=dtype_t,
                audio_condition_latent=_audio_condition_for_block(block),
            )

            cur_video, cur_audio = kv_pipeline._maybe_renorm_block(
                runtime_block.block_idx, cur_video, cur_audio,
            )
            clean_audio = _audio_condition_for_block(block)
            if clean_audio is not None:
                cur_audio = clean_audio
            if learned_memory_state is not None:
                cur_video = learned_memory_state.apply_color_memory(cur_video)
            video[:, block.video_start:block.video_end] = cur_video
            if audio is not None and cur_audio is not None:
                audio[:, block.audio_start:block.audio_end] = cur_audio
            # The emitted latent is final at this point. Hand it to the
            # realtime streamer before the sigma=0 clean-KV commit below;
            # the commit still runs before the next block, so generation
            # quality and train/infer KV semantics stay unchanged.
            stop_after_block = False
            if block_callback is not None:
                callback_result = block_callback(cur_video, cur_audio, block)
                stop_after_block = _block_callback_requests_stop(callback_result)
                if stop_after_block:
                    stop_video_end = int(block.video_end)
                    stop_audio_end = int(block.audio_end)
            # Training commits clean K/V with the pre-block memory state,
            # then updates memory for the following block. Keep the 1min
            # path aligned with that ordering.
            clean_commit_conditional_dict = block_conditional_dict

            # No following block in this window consumes the clean K/V.
            # Cross-window continuation normally re-prefills K/V from saved
            # latents.  The interactive service can opt into committing the
            # final block here, after the block has already been handed to
            # the realtime streamer, so the clean commit overlaps final
            # VAE/mux work without changing next-turn KV semantics.
            if is_final_gen_block:
                kv_cache.layers = list(pre_block_cache_layers)
                vs = torch.zeros((B, cur_video.shape[1]), device=device_t, dtype=dtype_t)
                a_s = (
                    torch.zeros((B, cur_audio.shape[1]), device=device_t, dtype=dtype_t)
                    if cur_audio is not None else None
                )
                kv_pipeline._profile_sync(device_t)
                _t_final_commit = (
                    time.perf_counter()
                    if kv_pipeline.profile_callback is not None else None
                )
                _, _, kv_cache = kv_pipeline.generator.model.forward_inference(
                    video_latent=cur_video, audio_latent=cur_audio,
                    timesteps=vs, audio_timesteps=a_s,
                    video_context=clean_commit_conditional_dict["video_context"],
                    audio_context=clean_commit_conditional_dict["audio_context"],
                    video_context_mask=clean_commit_conditional_dict.get("video_context_mask"),
                    audio_context_mask=clean_commit_conditional_dict.get("audio_context_mask"),
                    learned_memory_video=clean_commit_conditional_dict.get("learned_memory_video"),
                    learned_memory_audio=clean_commit_conditional_dict.get("learned_memory_audio"),
                    learned_memory_color=clean_commit_conditional_dict.get("learned_memory_color"),
                    kv_cache=kv_cache,
                    video_start_frame=runtime_block.video_start,
                    audio_start_frame=runtime_block.audio_start,
                    include_audio_sinks=False,
                    pyramid_policy=kv_pipeline.pyramid_policy,
                    kv_cache_only=True,
                )
                if _t_final_commit is not None:
                    kv_pipeline._profile_sync(device_t)
                    kv_pipeline._profile(
                        "window_final_kv_clean_commit_done",
                        block=runtime_block.block_idx,
                        video_frames=cur_video.shape[1],
                        audio_frames=cur_audio.shape[1] if cur_audio is not None else 0,
                        elapsed=f"{time.perf_counter() - _t_final_commit:.3f}s",
                    )
                cache_segments.append(
                    _make_interactive_kv_segment(
                        kind="generated",
                        block=runtime_block,
                        video_hw=video_hw,
                        include_audio_sinks=False,
                        num_audio_sink_tokens=num_audio_sink_tokens,
                        window_slot=gen_block_i,
                    )
                )
                _append_latest_clean_kv_to_journal(
                    speculative_kv_journal,
                    kv_cache,
                    cache_segments,
                )
                _attach_interactive_kv_segments(kv_cache, cache_segments)
                pruned_cache = _prune_clean_kv_to_sink_recent(
                    kv_cache,
                    recent_blocks=cross_chunk_recent,
                )
                if pruned_cache is None:
                    raise RuntimeError("Failed to prune final committed clean KV")
                kv_cache = pruned_cache
                cache_segments = _clone_interactive_kv_segments(kv_cache)
                if learned_memory_state is not None:
                    learned_memory_state.update(cur_video, cur_audio)
                if stop_after_block:
                    break
                continue

            # Clean cache commit
            kv_cache.layers = list(pre_block_cache_layers)
            vs = torch.zeros((B, cur_video.shape[1]), device=device_t, dtype=dtype_t)
            a_s = (
                torch.zeros((B, cur_audio.shape[1]), device=device_t, dtype=dtype_t)
                if cur_audio is not None else None
            )
            kv_pipeline._profile_sync(device_t)
            _t_clean_commit = (
                time.perf_counter()
                if kv_pipeline.profile_callback is not None else None
            )
            _, _, kv_cache = kv_pipeline.generator.model.forward_inference(
                video_latent=cur_video, audio_latent=cur_audio,
                timesteps=vs, audio_timesteps=a_s,
                video_context=clean_commit_conditional_dict["video_context"],
                audio_context=clean_commit_conditional_dict["audio_context"],
                video_context_mask=clean_commit_conditional_dict.get("video_context_mask"),
                audio_context_mask=clean_commit_conditional_dict.get("audio_context_mask"),
                learned_memory_video=clean_commit_conditional_dict.get("learned_memory_video"),
                learned_memory_audio=clean_commit_conditional_dict.get("learned_memory_audio"),
                learned_memory_color=clean_commit_conditional_dict.get("learned_memory_color"),
                kv_cache=kv_cache,
                # ``block`` indexes the local prefix+output buffer.  A
                # persistent cross-chunk cache must instead commit at the
                # absolute runtime position used by the denoise pass;
                # otherwise its metadata says global block N while its K/V
                # carries the repeated local-window RoPE position.
                video_start_frame=runtime_block.video_start,
                audio_start_frame=runtime_block.audio_start,
                include_audio_sinks=False,
                pyramid_policy=kv_pipeline.pyramid_policy,
                kv_cache_only=True,
            )
            if _t_clean_commit is not None:
                kv_pipeline._profile_sync(device_t)
                kv_pipeline._profile(
                    "window_kv_clean_commit_done",
                    block=runtime_block.block_idx,
                    video_frames=cur_video.shape[1],
                    audio_frames=cur_audio.shape[1] if cur_audio is not None else 0,
                    elapsed=f"{time.perf_counter() - _t_clean_commit:.3f}s",
                )
            cache_segments.append(
                _make_interactive_kv_segment(
                    kind="generated",
                    block=runtime_block,
                    video_hw=video_hw,
                    include_audio_sinks=False,
                    num_audio_sink_tokens=num_audio_sink_tokens,
                    window_slot=gen_block_i,
                )
            )
            _append_latest_clean_kv_to_journal(
                speculative_kv_journal,
                kv_cache,
                cache_segments,
            )
            _attach_interactive_kv_segments(kv_cache, cache_segments)
            pruned_cache = _prune_clean_kv_to_sink_recent(
                kv_cache,
                recent_blocks=cross_chunk_recent,
            )
            if pruned_cache is None:
                raise RuntimeError("Failed to prune committed clean KV")
            kv_cache = pruned_cache
            cache_segments = _clone_interactive_kv_segments(kv_cache)
            if learned_memory_state is not None:
                learned_memory_state.update(cur_video, cur_audio)
            if stop_after_block:
                break

    gen_video_end = F_total if stop_video_end is None else stop_video_end
    gen_audio_end = (
        F_audio_prefix + audio_shape[1]
        if stop_audio_end is None and audio_shape is not None
        else stop_audio_end
    )
    gen_video = video[:, F_prefix:gen_video_end]
    gen_audio = None
    if audio is not None and audio_shape is not None and gen_audio_end is not None:
        gen_audio = audio[:, F_audio_prefix:gen_audio_end]
    _attach_interactive_kv_segments(kv_cache, cache_segments)
    kv_pipeline.last_kv_cache = kv_cache
    kv_pipeline.last_kv_cache_segments = [dict(s) for s in cache_segments]
    return gen_video, gen_audio


def build_interactive_prefix_kv_cache(
    *,
    kv_pipeline: KVCacheCausalPipeline,
    selected_conditional_dicts: List[Dict[str, Any]],
    prefix_video_latent: Optional[torch.Tensor],
    prefix_audio_latent: Optional[torch.Tensor],
    next_video_shape: Tuple[int, ...],
    next_audio_shape: Optional[Tuple[int, ...]],
    device: str,
    dtype: torch.dtype,
    num_frame_per_block: int,
    num_frame_per_block_first: int,
    num_audio_sink_tokens: int,
    learned_memory_state: Optional[LearnedMemoryState] = None,
    profiler: Optional[_FirstStreamProfiler] = None,
    global_block_indices: Optional[List[int]] = None,
) -> Optional[Any]:
    """Build the exact clean prefix KV expected by the next interactive turn."""
    if prefix_video_latent is None or not _can_prepare_interactive_prefix_kv(kv_pipeline):
        return None

    device_t = next(kv_pipeline.generator.parameters()).device
    dtype_t = next(kv_pipeline.generator.parameters()).dtype
    B = int(next_video_shape[0])
    F_prefix = int(prefix_video_latent.shape[1])
    F_gen = int(next_video_shape[1])
    F_total = F_prefix + F_gen

    blocks = compute_av_blocks(
        total_video_latent_frames=F_total,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
    )
    prefix_blocks = [block for block in blocks if block.video_end <= F_prefix]
    if not prefix_blocks:
        return None

    max_warm_prefix = getattr(kv_pipeline, "_max_prefix_blocks", None)
    if max_warm_prefix is not None:
        if kv_pipeline.block0_sink_enabled:
            sink_block = prefix_blocks[0]
            warm_candidates = [
                b for b in prefix_blocks
                if b.block_idx != sink_block.block_idx
            ]
            warm_prefix = (
                warm_candidates[-int(max_warm_prefix):]
                if int(max_warm_prefix) > 0 else []
            )
            selected_blocks = [sink_block, *warm_prefix]
        else:
            selected_blocks = (
                prefix_blocks[-int(max_warm_prefix):]
                if int(max_warm_prefix) > 0 else []
            )
    else:
        selected_blocks = list(prefix_blocks)

    if not selected_blocks:
        return None
    if len(selected_conditional_dicts) != len(selected_blocks):
        raise ValueError(
            "selected_conditional_dicts must match selected interactive "
            f"prefix blocks: {len(selected_conditional_dicts)} vs {len(selected_blocks)}"
        )
    if global_block_indices is not None:
        if len(global_block_indices) != len(selected_blocks):
            raise ValueError(
                "global_block_indices must match selected interactive prefix "
                f"blocks: {len(global_block_indices)} vs {len(selected_blocks)}"
            )
        runtime_blocks = [
            _global_block_at_index(
                block_idx=block_idx,
                num_frame_per_block=num_frame_per_block,
                num_frame_per_block_first=num_frame_per_block_first,
            )
            for block_idx in global_block_indices
        ]
    else:
        runtime_blocks = list(selected_blocks)

    audio_channels = None
    if next_audio_shape is not None:
        audio_channels = int(next_audio_shape[2])
    elif prefix_audio_latent is not None:
        audio_channels = int(prefix_audio_latent.shape[2])

    F_audio_total = compute_aligned_audio_frames(
        F_total,
        num_frame_per_block,
        num_frame_per_block_first,
    )
    F_audio_prefix = (
        int(prefix_audio_latent.shape[1])
        if prefix_audio_latent is not None else 0
    )
    _validate_audio_prefix_alignment(
        prefix_video_latent,
        prefix_audio_latent,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
        label="interactive prefix KV build",
    )

    video = torch.zeros(
        (B, F_total, *next_video_shape[2:]),
        device=device_t,
        dtype=dtype_t,
    )
    video[:, :F_prefix] = prefix_video_latent.to(device=device_t, dtype=dtype_t)

    audio = None
    if audio_channels is not None:
        audio = torch.zeros(
            (B, F_audio_total, audio_channels),
            device=device_t,
            dtype=dtype_t,
        )
        if prefix_audio_latent is not None:
            audio[:, :F_audio_prefix] = prefix_audio_latent.to(
                device=device_t,
                dtype=dtype_t,
            )

    kv_cache = None
    cache_segments: List[Dict[str, Any]] = []
    kv_pipeline._profile_sync(device_t)
    t0 = time.perf_counter()
    with torch.no_grad():
        for pb, runtime_block, conditional_dict in zip(
            selected_blocks,
            runtime_blocks,
            selected_conditional_dicts,
        ):
            if (
                int(pb.video_end) - int(pb.video_start)
                != int(runtime_block.video_end) - int(runtime_block.video_start)
                or int(pb.audio_end) - int(pb.audio_start)
                != int(runtime_block.audio_end) - int(runtime_block.audio_start)
            ):
                raise RuntimeError(
                    "Interactive prefix local/global block shape mismatch: "
                    f"local={pb.block_idx}, global={runtime_block.block_idx}"
                )
            vb = video[:, pb.video_start:pb.video_end]
            ab = (
                audio[:, pb.audio_start:pb.audio_end]
                if audio is not None else None
            )
            vs = torch.zeros((B, vb.shape[1]), device=device_t, dtype=dtype_t)
            a_s = (
                torch.zeros((B, ab.shape[1]), device=device_t, dtype=dtype_t)
                if ab is not None else None
            )
            block_conditional_dict = conditional_dict
            if learned_memory_state is not None:
                block_conditional_dict = learned_memory_state.with_conditional_memory(
                    block_conditional_dict,
                    device=device_t,
                    dtype=dtype_t,
                )
            _, _, kv_cache = kv_pipeline.generator.model.forward_inference(
                video_latent=vb,
                audio_latent=ab,
                timesteps=vs,
                audio_timesteps=a_s,
                video_context=block_conditional_dict["video_context"],
                audio_context=block_conditional_dict["audio_context"],
                video_context_mask=block_conditional_dict.get("video_context_mask"),
                audio_context_mask=block_conditional_dict.get("audio_context_mask"),
                learned_memory_video=block_conditional_dict.get("learned_memory_video"),
                learned_memory_audio=block_conditional_dict.get("learned_memory_audio"),
                learned_memory_color=block_conditional_dict.get("learned_memory_color"),
                kv_cache=kv_cache,
                video_start_frame=runtime_block.video_start,
                audio_start_frame=runtime_block.audio_start,
                include_audio_sinks=(
                    runtime_block.block_idx == 0 and num_audio_sink_tokens > 0
                ),
                pyramid_policy=kv_pipeline.pyramid_policy,
                kv_cache_only=True,
            )
            if profiler is not None:
                kv_pipeline._profile_sync(device_t)
                profiler.mark(
                    "interactive_prefix_kv_prefill_block_done",
                    block=runtime_block.block_idx,
                    video_frames=vb.shape[1],
                    audio_frames=ab.shape[1] if ab is not None else 0,
                )
            cache_segments.append(
                _make_interactive_kv_segment(
                    kind="prefix",
                    block=runtime_block,
                    video_hw=(int(next_video_shape[-2]), int(next_video_shape[-1])),
                    include_audio_sinks=(
                        runtime_block.block_idx == 0 and num_audio_sink_tokens > 0
                    ),
                    num_audio_sink_tokens=num_audio_sink_tokens,
                    prefix_slot=pb.block_idx,
                )
            )
    kv_pipeline._profile_sync(device_t)
    if profiler is not None:
        profiler.mark(
            "interactive_prefix_kv_built",
            prefix_blocks=len(selected_blocks),
            block_ids=",".join(str(b.block_idx) for b in runtime_blocks),
            elapsed=f"{time.perf_counter() - t0:.3f}s",
        )
    del video, audio
    return _attach_interactive_kv_segments(kv_cache, cache_segments)


@dataclass
class WindowedModelRuntime:
    generator: Any
    train_step: int
    video_vae: Any
    audio_vae: Any
    denoising_sigmas: torch.Tensor
    kv_pipeline: KVCacheCausalPipeline
    video_shape_window0: List[int]
    audio_shape_window0: List[int]
    video_shape_gen: List[int]
    audio_shape_gen: List[int]
    tiling_config: Any


@dataclass
class TextEncoderRuntime:
    text_encoder: Any
    device: str
    dtype: torch.dtype


_WINDOWED_MODEL_RUNTIME_CACHE: "OrderedDict[Tuple[Any, ...], WindowedModelRuntime]" = OrderedDict()
_TEXT_ENCODER_RUNTIME_CACHE: Dict[Tuple[Any, ...], TextEncoderRuntime] = {}
_RESIDENT_PROMPT_EMBEDDING_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, Any]]" = OrderedDict()


def _prompt_embedding_cache_key(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
    prompt: str,
) -> Tuple[Any, ...]:
    return (
        os.path.abspath(args.original_ckpt),
        os.path.abspath(args.gemma_path),
        str(device),
        str(dtype),
        str(prompt),
    )


def _get_resident_prompt_embedding(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
    prompt: str,
) -> Optional[Dict[str, Any]]:
    max_items = max(0, int(getattr(args, "resident_prompt_cache_size", 0)))
    if max_items <= 0:
        return None
    key = _prompt_embedding_cache_key(args, dtype=dtype, device=device, prompt=prompt)
    cached = _RESIDENT_PROMPT_EMBEDDING_CACHE.get(key)
    if cached is None:
        return None
    _RESIDENT_PROMPT_EMBEDDING_CACHE.move_to_end(key)
    return cached


def _put_resident_prompt_embedding(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
    prompt: str,
    encoded: Dict[str, Any],
) -> None:
    max_items = max(0, int(getattr(args, "resident_prompt_cache_size", 0)))
    if max_items <= 0:
        return
    key = _prompt_embedding_cache_key(args, dtype=dtype, device=device, prompt=prompt)
    _RESIDENT_PROMPT_EMBEDDING_CACHE[key] = encoded
    _RESIDENT_PROMPT_EMBEDDING_CACHE.move_to_end(key)
    while len(_RESIDENT_PROMPT_EMBEDDING_CACHE) > max_items:
        _RESIDENT_PROMPT_EMBEDDING_CACHE.popitem(last=False)


def _runtime_cache_key(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
) -> Tuple[Any, ...]:
    return (
        os.path.abspath(args.model_ckpt),
        os.path.abspath(args.original_ckpt),
        str(device),
        str(dtype),
        int(args.video_height),
        int(args.video_width),
    )


def _text_encoder_cache_key(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
) -> Tuple[Any, ...]:
    return (
        os.path.abspath(args.original_ckpt),
        os.path.abspath(args.gemma_path),
        str(device),
        str(dtype),
    )


def _get_text_encoder_runtime(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
) -> TextEncoderRuntime:
    key = _text_encoder_cache_key(args, dtype=dtype, device=device)
    cached = _TEXT_ENCODER_RUNTIME_CACHE.get(key)
    if cached is not None:
        print("[Load] Reusing resident text encoder runtime from process cache")
        return cached

    print(f"\n[Load] Loading resident text encoder on {device}...")
    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=args.original_ckpt,
        gemma_path=args.gemma_path,
        device=torch.device(device),
        dtype=dtype,
        place_on_device=str(device).startswith("cuda"),
    )
    text_encoder.to(device)
    text_encoder.eval()
    runtime = TextEncoderRuntime(
        text_encoder=text_encoder,
        device=str(device),
        dtype=dtype,
    )
    _TEXT_ENCODER_RUNTIME_CACHE[key] = runtime
    print("[Load] Cached text encoder runtime for subsequent prompt encoding")
    return runtime


def _resolve_prompt_encode_device(
    args: argparse.Namespace,
    *,
    inference_device: str,
    runtime_cached: bool,
) -> str:
    requested = str(args.prompt_encode_device or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        return inference_device
    if re.fullmatch(r"cuda:[0-9]+", requested):
        return requested
    if requested not in {"", "auto"}:
        raise ValueError(
            "--prompt_encode_device must be auto, cuda, cpu, or an explicit cuda:N device"
        )
    if runtime_cached:
        return "cpu"
    return inference_device


def _get_windowed_model_runtime(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
    use_sink_block: bool,
) -> WindowedModelRuntime:
    key = _runtime_cache_key(args, dtype=dtype, device=device)
    if key in _WINDOWED_MODEL_RUNTIME_CACHE:
        print("[Load] Reusing resident TaoMate runtime from process cache")
        _WINDOWED_MODEL_RUNTIME_CACHE.move_to_end(key)
        return _WINDOWED_MODEL_RUNTIME_CACHE[key]

    runtime = _load_windowed_model_runtime(
        args,
        dtype=dtype,
        device=device,
        use_sink_block=use_sink_block,
    )
    while _WINDOWED_MODEL_RUNTIME_CACHE:
        _, old_runtime = _WINDOWED_MODEL_RUNTIME_CACHE.popitem(last=False)
        del old_runtime
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _WINDOWED_MODEL_RUNTIME_CACHE[key] = runtime
    print("[Load] Cached TaoMate runtime for subsequent requests in this process")
    return runtime


def _refresh_runtime_denoising_sigmas(
    runtime: WindowedModelRuntime,
    args: argparse.Namespace,
    *,
    device: str,
) -> None:
    denoising_sigmas = compute_denoising_sigmas(
        args.denoising_steps,
        args.num_inference_steps,
        device,
    )
    current = runtime.denoising_sigmas
    needs_update = (
        tuple(current.shape) != tuple(denoising_sigmas.shape)
        or not torch.allclose(current.to(device=denoising_sigmas.device), denoising_sigmas)
    )
    if needs_update:
        print(f"[Config] Updating resident denoising steps: {args.denoising_steps}")
        print(f"[Config] Sigmas: {[f'{s:.4f}' for s in denoising_sigmas.tolist()]}")
    runtime.denoising_sigmas = denoising_sigmas
    runtime.kv_pipeline.denoising_sigmas = denoising_sigmas


def _refresh_runtime_inference_policy(
    runtime: WindowedModelRuntime,
    args: argparse.Namespace,
    *,
    use_sink_block: bool,
) -> None:
    """Apply per-request inference policy to a resident runtime.

    These fields do not require rebuilding the generator or VAE. Keeping them
    out of ``_runtime_cache_key`` prevents the persistent worker from loading a
    duplicate model when a service request toggles a runtime-only strategy.
    """
    kv_pipeline = runtime.kv_pipeline
    kv_pipeline.context_noise = float(args.context_noise)
    kv_pipeline.context_noise_max = (
        float(args.context_noise_max)
        if args.context_noise_max is not None
        else float(args.context_noise)
    )
    schedule = str(args.context_noise_schedule or "constant").lower()
    if schedule not in ("constant", "linear", "sqrt"):
        raise ValueError(
            f"context_noise_schedule must be one of constant|linear|sqrt, got {args.context_noise_schedule!r}"
        )
    kv_pipeline.context_noise_schedule = schedule
    kv_pipeline._max_prefix_blocks = (
        int(args.max_prefix_blocks)
        if args.max_prefix_blocks is not None and int(args.max_prefix_blocks) >= 0
        else None
    )
    kv_pipeline.block0_sink_enabled = bool(use_sink_block)
    kv_pipeline.prefix_renorm = bool(args.prefix_renorm)
    kv_pipeline.prefix_renorm_alpha = max(0.0, min(1.0, float(args.prefix_renorm_alpha)))
    kv_pipeline.learned_memory_video_downsample = max(1, int(args.learned_memory_video_downsample))
    kv_pipeline.learned_memory_audio_tokens = max(1, int(args.learned_memory_audio_tokens))
    kv_pipeline.learned_memory_video_beta = max(0.0, min(1.0, float(args.learned_memory_video_beta)))
    kv_pipeline.learned_memory_audio_beta = max(0.0, min(1.0, float(args.learned_memory_audio_beta)))
    kv_pipeline.learned_memory_video_anchor_tether = max(
        0.0, min(1.0, float(args.learned_memory_video_anchor_tether))
    )
    kv_pipeline.learned_memory_audio_anchor_tether = max(
        0.0, min(1.0, float(args.learned_memory_audio_anchor_tether))
    )
    kv_pipeline.learned_memory_identity_anchor = bool(args.learned_memory_identity_anchor)
    kv_pipeline.learned_memory_identity_anchor_scale = max(
        0.0, float(args.learned_memory_identity_anchor_scale)
    )
    kv_pipeline.learned_memory_ref_video_anchor = bool(args.learned_memory_ref_video_anchor)
    kv_pipeline.learned_memory_drift_gate = bool(args.learned_memory_drift_gate)
    kv_pipeline.learned_memory_drift_gate_threshold = float(args.learned_memory_drift_gate_threshold)
    kv_pipeline.learned_memory_drift_gate_temperature = max(
        0.0, float(args.learned_memory_drift_gate_temperature)
    )
    kv_pipeline.learned_memory_drift_gate_min = max(
        0.0, min(1.0, float(args.learned_memory_drift_gate_min))
    )
    kv_pipeline.learned_memory_drift_gate_apply_to_color = bool(
        args.learned_memory_drift_gate_apply_to_color
    )
    kv_pipeline.learned_memory_color_alpha = max(
        0.0, min(1.0, float(args.learned_memory_color_alpha))
    )
    kv_pipeline.learned_memory_color_proto_alpha = max(
        0.0, min(1.0, float(args.learned_memory_color_proto_alpha))
    )
    kv_pipeline.learned_memory_color_update_beta = max(
        0.0, min(1.0, float(args.learned_memory_color_update_beta))
    )
    kv_pipeline.learned_memory_color_anchor_tether = max(
        0.0, min(1.0, float(args.learned_memory_color_anchor_tether))
    )
    kv_pipeline.learned_memory_color_proto_grid = max(1, int(args.learned_memory_color_proto_grid))
    kv_pipeline.learned_memory_color_drift_threshold = float(args.learned_memory_color_drift_threshold)
    kv_pipeline.learned_memory_color_max_correction = max(
        0.0, float(args.learned_memory_color_max_correction)
    )
    kv_pipeline.learned_memory_color_enabled = bool(
        kv_pipeline.learned_memory
        and (
            kv_pipeline.learned_memory_color_alpha > 0.0
            or kv_pipeline.learned_memory_color_proto_alpha > 0.0
            or kv_pipeline.learned_memory_color_film
        )
    )


def _load_windowed_model_runtime(
    args: argparse.Namespace,
    *,
    dtype: torch.dtype,
    device: str,
    use_sink_block: bool,
) -> WindowedModelRuntime:
    """Load the resident interactive inference runtime."""
    print("\n[Load] Loading TaoMate generator...")
    world_rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(f"TaoMate requires two model workers, got {world_size}.")
    generator, train_step = load_model_generator(
        model_ckpt_path=args.model_ckpt,
        original_ckpt_path=args.original_ckpt,
        video_height=args.video_height,
        video_width=args.video_width,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        num_audio_sink_tokens=args.num_audio_sink_tokens,
        use_flex_attention=True,
        device="cpu",
        dtype=dtype,
        causal_rope_type=args.causal_rope_type,
        use_mmap=True,
        use_ema=True,
        learned_memory_enabled=args.learned_memory,
        learned_memory_mode=args.learned_memory_mode,
        learned_memory_layer_interval=args.learned_memory_layer_interval,
        learned_memory_video_dim=args.learned_memory_video_dim,
        learned_memory_audio_dim=args.learned_memory_audio_dim,
        learned_memory_heads=args.learned_memory_heads,
        learned_memory_color_film=args.learned_memory_color_film,
        learned_memory_color_film_hidden_dim=args.learned_memory_color_film_hidden_dim,
    )
    from ltx_causal.tensor_parallel import shard_model

    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError("The distributed process group is not initialized.")
    model_group = dist.new_group([0, 1])
    if world_rank == 0:
        print("[Distributed] Sharding the generator across two workers...")
    shard_model(generator, tp_rank=world_rank, tp_size=2, tp_group=model_group)
    generator = generator.to(device)
    generator.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if world_rank == 0:
        print("[Distributed] Generator is resident on both model workers")

    print("[Load] Loading VAEs...")
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=args.original_ckpt,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    denoising_sigmas = compute_denoising_sigmas(
        args.denoising_steps, args.num_inference_steps, device,
    )
    print(f"[Config] Denoising steps: {args.denoising_steps}")
    print(f"[Config] Sigmas: {[f'{s:.4f}' for s in denoising_sigmas.tolist()]}")

    kv_pipeline = KVCacheCausalPipeline(
        generator=generator,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
        num_audio_sink_tokens=args.num_audio_sink_tokens,
        context_noise=args.context_noise,
        context_noise_max=args.context_noise_max,
        context_noise_schedule=args.context_noise_schedule,
        max_prefix_blocks=args.max_prefix_blocks,
        block0_sink_enabled=use_sink_block,
        prefix_renorm=args.prefix_renorm,
        prefix_renorm_alpha=args.prefix_renorm_alpha,
        learned_memory=args.learned_memory,
        learned_memory_video_downsample=args.learned_memory_video_downsample,
        learned_memory_audio_tokens=args.learned_memory_audio_tokens,
        learned_memory_video_beta=args.learned_memory_video_beta,
        learned_memory_audio_beta=args.learned_memory_audio_beta,
        learned_memory_video_anchor_tether=args.learned_memory_video_anchor_tether,
        learned_memory_audio_anchor_tether=args.learned_memory_audio_anchor_tether,
        learned_memory_identity_anchor=args.learned_memory_identity_anchor,
        learned_memory_identity_anchor_scale=args.learned_memory_identity_anchor_scale,
        learned_memory_ref_video_anchor=args.learned_memory_ref_video_anchor,
        learned_memory_drift_gate=args.learned_memory_drift_gate,
        learned_memory_drift_gate_threshold=args.learned_memory_drift_gate_threshold,
        learned_memory_drift_gate_temperature=args.learned_memory_drift_gate_temperature,
        learned_memory_drift_gate_min=args.learned_memory_drift_gate_min,
        learned_memory_drift_gate_apply_to_color=args.learned_memory_drift_gate_apply_to_color,
        learned_memory_color_alpha=args.learned_memory_color_alpha,
        learned_memory_color_proto_alpha=args.learned_memory_color_proto_alpha,
        learned_memory_color_update_beta=args.learned_memory_color_update_beta,
        learned_memory_color_anchor_tether=args.learned_memory_color_anchor_tether,
        learned_memory_color_proto_grid=args.learned_memory_color_proto_grid,
        learned_memory_color_drift_threshold=args.learned_memory_color_drift_threshold,
        learned_memory_color_max_correction=args.learned_memory_color_max_correction,
        learned_memory_color_film=args.learned_memory_color_film,
    )

    video_shape_window0, audio_shape_window0 = compute_latent_shapes(
        num_frames=FRAMES_PER_WINDOW,
        video_height=args.video_height,
        video_width=args.video_width,
        batch_size=1,
    )
    aligned_audio_w0 = compute_aligned_audio_frames(
        total_video_latent_frames=video_shape_window0[1],
        num_frame_per_block=args.num_frame_per_block,
        num_frame_per_block_first=args.num_frame_per_block_first,
    )
    audio_shape_window0[1] = aligned_audio_w0

    video_shape_gen = list(video_shape_window0)
    video_shape_gen[1] = LATENT_FRAMES_PER_GEN
    audio_shape_gen = list(audio_shape_window0)
    audio_shape_gen[1] = BLOCKS_PER_WINDOW * AUDIO_FRAMES_PER_BLOCK

    print(f"[Config] Window 0 shapes: video={video_shape_window0}  audio={audio_shape_window0}")
    print(f"[Config] Window 1+ shapes: video={video_shape_gen}  audio={audio_shape_gen}")

    from ltx_core.model.video_vae.tiling import TilingConfig

    return WindowedModelRuntime(
        generator=generator,
        train_step=train_step,
        video_vae=video_vae,
        audio_vae=audio_vae,
        denoising_sigmas=denoising_sigmas,
        kv_pipeline=kv_pipeline,
        video_shape_window0=video_shape_window0,
        audio_shape_window0=audio_shape_window0,
        video_shape_gen=video_shape_gen,
        audio_shape_gen=audio_shape_gen,
        tiling_config=TilingConfig.default(),
    )


def _prepare_windowed_runtime_context(
    args: argparse.Namespace,
) -> Tuple[int, int, int, bool, torch.dtype, str, bool]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != 2:
        raise RuntimeError(
            f"TaoMate requires exactly two worker processes, got {world_size}."
        )
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    args.device = f"cuda:{local_rank}"
    args.max_prefix_blocks = 2
    return (
        rank,
        local_rank,
        world_size,
        rank == 0,
        torch.bfloat16,
        args.device,
        True,
    )



def warmup_windowed_model_runtime(
    args: argparse.Namespace,
    *,
    destroy_process_group: bool = False,
) -> None:
    """Load and cache the reusable TaoMate runtime without generating media."""
    (
        rank,
        _local_rank,
        _world_size,
        _should_write_outputs,
        dtype,
        device,
        use_sink_block,
    ) = _prepare_windowed_runtime_context(args)
    if rank == 0:
        print("[Warmup] Loading resident 1min TaoMate runtime into GPU memory...")
    _get_windowed_model_runtime(
        args,
        dtype=dtype,
        device=device,
        use_sink_block=use_sink_block,
    )
    if rank == 0:
        encode_device = _resolve_prompt_encode_device(
            args,
            inference_device=device,
            runtime_cached=False,
        )
        if str(encode_device).startswith("cuda"):
            try:
                _get_text_encoder_runtime(
                    args,
                    dtype=dtype,
                    device=encode_device,
                )
                print("[Warmup] Resident text encoder runtime is ready")
            except RuntimeError as exc:
                if _is_cuda_oom(exc):
                    print(
                        "[Warmup][WARN] CUDA text encoder warmup ran out of memory; "
                        "future requests will fall back to the legacy CPU encode path."
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    raise
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if rank == 0:
        print("[Warmup] Resident 1min TaoMate runtime is ready")
    if destroy_process_group and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def run_inference(args: argparse.Namespace, *, destroy_process_group: bool = True) -> None:
    (
        rank,
        local_rank,
        world_size,
        should_write_outputs,
        dtype,
        device,
        use_sink_block,
    ) = _prepare_windowed_runtime_context(args)
    first_stream_profiler = _FirstStreamProfiler(
        enabled=False,
        device=device,
    )
    first_stream_profiler.mark(
        "run_inference_start",
        rank=rank,
        device=device,
    )

    cross_chunk_recent = 2

    print(f"{'=' * 70}")
    print("  TaoMate Streaming Inference")
    print(f"  Windows: {NUM_WINDOWS} x 5s = 60s")
    print(f"  Frames per window: {FRAMES_PER_WINDOW} pixel ({LATENT_FRAMES_PER_WINDOW} latent)")
    print(f"  Blocks per window: {BLOCKS_PER_WINDOW}")
    print(f"  Warm Prefix Blocks: {args.max_prefix_blocks} (non-sink, training-aligned)")
    print(f"  Clean KV: sink + {cross_chunk_recent} recent blocks")
    print(f"  Learned Memory: {args.learned_memory_mode}")
    if args.interactive_prefix_state_in or args.interactive_prefix_state_out:
        print(
            "  Interactive Prefix State: "
            f"in={args.interactive_prefix_state_in or 'none'} "
            f"out={args.interactive_prefix_state_out or 'none'}"
        )
    print(f"  Distributed model workers: {world_size}")
    print(f"{'=' * 70}")

    # ── Load benchmark tasks ──
    with open(args.benchmark_json, "r") as f:
        cases = json.load(f)

    if args.max_cases:
        cases = cases[:args.max_cases]
    prepared_cases = [
        (case, _normalize_case_segments(case))
        for case in cases
    ]

    case_chunk_counts = [len(segments) for _, segments in prepared_cases]
    if case_chunk_counts:
        if min(case_chunk_counts) == max(case_chunk_counts):
            chunk_summary = f"{case_chunk_counts[0]} generation chunks each"
        else:
            chunk_summary = (
                f"{min(case_chunk_counts)}-{max(case_chunk_counts)} generation chunks per case"
            )
    else:
        chunk_summary = "0 generation chunks"
    print(f"\n[Tasks] Loaded {len(cases)} cases, {chunk_summary}")
    if args.log_model_prompts and should_write_outputs:
        print("[ModelPrompt] Exact prompt logging is ON")
    first_stream_profiler.mark(
        "tasks_prepared",
        cases=len(prepared_cases),
        generation_chunks=sum(case_chunk_counts) if case_chunk_counts else 0,
    )
    if (args.interactive_prefix_state_in or args.interactive_prefix_state_out) and len(prepared_cases) != 1:
        raise ValueError(
            "interactive prefix state is only supported for a single interactive case"
        )

    runtime_key = _runtime_cache_key(args, dtype=dtype, device=device)
    runtime_cached = runtime_key in _WINDOWED_MODEL_RUNTIME_CACHE
    prompt_encode_device = _resolve_prompt_encode_device(
        args,
        inference_device=device,
        runtime_cached=runtime_cached,
    )
    first_stream_profiler.mark(
        "prompt_encode_device_resolved",
        requested=args.prompt_encode_device,
        effective=prompt_encode_device,
        runtime_cached=runtime_cached,
    )
    prompt_cache: Dict[str, Dict[str, Any]] = {}
    unique_prompts = set()
    for _, segments in prepared_cases:
        for seg in segments:
            unique_prompts.add(seg["prompt"])
    first_stream_profiler.mark("unique_prompts_collected", unique_prompts=len(unique_prompts))

    prompt_cache_path = os.path.abspath(args.prompt_cache_path)
    prompt_cache_src_rank = 0
    prompt_cache_is_src = (rank == prompt_cache_src_rank)
    first_stream_profiler.mark(
        "prompt_cache_transport_selected",
        transport="file",
    )
    if os.path.exists(prompt_cache_path):
        print(f"\n[Encode] Loading prompt cache from {prompt_cache_path}")
        first_stream_profiler.mark("prompt_cache_load_file_begin")
        prompt_cache = torch.load(prompt_cache_path, map_location="cpu", weights_only=False)
        print(f"[Encode] Loaded {len(prompt_cache)} cached prompts")
        first_stream_profiler.mark("prompt_cache_load_file_done", cached_prompts=len(prompt_cache))
    elif not prompt_cache_is_src:
        print(f"\n[Encode] Waiting for prompt cache file from rank {prompt_cache_src_rank}: {prompt_cache_path}")
        first_stream_profiler.mark("prompt_cache_wait_file_begin")
        wait_start = time.time()
        wait_poll = max(0.001, float(args.prompt_cache_wait_poll_interval))
        while not os.path.exists(prompt_cache_path):
            elapsed_wait = time.time() - wait_start
            if elapsed_wait > float(args.prompt_cache_wait_timeout):
                raise TimeoutError(
                    f"Timed out waiting for prompt cache file: {prompt_cache_path}"
                )
            remaining = max(0.0, float(args.prompt_cache_wait_timeout) - elapsed_wait)
            time.sleep(min(wait_poll, remaining))
        prompt_cache = torch.load(prompt_cache_path, map_location="cpu", weights_only=False)
        print(f"[Encode] Loaded {len(prompt_cache)} cached prompts")
        first_stream_profiler.mark("prompt_cache_wait_file_done", cached_prompts=len(prompt_cache))
    else:
        encode_devices = [prompt_encode_device]
        if str(prompt_encode_device).startswith("cuda"):
            encode_devices.append("cpu")
        encode_error: Optional[RuntimeError] = None
        for encode_device in encode_devices:
            text_encoder = None
            keep_text_encoder_resident = str(encode_device).startswith("cuda")
            prompt_cache.clear()
            try:
                if keep_text_encoder_resident:
                    first_stream_profiler.mark(
                        "text_encoder_runtime_get_begin",
                        device=encode_device,
                    )
                    text_encoder = _get_text_encoder_runtime(
                        args,
                        dtype=dtype,
                        device=encode_device,
                    ).text_encoder
                    first_stream_profiler.mark(
                        "text_encoder_runtime_get_done",
                        device=encode_device,
                    )
                else:
                    print(f"\n[Load] Loading text encoder for prompt cache on {encode_device}...")
                    first_stream_profiler.mark("text_encoder_load_begin", device=encode_device)
                    text_encoder = create_text_encoder_wrapper(
                        checkpoint_path=args.original_ckpt,
                        gemma_path=args.gemma_path,
                        device=torch.device(encode_device),
                        dtype=dtype,
                        place_on_device=str(encode_device).startswith("cuda"),
                    )
                    first_stream_profiler.mark("text_encoder_load_done", device=encode_device)
                    text_encoder.to(encode_device)
                    text_encoder.eval()
                    first_stream_profiler.mark("text_encoder_to_device_done", device=encode_device)

                print("[Encode] Pre-encoding prompts before generator load...")
                with torch.no_grad():
                    resident_cache_hits = 0
                    resident_cache_misses = 0
                    first_encode_marked = False
                    for i, p in enumerate(unique_prompts):
                        if p not in prompt_cache:
                            cached_prompt = _get_resident_prompt_embedding(
                                args,
                                dtype=dtype,
                                device=encode_device,
                                prompt=p,
                            )
                            if cached_prompt is not None:
                                resident_cache_hits += 1
                                prompt_cache[p] = cached_prompt
                                if i == 0:
                                    first_stream_profiler.mark(
                                        "first_prompt_resident_cache_hit",
                                        chars=len(p),
                                        device=encode_device,
                                    )
                            else:
                                resident_cache_misses += 1
                                if not first_encode_marked:
                                    first_encode_marked = True
                                    first_stream_profiler.mark(
                                        "first_prompt_encode_begin",
                                        chars=len(p),
                                        device=encode_device,
                                    )
                                encoded_prompt = _tensor_tree_to_cpu(text_encoder(text_prompts=[p]))
                                prompt_cache[p] = encoded_prompt
                                _put_resident_prompt_embedding(
                                    args,
                                    dtype=dtype,
                                    device=encode_device,
                                    prompt=p,
                                    encoded=encoded_prompt,
                                )
                                first_stream_profiler.mark(
                                    "first_prompt_encode_done",
                                    cache_miss_index=resident_cache_misses,
                                )
                        if (i + 1) % 20 == 0:
                            print(f"  Encoded {i+1}/{len(unique_prompts)} prompts...")
                    first_stream_profiler.mark(
                        "resident_prompt_cache_stats",
                        hits=resident_cache_hits,
                        misses=resident_cache_misses,
                        size=len(_RESIDENT_PROMPT_EMBEDDING_CACHE),
                    )
                    if resident_cache_hits:
                        print(
                            "[Encode] Resident prompt cache "
                            f"hits={resident_cache_hits} misses={resident_cache_misses} "
                            f"size={len(_RESIDENT_PROMPT_EMBEDDING_CACHE)}"
                        )
                if keep_text_encoder_resident:
                    print("[Encode] Keeping resident text encoder runtime in process cache")
                else:
                    text_encoder.to("cpu")
                    del text_encoder
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                prompt_encode_device = encode_device
                print(f"[Encode] Cached {len(prompt_cache)} unique prompts")
                os.makedirs(os.path.dirname(prompt_cache_path), exist_ok=True)
                tmp_cache_path = f"{prompt_cache_path}.tmp.{os.getpid()}"
                torch.save(prompt_cache, tmp_cache_path)
                os.replace(tmp_cache_path, prompt_cache_path)
                print(f"[Encode] Wrote prompt cache to {prompt_cache_path}")
                break
            except RuntimeError as exc:
                if str(encode_device).startswith("cuda") and _is_cuda_oom(exc):
                    encode_error = exc
                    print(
                        "[Encode][WARN] CUDA prompt encoding ran out of memory; "
                        "falling back to CPU for this request."
                    )
                    if text_encoder is not None:
                        try:
                            text_encoder.to("cpu")
                        except Exception:
                            pass
                    if keep_text_encoder_resident:
                        _TEXT_ENCODER_RUNTIME_CACHE.pop(
                            _text_encoder_cache_key(
                                args,
                                dtype=dtype,
                                device=encode_device,
                            ),
                            None,
                        )
                    del text_encoder
                    prompt_cache.clear()
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise
        else:
            if encode_error is not None:
                raise encode_error
    first_stream_profiler.mark("prompt_cache_ready", cached_prompts=len(prompt_cache))

    first_stream_profiler.mark("runtime_get_begin", runtime_cached=runtime_cached)
    runtime = _get_windowed_model_runtime(
        args,
        dtype=dtype,
        device=device,
        use_sink_block=use_sink_block,
    )
    _refresh_runtime_denoising_sigmas(runtime, args, device=device)
    _refresh_runtime_inference_policy(runtime, args, use_sink_block=use_sink_block)
    first_stream_profiler.mark("runtime_get_done", runtime_cached=runtime_cached)
    train_step = runtime.train_step
    video_vae = runtime.video_vae
    audio_vae = runtime.audio_vae
    kv_pipeline = runtime.kv_pipeline
    video_shape_window0 = runtime.video_shape_window0
    audio_shape_window0 = runtime.audio_shape_window0
    video_shape_gen = runtime.video_shape_gen
    audio_shape_gen = runtime.audio_shape_gen
    tiling_config = runtime.tiling_config
    kv_pipeline.profile_callback = (
        first_stream_profiler.mark if args.profile_first_stream else None
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Run inference --
    from torchvision.io import write_video

    for case_idx, (case, segments) in enumerate(prepared_cases):
        case_id = case.get("case_id") or case.get("id") or f"case_{case_idx + 1:03d}"
        description = case.get("description") or case.get("desc") or case_id
        seed = segments[0]["seed"] + args.seed_offset

        print(f"\n{'_' * 70}")
        print(f"[Case {case_idx+1}/{len(cases)}] {case_id}: {description}")
        prompt_window_count = len({int(seg.get("prompt_window_id", idx)) for idx, seg in enumerate(segments)})
        print(
            f"  Seed: {seed}, Prompt Windows: {prompt_window_count}, "
            f"Generation Chunks: {len(segments)}"
        )
        if args.log_model_prompts and should_write_outputs:
            for dbg_idx, dbg_segment in enumerate(segments):
                _log_model_prompt_debug(
                    output_dir=args.output_dir,
                    case_id=case_id,
                    window_idx=dbg_idx,
                    segment=dbg_segment,
                )
        first_stream_profiler.mark(
            "case_start",
            case_id=case_id,
            prompt_windows=prompt_window_count,
            generation_chunks=len(segments),
        )

        cond_task = _case_condition_task(case, segments)
        (
            cond_mode_case,
            first_p,
            end_p,
            audio_p,
            audio_latent_p,
        ) = _resolve_cond_inputs(cond_task, args)
        first_stream_profiler.mark("cond_inputs_resolved", cond_mode=cond_mode_case)
        first_frame_latent: Optional[torch.Tensor] = None
        end_frame_latent: Optional[torch.Tensor] = None
        full_audio_condition_latent: Optional[torch.Tensor] = None
        if cond_mode_case in ("i2v", "ii2v", "tia2v"):
            t_cond = time.perf_counter()
            first_frame_latent = _encode_image_to_latent(
                first_p, video_vae, args.video_height, args.video_width,
                device, dtype, checkpoint_path=args.original_ckpt,
            )
            if cond_mode_case == "ii2v" and end_p:
                end_frame_latent = _encode_image_to_latent(
                    end_p, video_vae, args.video_height, args.video_width,
                    device, dtype, checkpoint_path=args.original_ckpt,
                )
            print(
                f"  [Cond] mode={cond_mode_case} first={os.path.basename(first_p)}"
                + (f" end={os.path.basename(end_p)}" if end_p else "")
                + f" encode={time.perf_counter() - t_cond:.1f}s"
            )
            first_stream_profiler.mark("cond_image_encoded", cond_mode=cond_mode_case)
        if cond_mode_case in ("ta2v", "tia2v"):
            t_audio_cond = time.perf_counter()
            total_audio_condition_frames = int(audio_shape_window0[1]) + max(
                0, len(segments) - 1
            ) * int(audio_shape_gen[1])
            full_audio_condition_latent = _load_or_encode_audio_condition(
                audio_vae=audio_vae,
                checkpoint_path=args.original_ckpt,
                dtype=dtype,
                device=device,
                target_frames=total_audio_condition_frames,
                batch_size=int(video_shape_window0[0]),
                audio_path=audio_p,
                audio_latent_path=audio_latent_p,
                audio_start_time=float(
                    cond_task.get("audio_start_time", args.audio_start_time)
                ),
            )
            print(
                f"  [Cond] clean_audio={os.path.basename(audio_p or audio_latent_p)} "
                f"latent={tuple(full_audio_condition_latent.shape)} "
                f"encode={time.perf_counter() - t_audio_cond:.1f}s"
            )
            first_stream_profiler.mark(
                "cond_audio_encoded",
                cond_mode=cond_mode_case,
                audio_frames=total_audio_condition_frames,
            )

        t_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prefix_video_latent = None
        prefix_audio_latent = None
        decode_prefix_video_latent = None
        prefix_source_prompts: Optional[List[str]] = None
        sink_video_latent = None
        sink_audio_latent = None
        sink_source_prompt: Optional[str] = None
        audio_condition_offset = 0
        learned_memory_state = kv_pipeline._new_learned_memory_state()
        window_offset = 0
        global_block_offset = 0
        cross_chunk_kv_cache = None
        cross_chunk_kv_prompt: Optional[str] = None
        kv_pipeline.last_kv_cache = None
        kv_pipeline.last_kv_cache_segments = []
        deferred_realtime_streamers: List[Tuple[_RealtimeBlockStreamer, int]] = []
        dynamic_stop_enabled = bool(
            args.interactive_dynamic_stop and args.interactive_stop_control_path
        )
        dynamic_stop_max_windows = max(1, int(args.interactive_dynamic_stop_max_windows))
        dynamic_stop_state: Dict[str, Any] = {
            "emitted_blocks": 0,
            "committed_blocks": 0,
            "stop_after_blocks": None,
            "requested_stop_after_blocks": None,
            "overshoot_blocks": 0,
            "stop": False,
            "disabled": False,
            "last_control": None,
        }

        def _poll_dynamic_stop_control() -> Optional[Dict[str, Any]]:
            if not dynamic_stop_enabled:
                return None
            control = _read_json_if_exists(args.interactive_stop_control_path) if rank == 0 else None
            if dist.is_available() and dist.is_initialized():
                obj = [control]
                dist.broadcast_object_list(obj, src=0)
                control = obj[0]
            if isinstance(control, dict):
                dynamic_stop_state["last_control"] = control
            return control if isinstance(control, dict) else None

        def _apply_dynamic_stop_control(control: Optional[Dict[str, Any]]) -> None:
            if not control:
                return
            if control.get("available") is False or control.get("disabled"):
                dynamic_stop_state["disabled"] = True
                return
            stop_after = (
                control.get("stop_after_block_count")
                or control.get("stop_after_chunk_count")
            )
            if not control.get("completed") or stop_after is None:
                return
            emitted_blocks = int(dynamic_stop_state["emitted_blocks"])
            try:
                requested_stop_after = int(stop_after)
            except (TypeError, ValueError):
                requested_stop_after = emitted_blocks
            requested_stop_after = max(0, requested_stop_after)
            previous_stop_after = dynamic_stop_state.get("stop_after_blocks")
            dynamic_stop_state["requested_stop_after_blocks"] = requested_stop_after
            dynamic_stop_state["stop_after_blocks"] = requested_stop_after
            dynamic_stop_state["overshoot_blocks"] = max(
                0,
                emitted_blocks - requested_stop_after,
            )
            if previous_stop_after != requested_stop_after:
                first_stream_profiler.mark(
                    "interactive_dynamic_stop_control",
                    emitted_blocks=emitted_blocks,
                    requested_stop_after_blocks=requested_stop_after,
                    stop_after_blocks=requested_stop_after,
                    overshoot_blocks=int(dynamic_stop_state["overshoot_blocks"]),
                )
            if emitted_blocks >= requested_stop_after:
                dynamic_stop_state["stop"] = True

        def _dynamic_stop_after_block() -> bool:
            if not dynamic_stop_enabled:
                return False
            dynamic_stop_state["emitted_blocks"] = int(dynamic_stop_state["emitted_blocks"]) + 1
            control = _poll_dynamic_stop_control()
            _apply_dynamic_stop_control(control)
            return bool(dynamic_stop_state["stop"])

        def _close_deferred_realtime_streamers(*, force: bool = False) -> None:
            nonlocal deferred_realtime_streamers
            max_open = max(1, int(args.realtime_max_open_streamers))
            while deferred_realtime_streamers and (
                force or len(deferred_realtime_streamers) > max_open
            ):
                deferred_streamer, deferred_seg = deferred_realtime_streamers.pop(0)
                first_stream_profiler.mark(
                    "realtime_deferred_streamer_close_begin",
                    segment=deferred_seg,
                    force=force,
                    remaining=len(deferred_realtime_streamers),
                )
                deferred_streamer.close()
                first_stream_profiler.mark(
                    "realtime_deferred_streamer_close_done",
                    segment=deferred_seg,
                    force=force,
                    remaining=len(deferred_realtime_streamers),
                )

        interactive_state = _load_interactive_prefix_state(
            args.interactive_prefix_state_in,
            device=device,
            dtype=dtype,
        )
        first_stream_profiler.mark(
            "interactive_state_loaded",
            has_state=bool(interactive_state),
        )
        if interactive_state:
            prefix_video_latent = interactive_state.get("prefix_video_latent")
            prefix_audio_latent = interactive_state.get("prefix_audio_latent")
            decode_prefix_video_latent = interactive_state.get("decode_prefix_video_latent")
            prefix_source_prompts = interactive_state.get("prefix_source_prompts")
            sink_video_latent = interactive_state.get("sink_video_latent")
            sink_audio_latent = interactive_state.get("sink_audio_latent")
            sink_source_prompt = interactive_state.get("sink_source_prompt")
            window_offset = int(interactive_state.get("window_count") or 0)
            saved_block_count = interactive_state.get("block_count")
            global_block_offset = (
                int(saved_block_count)
                if saved_block_count is not None
                else int(window_offset) * BLOCKS_PER_WINDOW
            )
            loaded_learned_memory = interactive_state.get("learned_memory_state")
            if isinstance(loaded_learned_memory, LearnedMemoryState):
                learned_memory_state = loaded_learned_memory
            renorm_anchor_restored = _restore_prefix_renorm_anchor(
                kv_pipeline,
                sink_video_latent,
            )
            first_stream_profiler.mark(
                "interactive_renorm_anchor_restored",
                restored=renorm_anchor_restored,
            )
            if prefix_video_latent is not None:
                print(
                    f"  [Interactive] Continuing from {args.interactive_prefix_state_in} "
                    f"(window_offset={window_offset}, block_offset={global_block_offset})"
                )
                if renorm_anchor_restored:
                    print("  [Interactive] Restored prefix-renorm anchor from block0 sink")
                selected_prefix_prompts_for_kv = _interactive_selected_prefix_prompts_for_kv(
                    prefix_source_prompts,
                    kv_pipeline,
                )
                cross_chunk_kv_cache = _get_interactive_prefix_kv_cache(
                    args.interactive_prefix_state_in,
                    prefix_source_prompts=selected_prefix_prompts_for_kv,
                    prefix_video_shape=(
                        tuple(prefix_video_latent.shape)
                        if torch.is_tensor(prefix_video_latent) else None
                    ),
                    prefix_audio_shape=(
                        tuple(prefix_audio_latent.shape)
                        if torch.is_tensor(prefix_audio_latent) else None
                    ),
                    window_count=global_block_offset,
                    device=device,
                    dtype=dtype,
                )
                first_stream_profiler.mark(
                    "interactive_prefix_kv_cache_lookup",
                    hit=bool(cross_chunk_kv_cache is not None),
                    window_count=window_offset,
                    selected_prompts=len(selected_prefix_prompts_for_kv or []),
                )
                cross_chunk_kv_prompt = (
                    prefix_source_prompts[-1] if prefix_source_prompts else None
                )

        request_start_block_offset = int(global_block_offset)
        request_initial_prefix_video = prefix_video_latent
        request_initial_prefix_audio = prefix_audio_latent
        request_initial_decode_prefix_video = decode_prefix_video_latent
        request_initial_prefix_source_prompts = (
            list(prefix_source_prompts) if prefix_source_prompts is not None else None
        )
        request_initial_sink_video = sink_video_latent
        request_initial_sink_audio = sink_audio_latent
        request_initial_sink_source_prompt = sink_source_prompt
        request_initial_learned_memory = (
            _clone_learned_memory_state(learned_memory_state)
            if dynamic_stop_enabled else None
        )
        generated_request_records: List[Dict[str, Any]] = []
        speculative_kv_journal: Optional[Dict[str, Any]] = (
            {"kv_cache": None, "segments": []}
            if dynamic_stop_enabled else None
        )

        seg_idx = 0
        while seg_idx < len(segments):
            segment = segments[seg_idx]
            first_stream_profiler.mark(
                "window_setup_begin",
                segment=seg_idx,
                prefix=prefix_video_latent is not None,
            )
            seg_prompt = segment["prompt"]
            seg_seed = segment["seed"] + args.seed_offset
            seg_seed += window_offset + seg_idx

            # Get text encoding
            conditional_dict = _tensor_tree_to_device(prompt_cache[seg_prompt], device)
            first_stream_profiler.mark("window_prompt_to_device_done", segment=seg_idx)
            window_prefix_kv_cache = (
                cross_chunk_kv_cache if prefix_video_latent is not None else None
            )
            prefix_conditional_dicts = None
            first_stream_profiler.mark(
                "window_prefix_cond_ready",
                segment=seg_idx,
                prefix_cond=prefix_conditional_dicts is not None,
            )

            # Select shapes based on whether this is the first window or not
            is_bootstrap_window = prefix_video_latent is None
            if is_bootstrap_window:
                cur_video_shape = tuple(video_shape_window0)
                cur_audio_shape = tuple(audio_shape_window0)
            else:
                cur_video_shape = tuple(video_shape_gen)
                cur_audio_shape = tuple(audio_shape_gen)

            cache_refresh_elapsed = None
            if not is_bootstrap_window:
                cache_refresh_start = time.perf_counter()
                if cross_chunk_kv_cache is None:
                    selected_prefix_prompts_for_kv = (
                        _interactive_selected_prefix_prompts_for_kv(
                            prefix_source_prompts,
                            kv_pipeline,
                        )
                    )
                    selected_count = len(selected_prefix_prompts_for_kv or [])
                    if selected_count <= 0:
                        raise RuntimeError(
                            "Retained KV fallback requires a tracked latent prefix"
                        )
                    recent_count = max(0, selected_count - 1)
                    if global_block_offset < selected_count:
                        raise RuntimeError(
                            "Retained KV prefix exceeds generated global history: "
                            f"selected={selected_count}, generated={global_block_offset}"
                        )
                    global_prefix_ids = [0]
                    if recent_count > 0:
                        global_prefix_ids.extend(
                            range(
                                global_block_offset - recent_count,
                                global_block_offset,
                            )
                        )
                    cross_chunk_kv_cache = build_interactive_prefix_kv_cache(
                        kv_pipeline=kv_pipeline,
                        selected_conditional_dicts=[conditional_dict] * selected_count,
                        prefix_video_latent=prefix_video_latent,
                        prefix_audio_latent=prefix_audio_latent,
                        next_video_shape=cur_video_shape,
                        next_audio_shape=cur_audio_shape,
                        device=device,
                        dtype=dtype,
                        num_frame_per_block=args.num_frame_per_block,
                        num_frame_per_block_first=args.num_frame_per_block_first,
                        num_audio_sink_tokens=args.num_audio_sink_tokens,
                        learned_memory_state=learned_memory_state,
                        profiler=first_stream_profiler,
                        global_block_indices=global_prefix_ids,
                    )
                    if cross_chunk_kv_cache is None:
                        raise RuntimeError("Global-position retained KV prefill failed")
                    print(
                        " retained_kv_prefill="
                        + ",".join(str(block_id) for block_id in global_prefix_ids),
                        end="",
                        flush=True,
                    )
                elif cross_chunk_kv_prompt != seg_prompt:
                    cross_chunk_kv_cache = _refresh_retained_modal_kv(
                        kv_pipeline=kv_pipeline,
                        persistent_kv_cache=cross_chunk_kv_cache,
                        conditional_dict=conditional_dict,
                        prefix_video_latent=prefix_video_latent,
                        prefix_audio_latent=prefix_audio_latent,
                        next_video_shape=cur_video_shape,
                        next_audio_shape=cur_audio_shape,
                        num_frame_per_block=args.num_frame_per_block,
                        num_frame_per_block_first=args.num_frame_per_block_first,
                        num_audio_sink_tokens=args.num_audio_sink_tokens,
                        learned_memory_state=learned_memory_state,
                    )
                cross_chunk_kv_prompt = seg_prompt
                window_prefix_kv_cache = cross_chunk_kv_cache
                cache_refresh_elapsed = time.perf_counter() - cache_refresh_start

            if (
                speculative_kv_journal is not None
                and speculative_kv_journal.get("kv_cache") is None
                and window_prefix_kv_cache is not None
            ):
                _reset_speculative_kv_journal(
                    speculative_kv_journal,
                    window_prefix_kv_cache,
                )

            prompt_window_id = int(segment.get("prompt_window_id", 0) or 0)
            chunk_index = int(segment.get("chunk_index", seg_idx) or 0)
            print(
                f"  [Prompt {prompt_window_id + 1} / Chunk {chunk_index + 1}/{len(segments)}] "
                "Generating 5 causal blocks (KV-cache)...",
                end="",
                flush=True,
            )
            if cache_refresh_elapsed is not None:
                print(
                    f" kv_refresh={cache_refresh_elapsed:.1f}s",
                    end="",
                    flush=True,
                )
            t_win = time.perf_counter()

            # Generate this window using KV-cache (fast path). Image anchoring
            # is bootstrap-only, while clean-audio conditioning spans every
            # window and is sliced on the global 25 Hz audio-latent timeline.
            if cond_mode_case in ("ta2v", "tia2v"):
                window_cond_mode = cond_mode_case if is_bootstrap_window else "ta2v"
            else:
                window_cond_mode = cond_mode_case if is_bootstrap_window else "t2v"
            window_audio_condition: Optional[torch.Tensor] = None
            if full_audio_condition_latent is not None:
                audio_condition_end = audio_condition_offset + int(cur_audio_shape[1])
                window_audio_condition = full_audio_condition_latent[
                    :, audio_condition_offset:audio_condition_end
                ]
                if window_audio_condition.shape[1] != int(cur_audio_shape[1]):
                    raise ValueError(
                        "Insufficient clean-audio latent frames for window "
                        f"{seg_idx}: got {window_audio_condition.shape[1]}, "
                        f"expected {cur_audio_shape[1]}."
                    )
            realtime_block_callback = None
            realtime_streamer = None
            if should_write_outputs:
                realtime_video_prefix = (
                    decode_prefix_video_latent.detach()
                    if decode_prefix_video_latent is not None else None
                )
                realtime_state: Dict[str, Any] = {"block_idx": 0}
                realtime_stream_dir = os.path.join(args.output_dir, f"{case_id}_streams")
                os.makedirs(realtime_stream_dir, exist_ok=True)
                realtime_streamer = _RealtimeBlockStreamer(
                    write_video_fn=write_video,
                    stream_dir=realtime_stream_dir,
                    case_id=case_id,
                    segment_index=seg_idx,
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                    tiling_config=tiling_config,
                    initial_video_prefix=realtime_video_prefix,
                    max_context_latents=args.realtime_decode_context_latents,
                    queue_size=args.realtime_stream_queue_size,
                    validate_streams=False,
                    fragmented_mp4=True,
                    stream_workers=args.realtime_stream_workers,
                    blocks_per_chunk=args.realtime_stream_blocks_per_chunk,
                    first_chunk_blocks=args.realtime_first_chunk_blocks,
                    output_video_height=args.output_video_height,
                    output_video_width=args.output_video_width,
                    device=device,
                    decode_device="cuda:2",
                    audio_decode_device="cuda:2",
                    fast_preview_frame=False,
                    write_preview_frame=False,
                    keep_cuda_latents=True,
                    write_asr_audio_sidecar=True,
                    profiler=first_stream_profiler,
                )
                first_stream_profiler.mark(
                    "realtime_streamer_ready",
                    segment=seg_idx,
                    async_stream=True,
                    stream_workers=args.realtime_stream_workers,
                    blocks_per_chunk=args.realtime_stream_blocks_per_chunk,
                    first_chunk_blocks=args.realtime_first_chunk_blocks,
                    keep_cuda_latents=True,
                )

                def realtime_block_callback(
                    block_video: torch.Tensor,
                    block_audio: Optional[torch.Tensor],
                    block: Any,
                ) -> None:
                    block_idx = int(realtime_state["block_idx"])
                    first_stream_profiler.mark(
                        "block_callback_enter",
                        segment=seg_idx,
                        block_idx=block_idx,
                    )
                    if block_idx == 0:
                        first_stream_profiler.mark(
                            "first_block_callback_enter",
                            segment=seg_idx,
                            video_shape=tuple(block_video.shape),
                            audio_shape=(
                                tuple(block_audio.shape)
                                if block_audio is not None else None
                            ),
                        )
                    realtime_streamer.submit(block_idx, block_video, block_audio)
                    realtime_state["block_idx"] = block_idx + 1

            if dynamic_stop_enabled:
                stream_block_callback = realtime_block_callback

                def realtime_block_callback(
                    block_video: torch.Tensor,
                    block_audio: Optional[torch.Tensor],
                    block: Any,
                ) -> Dict[str, bool]:
                    stream_result = None
                    if stream_block_callback is not None:
                        stream_result = stream_block_callback(block_video, block_audio, block)
                    stop_now = _dynamic_stop_after_block()
                    return {
                        "stop_after_block": bool(
                            stop_now or _block_callback_requests_stop(stream_result)
                        )
                    }

            segment_emitted_start = int(dynamic_stop_state["emitted_blocks"])
            segment_learned_memory_snapshot = (
                _clone_learned_memory_state(learned_memory_state)
                if dynamic_stop_enabled else None
            )
            raw_audio_frame_count = 0
            generation_succeeded = False
            realtime_streamer_needs_close = False
            try:
                first_stream_profiler.mark(
                    "dit_generate_begin",
                    segment=seg_idx,
                    bootstrap=is_bootstrap_window,
                    video_shape=cur_video_shape,
                    audio_shape=cur_audio_shape,
                )
                video_latent, audio_latent = generate_window_kvcache(
                    kv_pipeline=kv_pipeline,
                    conditional_dict=conditional_dict,
                    prefix_conditional_dicts=prefix_conditional_dicts,
                    video_shape=cur_video_shape,
                    audio_shape=cur_audio_shape,
                    prefix_video_latent=prefix_video_latent,
                    prefix_audio_latent=prefix_audio_latent,
                    seed=seg_seed,
                    device=device,
                    dtype=dtype,
                    num_frame_per_block=args.num_frame_per_block,
                    num_frame_per_block_first=args.num_frame_per_block_first,
                    num_audio_sink_tokens=args.num_audio_sink_tokens,
                    conditioning_mode=window_cond_mode,
                    first_frame_latent=first_frame_latent if is_bootstrap_window else None,
                    end_frame_latent=end_frame_latent if is_bootstrap_window else None,
                    audio_condition_latent=window_audio_condition,
                    learned_memory_state=learned_memory_state,
                    prefix_kv_cache=window_prefix_kv_cache,
                    global_window_index=window_offset + seg_idx,
                    global_block_offset=global_block_offset,
                    block_callback=realtime_block_callback,
                    speculative_kv_journal=speculative_kv_journal,
                )
                raw_audio_frame_count = (
                    int(audio_latent.shape[1]) if audio_latent is not None else 0
                )
                raw_generated_block_count = _generated_block_count_from_latents(
                    video_latent,
                    audio_latent,
                    is_bootstrap_window=is_bootstrap_window,
                )
                raw_records = _generated_block_slices(
                    video_latent,
                    audio_latent,
                    first_global_block_idx=global_block_offset,
                    block_count=raw_generated_block_count,
                    num_frame_per_block=args.num_frame_per_block,
                    num_frame_per_block_first=args.num_frame_per_block_first,
                )
                for block_idx, block_video, block_audio in raw_records:
                    generated_request_records.append(
                        {
                            "block_idx": int(block_idx),
                            "segment_idx": int(seg_idx),
                            "prompt": seg_prompt,
                            "video": block_video.detach(),
                            "audio": (
                                block_audio.detach() if block_audio is not None else None
                            ),
                        }
                    )

                generated_block_count = raw_generated_block_count
                if dynamic_stop_enabled and dynamic_stop_state.get("stop"):
                    requested_stop = int(
                        dynamic_stop_state.get("requested_stop_after_blocks") or 0
                    )
                    generated_block_count = max(
                        0,
                        min(
                            raw_generated_block_count,
                            requested_stop - segment_emitted_start,
                        ),
                    )
                if generated_block_count < raw_generated_block_count:
                    video_latent, audio_latent, _ = _trim_generated_latents_to_blocks(
                        video_latent,
                        audio_latent,
                        first_global_block_idx=global_block_offset,
                        generated_block_count=raw_generated_block_count,
                        keep_block_count=generated_block_count,
                        num_frame_per_block=args.num_frame_per_block,
                        num_frame_per_block_first=args.num_frame_per_block_first,
                    )
                    learned_memory_state = _replay_learned_memory_blocks(
                        segment_learned_memory_snapshot,
                        raw_records[:generated_block_count],
                    )
                    first_stream_profiler.mark(
                        "interactive_speculative_latent_rollback",
                        segment=seg_idx,
                        emitted=raw_generated_block_count,
                        committed=generated_block_count,
                        discarded=raw_generated_block_count - generated_block_count,
                    )

                if dynamic_stop_enabled:
                    dynamic_stop_state["committed_blocks"] = (
                        min(
                            int(dynamic_stop_state["emitted_blocks"]),
                            int(dynamic_stop_state.get("requested_stop_after_blocks") or 0),
                        )
                        if dynamic_stop_state.get("stop") else
                        int(dynamic_stop_state["emitted_blocks"])
                    )
                committed_cache = getattr(kv_pipeline, "last_kv_cache", None)
                if committed_cache is None:
                    raise RuntimeError("Generation did not commit clean KV")
                if (
                    speculative_kv_journal is not None
                    and speculative_kv_journal.get("kv_cache") is None
                ):
                    _reset_speculative_kv_journal(
                        speculative_kv_journal,
                        committed_cache,
                    )
                if dynamic_stop_enabled and dynamic_stop_state.get("stop"):
                    absolute_block_limit = (
                        request_start_block_offset
                        + int(dynamic_stop_state["committed_blocks"])
                    )
                    committed_cache = _commit_speculative_kv_journal(
                        speculative_kv_journal,
                        absolute_block_limit=absolute_block_limit,
                        recent_blocks=cross_chunk_recent,
                    )
                    if committed_cache is None:
                        raise RuntimeError(
                            "Unable to commit clean KV at ASR boundary "
                            f"block {absolute_block_limit}"
                        )
                else:
                    committed_cache = _prune_clean_kv_to_sink_recent(
                        committed_cache,
                        recent_blocks=cross_chunk_recent,
                    )
                    if committed_cache is None:
                        raise RuntimeError(
                            "Unable to retain sink + recent clean KV after window"
                        )
                cross_chunk_kv_cache = committed_cache
                cross_chunk_kv_prompt = seg_prompt
                kv_pipeline.last_kv_cache = committed_cache
                kv_pipeline.last_kv_cache_segments = (
                    _clone_interactive_kv_segments(committed_cache)
                )
                committed_segments = kv_pipeline.last_kv_cache_segments
                block_ids = ",".join(
                    str(segment.get("block_idx", "?"))
                    for segment in committed_segments
                )
                print(
                    f" kv=retained[{block_ids}]",
                    end="",
                    flush=True,
                )
                generation_succeeded = True
            finally:
                if realtime_streamer is not None:
                    if generation_succeeded:
                        first_stream_profiler.mark("realtime_streamer_flush_begin", segment=seg_idx)
                        realtime_streamer.flush()
                        realtime_streamer_needs_close = True
                        first_stream_profiler.mark("realtime_streamer_flush_done", segment=seg_idx)
                    else:
                        first_stream_profiler.mark("realtime_streamer_close_begin", segment=seg_idx)
                        realtime_streamer.close()
                        first_stream_profiler.mark("realtime_streamer_close_done", segment=seg_idx)
            first_stream_profiler.mark("dit_generate_done", segment=seg_idx)
            if window_audio_condition is not None:
                audio_condition_offset += raw_audio_frame_count
            del conditional_dict, prefix_conditional_dicts
            elapsed_win = time.perf_counter() - t_win
            print(f" gen={elapsed_win:.1f}s", end="", flush=True)
            if generated_block_count != BLOCKS_PER_WINDOW:
                print(f" blocks={generated_block_count}", end="", flush=True)
            global_block_offset += generated_block_count

            # -- Next-window prefix --
            SINK_FRAMES = args.num_frame_per_block_first
            LATENT_RECENT_BLOCKS = max(4, int(cross_chunk_recent))
            RECENT_FRAMES = LATENT_RECENT_BLOCKS * args.num_frame_per_block

            if prefix_video_latent is None:
                next_prefix_video = video_latent.detach()
                next_prefix_audio = audio_latent.detach() if audio_latent is not None else None
                next_prefix_source_prompts = [seg_prompt] * generated_block_count
                sink_video_latent = video_latent[:, :SINK_FRAMES].detach().clone()
                sink_source_prompt = seg_prompt
                if audio_latent is not None:
                    sink_audio_frames = compute_aligned_audio_frames(
                        SINK_FRAMES,
                        args.num_frame_per_block,
                        args.num_frame_per_block_first,
                    )
                    sink_audio_latent = audio_latent[:, :sink_audio_frames].detach().clone()
                print(f" [sink saved: {SINK_FRAMES}v frames]", end="")
            else:
                if sink_video_latent is None:
                    raise RuntimeError("Missing sink latent for continuation prefix")
                previous_recent_video = prefix_video_latent[:, SINK_FRAMES:]
                recent_video_candidates = torch.cat(
                    [previous_recent_video, video_latent.detach()], dim=1,
                )
                recent_video = recent_video_candidates[:, -RECENT_FRAMES:]
                next_prefix_video = torch.cat([sink_video_latent, recent_video], dim=1)
                del recent_video_candidates, previous_recent_video, recent_video
                previous_recent_prompts = (
                    list(prefix_source_prompts[1:]) if prefix_source_prompts else []
                )
                recent_prompt_candidates = [
                    *previous_recent_prompts,
                    *([seg_prompt] * generated_block_count),
                ]
                next_prefix_source_prompts = [
                    sink_source_prompt if sink_source_prompt is not None else segments[0]["prompt"],
                    *recent_prompt_candidates[-LATENT_RECENT_BLOCKS:],
                ]

                if sink_audio_latent is not None and audio_latent is not None:
                    expected_prefix_audio_frames = compute_aligned_audio_frames(
                        int(next_prefix_video.shape[1]),
                        args.num_frame_per_block,
                        args.num_frame_per_block_first,
                    )
                    recent_audio_frames = (
                        expected_prefix_audio_frames - int(sink_audio_latent.shape[1])
                    )
                    previous_recent_audio = (
                        prefix_audio_latent[:, int(sink_audio_latent.shape[1]):]
                        if prefix_audio_latent is not None else None
                    )
                    recent_audio_candidates = (
                        torch.cat([previous_recent_audio, audio_latent.detach()], dim=1)
                        if previous_recent_audio is not None
                        else audio_latent.detach()
                    )
                    recent_audio = recent_audio_candidates[:, -recent_audio_frames:]
                    next_prefix_audio = torch.cat([sink_audio_latent, recent_audio], dim=1)
                    del previous_recent_audio, recent_audio_candidates, recent_audio
                else:
                    next_prefix_audio = audio_latent.detach() if audio_latent is not None else None

            _validate_audio_prefix_alignment(
                next_prefix_video,
                next_prefix_audio,
                num_frame_per_block=args.num_frame_per_block,
                num_frame_per_block_first=args.num_frame_per_block_first,
                label=f"case {case_id} window {window_offset + seg_idx} next prefix",
            )

            t_dec = time.perf_counter()
            if should_write_outputs and int(args.realtime_decode_context_latents) > 0:
                context_video = (
                    video_latent.detach()
                    if decode_prefix_video_latent is None
                    else torch.cat(
                        [
                            decode_prefix_video_latent.detach(),
                            video_latent.detach(),
                        ],
                        dim=1,
                    )
                )
                next_decode_prefix_video = context_video[
                    :, -int(args.realtime_decode_context_latents):
                ].contiguous()
                del context_video
            else:
                next_decode_prefix_video = None

            # Update prefix for next window
            prefix_video_latent = next_prefix_video
            prefix_audio_latent = next_prefix_audio
            decode_prefix_video_latent = next_decode_prefix_video
            prefix_source_prompts = next_prefix_source_prompts
            # The next normal window rebuilds prefix KV from the compact
            # sink+recent latent prefix. Keep only an explicitly prepared
            # session cache; otherwise this reference retains the previous
            # window's full KV alongside the new one and can double peak VRAM.
            kv_pipeline.last_kv_cache = None
            kv_pipeline.last_kv_cache_segments = []

            defer_realtime_close_until_state_ready = (
                bool(args.interactive_prefix_state_out)
                and seg_idx == len(segments) - 1
            )
            if (
                realtime_streamer is not None
                and realtime_streamer_needs_close
                and not defer_realtime_close_until_state_ready
            ):
                deferred_realtime_streamers.append((realtime_streamer, seg_idx))
                first_stream_profiler.mark(
                    "realtime_streamer_close_deferred",
                    segment=seg_idx,
                    pending=len(deferred_realtime_streamers),
                )
                _close_deferred_realtime_streamers(force=False)
                realtime_streamer_needs_close = False

            del video_latent, audio_latent, next_prefix_video, next_prefix_audio
            del next_decode_prefix_video, next_prefix_source_prompts
            elapsed_dec = time.perf_counter() - t_dec
            print(f" dec={elapsed_dec:.1f}s")

            if dynamic_stop_enabled and dynamic_stop_state.get("stop"):
                if should_write_outputs:
                    print(
                        "  [DynamicStop] ASR completion reached; "
                        f"emitted_blocks={dynamic_stop_state.get('emitted_blocks')} "
                        f"requested={dynamic_stop_state.get('requested_stop_after_blocks')} "
                        f"stop_after={dynamic_stop_state.get('stop_after_blocks')} "
                        f"committed={dynamic_stop_state.get('committed_blocks')} "
                        f"overshoot={dynamic_stop_state.get('overshoot_blocks')}"
                    )
                seg_idx += 1
                break

            if (
                dynamic_stop_enabled
                and seg_idx == len(segments) - 1
                and not dynamic_stop_state.get("disabled")
            ):
                if len(segments) < dynamic_stop_max_windows:
                    extra_segment = dict(segment)
                    extra_segment["segment_id"] = len(segments)
                    extra_segment["chunk_index"] = len(segments)
                    extra_segment["prompt_window_id"] = int(
                        segment.get("prompt_window_id", 0) or 0
                    )
                    extra_segment["is_prompt_repeat"] = True
                    segments.append(extra_segment)
                    if should_write_outputs:
                        print(
                            "  [DynamicStop] ASR has not reported completion; "
                            f"extending current prompt to generation chunk {len(segments)}/"
                            f"{dynamic_stop_max_windows}"
                        )
                elif should_write_outputs:
                    print(
                        "  [DynamicStop] Reached max continuation chunks without "
                        "ASR completion; stopping at safety cap."
                    )

            seg_idx += 1

        emitted_request_blocks = int(dynamic_stop_state.get("emitted_blocks") or 0)
        committed_request_blocks = int(
            dynamic_stop_state.get("committed_blocks", emitted_request_blocks)
        )
        discarded_speculative_blocks = max(
            0,
            emitted_request_blocks - committed_request_blocks,
        )
        if dynamic_stop_enabled and discarded_speculative_blocks > 0:
            committed_records = generated_request_records[:committed_request_blocks]
            rebuilt_prefix = _rebuild_committed_interactive_prefix(
                initial_prefix_video=request_initial_prefix_video,
                initial_prefix_audio=request_initial_prefix_audio,
                initial_decode_prefix_video=request_initial_decode_prefix_video,
                initial_prefix_source_prompts=request_initial_prefix_source_prompts,
                initial_sink_video=request_initial_sink_video,
                initial_sink_audio=request_initial_sink_audio,
                initial_sink_source_prompt=request_initial_sink_source_prompt,
                request_start_block_offset=request_start_block_offset,
                generated_records=generated_request_records,
                committed_generated_blocks=committed_request_blocks,
                recent_blocks=max(4, int(cross_chunk_recent)),
                decode_context_latents=args.realtime_decode_context_latents,
                num_frame_per_block=args.num_frame_per_block,
                num_frame_per_block_first=args.num_frame_per_block_first,
            )
            prefix_video_latent = rebuilt_prefix["prefix_video_latent"]
            prefix_audio_latent = rebuilt_prefix["prefix_audio_latent"]
            decode_prefix_video_latent = rebuilt_prefix["decode_prefix_video_latent"]
            prefix_source_prompts = rebuilt_prefix["prefix_source_prompts"]
            sink_video_latent = rebuilt_prefix["sink_video_latent"]
            sink_audio_latent = rebuilt_prefix["sink_audio_latent"]
            sink_source_prompt = rebuilt_prefix["sink_source_prompt"]
            learned_memory_state = _replay_learned_memory_blocks(
                _clone_learned_memory_state(request_initial_learned_memory),
                [
                    (
                        int(record["block_idx"]),
                        record["video"],
                        record.get("audio"),
                    )
                    for record in committed_records
                ],
            )
            global_block_offset = request_start_block_offset + committed_request_blocks
            cross_chunk_kv_cache = _commit_speculative_kv_journal(
                speculative_kv_journal,
                absolute_block_limit=global_block_offset,
                recent_blocks=cross_chunk_recent,
            )
            if cross_chunk_kv_cache is None:
                raise RuntimeError(
                    "Speculative KV journal could not reproduce the visible ASR endpoint"
                )
            kv_pipeline.last_kv_cache = cross_chunk_kv_cache
            kv_pipeline.last_kv_cache_segments = (
                _clone_interactive_kv_segments(cross_chunk_kv_cache)
            )
            _validate_audio_prefix_alignment(
                prefix_video_latent,
                prefix_audio_latent,
                num_frame_per_block=args.num_frame_per_block,
                num_frame_per_block_first=args.num_frame_per_block_first,
                label=f"case {case_id} speculative ASR commit",
            )
            if should_write_outputs:
                committed_ids = ",".join(
                    str(segment.get("block_idx", "?"))
                    for segment in _clone_interactive_kv_segments(cross_chunk_kv_cache)
                )
                print(
                    "  [SpecCommit] "
                    f"emitted={emitted_request_blocks} "
                    f"committed={committed_request_blocks} "
                    f"discarded={discarded_speculative_blocks} "
                    f"prefix_block_offset={global_block_offset} "
                    f"kv=[{committed_ids}]"
                )

        completed_window_count = window_offset + seg_idx

        if dynamic_stop_enabled and rank == 0 and args.interactive_stop_control_path:
            generation_result_path = os.path.join(
                os.path.dirname(os.path.abspath(args.interactive_stop_control_path)),
                "interactive_generation_result.json",
            )
            last_control = dynamic_stop_state.get("last_control") or {}
            _write_json_atomic(
                generation_result_path,
                {
                    "generated_block_count": emitted_request_blocks,
                    "committed_generated_block_count": int(
                        committed_request_blocks
                    ),
                    "requested_stop_after_block_count": dynamic_stop_state.get(
                        "requested_stop_after_blocks"
                    ),
                    "committed_prefix_total_block_count": int(global_block_offset),
                    "discarded_speculative_block_count": discarded_speculative_blocks,
                    "overshoot_block_count": discarded_speculative_blocks,
                    "completion_source": last_control.get("completion_source"),
                    "asr_observed_block_count": last_control.get("observed_block_count"),
                    "time": time.time(),
                },
            )

        if rank == 0 and is_inference_profile_enabled():
            dump_inference_profile(
                label=f"case={case_id}",
                log_fn=lambda m: print(f"  {m}", flush=True),
            )

        if args.interactive_prefix_state_out:
            cache_payload = _build_interactive_prefix_state_payload(
                case_id=case_id,
                window_count=completed_window_count,
                prefix_video_latent=prefix_video_latent,
                prefix_audio_latent=prefix_audio_latent,
                decode_prefix_video_latent=decode_prefix_video_latent,
                prefix_source_prompts=prefix_source_prompts,
                sink_video_latent=sink_video_latent,
                sink_audio_latent=sink_audio_latent,
                sink_source_prompt=sink_source_prompt,
                learned_memory_state=learned_memory_state,
                cpu=False,
                block_count=global_block_offset,
            )
            _put_interactive_prefix_state_cache(
                args.interactive_prefix_state_out,
                cache_payload,
                device=device,
                dtype=dtype,
            )
            selected_prefix_prompts_for_kv = _interactive_selected_prefix_prompts_for_kv(
                prefix_source_prompts,
                kv_pipeline,
            )
            if cross_chunk_kv_cache is not None:
                _put_interactive_prefix_kv_cache(
                    args.interactive_prefix_state_out,
                    kv_cache=cross_chunk_kv_cache,
                    prefix_source_prompts=selected_prefix_prompts_for_kv,
                    prefix_video_shape=(
                        tuple(prefix_video_latent.shape)
                        if torch.is_tensor(prefix_video_latent) else None
                    ),
                    prefix_audio_shape=(
                        tuple(prefix_audio_latent.shape)
                        if torch.is_tensor(prefix_audio_latent) else None
                    ),
                    window_count=global_block_offset,
                    device=device,
                    dtype=dtype,
                )
                first_stream_profiler.mark(
                    "interactive_prefix_kv_cache_stored",
                    window_count=completed_window_count,
                    block_count=global_block_offset,
                    selected_prompts=len(selected_prefix_prompts_for_kv or []),
                )
            first_stream_profiler.mark(
                "interactive_state_cached",
                window_count=completed_window_count,
                block_count=global_block_offset,
            )

        if not should_write_outputs:
            print(f"  [Distributed] rank={rank} generated latents; skipping decode/save")
            del prefix_video_latent, prefix_audio_latent, decode_prefix_video_latent
            del prefix_source_prompts
            del first_frame_latent, end_frame_latent
            del sink_video_latent, sink_audio_latent, sink_source_prompt
            del learned_memory_state
            gc.collect()
            torch.cuda.empty_cache()
            continue

        elapsed_total = time.perf_counter() - t_start
        print(f"  [Interactive] stream complete after {elapsed_total:.1f}s")
        if args.interactive_prefix_state_out:
            state_payload = _build_interactive_prefix_state_payload(
                case_id=case_id,
                window_count=completed_window_count,
                prefix_video_latent=prefix_video_latent,
                prefix_audio_latent=prefix_audio_latent,
                decode_prefix_video_latent=decode_prefix_video_latent,
                prefix_source_prompts=prefix_source_prompts,
                sink_video_latent=sink_video_latent,
                sink_audio_latent=sink_audio_latent,
                sink_source_prompt=sink_source_prompt,
                learned_memory_state=learned_memory_state,
                cpu=True,
                block_count=global_block_offset,
            )
            first_stream_profiler.mark("interactive_prefix_state_cpu_payload_built")
            _save_interactive_prefix_state(args.interactive_prefix_state_out, state_payload)
            _put_interactive_prefix_state_cache(
                args.interactive_prefix_state_out,
                cache_payload,
                device=device,
                dtype=dtype,
            )
            first_stream_profiler.mark("interactive_prefix_state_saved")
            print(f"  [Interactive] Saved prefix state: {args.interactive_prefix_state_out}")
            _notify_avatar_worker_interactive_ready(
                prefix_state_out=args.interactive_prefix_state_out,
                output_dir=args.output_dir,
                window_count=completed_window_count,
            )
            first_stream_profiler.mark("interactive_ready_status_written")

        if realtime_streamer is not None and realtime_streamer_needs_close:
            deferred_realtime_streamers.append((realtime_streamer, seg_idx))
            first_stream_profiler.mark(
                "realtime_streamer_close_deferred",
                segment=seg_idx,
                pending=len(deferred_realtime_streamers),
                final_window=True,
            )
        _close_deferred_realtime_streamers(force=True)
        del prefix_video_latent, prefix_audio_latent, decode_prefix_video_latent
        del prefix_source_prompts
        del first_frame_latent, end_frame_latent
        del sink_video_latent, sink_audio_latent, sink_source_prompt
        del learned_memory_state
        gc.collect()
        torch.cuda.empty_cache()
        continue

    print(f"\n{'=' * 70}")
    print(f"  All cases complete. Output: {args.output_dir}")
    print(f"{'=' * 70}")
    if destroy_process_group and world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def main():
    run_inference(parse_args())


if __name__ == "__main__":
    main()
