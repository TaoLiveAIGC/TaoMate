"""
KV Cache inference engine for CausalLTXModel.

All KV-cache-related logic lives in this file. Existing training/inference code
is never modified — only a thin ``forward_inference()`` delegator is added to
CausalLTXModel (see causal_model.py).

Architecture:
    attention_with_cache()       — single attention op with KV cache
    block_forward_with_cache()   — one transformer block with KV cache
    model_forward_inference()    — full model forward (entry point)

Why KV cache is *exact* (not approximate):
    1. Causal mask: prefix tokens cannot attend to future blocks → their K/V
       are independent of the current block.
    2. Prefix σ=0 → AdaLN output is constant → K/V are identical across
       denoising steps.
    3. RoPE is baked into K after projection → cached K carries correct
       positional information.
"""

from collections import OrderedDict
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Optional, Tuple, List, TYPE_CHECKING, Callable
import weakref

import torch
import torch.utils.checkpoint
from torch import Tensor

from ltx_causal.attention.flex_attention_utils import standard_attention_forward
from ltx_causal.attention.causal_attention import CausalLTXAttention
from ltx_causal.transformer.causal_block import CausalAVTransformerBlock, rms_norm
from ltx_causal.rope.causal_rope import causal_precompute_freqs_cis, CausalRopeType
from ltx_causal.tensor_parallel import (
    RowParallelLinear,
    TPRMSNorm,
    tp_qk_norm_fused,
    is_compile_sync_qk_norm,
)
from ltx_causal.transformer.pyramid_kv import PyramidKVPolicy, get_active_capture_hook

# === Optional Sequence Parallel support ===
# Mirror the lazy-import pattern used by causal_attention.py / causal_model.py
# so this file stays usable in inference-only environments where the heavy
# ltx_distillation training deps (lmdb, etc.) are not installed.
try:
    from taomate.runtime_support.parallel import (  # type: ignore[import-not-found]
        is_sp_enabled as _is_sp_enabled,
        get_sp_world_size as _get_sp_world_size,
        get_sp_rank as _get_sp_rank,
        get_sp_group as _get_sp_group,
        seq_all_to_all_head as _seq_all_to_all_head,
        seq_all_to_all_head_many as _seq_all_to_all_head_many,
        seq_all_to_all_head_many_async as _seq_all_to_all_head_many_async,
        head_all_to_all_seq as _head_all_to_all_seq,
        seq_all_to_all_head_async as _seq_all_to_all_head_async,
        head_all_to_all_seq_async as _head_all_to_all_seq_async,
        split_sequence as _split_sequence,
        gather_sequence as _gather_sequence,
    )
except Exception:  # ImportError or transitive dep missing
    def _is_sp_enabled() -> bool:  # type: ignore[misc]
        return False

    def _get_sp_world_size() -> int:  # type: ignore[misc]
        return 1

    def _get_sp_rank() -> int:  # type: ignore[misc]
        return 0

    def _get_sp_group():  # type: ignore[misc]
        return None

    def _seq_all_to_all_head(x, group=None, sp_size=None):  # type: ignore[misc]
        return x

    def _seq_all_to_all_head_many(tensors, group=None, sp_size=None):  # type: ignore[misc]
        return tuple(tensors)

    def _seq_all_to_all_head_many_async(tensors, group=None, sp_size=None):  # type: ignore[misc]
        class _ImmediateMany:
            def __init__(self, values):
                self.values = tuple(values)
            def wait(self):
                return self.values
        return _ImmediateMany(tensors)

    def _head_all_to_all_seq(x, group=None, sp_size=None):  # type: ignore[misc]
        return x

    def _seq_all_to_all_head_async(x, group=None, sp_size=None):  # type: ignore[misc]
        class _Immediate:
            def __init__(self, value):
                self.value = value
            def wait(self):
                return self.value
        return _Immediate(x)

    def _head_all_to_all_seq_async(x, group=None, sp_size=None):  # type: ignore[misc]
        class _Immediate:
            def __init__(self, value):
                self.value = value
            def wait(self):
                return self.value
        return _Immediate(x)

    def _split_sequence(x, dim=1, sp_size=None, sp_rank=None):  # type: ignore[misc]
        return x

    def _gather_sequence(x, dim=1, group=None, sp_size=None, sp_rank=None):  # type: ignore[misc]
        return x

if TYPE_CHECKING:
    from ltx_causal.transformer.causal_model import CausalLTXModel


_LTX_PROFILE_ENABLED = False
_LTX_PROFILE_DETAIL_ENABLED = False
_LTX_COMPILE_ENABLED = False
_ASYNC_ULYSSES_ENABLED = False
_ASYNC_ULYSSES_STRICT_V_FIRST = False
_ASYNC_ULYSSES_PACK_QK = False
_FUSE_ULYSSES_A2A = True
_FUSE_CROSS_MODAL_A2A = True
_CROSS_MODAL_PAIR_ASYNC_H2S = False
_CROSS_MODAL_TP_OUT_OVERLAP = False
_CACHE_CONTEXT_PROJECTION = False
_CONTEXT_PROJECTION_CACHE_SIZE = 16
_CACHE_ROPE = False
_ROPE_CACHE_SIZE = 16
_CACHE_TEXT_CROSS_KV = False
_TEXT_CROSS_KV_CACHE_SIZE = 4
_CACHE_LEARNED_MEMORY_PROJECTION = False
_LEARNED_MEMORY_PROJECTION_CACHE_SIZE = 8
_PAIR_TEXT_CROSS_ASYNC_H2S = False
_TEXT_CROSS_TP_OUT_OVERLAP = False
_TP_FF_OVERLAP = False
_TP_AUDIO_FF_OVERLAP = False

def _mark_dyn(t: Optional[Tensor], *dims: int) -> None:
    """对 tensor 的指定 dims 标记为动态维度（仅 LTX_COMPILE=1 生效）。

    Dynamo 对每个未标记的具体 size 都生成专门的 guard，shape 变化即
    recompile。mark_dynamic 后该维度走 symbolic 形状，不再触发 recompile。
    None / 非 tensor 输入直接跳过，方便在 forward 入口批量调用。
    """
    if not _LTX_COMPILE_ENABLED or t is None or not torch.is_tensor(t):
        return
    for d in dims:
        if d < t.dim():
            try:
                torch._dynamo.mark_dynamic(t, d)
            except Exception:
                # mark_dynamic 在某些 PyTorch 版本下对已编译 tensor 抛错；
                # 静默忽略不影响正确性，最差只是退化为 recompile。
                pass


class _BlockProfiler:
    """Per-block forward CUDA timing collector.

    Enabled by ``LTX_PROFILE=1``. Records ``cuda.Event`` pairs around each
    transformer block plus aggregate "pre_blocks" / "post_blocks" phases.
    All events are queued without host sync; aggregation/sync happens only
    when ``dump()`` is called, so the hot loop is barely affected.

    Usage from inference scripts::

        from ltx_causal.transformer.kv_cache import dump_inference_profile
        dump_inference_profile(label="task gXX", log_fn=log)
    """

    def __init__(self) -> None:
        # (block_idx, start_event, end_event)
        self.block_records: List[Tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        # (phase_name, start_event, end_event)
        self.phase_records: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        # (block_idx, detail_name, start_event, end_event)
        self.detail_records: List[Tuple[int, str, torch.cuda.Event, torch.cuda.Event]] = []
        self.forward_calls: int = 0

    @staticmethod
    def _new_event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def record_phase(
        self, name: str, start: torch.cuda.Event, end: torch.cuda.Event,
    ) -> None:
        self.phase_records.append((name, start, end))

    def record_block(
        self, idx: int, start: torch.cuda.Event, end: torch.cuda.Event,
    ) -> None:
        self.block_records.append((idx, start, end))

    def record_detail(
        self, idx: int, name: str, start: torch.cuda.Event, end: torch.cuda.Event,
    ) -> None:
        self.detail_records.append((idx, name, start, end))

    def dump(self, label: str = "", log_fn: Callable[[str], None] = print) -> None:
        if not self.block_records and not self.phase_records and not self.detail_records:
            return
        # All recorded events have already been queued; sync to make
        # elapsed_time() valid.
        torch.cuda.synchronize()

        prefix = f"[Profile{(' ' + label) if label else ''}]"
        log_fn(f"{prefix} forward_inference calls: {self.forward_calls}")

        # ── Phase breakdown ──
        if self.phase_records:
            by_name: dict = {}
            for name, s, e in self.phase_records:
                by_name.setdefault(name, []).append(s.elapsed_time(e))
            log_fn(f"{prefix} phase breakdown (avg ms / call):")
            for name in sorted(by_name.keys()):
                ts = by_name[name]
                avg = sum(ts) / len(ts)
                tot = sum(ts)
                log_fn(f"{prefix}   {name:<20s} avg={avg:7.3f} ms  "
                       f"total={tot:8.1f} ms  n={len(ts)}")

        # ── Per-block stats ──
        if self.block_records:
            per_block: dict = {}
            for idx, s, e in self.block_records:
                per_block.setdefault(idx, []).append(s.elapsed_time(e))
            block_stats = []
            for idx in sorted(per_block.keys()):
                ts = per_block[idx]
                avg = sum(ts) / len(ts)
                block_stats.append((idx, avg, sum(ts)))
            avg_all = sum(a for _, a, _ in block_stats) / max(1, len(block_stats))
            tot_all = sum(t for _, _, t in block_stats)
            log_fn(f"{prefix} per-block: {len(block_stats)} blocks, "
                   f"avg={avg_all:.3f} ms / block, total={tot_all:.1f} ms")
            log_fn(f"{prefix} top-3 slowest blocks:")
            for idx, avg, tot in sorted(block_stats, key=lambda x: x[1], reverse=True)[:3]:
                log_fn(f"{prefix}   block {idx:02d}: avg={avg:.3f} ms  total={tot:.1f} ms")

        # ── Optional intra-block detail stats ──
        if self.detail_records:
            by_name: dict = {}
            by_pair: dict = {}
            for idx, name, s, e in self.detail_records:
                dt = s.elapsed_time(e)
                by_name.setdefault(name, []).append(dt)
                by_pair.setdefault((idx, name), []).append(dt)

            log_fn(f"{prefix} detail breakdown (avg ms / section):")
            detail_stats = []
            for name in sorted(by_name.keys()):
                ts = by_name[name]
                avg = sum(ts) / len(ts)
                tot = sum(ts)
                detail_stats.append((name, avg, tot, len(ts)))
            for name, avg, tot, count in sorted(detail_stats, key=lambda x: x[2], reverse=True):
                log_fn(f"{prefix}   {name:<20s} avg={avg:7.3f} ms  "
                       f"total={tot:8.1f} ms  n={count}")

            pair_stats = []
            for (idx, name), ts in by_pair.items():
                pair_stats.append((idx, name, sum(ts) / len(ts), sum(ts)))
            log_fn(f"{prefix} top-8 slowest block sections:")
            for idx, name, avg, tot in sorted(pair_stats, key=lambda x: x[2], reverse=True)[:8]:
                log_fn(f"{prefix}   block {idx:02d} {name:<16s} "
                       f"avg={avg:.3f} ms  total={tot:.1f} ms")

        # ── Reset for next task ──
        self.block_records.clear()
        self.phase_records.clear()
        self.detail_records.clear()
        self.forward_calls = 0


_PROFILER: Optional[_BlockProfiler] = (
    _BlockProfiler() if _LTX_PROFILE_ENABLED else None
)


def dump_inference_profile(
    label: str = "",
    log_fn: Callable[[str], None] = print,
) -> None:
    """Public dump hook for inference scripts (no-op if LTX_PROFILE not set)."""
    if _PROFILER is not None:
        _PROFILER.dump(label=label, log_fn=log_fn)


def is_profiling_enabled() -> bool:
    """Return True iff LTX_PROFILE=1 was set at process start."""
    return _PROFILER is not None


def _profile_detail_start() -> Optional[torch.cuda.Event]:
    if _PROFILER is None or not _LTX_PROFILE_DETAIL_ENABLED:
        return None
    event = _PROFILER._new_event()
    event.record()
    return event


def _profile_detail_end(
    layer_idx: Optional[int],
    name: str,
    start: Optional[torch.cuda.Event],
) -> None:
    if _PROFILER is None or start is None:
        return
    end = _PROFILER._new_event()
    end.record()
    _PROFILER.record_detail(-1 if layer_idx is None else int(layer_idx), name, start, end)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LayerKVCache:
    """Per-layer KV cache for 4 attention types (text K/V are not cached).

    Shapes (after reshape to multi-head format):
        video_self_{k,v}: [B, L_v_cached, H_video, D_h_video]
        audio_self_{k,v}: [B, L_a_cached, H_audio, D_h_audio]
        a2v_{k,v}:        [B, L_a_cached, H_cross, D_h_cross]   (audio K for A2V)
        v2a_{k,v}:        [B, L_v_cached, H_cross, D_h_cross]   (video K for V2A)

    Ulysses Sequence Parallel layout:
        When sp_size > 1 the cache is stored in **head-sharded** form, i.e.
        ``H`` above is replaced by ``H_local = H // sp_size`` and the
        sequence dimension is FULL (not local). Each rank persists 1/sp
        of the heads on the full sequence; concatenation along dim=1 with
        the per-step ``new_k/new_v`` (which are also head-sharded after the
        seq->head all-to-all in ``attention_with_cache``) Just Works.
        Cross-rank communication is therefore NOT required to read/write
        the cache, and CPU-offload remains independent per-rank.
    """
    video_self_k: Optional[Tensor] = None
    video_self_v: Optional[Tensor] = None
    audio_self_k: Optional[Tensor] = None
    audio_self_v: Optional[Tensor] = None
    a2v_k: Optional[Tensor] = None
    a2v_v: Optional[Tensor] = None
    v2a_k: Optional[Tensor] = None
    v2a_v: Optional[Tensor] = None


@dataclass
class KVCache:
    """Full KV cache across all transformer layers."""
    layers: List[LayerKVCache] = field(default_factory=list)
    # Training-only escape hatch for long self-forcing rollouts.  When enabled,
    # model_forward_inference materializes one layer cache on CUDA at a time and
    # immediately stores the detached updated layer back on pageable CPU.
    layerwise_cpu_offload: bool = False


def _move_layer_cache(
    layer_cache: LayerKVCache,
    device: torch.device | str,
    *,
    detach: bool,
) -> LayerKVCache:
    """Copy one layer cache without mutating snapshots held by the caller."""
    values = {}
    for cache_field in dataclass_fields(layer_cache):
        value = getattr(layer_cache, cache_field.name)
        if value is not None:
            if detach:
                value = value.detach()
            value = value.to(device=device)
        values[cache_field.name] = value
    return LayerKVCache(**values)


def enable_layerwise_cpu_offload(kv_cache: KVCache) -> None:
    """Move an existing cache to CPU and enable bounded per-layer materialization."""
    if kv_cache.layerwise_cpu_offload:
        return
    kv_cache.layers = [
        _move_layer_cache(layer, "cpu", detach=True)
        for layer in kv_cache.layers
    ]
    kv_cache.layerwise_cpu_offload = True


# ============================================================================
# Helpers
# ============================================================================

def _cat_cache(old: Optional[Tensor], new: Tensor) -> Tensor:
    """Append *new* K/V to the existing cache along the sequence dimension."""
    if old is None:
        return new
    return torch.cat([old, new], dim=1)


def _prepare_context_for_inference(
    model: "CausalLTXModel",
    context: Tensor,
    projection,
    target_dim: int,
    batch_size: int,
    *,
    cache_name: str,
) -> Tensor:
    """Project text context with an inference-only object-identity cache.

    A realtime request calls ``forward_inference`` many times with the same
    prompt embedding tensor.  The caption projections are deterministic in eval
    mode, so reusing their projected output is exact.  The cache is opt-in and
    disabled whenever autograd/training could observe tensor identity or graph
    history.
    """
    if (
        not _CACHE_CONTEXT_PROJECTION
        or torch.is_grad_enabled()
        or getattr(model, "training", False)
        or context is None
    ):
        return model._prepare_context(context, projection, target_dim, batch_size)

    try:
        context_ref = weakref.ref(context)
    except TypeError:
        return model._prepare_context(context, projection, target_dim, batch_size)

    key = (
        cache_name,
        id(projection),
        id(context),
        int(context.data_ptr()),
        tuple(context.shape),
        tuple(context.stride()),
        str(context.dtype),
        str(context.device),
        int(target_dim),
        int(batch_size),
    )
    cache = getattr(model, "_kv_context_projection_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(model, "_kv_context_projection_cache", cache)

    cached = cache.get(key)
    if cached is not None:
        cached_ref, cached_value = cached
        if cached_ref() is context:
            cache.move_to_end(key)
            return cached_value
        cache.pop(key, None)

    out = model._prepare_context(context, projection, target_dim, batch_size)
    cache[key] = (context_ref, out)
    cache.move_to_end(key)
    while len(cache) > _CONTEXT_PROJECTION_CACHE_SIZE:
        cache.popitem(last=False)
    return out


def _project_learned_memory_for_inference(
    model: "CausalLTXModel",
    memory: Optional[Tensor],
    projection,
    *,
    cache_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    """Project learned-memory tokens with an exact inference-only LRU cache.

    Learned-memory state is constant across the denoise steps of one block and
    across the following sigma=0 clean-KV commit.  The adapter input projection
    is deterministic in eval/no-grad mode, so caching by tensor identity is
    output-equivalent and only removes repeated small GEMMs.
    """
    if memory is None or projection is None:
        return None
    if (
        not _CACHE_LEARNED_MEMORY_PROJECTION
        or torch.is_grad_enabled()
        or getattr(model, "training", False)
    ):
        return projection(memory.to(device=device, dtype=dtype))

    try:
        memory_ref = weakref.ref(memory)
    except TypeError:
        return projection(memory.to(device=device, dtype=dtype))

    device_obj = torch.device(device)
    key = (
        cache_name,
        id(projection),
        id(memory),
        int(memory.data_ptr()),
        tuple(memory.shape),
        tuple(memory.stride()),
        str(memory.dtype),
        str(memory.device),
        str(device_obj.type),
        int(device_obj.index) if device_obj.index is not None else -1,
        str(dtype),
    )
    cache = getattr(model, "_kv_learned_memory_projection_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(model, "_kv_learned_memory_projection_cache", cache)

    cached = cache.get(key)
    if cached is not None:
        cached_ref, cached_value = cached
        if cached_ref() is memory:
            cache.move_to_end(key)
            return cached_value
        cache.pop(key, None)

    out = projection(memory.to(device=device, dtype=dtype))
    cache[key] = (memory_ref, out)
    cache.move_to_end(key)
    while len(cache) > _LEARNED_MEMORY_PROJECTION_CACHE_SIZE:
        cache.popitem(last=False)
    return out


def _prepare_learned_memory_for_inference(
    model: "CausalLTXModel",
    video_memory: Optional[Tensor],
    audio_memory: Optional[Tensor],
    color_memory: Optional[Tensor],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    if not model.config.learned_memory_enabled:
        return None, None, None
    vm = _project_learned_memory_for_inference(
        model,
        video_memory,
        model.learned_memory_video_encoder,
        cache_name="video",
        device=device,
        dtype=dtype,
    )
    am = _project_learned_memory_for_inference(
        model,
        audio_memory,
        model.learned_memory_audio_encoder,
        cache_name="audio",
        device=device,
        dtype=dtype,
    )
    cm = None
    if color_memory is not None and model.config.learned_memory_color_film_enabled:
        cm = color_memory.to(device=device, dtype=dtype)
    return vm, am, cm


def _causal_precompute_freqs_cis_for_inference(
    model: "CausalLTXModel",
    cache_name: str,
    grid_key: Tuple[int, ...],
    grid_sizes: Tensor,
    dim: int,
    *,
    theta: float,
    max_pos: List[int],
    start_frame: int,
    rope_type: CausalRopeType,
    rope_extrapolation: str,
    rope_train_max_seconds: float,
    device: torch.device,
    dtype: torch.dtype,
    is_audio: bool = False,
    num_attention_heads: int,
) -> Tuple[Tensor, Tensor]:
    """Precompute RoPE with an exact inference-only LRU cache.

    Realtime denoising calls ``forward_inference`` several times with the same
    token grid and absolute start frame.  RoPE depends only on those integer
    layout values and dtype/device, not on the latent content or timestep, so
    reusing the generated cos/sin tables is exact in eval/no-grad inference.
    """
    if (
        not _CACHE_ROPE
        or torch.is_grad_enabled()
        or getattr(model, "training", False)
    ):
        return causal_precompute_freqs_cis(
            grid_sizes,
            dim,
            theta=theta,
            max_pos=max_pos,
            start_frame=start_frame,
            rope_type=rope_type,
            rope_extrapolation=rope_extrapolation,
            rope_train_max_seconds=rope_train_max_seconds,
            device=device,
            dtype=dtype,
            is_audio=is_audio,
            num_attention_heads=num_attention_heads,
        )

    device_obj = torch.device(device)
    key = (
        cache_name,
        tuple(int(v) for v in grid_key),
        int(dim),
        float(theta),
        tuple(int(v) for v in max_pos),
        int(start_frame),
        rope_type.value if isinstance(rope_type, CausalRopeType) else str(rope_type),
        str(device_obj.type),
        int(device_obj.index) if device_obj.index is not None else -1,
        str(dtype),
        bool(is_audio),
        str(rope_extrapolation),
        float(rope_train_max_seconds),
        int(num_attention_heads),
    )
    cache = getattr(model, "_kv_rope_cache", None)
    if cache is None:
        cache = OrderedDict()
        setattr(model, "_kv_rope_cache", cache)

    cached = cache.get(key)
    if cached is not None:
        cache.move_to_end(key)
        return cached

    out = causal_precompute_freqs_cis(
        grid_sizes,
        dim,
        theta=theta,
        max_pos=max_pos,
        start_frame=start_frame,
        rope_type=rope_type,
        rope_extrapolation=rope_extrapolation,
        rope_train_max_seconds=rope_train_max_seconds,
        device=device,
        dtype=dtype,
        is_audio=is_audio,
        num_attention_heads=num_attention_heads,
    )
    cache[key] = out
    cache.move_to_end(key)
    while len(cache) > _ROPE_CACHE_SIZE:
        cache.popitem(last=False)
    return out


def _small_tensor_value_key(tensor: Optional[Tensor]) -> Optional[Tuple[float, ...]]:
    """Return a compact value key for tiny timestep tensors.

    Text-cross AdaLN changes the context K/V by denoise timestep.  The tensor
    object is recreated for each forward, but the scalar timestep values repeat
    across blocks inside one segment, so an object-identity key would miss.  This
    helper is only used for tiny `[B, 1]` timestep tensors in no-grad inference.
    """
    if tensor is None:
        return None
    try:
        flat = tensor.detach().reshape(-1)
        if flat.numel() > 16:
            return None
        values = flat.float().cpu().tolist()
    except Exception:
        return None
    return tuple(round(float(v), 6) for v in values)


def _text_context_tensor_key(
    context: Tensor,
    *,
    base_context: Tensor,
    prompt_cache_key: Optional[Tuple[float, ...]],
    cache_name: str,
    attn: CausalLTXAttention,
    sp_size: int,
) -> Optional[Tuple]:
    try:
        weakref.ref(base_context)
    except TypeError:
        return None

    rank = _get_sp_rank() if sp_size > 1 else 0
    return (
        cache_name,
        id(attn),
        id(base_context),
        int(base_context.data_ptr()),
        tuple(base_context.shape),
        tuple(base_context.stride()),
        str(base_context.dtype),
        str(base_context.device),
        prompt_cache_key,
        tuple(context.shape),
        str(context.dtype),
        str(context.device),
        int(sp_size),
        int(rank),
    )


def _get_cached_text_cross_kv(
    attn: CausalLTXAttention,
    context: Tensor,
    *,
    base_context: Tensor,
    prompt_cache_key: Optional[Tuple[float, ...]],
    cache_name: str,
    layer_idx: Optional[int],
) -> Tuple[Tensor, Tensor]:
    """Project text context to K/V with an exact inference-only LRU cache."""
    B = context.shape[0]
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
    key = _text_context_tensor_key(
        context,
        base_context=base_context,
        prompt_cache_key=prompt_cache_key,
        cache_name=cache_name,
        attn=attn,
        sp_size=sp_size,
    )

    cache = getattr(attn, "_ltx_text_cross_kv_cache", None)
    if key is not None:
        if cache is None:
            cache = OrderedDict()
            setattr(attn, "_ltx_text_cross_kv_cache", cache)
        cached = cache.get(key)
        if cached is not None:
            context_ref, k_cached, v_cached = cached
            if context_ref() is base_context:
                cache.move_to_end(key)
                _profile_detail_end(layer_idx, f"{cache_name}.text_kv_cache_hit", None)
                return k_cached, v_cached
            cache.pop(key, None)

    _prof_s = _profile_detail_start()
    k = attn.k_norm(attn.to_k(context))
    v = attn.to_v(context)
    k = k.view(B, -1, attn.heads, attn.dim_head)
    v = v.view(B, -1, attn.heads, attn.dim_head)

    if sp_size > 1:
        H_local = attn.heads // sp_size
        head_start = _get_sp_rank() * H_local
        k = k.narrow(2, head_start, H_local).contiguous()
        v = v.narrow(2, head_start, H_local).contiguous()
    _profile_detail_end(layer_idx, f"{cache_name}.text_kv_project", _prof_s)

    if key is not None and cache is not None:
        cache[key] = (weakref.ref(base_context), k, v)
        cache.move_to_end(key)
        while len(cache) > _TEXT_CROSS_KV_CACHE_SIZE:
            cache.popitem(last=False)
    return k, v


def _text_cross_attention_with_kv_cache(
    attn: CausalLTXAttention,
    x: Tensor,
    context: Tensor,
    *,
    mask: Optional[Tensor],
    base_context: Tensor,
    prompt_cache_key: Optional[Tuple[float, ...]],
    cache_name: str,
    layer_idx: Optional[int],
) -> Tensor:
    """Text cross-attention fast path with cached K/V projections.

    This mirrors ``CausalLTXAttention.forward(..., sp_context_sharded=False)``
    exactly for inference.  Only K/V projection is cached; Q, mask application,
    attention, gating and output projection remain live per call.
    """
    if (
        not _CACHE_TEXT_CROSS_KV
        or torch.is_grad_enabled()
        or getattr(attn, "training", False)
    ):
        return attn(x, context=context, mask=mask, sp_context_sharded=False)

    B, _, _ = x.shape
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1

    _prof_s = _profile_detail_start()
    q = attn.q_norm(attn.to_q(x))
    q = q.view(B, -1, attn.heads, attn.dim_head)
    if sp_size > 1:
        q = _seq_all_to_all_head(q, _get_sp_group(), sp_size)
    _profile_detail_end(layer_idx, f"{cache_name}.text_q_project", _prof_s)

    k, v = _get_cached_text_cross_kv(
        attn,
        context,
        base_context=base_context,
        prompt_cache_key=prompt_cache_key,
        cache_name=cache_name,
        layer_idx=layer_idx,
    )

    _prof_s = _profile_detail_start()
    out = standard_attention_forward(q, k, v, mask=mask)
    _profile_detail_end(layer_idx, f"{cache_name}.text_sdpa", _prof_s)

    if sp_size > 1:
        _prof_s = _profile_detail_start()
        out = _head_all_to_all_seq(out, _get_sp_group(), sp_size)
        _profile_detail_end(layer_idx, f"{cache_name}.text_a2a_head2seq", _prof_s)

    _prof_s = _profile_detail_start()
    out = out.reshape(B, -1, attn.inner_dim)
    if attn.to_gate_logits is not None:
        gate_logits = attn.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, attn.heads, attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, attn.inner_dim)
    out = attn.to_out(out)
    _profile_detail_end(layer_idx, f"{cache_name}.text_out_proj", _prof_s)
    return out


@dataclass
class _TextCrossHeadLayout:
    attn: CausalLTXAttention
    x: Tensor
    out: Tensor
    gate: Optional[Tensor]
    cache_name: str
    layer_idx: Optional[int]
    sp_size: int


class _AsyncTextCrossOutput:
    def __init__(
        self,
        handle,
        gate: Optional[Tensor],
        layer_idx: Optional[int],
        profile_name: str,
    ) -> None:
        self.handle = handle
        self.gate = gate
        self.layer_idx = layer_idx
        self.profile_name = profile_name

    def wait(self) -> Tensor:
        _prof_s = _profile_detail_start()
        out = self.handle.wait()
        if self.gate is not None:
            out = out * self.gate
        _profile_detail_end(self.layer_idx, self.profile_name, _prof_s)
        return out


def _can_pair_text_cross_attention(attn: CausalLTXAttention) -> bool:
    return (
        _CACHE_TEXT_CROSS_KV
        and _PAIR_TEXT_CROSS_ASYNC_H2S
        and _is_sp_enabled()
        and _get_sp_world_size() > 1
        and not torch.is_grad_enabled()
        and not getattr(attn, "training", False)
    )


def _prepare_text_cross_inputs_for_inference(
    block: CausalAVTransformerBlock,
    x: Tensor,
    context: Tensor,
    scale_shift_table: Tensor,
    prompt_scale_shift_table: Optional[Tensor],
    timestep: Tensor,
    prompt_timestep: Optional[Tensor],
) -> Tuple[Tensor, Tensor, Tensor, Optional[Tuple[float, ...]], Optional[Tensor]]:
    """Prepare text-cross Q input/context while preserving prompt AdaLN."""
    if block.cross_attention_adaln:
        shift_q, scale_q, gate = block.get_ada_values(
            scale_shift_table, x.shape[0], timestep, slice(6, 9)
        )
        attn_input = rms_norm(x, eps=block.norm_eps) * (1 + scale_q) + shift_q

        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None]
            .to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(
                batch_size, prompt_timestep.shape[1], 2, -1
            )
        ).unbind(dim=2)
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        return attn_input, encoder_hidden_states, context, None, gate

    return rms_norm(x, eps=block.norm_eps), context, context, None, None


def _text_cross_attention_head_layout(
    attn: CausalLTXAttention,
    x: Tensor,
    context: Tensor,
    *,
    mask: Optional[Tensor],
    base_context: Tensor,
    prompt_cache_key: Optional[Tuple[float, ...]],
    gate: Optional[Tensor],
    cache_name: str,
    layer_idx: Optional[int],
) -> _TextCrossHeadLayout:
    """Run text cross-attention through SDPA and keep output head-sharded."""
    B, _, _ = x.shape
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1

    _prof_s = _profile_detail_start()
    q = attn.q_norm(attn.to_q(x))
    q = q.view(B, -1, attn.heads, attn.dim_head)
    q = _seq_all_to_all_head(q, _get_sp_group(), sp_size)
    _profile_detail_end(layer_idx, f"{cache_name}.pair_text_q_project", _prof_s)

    k, v = _get_cached_text_cross_kv(
        attn,
        context,
        base_context=base_context,
        prompt_cache_key=prompt_cache_key,
        cache_name=cache_name,
        layer_idx=layer_idx,
    )

    _prof_s = _profile_detail_start()
    out = standard_attention_forward(q, k, v, mask=mask)
    _profile_detail_end(layer_idx, f"{cache_name}.pair_text_sdpa", _prof_s)
    return _TextCrossHeadLayout(
        attn=attn,
        x=x,
        out=out,
        gate=gate,
        cache_name=cache_name,
        layer_idx=layer_idx,
        sp_size=sp_size,
    )


def _finish_text_cross_head_layout(
    item: _TextCrossHeadLayout,
    h2s_async=None,
) -> Tensor:
    """Finish text cross-attention from head-sharded layout."""
    if h2s_async is not None:
        _prof_s = _profile_detail_start()
        out = h2s_async.wait()
        _profile_detail_end(item.layer_idx, f"{item.cache_name}.pair_text_a2a_head2seq_wait", _prof_s)
    else:
        _prof_s = _profile_detail_start()
        out = _head_all_to_all_seq(item.out, _get_sp_group(), item.sp_size)
        _profile_detail_end(item.layer_idx, f"{item.cache_name}.pair_text_a2a_head2seq", _prof_s)

    _prof_s = _profile_detail_start()
    out = out.reshape(item.x.shape[0], -1, item.attn.inner_dim)
    if item.attn.to_gate_logits is not None:
        gate_logits = item.attn.to_gate_logits(item.x)
        b, t, _ = out.shape
        out = out.view(b, t, item.attn.heads, item.attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, item.attn.inner_dim)
    out = item.attn.to_out(out)
    if item.gate is not None:
        out = out * item.gate
    _profile_detail_end(item.layer_idx, f"{item.cache_name}.pair_text_out_proj", _prof_s)
    return out


def _finish_text_cross_head_layout_async_tp(
    item: _TextCrossHeadLayout,
) -> Optional[_AsyncTextCrossOutput]:
    if (
        torch.is_grad_enabled()
        or item.sp_size != 1
        or not isinstance(item.attn.to_out, RowParallelLinear)
    ):
        return None

    _prof_s = _profile_detail_start()
    out = item.out.reshape(item.x.shape[0], -1, item.attn.inner_dim)
    if item.attn.to_gate_logits is not None:
        gate_logits = item.attn.to_gate_logits(item.x)
        b, t, _ = out.shape
        out = out.view(b, t, item.attn.heads, item.attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, item.attn.inner_dim)
    handle = item.attn.to_out.forward_async(out)
    _profile_detail_end(item.layer_idx, f"{item.cache_name}.tp_out_proj_launch", _prof_s)
    return _AsyncTextCrossOutput(
        handle,
        item.gate,
        item.layer_idx,
        f"{item.cache_name}.tp_out_proj_wait",
    )


def _apply_paired_text_cross_attention_for_inference(
    block: CausalAVTransformerBlock,
    vx: Tensor,
    ax: Tensor,
    video_ctx: Tensor,
    audio_ctx: Tensor,
    video_timestep_6d: Tensor,
    audio_timestep_6d: Tensor,
    video_prompt_ts: Optional[Tensor],
    audio_prompt_ts: Optional[Tensor],
    video_context_mask: Optional[Tensor],
    audio_context_mask: Optional[Tensor],
    video_prompt_cache_key: Optional[Tuple[float, ...]],
    audio_prompt_cache_key: Optional[Tuple[float, ...]],
    layer_idx: Optional[int],
) -> Optional[Tuple[Tensor, Tensor]]:
    """Pair independent video/audio text-cross paths to hide one H2S A2A.

    This preserves the original math: self-attention has already updated both
    streams, text K/V projections are exact cache hits/misses, and the only
    difference is scheduling the head->seq all-to-all asynchronously.
    """
    if not (
        _can_pair_text_cross_attention(block.attn2)
        and _can_pair_text_cross_attention(block.audio_attn2)
    ):
        return None

    v_input, v_ctx, v_base_ctx, _, v_gate = _prepare_text_cross_inputs_for_inference(
        block,
        vx,
        video_ctx,
        block.scale_shift_table,
        getattr(block, "prompt_scale_shift_table", None),
        video_timestep_6d,
        video_prompt_ts,
    )
    a_input, a_ctx, a_base_ctx, _, a_gate = _prepare_text_cross_inputs_for_inference(
        block,
        ax,
        audio_ctx,
        block.audio_scale_shift_table,
        getattr(block, "audio_prompt_scale_shift_table", None),
        audio_timestep_6d,
        audio_prompt_ts,
    )

    video_item = _text_cross_attention_head_layout(
        block.attn2,
        v_input,
        v_ctx,
        mask=video_context_mask,
        base_context=v_base_ctx,
        prompt_cache_key=video_prompt_cache_key,
        gate=v_gate,
        cache_name="video_text_cross",
        layer_idx=layer_idx,
    )
    _prof_s = _profile_detail_start()
    video_h2s = _head_all_to_all_seq_async(video_item.out, _get_sp_group(), video_item.sp_size)
    _profile_detail_end(layer_idx, "video_text_cross.pair_text_a2a_head2seq_launch", _prof_s)

    audio_item = _text_cross_attention_head_layout(
        block.audio_attn2,
        a_input,
        a_ctx,
        mask=audio_context_mask,
        base_context=a_base_ctx,
        prompt_cache_key=audio_prompt_cache_key,
        gate=a_gate,
        cache_name="audio_text_cross",
        layer_idx=layer_idx,
    )
    _prof_s = _profile_detail_start()
    audio_h2s = _head_all_to_all_seq_async(audio_item.out, _get_sp_group(), audio_item.sp_size)
    _profile_detail_end(layer_idx, "audio_text_cross.pair_text_a2a_head2seq_launch", _prof_s)

    vx_text_attn = _finish_text_cross_head_layout(video_item, video_h2s)
    ax_text_attn = _finish_text_cross_head_layout(audio_item, audio_h2s)
    return vx_text_attn, ax_text_attn


def _apply_text_cross_attention_for_inference(
    block: CausalAVTransformerBlock,
    x: Tensor,
    context: Tensor,
    attn: CausalLTXAttention,
    scale_shift_table: Tensor,
    prompt_scale_shift_table: Optional[Tensor],
    timestep: Tensor,
    prompt_timestep: Optional[Tensor],
    context_mask: Optional[Tensor],
    *,
    prompt_cache_key: Optional[Tuple[float, ...]],
    cache_name: str,
    layer_idx: Optional[int],
) -> Tensor:
    """Apply text cross-attention, preserving LTX-2.3 prompt AdaLN semantics."""
    if block.cross_attention_adaln:
        shift_q, scale_q, gate = block.get_ada_values(
            scale_shift_table, x.shape[0], timestep, slice(6, 9)
        )
        attn_input = rms_norm(x, eps=block.norm_eps) * (1 + scale_q) + shift_q

        batch_size = x.shape[0]
        shift_kv, scale_kv = (
            prompt_scale_shift_table[None, None]
            .to(device=x.device, dtype=x.dtype)
            + prompt_timestep.reshape(
                batch_size, prompt_timestep.shape[1], 2, -1
            )
        ).unbind(dim=2)
        encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        return _text_cross_attention_with_kv_cache(
            attn,
            attn_input,
            encoder_hidden_states,
            mask=context_mask,
            base_context=context,
            prompt_cache_key=prompt_cache_key,
            cache_name=cache_name,
            layer_idx=layer_idx,
        ) * gate

    return _text_cross_attention_with_kv_cache(
        attn,
        rms_norm(x, eps=block.norm_eps),
        context,
        mask=context_mask,
        base_context=context,
        prompt_cache_key=None,
        cache_name=cache_name,
        layer_idx=layer_idx,
    )


# ============================================================================
# Attention with KV Cache
# ============================================================================

def attention_with_cache(
    attn: CausalLTXAttention,
    x: Tensor,
    context: Optional[Tensor],
    pe: Optional[Tuple[Tensor, Tensor]],
    k_pe: Optional[Tuple[Tensor, Tensor]],
    cached_k: Optional[Tensor],
    cached_v: Optional[Tensor],
    logit_log_scale: Optional[Tensor] = None,
    pyramid_policy: Optional[PyramidKVPolicy] = None,
    layer_idx: Optional[int] = None,
    modality: Optional[str] = None,
    frame_seqlen: Optional[int] = None,
    sp_context_sharded: bool = True,
    new_kv_real_len: Optional[int] = None,
    profile_name: Optional[str] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Run one attention operation with KV cache.

    Args:
        attn: The ``CausalLTXAttention`` module (weights are accessed via it).
        x: Query input — only *new* tokens.  ``[B, L_new_local, D_q]``
            (already SP-sharded along seq when SP is on; ``L_new_local == L_new``
            when SP is off).
        context: Context for cross-attention ``[B, L_new_ctx_local, D_ctx]``.
                 ``None`` for self-attention (K/V come from *x*).
        pe: RoPE ``(cos, sin)`` for Q positions (new tokens only, local shard).
        k_pe: RoPE for K positions.  ``None`` → reuse *pe*.
        cached_k: Cached key tensor.
            - SP off: ``[B, L_cached, H, D_h]``
            - SP on : ``[B, L_cached_full, H_local, D_h]`` (head-sharded,
              full sequence per rank).
        cached_v: Cached value tensor (same convention).
        logit_log_scale: Optional per-position temperature scale for Q.
            Must be FULL-sequence ``[1, L_new_full, 1]`` when SP is on (it
            is applied AFTER the seq->head all-to-all).
        pyramid_policy: Optional Pyramid Forcing head-aware KV selector.
        layer_idx / modality / frame_seqlen: Required by pyramid_policy.
        sp_context_sharded: When True (self-attn / A2V / V2A) the K/V context
            is SP-sharded along seq just like Q, and we run a seq->head
            all-to-all on K/V so the cache is in head-sharded layout. When
            False (text cross-attn — currently NOT routed through this
            function but kept for symmetry) K/V are full-replica; we only
            ``narrow`` the head dim. No-op when SP is off.

    Returns:
        ``(output, new_k, new_v)`` where:
        - ``output`` matches the local seq layout of *x* (i.e. SP-sharded
          along seq when SP is on);
        - ``new_k/new_v`` are in the same layout as ``cached_k/cached_v``
          (head-sharded full-seq under SP, plain heads-full under no-SP),
          so callers can ``cat(old, new, dim=1)`` without further changes.
    """
    B, L, _ = x.shape
    ctx = x if context is None else context
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
    use_async_ulysses = (
        _ASYNC_ULYSSES_ENABLED
        and sp_size > 1
        and sp_context_sharded
        and not torch.is_grad_enabled()
    )
    use_tp_qk_fused = (
        isinstance(attn.q_norm, TPRMSNorm)
        and not is_compile_sync_qk_norm()
    )
    use_async_tp_qk_fused = use_tp_qk_fused and not _ASYNC_ULYSSES_STRICT_V_FIRST
    profile_base = profile_name or ("self" if context is None else "cross")

    # ASYNC_ULYSSES inference path: when TP-aware q/k RMSNorm is active, the
    # default path keeps the existing q/k all-reduce fusion and launches V
    # projection + scatter from its overlap callback.  The strict V-first mode
    # below starts V projection + scatter before Q/K projection and RoPE, which
    # matches the explicit "V communication overlaps Q/K compute" schedule.
    # Training and text-cross paths stay on the existing synchronous route
    # unless the explicit env switch is enabled.
    if use_async_ulysses and use_async_tp_qk_fused:
        sp_group = _get_sp_group()

        _prof_s = _profile_detail_start()
        q = attn.to_q(x)
        k = attn.to_k(ctx)

        def _compute_v_async(_ctx=ctx, _to_v=attn.to_v):
            _v = _to_v(_ctx).view(B, -1, attn.heads, attn.dim_head)
            return _seq_all_to_all_head_async(_v, sp_group, sp_size)

        q, k, v_async = tp_qk_norm_fused(
            attn.q_norm, attn.k_norm, q, k,
            overlap_compute=_compute_v_async,
        )
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_qk_norm_v_launch", _prof_s)

        _prof_s = _profile_detail_start()
        if pe is not None:
            q = attn._apply_rope(q, pe)
            k = attn._apply_rope(k, k_pe if k_pe is not None else pe)
        q = q.view(B, -1, attn.heads, attn.dim_head)
        k = k.view(B, -1, attn.heads, attn.dim_head)
        q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)
        k_async = _seq_all_to_all_head_async(k, sp_group, sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_qk_launch", _prof_s)

        _prof_s = _profile_detail_start()
        v = v_async.wait()
        q = q_async.wait()
        k = k_async.wait()
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_wait", _prof_s)

        # Apply logit_log_scale on the head-sharded layout (full seq).
        if logit_log_scale is not None:
            if logit_log_scale.dim() == 3:
                scale_4d = logit_log_scale.unsqueeze(-1)
            elif logit_log_scale.dim() == 4:
                scale_4d = logit_log_scale
            else:
                scale_4d = logit_log_scale.view(1, -1, 1, 1)
            q = q * scale_4d

        ulysses_prefetched = True
    elif use_async_ulysses and use_tp_qk_fused and _ASYNC_ULYSSES_STRICT_V_FIRST:
        sp_group = _get_sp_group()

        _prof_s = _profile_detail_start()
        v = attn.to_v(ctx).view(B, -1, attn.heads, attn.dim_head)
        v_async = _seq_all_to_all_head_async(v, sp_group, sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_strict_v_launch", _prof_s)

        _prof_s = _profile_detail_start()
        q = attn.to_q(x)
        q = attn.q_norm(q)
        if pe is not None:
            q = attn._apply_rope(q, pe)
        q = q.view(B, -1, attn.heads, attn.dim_head)
        q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_strict_q_launch", _prof_s)

        _prof_s = _profile_detail_start()
        k = attn.to_k(ctx)
        k = attn.k_norm(k)
        if pe is not None:
            k = attn._apply_rope(k, k_pe if k_pe is not None else pe)
        k = k.view(B, -1, attn.heads, attn.dim_head)
        k_async = _seq_all_to_all_head_async(k, sp_group, sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_strict_k_launch", _prof_s)

        _prof_s = _profile_detail_start()
        v = v_async.wait()
        q = q_async.wait()
        k = k_async.wait()
        _profile_detail_end(layer_idx, f"{profile_base}.async_tp_strict_wait", _prof_s)

        # Apply logit_log_scale on the head-sharded layout (full seq).
        if logit_log_scale is not None:
            if logit_log_scale.dim() == 3:
                scale_4d = logit_log_scale.unsqueeze(-1)
            elif logit_log_scale.dim() == 4:
                scale_4d = logit_log_scale
            else:
                scale_4d = logit_log_scale.view(1, -1, 1, 1)
            q = q * scale_4d

        ulysses_prefetched = True
    elif use_async_ulysses and not use_tp_qk_fused:
        sp_group = _get_sp_group()

        _prof_s = _profile_detail_start()
        v = attn.to_v(ctx).view(B, -1, attn.heads, attn.dim_head)
        v_async = _seq_all_to_all_head_async(v, sp_group, sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.async_v_launch", _prof_s)

        _prof_s = _profile_detail_start()
        q = attn.q_norm(attn.to_q(x))
        if pe is not None:
            q = attn._apply_rope(q, pe)
        q = q.view(B, -1, attn.heads, attn.dim_head)
        pack_qk_candidate = _ASYNC_ULYSSES_PACK_QK and not torch.is_grad_enabled()
        q_async = None
        if not pack_qk_candidate:
            q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)
        _profile_detail_end(
            layer_idx,
            f"{profile_base}.async_q_defer" if pack_qk_candidate else f"{profile_base}.async_q_launch",
            _prof_s,
        )

        _prof_s = _profile_detail_start()
        k = attn.k_norm(attn.to_k(ctx))
        if pe is not None:
            k = attn._apply_rope(k, k_pe if k_pe is not None else pe)
        k = k.view(B, -1, attn.heads, attn.dim_head)
        pack_qk = (
            _ASYNC_ULYSSES_PACK_QK
            and not torch.is_grad_enabled()
            and q.shape == k.shape
        )
        if pack_qk:
            qk_async = _seq_all_to_all_head_many_async((q, k), sp_group, sp_size)
        else:
            if q_async is None:
                q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)
            k_async = _seq_all_to_all_head_async(k, sp_group, sp_size)
        _profile_detail_end(
            layer_idx,
            f"{profile_base}.async_qk_launch" if pack_qk else f"{profile_base}.async_k_launch",
            _prof_s,
        )

        _prof_s = _profile_detail_start()
        v = v_async.wait()
        if pack_qk:
            q, k = qk_async.wait()
        else:
            q = q_async.wait()
            k = k_async.wait()
        _profile_detail_end(layer_idx, f"{profile_base}.async_wait", _prof_s)

        # Apply logit_log_scale on the head-sharded layout (full seq).
        if logit_log_scale is not None:
            if logit_log_scale.dim() == 3:
                scale_4d = logit_log_scale.unsqueeze(-1)
            elif logit_log_scale.dim() == 4:
                scale_4d = logit_log_scale
            else:
                scale_4d = logit_log_scale.view(1, -1, 1, 1)
            q = q * scale_4d

        ulysses_prefetched = True
    else:
        ulysses_prefetched = False

    # --- Projection + Norm ---
    # TP fast path: 当 q_norm/k_norm 是 TPRMSNorm 时，把它们的 all_reduce
    # 融合成一次 NCCL 调用，并让 to_v(ctx) 的投影与该 all_reduce 重叠。
    # 单条数据 8 卡推理下，每个 attention 由 3 次 all_reduce 降到 2 次，
    # 同时把 to_v 的 GEMM 隐藏在通信延迟里。
    #
    # 但当 LTX_COMPILE=1（torch.compile 启用）时，async_op + Python closure
    # 会让 Dynamo 严重 graph break，反而抵消掉 compile 收益。此时退化为
    # 标准同步路径——三个独立 all_reduce 都进编译图，由 Inductor 统一融合。
    _prof_s = _profile_detail_start()
    if not ulysses_prefetched and use_tp_qk_fused:
        # Async fused fast path（无 torch.compile 时）
        # to_q / to_k 先算（q_norm/k_norm 依赖之），to_v 在 all_reduce flight
        # 期间发射，与 NCCL kernel 在 default stream / NCCL stream 上并行。
        q = attn.to_q(x)
        k = attn.to_k(ctx)

        # 直接以闭包形式发射 to_v；不再走 dict[holder]，省一次字典查找
        def _compute_v(_ctx=ctx, _to_v=attn.to_v):
            return _to_v(_ctx)

        q, k, v = tp_qk_norm_fused(
            attn.q_norm, attn.k_norm, q, k,
            overlap_compute=_compute_v,
        )
    elif not ulysses_prefetched:
        # 同步路径：兼容 torch.compile，也兼容标准 nn.RMSNorm
        q = attn.q_norm(attn.to_q(x))
        k = attn.k_norm(attn.to_k(ctx))
        v = attn.to_v(ctx)
    if not ulysses_prefetched:
        _profile_detail_end(layer_idx, f"{profile_base}.qkv_proj_norm", _prof_s)

    if not ulysses_prefetched:
        _prof_s = _profile_detail_start()
        # --- RoPE on the LOCAL shard (each rank rotates its own tokens) ---
        if pe is not None:
            q = attn._apply_rope(q, pe)
            k = attn._apply_rope(k, k_pe if k_pe is not None else pe)

        # --- Optional log-ratio scaling (Q) ---
        # Bit-equal preservation when sp_size==1: keep the original ordering of
        # the multiply BEFORE reshape, exactly like the pre-SP code path.
        if sp_size == 1 and logit_log_scale is not None:
            q = q * logit_log_scale

        # --- Reshape to [B, L, H, D_h] ---
        q = q.view(B, -1, attn.heads, attn.dim_head)
        k = k.view(B, -1, attn.heads, attn.dim_head)
        v = v.view(B, -1, attn.heads, attn.dim_head)
        _profile_detail_end(layer_idx, f"{profile_base}.rope_reshape", _prof_s)

        # --- Ulysses SP all-to-all: Seq-sharded -> Head-sharded ---
        if sp_size > 1:
            _prof_s = _profile_detail_start()
            sp_group = _get_sp_group()
            if sp_context_sharded:
                fuse_a2a = _FUSE_ULYSSES_A2A and not torch.is_grad_enabled()
                if fuse_a2a and q.shape == k.shape == v.shape:
                    q, k, v = _seq_all_to_all_head_many((q, k, v), sp_group, sp_size)
                elif fuse_a2a and k.shape == v.shape:
                    q = _seq_all_to_all_head(q, sp_group, sp_size)
                    k, v = _seq_all_to_all_head_many((k, v), sp_group, sp_size)
                else:
                    q = _seq_all_to_all_head(q, sp_group, sp_size)
                    k = _seq_all_to_all_head(k, sp_group, sp_size)
                    v = _seq_all_to_all_head(v, sp_group, sp_size)
            else:
                q = _seq_all_to_all_head(q, sp_group, sp_size)
                H_local = attn.heads // sp_size
                head_start = _get_sp_rank() * H_local
                k = k.narrow(2, head_start, H_local).contiguous()
                v = v.narrow(2, head_start, H_local).contiguous()

            # Apply logit_log_scale on the head-sharded layout (full seq).
            if logit_log_scale is not None:
                if logit_log_scale.dim() == 3:
                    scale_4d = logit_log_scale.unsqueeze(-1)
                elif logit_log_scale.dim() == 4:
                    scale_4d = logit_log_scale
                else:
                    scale_4d = logit_log_scale.view(1, -1, 1, 1)
                q = q * scale_4d
            _profile_detail_end(layer_idx, f"{profile_base}.a2a_seq2head", _prof_s)

    # --- Audio SP-pad strip (NEW) -----------------------------------------
    # When the audio stream is end-padded so that A_total is divisible by
    # sp_size (see ``model_forward_inference`` entry), the a2a above brings
    # the FULL padded sequence onto every rank in head-sharded layout. Before
    # we (a) write into the persistent KV cache and (b) feed K/V into SDPA,
    # we narrow the new tokens back to the REAL length, so the cache stays
    # pollution-free across blocks and the softmax denominator only sums over
    # genuine positions. Padded query rows still go through SDPA but their
    # outputs are eventually dropped by the post-gather unpad in
    # ``model_forward_inference``.
    if new_kv_real_len is not None and new_kv_real_len < k.shape[1]:
        _prof_s = _profile_detail_start()
        k = k[:, :new_kv_real_len].contiguous()
        v = v[:, :new_kv_real_len].contiguous()
        _profile_detail_end(layer_idx, f"{profile_base}.real_len_strip", _prof_s)

    # Keep new K/V for cache update (already head-sharded under SP, plain
    # heads-full under no-SP). Storing them in this exact layout keeps
    # ``cat(cached, new, dim=1)`` valid across both modes.
    new_k, new_v = k, v

    # --- Concatenate cached + new K/V along seq dim ---
    _prof_s = _profile_detail_start()
    if cached_k is not None:
        full_k = torch.cat([cached_k, k], dim=1)
        full_v = torch.cat([cached_v, v], dim=1)
    else:
        full_k, full_v = k, v
    _profile_detail_end(layer_idx, f"{profile_base}.cache_cat", _prof_s)

    # --- Optional head-aware Pyramid KV selection ---
    attn_mask: Optional[Tensor] = None
    if pyramid_policy is not None:
        if layer_idx is None or modality is None or frame_seqlen is None:
            raise ValueError(
                "pyramid_policy requires layer_idx, modality, frame_seqlen"
            )
        sel_k, sel_v, attn_mask = pyramid_policy.select_for_layer(
            layer_idx=layer_idx,
            modality=modality,
            cached_k=full_k,
            cached_v=full_v,
            frame_seqlen=frame_seqlen,
        )
    else:
        sel_k, sel_v = full_k, full_v

    # --- Calibration capture hook (no-op when no collector installed) ---
    capture = get_active_capture_hook()
    if (
        capture is not None
        and layer_idx is not None
        and modality is not None
        and frame_seqlen is not None
    ):
        capture.record(
            layer_idx=layer_idx,
            modality=modality,
            q=q,
            full_k=full_k,
            frame_seqlen=frame_seqlen,
            attn_module=attn,
        )

    # --- Standard attention (no causal mask — causality enforced by cache;
    #     attn_mask carries head-aware kept/padded info when policy is on) ---
    _prof_s = _profile_detail_start()
    out = standard_attention_forward(q, sel_k, sel_v, mask=attn_mask)
    _profile_detail_end(layer_idx, f"{profile_base}.sdpa", _prof_s)

    # --- Ulysses SP all-to-all: Head-sharded -> Seq-sharded ---
    if sp_size > 1:
        _prof_s = _profile_detail_start()
        out = _head_all_to_all_seq(out, _get_sp_group(), sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.a2a_head2seq", _prof_s)

    out = _finish_attention_output_from_seq_layout(
        attn,
        out,
        x,
        layer_idx=layer_idx,
        profile_base=profile_base,
    )
    return out, new_k, new_v


def _project_qkv_local_for_cross_pair(
    attn: CausalLTXAttention,
    x: Tensor,
    context: Tensor,
    pe: Optional[Tuple[Tensor, Tensor]],
    k_pe: Optional[Tuple[Tensor, Tensor]],
    *,
    layer_idx: Optional[int],
    profile_base: str,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Project local Q/K/V without Ulysses communication.

    Used by the paired A2V/V2A inference path so same-shaped tensors from both
    cross-modal attentions can share seq->head all-to-all calls.
    """
    B = x.shape[0]
    use_tp_qk_fused = (
        isinstance(attn.q_norm, TPRMSNorm)
        and not is_compile_sync_qk_norm()
    )

    _prof_s = _profile_detail_start()
    if use_tp_qk_fused:
        q = attn.to_q(x)
        k = attn.to_k(context)

        def _compute_v(_ctx=context, _to_v=attn.to_v):
            return _to_v(_ctx)

        q, k, v = tp_qk_norm_fused(
            attn.q_norm, attn.k_norm, q, k,
            overlap_compute=_compute_v,
        )
    else:
        q = attn.q_norm(attn.to_q(x))
        k = attn.k_norm(attn.to_k(context))
        v = attn.to_v(context)
    _profile_detail_end(layer_idx, f"{profile_base}.qkv_proj_norm", _prof_s)

    _prof_s = _profile_detail_start()
    if pe is not None:
        q = attn._apply_rope(q, pe)
        k = attn._apply_rope(k, k_pe if k_pe is not None else pe)
    q = q.view(B, -1, attn.heads, attn.dim_head)
    k = k.view(B, -1, attn.heads, attn.dim_head)
    v = v.view(B, -1, attn.heads, attn.dim_head)
    _profile_detail_end(layer_idx, f"{profile_base}.rope_reshape", _prof_s)
    return q, k, v


def _finish_attention_from_head_layout(
    attn: CausalLTXAttention,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    x: Tensor,
    cached_k: Optional[Tensor],
    cached_v: Optional[Tensor],
    *,
    new_kv_real_len: Optional[int],
    layer_idx: Optional[int],
    profile_base: str,
    sp_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Finish attention after Q/K/V are already in head-sharded layout."""
    B = x.shape[0]

    if new_kv_real_len is not None and new_kv_real_len < k.shape[1]:
        _prof_s = _profile_detail_start()
        k = k[:, :new_kv_real_len].contiguous()
        v = v[:, :new_kv_real_len].contiguous()
        _profile_detail_end(layer_idx, f"{profile_base}.real_len_strip", _prof_s)

    new_k, new_v = k, v

    _prof_s = _profile_detail_start()
    if cached_k is not None:
        full_k = torch.cat([cached_k, k], dim=1)
        full_v = torch.cat([cached_v, v], dim=1)
    else:
        full_k, full_v = k, v
    _profile_detail_end(layer_idx, f"{profile_base}.cache_cat", _prof_s)

    _prof_s = _profile_detail_start()
    out = standard_attention_forward(q, full_k, full_v, mask=None)
    _profile_detail_end(layer_idx, f"{profile_base}.sdpa", _prof_s)

    if sp_size > 1:
        _prof_s = _profile_detail_start()
        out = _head_all_to_all_seq(out, _get_sp_group(), sp_size)
        _profile_detail_end(layer_idx, f"{profile_base}.a2a_head2seq", _prof_s)

    out = _finish_attention_output_from_seq_layout(
        attn,
        out,
        x,
        layer_idx=layer_idx,
        profile_base=profile_base,
    )
    return out, new_k, new_v


def _prepare_kv_for_head_layout_attention(
    k: Tensor,
    v: Tensor,
    cached_k: Optional[Tensor],
    cached_v: Optional[Tensor],
    *,
    new_kv_real_len: Optional[int],
    layer_idx: Optional[int],
    profile_base: str,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if new_kv_real_len is not None and new_kv_real_len < k.shape[1]:
        _prof_s = _profile_detail_start()
        k = k[:, :new_kv_real_len].contiguous()
        v = v[:, :new_kv_real_len].contiguous()
        _profile_detail_end(layer_idx, f"{profile_base}.real_len_strip", _prof_s)

    new_k, new_v = k, v

    _prof_s = _profile_detail_start()
    if cached_k is not None:
        full_k = torch.cat([cached_k, k], dim=1)
        full_v = torch.cat([cached_v, v], dim=1)
    else:
        full_k, full_v = k, v
    _profile_detail_end(layer_idx, f"{profile_base}.cache_cat", _prof_s)
    return new_k, new_v, full_k, full_v


def _finish_attention_output_from_seq_layout(
    attn: CausalLTXAttention,
    out: Tensor,
    x: Tensor,
    *,
    layer_idx: Optional[int],
    profile_base: str,
) -> Tensor:
    B = x.shape[0]
    _prof_s = _profile_detail_start()
    out = out.reshape(B, -1, attn.inner_dim)
    if attn.to_gate_logits is not None:
        gate_logits = attn.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, attn.heads, attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, attn.inner_dim)
    out = attn.to_out(out)
    _profile_detail_end(layer_idx, f"{profile_base}.out_proj", _prof_s)
    return out


class _AsyncAttentionOutput:
    def __init__(self, handle, layer_idx: Optional[int], profile_name: str) -> None:
        self.handle = handle
        self.layer_idx = layer_idx
        self.profile_name = profile_name

    def wait(self) -> Tensor:
        _prof_s = _profile_detail_start()
        out = self.handle.wait()
        _profile_detail_end(self.layer_idx, self.profile_name, _prof_s)
        return out


def _finish_attention_output_from_seq_layout_async(
    attn: CausalLTXAttention,
    out: Tensor,
    x: Tensor,
    *,
    layer_idx: Optional[int],
    profile_base: str,
) -> Optional[_AsyncAttentionOutput]:
    if torch.is_grad_enabled() or not isinstance(attn.to_out, RowParallelLinear):
        return None

    B = x.shape[0]
    _prof_s = _profile_detail_start()
    out = out.reshape(B, -1, attn.inner_dim)
    if attn.to_gate_logits is not None:
        gate_logits = attn.to_gate_logits(x)
        b, t, _ = out.shape
        out = out.view(b, t, attn.heads, attn.dim_head)
        gates = 2.0 * torch.sigmoid(gate_logits)
        out = out * gates.unsqueeze(-1)
        out = out.view(b, t, attn.inner_dim)
    handle = attn.to_out.forward_async(out)
    _profile_detail_end(layer_idx, f"{profile_base}.out_proj_async_launch", _prof_s)
    return _AsyncAttentionOutput(
        handle,
        layer_idx=layer_idx,
        profile_name=f"{profile_base}.out_proj_async_wait",
    )


class _AsyncFeedForwardOutput:
    def __init__(self, handle, layer_idx: Optional[int], profile_name: str) -> None:
        self.handle = handle
        self.layer_idx = layer_idx
        self.profile_name = profile_name

    def wait(self) -> Tensor:
        _prof_s = _profile_detail_start()
        out = self.handle.wait()
        _profile_detail_end(self.layer_idx, self.profile_name, _prof_s)
        return out


def _feedforward_async(
    ff: torch.nn.Module,
    x: Tensor,
    *,
    layer_idx: Optional[int],
    profile_base: str,
    enabled: bool = _TP_FF_OVERLAP,
):
    if not enabled or torch.is_grad_enabled():
        return None
    try:
        project_in = ff.net[0]
        middle = ff.net[1]
        project_out = ff.net[2]
    except Exception:
        return None
    if not isinstance(project_out, RowParallelLinear):
        return None
    _prof_s = _profile_detail_start()
    out = project_in(x)
    out = middle(out)
    handle = project_out.forward_async(out)
    _profile_detail_end(layer_idx, f"{profile_base}.launch", _prof_s)
    return _AsyncFeedForwardOutput(
        handle,
        layer_idx=layer_idx,
        profile_name=f"{profile_base}.wait",
    )


def _finish_cross_modal_pair_from_head_layout_async_h2s(
    audio_to_video_attn: CausalLTXAttention,
    video_to_audio_attn: CausalLTXAttention,
    a2v_q: Tensor,
    a2v_k: Tensor,
    a2v_v: Tensor,
    v2a_q: Tensor,
    v2a_k: Tensor,
    v2a_v: Tensor,
    vx_a2v: Tensor,
    ax_v2a: Tensor,
    layer_cache: LayerKVCache,
    *,
    video_real_len: Optional[int],
    audio_real_len: Optional[int],
    layer_idx: Optional[int],
    sp_size: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Finish paired A2V/V2A while overlapping A2V H2S with V2A SDPA.

    This is mathematically identical to calling
    ``_finish_attention_from_head_layout`` twice.  It only changes scheduling:
    after A2V SDPA, launch its head->seq all-to-all asynchronously, then compute
    V2A SDPA while that communication is in flight.
    """
    a2v_new_k, a2v_new_v, a2v_full_k, a2v_full_v = (
        _prepare_kv_for_head_layout_attention(
            a2v_k,
            a2v_v,
            layer_cache.a2v_k,
            layer_cache.a2v_v,
            new_kv_real_len=audio_real_len,
            layer_idx=layer_idx,
            profile_base="attn.audio_to_video",
        )
    )
    v2a_new_k, v2a_new_v, v2a_full_k, v2a_full_v = (
        _prepare_kv_for_head_layout_attention(
            v2a_k,
            v2a_v,
            layer_cache.v2a_k,
            layer_cache.v2a_v,
            new_kv_real_len=video_real_len,
            layer_idx=layer_idx,
            profile_base="attn.video_to_audio",
        )
    )

    _prof_s = _profile_detail_start()
    a2v_out = standard_attention_forward(a2v_q, a2v_full_k, a2v_full_v, mask=None)
    _profile_detail_end(layer_idx, "attn.audio_to_video.sdpa", _prof_s)

    a2v_h2s = None
    if sp_size > 1:
        _prof_s = _profile_detail_start()
        a2v_h2s = _head_all_to_all_seq_async(a2v_out, _get_sp_group(), sp_size)
        _profile_detail_end(layer_idx, "attn.audio_to_video.a2a_head2seq_launch", _prof_s)

    _prof_s = _profile_detail_start()
    v2a_out = standard_attention_forward(v2a_q, v2a_full_k, v2a_full_v, mask=None)
    _profile_detail_end(layer_idx, "attn.video_to_audio.sdpa", _prof_s)

    v2a_h2s = None
    if sp_size > 1:
        _prof_s = _profile_detail_start()
        v2a_h2s = _head_all_to_all_seq_async(v2a_out, _get_sp_group(), sp_size)
        _profile_detail_end(layer_idx, "attn.video_to_audio.a2a_head2seq_launch", _prof_s)

    if a2v_h2s is not None:
        _prof_s = _profile_detail_start()
        a2v_out = a2v_h2s.wait()
        _profile_detail_end(layer_idx, "attn.audio_to_video.a2a_head2seq_wait", _prof_s)
    if v2a_h2s is not None:
        _prof_s = _profile_detail_start()
        v2a_out = v2a_h2s.wait()
        _profile_detail_end(layer_idx, "attn.video_to_audio.a2a_head2seq_wait", _prof_s)

    a2v_out = _finish_attention_output_from_seq_layout(
        audio_to_video_attn,
        a2v_out,
        vx_a2v,
        layer_idx=layer_idx,
        profile_base="attn.audio_to_video",
    )
    v2a_out = _finish_attention_output_from_seq_layout(
        video_to_audio_attn,
        v2a_out,
        ax_v2a,
        layer_idx=layer_idx,
        profile_base="attn.video_to_audio",
    )
    return a2v_out, a2v_new_k, a2v_new_v, v2a_out, v2a_new_k, v2a_new_v


def _finish_cross_modal_pair_tp_out_overlap(
    audio_to_video_attn: CausalLTXAttention,
    video_to_audio_attn: CausalLTXAttention,
    a2v_q: Tensor,
    a2v_k: Tensor,
    a2v_v: Tensor,
    v2a_q: Tensor,
    v2a_k: Tensor,
    v2a_v: Tensor,
    vx_a2v: Tensor,
    ax_v2a: Tensor,
    layer_cache: LayerKVCache,
    *,
    video_real_len: Optional[int],
    audio_real_len: Optional[int],
    layer_idx: Optional[int],
) -> Optional[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]:
    """Overlap TP out-proj all-reduce for paired A2V/V2A cross attention.

    A2V and V2A both consume the pre-cross-modal states, so launching A2V's
    row-parallel output all-reduce before V2A SDPA is a pure scheduling change.
    """
    if not (
        isinstance(audio_to_video_attn.to_out, RowParallelLinear)
        and isinstance(video_to_audio_attn.to_out, RowParallelLinear)
    ):
        return None

    a2v_new_k, a2v_new_v, a2v_full_k, a2v_full_v = (
        _prepare_kv_for_head_layout_attention(
            a2v_k,
            a2v_v,
            layer_cache.a2v_k,
            layer_cache.a2v_v,
            new_kv_real_len=audio_real_len,
            layer_idx=layer_idx,
            profile_base="attn.audio_to_video",
        )
    )
    v2a_new_k, v2a_new_v, v2a_full_k, v2a_full_v = (
        _prepare_kv_for_head_layout_attention(
            v2a_k,
            v2a_v,
            layer_cache.v2a_k,
            layer_cache.v2a_v,
            new_kv_real_len=video_real_len,
            layer_idx=layer_idx,
            profile_base="attn.video_to_audio",
        )
    )

    _prof_s = _profile_detail_start()
    a2v_out = standard_attention_forward(a2v_q, a2v_full_k, a2v_full_v, mask=None)
    _profile_detail_end(layer_idx, "attn.audio_to_video.sdpa", _prof_s)

    a2v_async = _finish_attention_output_from_seq_layout_async(
        audio_to_video_attn,
        a2v_out,
        vx_a2v,
        layer_idx=layer_idx,
        profile_base="attn.audio_to_video",
    )
    if a2v_async is None:
        return None

    _prof_s = _profile_detail_start()
    v2a_out = standard_attention_forward(v2a_q, v2a_full_k, v2a_full_v, mask=None)
    _profile_detail_end(layer_idx, "attn.video_to_audio.sdpa", _prof_s)

    v2a_async = _finish_attention_output_from_seq_layout_async(
        video_to_audio_attn,
        v2a_out,
        ax_v2a,
        layer_idx=layer_idx,
        profile_base="attn.video_to_audio",
    )
    if v2a_async is None:
        return None

    a2v_out = a2v_async.wait()
    v2a_out = v2a_async.wait()
    return a2v_out, a2v_new_k, a2v_new_v, v2a_out, v2a_new_k, v2a_new_v


def _cross_modal_pair_attention_with_cache(
    audio_to_video_attn: CausalLTXAttention,
    video_to_audio_attn: CausalLTXAttention,
    vx_a2v: Tensor,
    ax_a2v: Tensor,
    ax_v2a: Tensor,
    vx_v2a: Tensor,
    video_cross_pe: Tuple[Tensor, Tensor],
    audio_cross_pe: Tuple[Tensor, Tensor],
    layer_cache: LayerKVCache,
    *,
    video_real_len: Optional[int],
    audio_real_len: Optional[int],
    layer_idx: Optional[int],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Run A2V and V2A with paired seq->head all-to-all.

    A2V and V2A both depend on the pre-cross-modal ``vx_norm3`` / ``ax_norm3``
    states.  They are therefore independent until their residual updates are
    applied, so their same-shaped Ulysses communications can be packed without
    changing attention math.
    """
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
    sp_group = _get_sp_group() if sp_size > 1 else None

    a2v_q, a2v_k, a2v_v = _project_qkv_local_for_cross_pair(
        audio_to_video_attn,
        vx_a2v,
        ax_a2v,
        video_cross_pe,
        audio_cross_pe,
        layer_idx=layer_idx,
        profile_base="attn.audio_to_video",
    )
    v2a_q, v2a_k, v2a_v = _project_qkv_local_for_cross_pair(
        video_to_audio_attn,
        ax_v2a,
        vx_v2a,
        audio_cross_pe,
        video_cross_pe,
        layer_idx=layer_idx,
        profile_base="attn.video_to_audio",
    )

    if sp_size > 1:
        _prof_s = _profile_detail_start()
        a2v_q, v2a_k, v2a_v = _seq_all_to_all_head_many(
            (a2v_q, v2a_k, v2a_v), sp_group, sp_size,
        )
        _profile_detail_end(layer_idx, "attn.cross_modal_pair.video_a2a_seq2head", _prof_s)

        _prof_s = _profile_detail_start()
        v2a_q, a2v_k, a2v_v = _seq_all_to_all_head_many(
            (v2a_q, a2v_k, a2v_v), sp_group, sp_size,
        )
        _profile_detail_end(layer_idx, "attn.cross_modal_pair.audio_a2a_seq2head", _prof_s)

    use_async_h2s = (
        _CROSS_MODAL_PAIR_ASYNC_H2S
        and sp_size > 1
        and not torch.is_grad_enabled()
    )
    if use_async_h2s:
        (
            a2v_out,
            new_a2v_k,
            new_a2v_v,
            v2a_out,
            new_v2a_k,
            new_v2a_v,
        ) = _finish_cross_modal_pair_from_head_layout_async_h2s(
            audio_to_video_attn,
            video_to_audio_attn,
            a2v_q,
            a2v_k,
            a2v_v,
            v2a_q,
            v2a_k,
            v2a_v,
            vx_a2v,
            ax_v2a,
            layer_cache,
            video_real_len=video_real_len,
            audio_real_len=audio_real_len,
            layer_idx=layer_idx,
            sp_size=sp_size,
        )
    elif (
        _CROSS_MODAL_TP_OUT_OVERLAP
        and sp_size == 1
        and not torch.is_grad_enabled()
    ):
        tp_overlap_result = _finish_cross_modal_pair_tp_out_overlap(
            audio_to_video_attn,
            video_to_audio_attn,
            a2v_q,
            a2v_k,
            a2v_v,
            v2a_q,
            v2a_k,
            v2a_v,
            vx_a2v,
            ax_v2a,
            layer_cache,
            video_real_len=video_real_len,
            audio_real_len=audio_real_len,
            layer_idx=layer_idx,
        )
        if tp_overlap_result is not None:
            (
                a2v_out,
                new_a2v_k,
                new_a2v_v,
                v2a_out,
                new_v2a_k,
                new_v2a_v,
            ) = tp_overlap_result
        else:
            a2v_out, new_a2v_k, new_a2v_v = _finish_attention_from_head_layout(
                audio_to_video_attn,
                a2v_q,
                a2v_k,
                a2v_v,
                vx_a2v,
                layer_cache.a2v_k,
                layer_cache.a2v_v,
                new_kv_real_len=audio_real_len,
                layer_idx=layer_idx,
                profile_base="attn.audio_to_video",
                sp_size=sp_size,
            )
            v2a_out, new_v2a_k, new_v2a_v = _finish_attention_from_head_layout(
                video_to_audio_attn,
                v2a_q,
                v2a_k,
                v2a_v,
                ax_v2a,
                layer_cache.v2a_k,
                layer_cache.v2a_v,
                new_kv_real_len=video_real_len,
                layer_idx=layer_idx,
                profile_base="attn.video_to_audio",
                sp_size=sp_size,
            )
    else:
        a2v_out, new_a2v_k, new_a2v_v = _finish_attention_from_head_layout(
            audio_to_video_attn,
            a2v_q,
            a2v_k,
            a2v_v,
            vx_a2v,
            layer_cache.a2v_k,
            layer_cache.a2v_v,
            new_kv_real_len=audio_real_len,
            layer_idx=layer_idx,
            profile_base="attn.audio_to_video",
            sp_size=sp_size,
        )
        v2a_out, new_v2a_k, new_v2a_v = _finish_attention_from_head_layout(
            video_to_audio_attn,
            v2a_q,
            v2a_k,
            v2a_v,
            ax_v2a,
            layer_cache.v2a_k,
            layer_cache.v2a_v,
            new_kv_real_len=video_real_len,
            layer_idx=layer_idx,
            profile_base="attn.video_to_audio",
            sp_size=sp_size,
        )
    return a2v_out, new_a2v_k, new_a2v_v, v2a_out, new_v2a_k, new_v2a_v


# ============================================================================
# Transformer Block with KV Cache
# ============================================================================

def block_forward_with_cache(
    block: CausalAVTransformerBlock,
    video_x: Tensor,
    audio_x: Tensor,
    video_timestep_6d: Tensor,
    audio_timestep_6d: Tensor,
    video_pe: Tuple[Tensor, Tensor],
    audio_pe: Tuple[Tensor, Tensor],
    video_cross_pe: Tuple[Tensor, Tensor],
    audio_cross_pe: Tuple[Tensor, Tensor],
    video_ctx: Tensor,
    audio_ctx: Tensor,
    video_context_mask: Optional[Tensor],
    audio_context_mask: Optional[Tensor],
    video_cross_ss: Tensor,
    video_cross_gate: Tensor,
    audio_cross_ss: Tensor,
    audio_cross_gate: Tensor,
    video_prompt_ts: Optional[Tensor],
    audio_prompt_ts: Optional[Tensor],
    video_prompt_cache_key: Optional[Tuple[float, ...]],
    audio_prompt_cache_key: Optional[Tuple[float, ...]],
    layer_cache: LayerKVCache,
    video_memory: Optional[Tensor] = None,
    audio_memory: Optional[Tensor] = None,
    color_memory: Optional[Tensor] = None,
    pyramid_policy: Optional[PyramidKVPolicy] = None,
    layer_idx: Optional[int] = None,
    video_frame_seqlen: Optional[int] = None,
    audio_frame_seqlen: Optional[int] = None,
    video_real_len: Optional[int] = None,
    audio_real_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, LayerKVCache]:
    """One transformer block forward with KV cache.

    Execution order mirrors ``CausalAVTransformerBlock.forward()`` exactly:
        Video self-attn → Video text-cross → Audio self-attn → Audio text-cross
        → A2V → V2A → Video FFN → Audio FFN

    Returns:
        ``(video_x, audio_x, updated_layer_cache)``
    """
    vx, ax = video_x, audio_x
    eps = block.norm_eps
    audio_self_precomputed = False

    # ``video_real_len`` / ``audio_real_len`` are the genuine token counts on the FULL
    # (post-a2a) sequence, set by ``model_forward_inference`` when SP is on
    # and a modality was end-padded so that its new-token count is divisible by
    # sp_size.  Forwarded verbatim to every ``attention_with_cache`` call whose
    # K/V come from that modality, so the persistent cache and the SDPA K
    # dimension stay pad-free. ``None`` (or the padded total when no pad was
    # applied) is a no-op inside ``attention_with_cache``.

    # ── Video Self-Attention ────────────────────────────────────────────
    _prof_s = _profile_detail_start()
    vshift_msa, vscale_msa, vgate_msa = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video_timestep_6d, slice(0, 3),
    )
    norm_vx = rms_norm(vx, eps=eps) * (1 + vscale_msa) + vshift_msa
    vx_attn, new_vs_k, new_vs_v = attention_with_cache(
        block.attn1, norm_vx, None,
        video_pe, None,
        layer_cache.video_self_k, layer_cache.video_self_v,
        pyramid_policy=pyramid_policy,
        layer_idx=layer_idx,
        modality="video",
        frame_seqlen=video_frame_seqlen,
        new_kv_real_len=video_real_len,
        profile_name="attn.video_self",
    )
    vx = vx + vx_attn * vgate_msa
    if getattr(block, "learned_memory_mode", None) == "memory_kv_side_branch":
        vx = block._apply_learned_memory(
            vx, video_memory, getattr(block, "video_memory_attn", None)
        )
        vx = block._apply_learned_color_film(
            vx, color_memory, getattr(block, "video_color_film", None)
        )
    _profile_detail_end(layer_idx, "video_self", _prof_s)

    if _PAIR_TEXT_CROSS_ASYNC_H2S:
        # Audio self-attention is independent
        # of video text-cross, so it can be computed before paired text-cross.
        if not audio_self_precomputed:
            _prof_s = _profile_detail_start()
            ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
                block.audio_scale_shift_table, ax.shape[0], audio_timestep_6d, slice(0, 3),
            )
            norm_ax = rms_norm(ax, eps=eps) * (1 + ascale_msa) + ashift_msa
            ax_attn, new_as_k, new_as_v = attention_with_cache(
                block.audio_attn1, norm_ax, None,
                audio_pe, None,
                layer_cache.audio_self_k, layer_cache.audio_self_v,
                pyramid_policy=pyramid_policy,
                layer_idx=layer_idx,
                modality="audio",
                frame_seqlen=audio_frame_seqlen,
                new_kv_real_len=audio_real_len,
                profile_name="attn.audio_self",
            )
            ax = ax + ax_attn * agate_msa
            if getattr(block, "learned_memory_mode", None) == "memory_kv_side_branch":
                ax = block._apply_learned_memory(
                    ax, audio_memory, getattr(block, "audio_memory_attn", None)
                )
            _profile_detail_end(layer_idx, "audio_self", _prof_s)

        paired_text = _apply_paired_text_cross_attention_for_inference(
            block,
            vx,
            ax,
            video_ctx,
            audio_ctx,
            video_timestep_6d,
            audio_timestep_6d,
            video_prompt_ts,
            audio_prompt_ts,
            video_context_mask,
            audio_context_mask,
            video_prompt_cache_key,
            audio_prompt_cache_key,
            layer_idx,
        )
        if paired_text is not None:
            _prof_s = _profile_detail_start()
            vx_text_attn, ax_text_attn = paired_text
            vx = vx + vx_text_attn
            ax = ax + ax_text_attn
            if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
                vx = block._apply_learned_memory(
                    vx, video_memory, getattr(block, "video_memory_attn", None)
                )
                vx = block._apply_learned_color_film(
                    vx, color_memory, getattr(block, "video_color_film", None)
                )
                ax = block._apply_learned_memory(
                    ax, audio_memory, getattr(block, "audio_memory_attn", None)
                )
            _profile_detail_end(layer_idx, "text_cross_pair", _prof_s)
        else:
            # Fallback for an unsupported fused update.
            _prof_s = _profile_detail_start()
            vx_text_attn = _apply_text_cross_attention_for_inference(
                block, vx, video_ctx, block.attn2,
                block.scale_shift_table,
                getattr(block, "prompt_scale_shift_table", None),
                video_timestep_6d, video_prompt_ts,
                video_context_mask,
                prompt_cache_key=video_prompt_cache_key,
                cache_name="video_text_cross",
                layer_idx=layer_idx,
            )
            vx = vx + vx_text_attn
            if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
                vx = block._apply_learned_memory(
                    vx, video_memory, getattr(block, "video_memory_attn", None)
                )
                vx = block._apply_learned_color_film(
                    vx, color_memory, getattr(block, "video_color_film", None)
                )
            _profile_detail_end(layer_idx, "video_text_cross", _prof_s)

            _prof_s = _profile_detail_start()
            ax_text_attn = _apply_text_cross_attention_for_inference(
                block, ax, audio_ctx, block.audio_attn2,
                block.audio_scale_shift_table,
                getattr(block, "audio_prompt_scale_shift_table", None),
                audio_timestep_6d, audio_prompt_ts,
                audio_context_mask,
                prompt_cache_key=audio_prompt_cache_key,
                cache_name="audio_text_cross",
                layer_idx=layer_idx,
            )
            ax = ax + ax_text_attn
            if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
                ax = block._apply_learned_memory(
                    ax, audio_memory, getattr(block, "audio_memory_attn", None)
                )
            _profile_detail_end(layer_idx, "audio_text_cross", _prof_s)

        del vshift_msa, vscale_msa, vgate_msa
        del ashift_msa, ascale_msa, agate_msa
    elif (
        _TEXT_CROSS_TP_OUT_OVERLAP
        and not torch.is_grad_enabled()
        and not getattr(block.attn2, "training", False)
        and isinstance(block.attn2.to_out, RowParallelLinear)
        and (_get_sp_world_size() if _is_sp_enabled() else 1) == 1
    ):
        v_text_async = None
        _prof_s = _profile_detail_start()
        v_input, v_ctx, v_base_ctx, _, v_gate = _prepare_text_cross_inputs_for_inference(
            block,
            vx,
            video_ctx,
            block.scale_shift_table,
            getattr(block, "prompt_scale_shift_table", None),
            video_timestep_6d,
            video_prompt_ts,
        )
        video_item = _text_cross_attention_head_layout(
            block.attn2,
            v_input,
            v_ctx,
            mask=video_context_mask,
            base_context=v_base_ctx,
            prompt_cache_key=video_prompt_cache_key,
            gate=v_gate,
            cache_name="video_text_cross",
            layer_idx=layer_idx,
        )
        v_text_async = _finish_text_cross_head_layout_async_tp(video_item)
        _profile_detail_end(layer_idx, "video_text_cross.tp_async_prefix", _prof_s)

        if v_text_async is None:
            _prof_s = _profile_detail_start()
            vx_text_attn = _apply_text_cross_attention_for_inference(
                block, vx, video_ctx, block.attn2,
                block.scale_shift_table,
                getattr(block, "prompt_scale_shift_table", None),
                video_timestep_6d, video_prompt_ts,
                video_context_mask,
                prompt_cache_key=video_prompt_cache_key,
                cache_name="video_text_cross",
                layer_idx=layer_idx,
            )
            vx = vx + vx_text_attn
            if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
                vx = block._apply_learned_memory(
                    vx, video_memory, getattr(block, "video_memory_attn", None)
                )
                vx = block._apply_learned_color_film(
                    vx, color_memory, getattr(block, "video_color_film", None)
                )
            _profile_detail_end(layer_idx, "video_text_cross", _prof_s)

        del vshift_msa, vscale_msa, vgate_msa

        # Audio self/text do not depend on the video text-cross residual, so
        # they can run while the video TP out-proj all-reduce is in flight.
        if not audio_self_precomputed:
            _prof_s = _profile_detail_start()
            ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
                block.audio_scale_shift_table, ax.shape[0], audio_timestep_6d, slice(0, 3),
            )
            norm_ax = rms_norm(ax, eps=eps) * (1 + ascale_msa) + ashift_msa
            ax_attn, new_as_k, new_as_v = attention_with_cache(
                block.audio_attn1, norm_ax, None,
                audio_pe, None,
                layer_cache.audio_self_k, layer_cache.audio_self_v,
                pyramid_policy=pyramid_policy,
                layer_idx=layer_idx,
                modality="audio",
                frame_seqlen=audio_frame_seqlen,
                new_kv_real_len=audio_real_len,
                profile_name="attn.audio_self",
            )
            ax = ax + ax_attn * agate_msa
            if getattr(block, "learned_memory_mode", None) == "memory_kv_side_branch":
                ax = block._apply_learned_memory(
                    ax, audio_memory, getattr(block, "audio_memory_attn", None)
                )
            _profile_detail_end(layer_idx, "audio_self", _prof_s)

        _prof_s = _profile_detail_start()
        ax_text_attn = _apply_text_cross_attention_for_inference(
            block, ax, audio_ctx, block.audio_attn2,
            block.audio_scale_shift_table,
            getattr(block, "audio_prompt_scale_shift_table", None),
            audio_timestep_6d, audio_prompt_ts,
            audio_context_mask,
            prompt_cache_key=audio_prompt_cache_key,
            cache_name="audio_text_cross",
            layer_idx=layer_idx,
        )
        ax = ax + ax_text_attn
        if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
            ax = block._apply_learned_memory(
                ax, audio_memory, getattr(block, "audio_memory_attn", None)
            )
        _profile_detail_end(layer_idx, "audio_text_cross", _prof_s)

        if v_text_async is not None:
            _prof_s = _profile_detail_start()
            vx = vx + v_text_async.wait()
            if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
                vx = block._apply_learned_memory(
                    vx, video_memory, getattr(block, "video_memory_attn", None)
                )
                vx = block._apply_learned_color_film(
                    vx, color_memory, getattr(block, "video_color_film", None)
                )
            _profile_detail_end(layer_idx, "video_text_cross", _prof_s)

        del ashift_msa, ascale_msa, agate_msa
    else:
        # ── Video-Text Cross-Attention ─────────────────────────────────
        _prof_s = _profile_detail_start()
        vx_text_attn = _apply_text_cross_attention_for_inference(
            block, vx, video_ctx, block.attn2,
            block.scale_shift_table,
            getattr(block, "prompt_scale_shift_table", None),
            video_timestep_6d, video_prompt_ts,
            video_context_mask,
            prompt_cache_key=video_prompt_cache_key,
            cache_name="video_text_cross",
            layer_idx=layer_idx,
        )
        vx = vx + vx_text_attn
        if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
            vx = block._apply_learned_memory(
                vx, video_memory, getattr(block, "video_memory_attn", None)
            )
            vx = block._apply_learned_color_film(
                vx, color_memory, getattr(block, "video_color_film", None)
            )
        _profile_detail_end(layer_idx, "video_text_cross", _prof_s)

        del vshift_msa, vscale_msa, vgate_msa

        if not audio_self_precomputed:
            # ── Audio Self-Attention ───────────────────────────────────────
            _prof_s = _profile_detail_start()
            ashift_msa, ascale_msa, agate_msa = block.get_ada_values(
                block.audio_scale_shift_table, ax.shape[0], audio_timestep_6d, slice(0, 3),
            )
            norm_ax = rms_norm(ax, eps=eps) * (1 + ascale_msa) + ashift_msa
            ax_attn, new_as_k, new_as_v = attention_with_cache(
                block.audio_attn1, norm_ax, None,
                audio_pe, None,
                layer_cache.audio_self_k, layer_cache.audio_self_v,
                pyramid_policy=pyramid_policy,
                layer_idx=layer_idx,
                modality="audio",
                frame_seqlen=audio_frame_seqlen,
                new_kv_real_len=audio_real_len,
                profile_name="attn.audio_self",
            )
            ax = ax + ax_attn * agate_msa
            if getattr(block, "learned_memory_mode", None) == "memory_kv_side_branch":
                ax = block._apply_learned_memory(
                    ax, audio_memory, getattr(block, "audio_memory_attn", None)
                )
            _profile_detail_end(layer_idx, "audio_self", _prof_s)

        # ── Audio-Text Cross-Attention ─────────────────────────────────
        _prof_s = _profile_detail_start()
        ax_text_attn = _apply_text_cross_attention_for_inference(
            block, ax, audio_ctx, block.audio_attn2,
            block.audio_scale_shift_table,
            getattr(block, "audio_prompt_scale_shift_table", None),
            audio_timestep_6d, audio_prompt_ts,
            audio_context_mask,
            prompt_cache_key=audio_prompt_cache_key,
            cache_name="audio_text_cross",
            layer_idx=layer_idx,
        )
        ax = ax + ax_text_attn
        if getattr(block, "learned_memory_mode", None) == "cross_attn_adapter":
            ax = block._apply_learned_memory(
                ax, audio_memory, getattr(block, "audio_memory_attn", None)
            )
        _profile_detail_end(layer_idx, "audio_text_cross", _prof_s)

        del ashift_msa, ascale_msa, agate_msa

    # ── Cross-Modal Attention (A2V & V2A) ──────────────────────────────
    new_a2v_k = new_a2v_v = new_v2a_k = new_v2a_v = None

    if not block.skip_cross_modal_attention:
        _prof_s = _profile_detail_start()
        vx_norm3 = rms_norm(vx, eps=eps)
        ax_norm3 = rms_norm(ax, eps=eps)

        # Cross-modal AdaLN values
        (scale_ca_audio_a2v, shift_ca_audio_a2v,
         scale_ca_audio_v2a, shift_ca_audio_v2a,
         gate_out_v2a) = block.get_av_ca_ada_values(
            block.scale_shift_table_a2v_ca_audio,
            ax.shape[0], audio_cross_ss, audio_cross_gate,
        )
        (scale_ca_video_a2v, shift_ca_video_a2v,
         scale_ca_video_v2a, shift_ca_video_v2a,
         gate_out_a2v) = block.get_av_ca_ada_values(
            block.scale_shift_table_a2v_ca_video,
            vx.shape[0], video_cross_ss, video_cross_gate,
        )
        _profile_detail_end(layer_idx, "cross_modal_setup", _prof_s)

        vx_a2v = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
        ax_a2v = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
        ax_v2a = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
        vx_v2a = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a

        use_cross_modal_pair = (
            _FUSE_CROSS_MODAL_A2A
            and not torch.is_grad_enabled()
            and (
                (_is_sp_enabled() and _get_sp_world_size() > 1)
                or _CROSS_MODAL_TP_OUT_OVERLAP
            )
        )
        if use_cross_modal_pair:
            _prof_s = _profile_detail_start()
            (
                a2v_out,
                new_a2v_k,
                new_a2v_v,
                v2a_out,
                new_v2a_k,
                new_v2a_v,
            ) = _cross_modal_pair_attention_with_cache(
                block.audio_to_video_attn,
                block.video_to_audio_attn,
                vx_a2v,
                ax_a2v,
                ax_v2a,
                vx_v2a,
                video_cross_pe,
                audio_cross_pe,
                layer_cache,
                video_real_len=video_real_len,
                audio_real_len=audio_real_len,
                layer_idx=layer_idx,
            )
            vx = vx + a2v_out * gate_out_a2v
            ax = ax + v2a_out * gate_out_v2a
            _profile_detail_end(layer_idx, "cross_modal_pair", _prof_s)
        else:
            # A2V: Video queries Audio
            _prof_s = _profile_detail_start()
            a2v_out, new_a2v_k, new_a2v_v = attention_with_cache(
                block.audio_to_video_attn,
                vx_a2v, ax_a2v,
                video_cross_pe, audio_cross_pe,
                layer_cache.a2v_k, layer_cache.a2v_v,
                new_kv_real_len=audio_real_len,
                profile_name="attn.audio_to_video",
            )
            vx = vx + a2v_out * gate_out_a2v
            _profile_detail_end(layer_idx, "audio_to_video", _prof_s)

            # V2A: Audio queries Video (use vx_norm3 from BEFORE A2V update)
            _v2a_prof_s = _profile_detail_start()
            v2a_out, new_v2a_k, new_v2a_v = attention_with_cache(
                block.video_to_audio_attn,
                ax_v2a, vx_v2a,
                audio_cross_pe, video_cross_pe,
                layer_cache.v2a_k, layer_cache.v2a_v,
                new_kv_real_len=video_real_len,
                profile_name="attn.video_to_audio",
            )
            ax = ax + v2a_out * gate_out_v2a
            _profile_detail_end(layer_idx, "video_to_audio", _v2a_prof_s)

    # ── Feed-Forward Networks ──────────────────────────────────────────
    _prof_s = _profile_detail_start()
    vshift_mlp, vscale_mlp, vgate_mlp = block.get_ada_values(
        block.scale_shift_table, vx.shape[0], video_timestep_6d, slice(3, 6),
    )
    v_ff_input = rms_norm(vx, eps=eps) * (1 + vscale_mlp) + vshift_mlp
    v_ff_total_s = _prof_s
    v_ff_async = _feedforward_async(
        block.ff,
        v_ff_input,
        layer_idx=layer_idx,
        profile_base="video_ff.async",
    )
    if v_ff_async is None:
        vx = vx + block.ff(v_ff_input) * vgate_mlp
        _profile_detail_end(layer_idx, "video_ff", _prof_s)

    _prof_s = _profile_detail_start()
    ashift_mlp, ascale_mlp, agate_mlp = block.get_ada_values(
        block.audio_scale_shift_table, ax.shape[0], audio_timestep_6d, slice(3, 6),
    )
    a_ff_input = rms_norm(ax, eps=eps) * (1 + ascale_mlp) + ashift_mlp
    a_ff_total_s = _prof_s
    a_ff_async = _feedforward_async(
        block.audio_ff,
        a_ff_input,
        layer_idx=layer_idx,
        profile_base="audio_ff.async",
        enabled=_TP_AUDIO_FF_OVERLAP,
    )
    if a_ff_async is None:
        ax = ax + block.audio_ff(a_ff_input) * agate_mlp
        _profile_detail_end(layer_idx, "audio_ff", _prof_s)

    if v_ff_async is not None:
        vx = vx + v_ff_async.wait() * vgate_mlp
        _profile_detail_end(layer_idx, "video_ff", v_ff_total_s)
    if a_ff_async is not None:
        ax = ax + a_ff_async.wait() * agate_mlp
        _profile_detail_end(layer_idx, "audio_ff", a_ff_total_s)

    # ── Update layer cache ─────────────────────────────────────────────
    _prof_s = _profile_detail_start()
    updated_cache = LayerKVCache(
        video_self_k=_cat_cache(layer_cache.video_self_k, new_vs_k),
        video_self_v=_cat_cache(layer_cache.video_self_v, new_vs_v),
        audio_self_k=_cat_cache(layer_cache.audio_self_k, new_as_k),
        audio_self_v=_cat_cache(layer_cache.audio_self_v, new_as_v),
        a2v_k=_cat_cache(layer_cache.a2v_k, new_a2v_k) if new_a2v_k is not None else layer_cache.a2v_k,
        a2v_v=_cat_cache(layer_cache.a2v_v, new_a2v_v) if new_a2v_v is not None else layer_cache.a2v_v,
        v2a_k=_cat_cache(layer_cache.v2a_k, new_v2a_k) if new_v2a_k is not None else layer_cache.v2a_k,
        v2a_v=_cat_cache(layer_cache.v2a_v, new_v2a_v) if new_v2a_v is not None else layer_cache.v2a_v,
    )
    _profile_detail_end(layer_idx, "cache_update", _prof_s)

    return vx, ax, updated_cache


# ============================================================================
# Gradient Checkpointing Helper
# ============================================================================

def _run_block_with_kv_cache(tblock, kv_kwargs):
    """Thin wrapper for gradient checkpointing compatibility.

    Separates the block FSDP ``__call__`` from the external cache assignment,
    so that ``torch.utils.checkpoint`` can recompute intermediates during
    backward without duplicating the cache-mutation side effect.

    During recomputation (backward), FSDP hooks are triggered again to
    all-gather the block's parameters — this is the standard FSDP + activation
    checkpointing pattern supported since PyTorch 2.0.
    """
    return tblock(_kv_cache_kwargs=kv_kwargs)


# ============================================================================
# Model-Level Forward Inference
# ============================================================================

def model_forward_inference(
    model: "CausalLTXModel",
    video_latent: Tensor,
    audio_latent: Tensor,
    timesteps: Tensor,
    audio_timesteps: Optional[Tensor],
    video_context: Tensor,
    audio_context: Tensor,
    video_context_mask: Optional[Tensor] = None,
    audio_context_mask: Optional[Tensor] = None,
    learned_memory_video: Optional[Tensor] = None,
    learned_memory_audio: Optional[Tensor] = None,
    learned_memory_color: Optional[Tensor] = None,
    kv_cache: Optional[KVCache] = None,
    video_start_frame: int = 0,
    audio_start_frame: int = 0,
    include_audio_sinks: bool = True,
    pyramid_policy: Optional[PyramidKVPolicy] = None,
    kv_cache_only: bool = False,
) -> Tuple[Optional[Tensor], Optional[Tensor], KVCache]:
    """Full model forward pass with KV cache — mirrors ``CausalLTXModel.forward()``.

    Only processes the *new* block's tokens.  Cached prefix K/V are reused.

    Args:
        model: The trained ``CausalLTXModel`` (weights accessed via it).
        video_latent: ``[B, F_v_new, C, H, W]`` — current block only.
        audio_latent: ``[B, F_a_new, C]`` — current block only.
        timesteps: ``[B, F_v_new]`` sigma values for video.
        audio_timesteps: ``[B, F_a_new]`` sigma values for audio (or None → use *timesteps*).
        video_context: ``[B, L_ctx, caption_channels]`` text embeddings.
        audio_context: Same shape as *video_context*.
        video_context_mask: Optional padding mask for video text context.
        audio_context_mask: Optional padding mask for audio text context.
        kv_cache: Accumulated cache from previous blocks (None → first call).
        video_start_frame: Absolute video frame index of the first new frame
            (for correct RoPE positioning).
        audio_start_frame: Absolute audio frame index.
        include_audio_sinks: Prepend audio sink tokens (True only for block 0).
        kv_cache_only: When True, return after transformer blocks update the
            cache and skip SP gather, output projection, and unpatchify. This is
            exact for clean prefix/KV commit calls whose outputs are discarded.

    Returns:
        ``(video_velocity, audio_velocity, updated_kv_cache)``
    """
    # Avoid circular import at module level
    from ltx_causal.transformer.causal_model import CausalLTXModel

    # ── torch.compile: 标记动态维度，避免 KV cache 长度 / chunk size 变化触发
    #    重编译。仅在 LTX_COMPILE=1 时生效，否则 _mark_dyn 是 no-op。
    #    诊断 44 次 recompile 后引入；目标：把 recompile 收敛到 1-2 次。
    if _LTX_COMPILE_ENABLED:
        # video_latent: [B, F_v_new, C, H, W] —— F_v_new 与 H/W 可能在 prefix
        # renorm 阶段 vs denoise 阶段、以及 4-frame vs 3-frame chunk 间变化
        _mark_dyn(video_latent, 1, 3, 4)
        # audio_latent: [B, F_a, audio_dim] —— F_a 同样可变
        _mark_dyn(audio_latent, 1)
        # context: [B, L_ctx, caption_ch] —— L_ctx 一般固定 512，但稳妥起见
        _mark_dyn(video_context, 1)
        _mark_dyn(audio_context, 1)
        # timesteps: [B, F_v_new] / [B, F_a]
        _mark_dyn(timesteps, 1)
        _mark_dyn(audio_timesteps, 1)
        # KV cache 是真正的变长源头：每 forward 长度递增
        if kv_cache is not None:
            for layer in kv_cache.layers:
                if layer is None:
                    continue
                for cached in (
                    layer.video_self_k, layer.video_self_v,
                    layer.audio_self_k, layer.audio_self_v,
                    layer.a2v_k, layer.a2v_v,
                    layer.v2a_k, layer.v2a_v,
                ):
                    _mark_dyn(cached, 1)  # seq_len dim

    # ── Profiler: forward entry ──
    if _PROFILER is not None:
        _PROFILER.forward_calls += 1
        _prof_pre_s = _PROFILER._new_event()
        _prof_pre_s.record()

    config = model.config
    B = video_latent.shape[0]
    device = video_latent.device
    hidden_dtype = video_latent.dtype

    # ================================================================
    # 1. Patch Embedding
    # ================================================================
    B_v, F_v, C_v, H_v, W_v = video_latent.shape
    video_flat = video_latent.permute(0, 2, 1, 3, 4).reshape(B_v, C_v, -1).permute(0, 2, 1)
    video_x = model.patchify_proj(video_flat)  # [B, F_v*H*W, D]
    video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

    audio_x = model.audio_patchify_proj(audio_latent)  # [B, F_a, audio_dim]
    F_a_original = audio_x.shape[1]
    audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

    # ================================================================
    # 2. Context Projection (recomputed each call — trivial cost)
    # ================================================================
    video_ctx = _prepare_context_for_inference(
        model,
        video_context,
        model.caption_projection,
        config.video_dim,
        B,
        cache_name="video",
    )
    audio_ctx = _prepare_context_for_inference(
        model,
        audio_context,
        model.audio_caption_projection,
        config.audio_dim,
        B,
        cache_name="audio",
    )
    video_memory, audio_memory, color_memory = _prepare_learned_memory_for_inference(
        model,
        learned_memory_video,
        learned_memory_audio,
        learned_memory_color,
        device=device,
        dtype=hidden_dtype,
    )

    # ================================================================
    # 3. Audio Sink Tokens (only block 0)
    # ================================================================
    num_sink = config.num_audio_sink_tokens
    if num_sink > 0 and include_audio_sinks:
        sink_expanded = model.audio_sink_tokens.expand(B, -1, -1).to(audio_x.dtype)
        if config.condition_sink_on_text and hasattr(model, "sink_text_condition"):
            ctx_pooled = audio_ctx.mean(dim=1)
            scale_shift = model.sink_text_condition(ctx_pooled)
            scale, shift = scale_shift.chunk(2, dim=-1)
            sink_expanded = (
                sink_expanded * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            )
        audio_x = torch.cat([sink_expanded, audio_x], dim=1)

    # ================================================================
    # 4. Attention Masks
    # ================================================================
    video_context_mask = model._prepare_attention_mask(video_context_mask, hidden_dtype)
    audio_context_mask = model._prepare_attention_mask(audio_context_mask, hidden_dtype)

    # ================================================================
    # 5. Timestep Embedding (AdaLN)
    # ================================================================
    video_ts = timesteps
    video_timestep_6d, video_embedded_ts = model._prepare_timestep(
        video_ts, model.adaln_single, B, hidden_dtype,
    )

    audio_ts = audio_timesteps if audio_timesteps is not None else timesteps
    # Expand audio timestep with sink entries when applicable
    if num_sink > 0 and include_audio_sinks and audio_ts.ndim == 2:
        sink_ts = audio_ts[:, :1].expand(-1, num_sink)
        audio_ts_expanded = torch.cat([sink_ts, audio_ts], dim=1)
    else:
        audio_ts_expanded = audio_ts

    audio_timestep_6d, audio_embedded_ts_full = model._prepare_timestep(
        audio_ts_expanded, model.audio_adaln_single, B, hidden_dtype,
    )
    # Strip sink entries from embedded timestep for output processing
    if num_sink > 0 and include_audio_sinks and audio_embedded_ts_full.shape[1] > 1:
        audio_embedded_ts = audio_embedded_ts_full[:, num_sink:]
    else:
        audio_embedded_ts = audio_embedded_ts_full

    # ================================================================
    # 6. Cross-Attention Timesteps
    # ================================================================
    video_cross_ss, video_cross_gate = model._prepare_cross_attention_timestep(
        video_ts,
        model.av_ca_video_scale_shift_adaln_single,
        model.av_ca_a2v_gate_adaln_single,
        B, hidden_dtype,
    )
    audio_cross_ss, audio_cross_gate = model._prepare_cross_attention_timestep(
        audio_ts_expanded,
        model.av_ca_audio_scale_shift_adaln_single,
        model.av_ca_v2a_gate_adaln_single,
        B, hidden_dtype,
    )

    # ================================================================
    # 7. Prompt Timestep (LTX-2.3 text cross-attention AdaLN)
    # ================================================================
    video_prompt_ts = None
    audio_prompt_ts = None
    video_prompt_cache_key = None
    audio_prompt_cache_key = None
    if model.cross_attention_adaln:
        video_ts_scalar = (
            video_ts.mean(dim=-1, keepdim=True) if video_ts.ndim == 2 else video_ts
        )
        audio_ts_scalar = (
            audio_ts_expanded.mean(dim=-1, keepdim=True)
            if audio_ts_expanded.ndim == 2 else audio_ts_expanded
        )
        video_prompt_ts, _ = model._prepare_timestep(
            video_ts_scalar, model.prompt_adaln_single, B, hidden_dtype,
        )
        audio_prompt_ts, _ = model._prepare_timestep(
            audio_ts_scalar, model.audio_prompt_adaln_single, B, hidden_dtype,
        )
        if _CACHE_TEXT_CROSS_KV and not torch.is_grad_enabled() and not model.training:
            video_prompt_cache_key = _small_tensor_value_key(video_ts_scalar)
            audio_prompt_cache_key = _small_tensor_value_key(audio_ts_scalar)

    # ================================================================
    # 8. Expand per-frame → per-token for video
    # ================================================================
    frame_seqlen = H_v * W_v
    _expand = CausalLTXModel._expand_per_frame_to_per_token
    video_timestep_6d = _expand(video_timestep_6d, frame_seqlen)
    video_embedded_ts = _expand(video_embedded_ts, frame_seqlen)
    video_cross_ss = _expand(video_cross_ss, frame_seqlen)
    video_cross_gate = _expand(video_cross_gate, frame_seqlen)

    # ================================================================
    # 9. RoPE (with start_frame offset for correct absolute positions)
    # ================================================================

    # -- Self-attention RoPE --
    video_pe = _causal_precompute_freqs_cis_for_inference(
        model, "video_self", (F_v, H_v, W_v),
        video_grid_sizes, config.video_d_head * config.video_heads,
        theta=config.pe_theta, max_pos=list(config.pe_max_pos),
        start_frame=video_start_frame, rope_type=config.rope_type,
        rope_extrapolation=getattr(config, "rope_extrapolation", "off"),
        rope_train_max_seconds=getattr(config, "rope_train_max_seconds", 8.0),
        device=device, dtype=video_x.dtype,
        num_attention_heads=config.video_heads,
    )
    audio_pe = _causal_precompute_freqs_cis_for_inference(
        model, "audio_self", (F_a_original,),
        audio_grid_sizes, config.audio_d_head * config.audio_heads,
        theta=config.pe_theta, max_pos=list(config.audio_pe_max_pos),
        start_frame=audio_start_frame, rope_type=config.rope_type,
        rope_extrapolation=getattr(config, "rope_extrapolation", "off"),
        rope_train_max_seconds=getattr(config, "rope_train_max_seconds", 8.0),
        device=device, dtype=audio_x.dtype, is_audio=True,
        num_attention_heads=config.audio_heads,
    )
    # Prepend identity RoPE for sink tokens (cos=1, sin=0 → no rotation)
    if num_sink > 0 and include_audio_sinks:
        if config.rope_type == CausalRopeType.SPLIT:
            # SPLIT: pe shape is (B, H, T, D_half)
            b, h, _, d_half = audio_pe[0].shape
            sink_cos = torch.ones(b, h, num_sink, d_half, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(b, h, num_sink, d_half, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=2),
                torch.cat([sink_sin, audio_pe[1]], dim=2),
            )
        else:
            # INTERLEAVED: pe shape is (B, T, dim)
            rd = audio_pe[0].shape[-1]
            sink_cos = torch.ones(1, num_sink, rd, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(1, num_sink, rd, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=1),
                torch.cat([sink_sin, audio_pe[1]], dim=1),
            )

    # -- Cross-modal RoPE (1-D temporal) --
    cross_pe_max_pos = max(config.pe_max_pos[0], config.audio_pe_max_pos[0])

    video_temporal_grid = torch.tensor([[F_v]], device=device, dtype=torch.long)
    video_cross_pe = _causal_precompute_freqs_cis_for_inference(
        model, "video_cross", (F_v,),
        video_temporal_grid, config.audio_cross_attention_dim,
        theta=config.pe_theta, max_pos=[cross_pe_max_pos],
        start_frame=video_start_frame, rope_type=config.rope_type,
        rope_extrapolation=getattr(config, "rope_extrapolation", "off"),
        rope_train_max_seconds=getattr(config, "rope_train_max_seconds", 8.0),
        device=device, dtype=video_x.dtype, is_audio=False,
        num_attention_heads=config.audio_heads,
    )
    # Expand temporal PE → per-token (each frame's tokens share same PE)
    # Use the actual H*W (frame_seqlen, computed at L551 from current
    # video_x shape) instead of config.video_frame_seqlen, which is the
    # __init__-time default (yaml-default 384). Required for multi-resolution.
    #
    # SPLIT mode returns [B, H, T, D_per_head] (4D);
    # INTERLEAVED mode returns [B, T, dim] (3D).
    if video_cross_pe[0].ndim == 4:
        # SPLIT: [B, H, T, D] → [B, H, T, frame_seqlen, D] → [B, H, T*frame_seqlen, D]
        B_pe, H_pe, T_pe, D_pe = video_cross_pe[0].shape
        video_cross_pe = (
            video_cross_pe[0]
            .unsqueeze(3).expand(-1, -1, -1, frame_seqlen, -1)
            .reshape(B_pe, H_pe, T_pe * frame_seqlen, D_pe),
            video_cross_pe[1]
            .unsqueeze(3).expand(-1, -1, -1, frame_seqlen, -1)
            .reshape(B_pe, H_pe, T_pe * frame_seqlen, D_pe),
        )
    else:
        # INTERLEAVED: [B, T, dim] → [B, T, frame_seqlen, dim] → [B, T*frame_seqlen, dim]
        video_cross_pe = (
            video_cross_pe[0]
            .unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1]
            .unsqueeze(2).expand(-1, -1, frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

    audio_temporal_grid = torch.tensor(
        [[F_a_original]], device=device, dtype=torch.long,
    )
    audio_cross_pe = _causal_precompute_freqs_cis_for_inference(
        model, "audio_cross", (F_a_original,),
        audio_temporal_grid, config.audio_cross_attention_dim,
        theta=config.pe_theta, max_pos=[cross_pe_max_pos],
        start_frame=audio_start_frame, rope_type=config.rope_type,
        rope_extrapolation=getattr(config, "rope_extrapolation", "off"),
        rope_train_max_seconds=getattr(config, "rope_train_max_seconds", 8.0),
        device=device, dtype=audio_x.dtype, is_audio=True,
        num_attention_heads=config.audio_heads,
    )
    # Prepend identity cross-PE for sink tokens
    if num_sink > 0 and include_audio_sinks:
        if config.rope_type == CausalRopeType.SPLIT:
            # SPLIT: shape (B, H, T, D_half)
            b, h, _, d_half = audio_cross_pe[0].shape
            sink_cc = torch.ones(b, h, num_sink, d_half, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cs = torch.zeros(b, h, num_sink, d_half, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cc, audio_cross_pe[0]], dim=2),
                torch.cat([sink_cs, audio_cross_pe[1]], dim=2),
            )
        else:
            # INTERLEAVED: shape (B, T, dim)
            crd = audio_cross_pe[0].shape[-1]
            sink_cc = torch.ones(1, num_sink, crd, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cs = torch.zeros(1, num_sink, crd, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cc, audio_cross_pe[0]], dim=1),
                torch.cat([sink_cs, audio_cross_pe[1]], dim=1),
            )
    # ================================================================
    # 9.5 Slice RoPE for Tensor Parallelism
    #     TP 按 head 切分 attention，每卡只需对应 head 的 RoPE 子集。
    #     用全 dim 计算频率后按 head 维度切片，保证频率分布与训练一致。
    # ================================================================
    tp_rank = getattr(model, '_tp_rank', None)
    tp_size = getattr(model, '_tp_size', None)
    if tp_rank is not None and tp_size is not None and tp_size > 1:
        def _slice_pe(pe, full_dim):
            cos, sin = pe
            if config.rope_type == CausalRopeType.SPLIT:
                # SPLIT RoPE layout is [B, H, T, D_head/2]. TP shards heads,
                # so keep the local head range and leave the per-head rotary
                # dimension unchanged. Slicing the last dimension would make
                # cos/sin incompatible with apply_split_rotary_emb().
                heads = cos.shape[1]
                head_chunk = heads // tp_size
                head_start = tp_rank * head_chunk
                head_end = head_start + head_chunk
                return (
                    cos[:, head_start:head_end, :, :].contiguous(),
                    sin[:, head_start:head_end, :, :].contiguous(),
                )

            # INTERLEAVED RoPE layout is [B, T, full_inner_dim], so TP sharding
            # maps to the last embedding dimension.
            chunk = full_dim // tp_size
            s = tp_rank * chunk
            return (cos[..., s:s + chunk].contiguous(), sin[..., s:s + chunk].contiguous())

        video_pe = _slice_pe(
            video_pe, config.video_d_head * config.video_heads,
        )
        audio_pe = _slice_pe(
            audio_pe, config.audio_d_head * config.audio_heads,
        )
        video_cross_pe = _slice_pe(
            video_cross_pe, config.audio_cross_attention_dim,
        )
        audio_cross_pe = _slice_pe(
            audio_cross_pe, config.audio_cross_attention_dim,
        )

    # ================================================================
    # 9.5 [Ulysses SP] Sequence-Parallel sharding (entry)
    # ================================================================
    # Same partitioning rules as ``CausalLTXModel.forward()``: split every
    # per-token tensor along the sequence dim into the local shard. The
    # *embedded_ts kept full on every rank are only consumed at the output
    # stage AFTER gather_sequence(); cache K/V transition into head-sharded
    # layout via the seq->head all-to-all inside ``attention_with_cache``.
    #
    # Padding policy (causal inference path):
    #   - VIDEO/AUDIO are end-padded only to make Ulysses sequence chunks even.
    #     The padded rows are query-only work rows: K/V are stripped back to the
    #     genuine length inside ``attention_with_cache`` before cache append and
    #     before SDPA, PE pad uses identity rotation (cos=1, sin=0), and gathered
    #     outputs are unpadded before projection/unpatchify. This keeps the
    #     persistent KV cache and visible output layout identical to the unpadded
    #     sequence.
    sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
    video_pad_extra = 0
    audio_pad_extra = 0
    V_total_real = video_x.shape[1]
    A_total_real = audio_x.shape[1]
    if sp_size > 1:
        V_total = video_x.shape[1]
        A_total = audio_x.shape[1]

        def _rope_seq_dim(pe):
            # SPLIT RoPE is [B, H, T, D]; interleaved RoPE is [B, T, D].
            return 2 if pe[0].ndim == 4 else 1

        def _pad_pe_identity(pe, extra_n):
            # cos pad = 1, sin pad = 0 -> identity rotation. Padded rows are
            # dropped as K/V and unpadded before output.
            if extra_n <= 0:
                return pe
            cos, sin = pe
            dim = _rope_seq_dim(pe)
            cos_pad_shape = list(cos.shape)
            cos_pad_shape[dim] = extra_n
            sin_pad_shape = list(sin.shape)
            sin_pad_shape[dim] = extra_n
            cos_pad = torch.ones(cos_pad_shape, device=cos.device, dtype=cos.dtype)
            sin_pad = torch.zeros(sin_pad_shape, device=sin.device, dtype=sin.dtype)
            return (
                torch.cat([cos, cos_pad], dim=dim),
                torch.cat([sin, sin_pad], dim=dim),
            )

        def _pad_repeat_last(t, extra_n):
            # Replicate the last real row for AdaLN scale/shift tensors. Padded
            # rows are unpadded before output, so this only keeps intermediate
            # math finite.
            if extra_n <= 0 or t.shape[1] == 1:
                return t
            last = t[:, -1:].expand(-1, extra_n, *([-1] * (t.ndim - 2)))
            return torch.cat([t, last.contiguous()], dim=1)

        def _pad_zero_rows(t, extra_n):
            if extra_n <= 0 or t.shape[1] == 1:
                return t
            pad_shape = list(t.shape)
            pad_shape[1] = extra_n
            return torch.cat(
                [t, torch.zeros(pad_shape, device=t.device, dtype=t.dtype)],
                dim=1,
            )

        if V_total % sp_size != 0:
            video_pad_extra = sp_size - (V_total % sp_size)
            B_vx, _, D_vx = video_x.shape
            video_x = torch.cat(
                [
                    video_x,
                    torch.zeros(
                        (B_vx, video_pad_extra, D_vx),
                        device=video_x.device,
                        dtype=video_x.dtype,
                    ),
                ],
                dim=1,
            )
            video_pe = _pad_pe_identity(video_pe, video_pad_extra)
            video_cross_pe = _pad_pe_identity(video_cross_pe, video_pad_extra)
            video_timestep_6d = _pad_repeat_last(video_timestep_6d, video_pad_extra)
            video_cross_ss = _pad_repeat_last(video_cross_ss, video_pad_extra)
            # A2V output for padded video query rows has no contribution.
            video_cross_gate = _pad_zero_rows(video_cross_gate, video_pad_extra)

        # ── Audio end-pad (NEW) ──────────────────────────────────────────
        if A_total % sp_size != 0:
            audio_pad_extra = sp_size - (A_total % sp_size)
            extra = audio_pad_extra
            B_a, _, D_a = audio_x.shape
            zero_x = torch.zeros(
                (B_a, extra, D_a), device=audio_x.device, dtype=audio_x.dtype,
            )
            audio_x = torch.cat([audio_x, zero_x], dim=1)

            audio_pe = _pad_pe_identity(audio_pe, extra)
            audio_cross_pe = _pad_pe_identity(audio_cross_pe, extra)
            audio_timestep_6d = _pad_repeat_last(audio_timestep_6d, extra)
            audio_cross_ss = _pad_repeat_last(audio_cross_ss, extra)

            # Gate of 0 means the cross-attn output for padded rows has zero
            # contribution. Padded ax rows feed back as zero anyway, but we
            # keep this defensive.
            audio_cross_gate = _pad_zero_rows(audio_cross_gate, extra)

        video_x = _split_sequence(video_x, dim=1, sp_size=sp_size)
        audio_x = _split_sequence(audio_x, dim=1, sp_size=sp_size)

        def _split_rope_sequence(pe):
            # SPLIT RoPE is [B, H, T, D]; interleaved RoPE is [B, T, D].
            # Sequence parallelism must shard the token dimension, not heads.
            seq_dim = 2 if pe[0].ndim == 4 else 1
            return (
                _split_sequence(pe[0], dim=seq_dim, sp_size=sp_size),
                _split_sequence(pe[1], dim=seq_dim, sp_size=sp_size),
            )

        video_pe = _split_rope_sequence(video_pe)
        audio_pe = _split_rope_sequence(audio_pe)
        video_cross_pe = _split_rope_sequence(video_cross_pe)
        audio_cross_pe = _split_rope_sequence(audio_cross_pe)

        if video_timestep_6d.shape[1] > 1:
            video_timestep_6d = _split_sequence(video_timestep_6d, dim=1, sp_size=sp_size)
        if video_cross_ss.shape[1] > 1:
            video_cross_ss = _split_sequence(video_cross_ss, dim=1, sp_size=sp_size)
        if video_cross_gate.shape[1] > 1:
            video_cross_gate = _split_sequence(video_cross_gate, dim=1, sp_size=sp_size)
        if audio_timestep_6d.shape[1] > 1:
            audio_timestep_6d = _split_sequence(audio_timestep_6d, dim=1, sp_size=sp_size)
        if audio_cross_ss.shape[1] > 1:
            audio_cross_ss = _split_sequence(audio_cross_ss, dim=1, sp_size=sp_size)
        if audio_cross_gate.shape[1] > 1:
            audio_cross_gate = _split_sequence(audio_cross_gate, dim=1, sp_size=sp_size)

    # ================================================================
    # 10. Initialise KV Cache (if first call)
    # ================================================================
    num_layers = len(model.transformer_blocks)
    if kv_cache is None:
        kv_cache = KVCache(layers=[LayerKVCache() for _ in range(num_layers)])

    # ================================================================
    # 11. Iterate through Transformer Blocks
    #     When gradient_checkpointing is enabled and we are in a grad-
    #     tracked context (exit step), wrap each block call with
    #     torch.utils.checkpoint to free intermediate activations.
    #     This is the standard FSDP + activation checkpointing pattern:
    #     during backward recomputation, FSDP re-gathers block params.
    # ================================================================
    use_ckpt = (
        getattr(model, 'gradient_checkpointing', False)
        and model.training
        and torch.is_grad_enabled()
    )

    for i, tblock in enumerate(model.transformer_blocks):
        layer_cache = kv_cache.layers[i]
        if kv_cache.layerwise_cpu_offload:
            layer_cache = _move_layer_cache(
                layer_cache,
                video_x.device,
                detach=False,
            )
        kv_kwargs = dict(
            video_x=video_x,
            audio_x=audio_x,
            video_timestep_6d=video_timestep_6d,
            audio_timestep_6d=audio_timestep_6d,
            video_pe=video_pe,
            audio_pe=audio_pe,
            video_cross_pe=video_cross_pe,
            audio_cross_pe=audio_cross_pe,
            video_ctx=video_ctx,
            audio_ctx=audio_ctx,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            video_cross_ss=video_cross_ss,
            video_cross_gate=video_cross_gate,
            audio_cross_ss=audio_cross_ss,
            audio_cross_gate=audio_cross_gate,
            video_prompt_ts=video_prompt_ts,
            audio_prompt_ts=audio_prompt_ts,
            video_prompt_cache_key=video_prompt_cache_key,
            audio_prompt_cache_key=audio_prompt_cache_key,
            video_memory=video_memory,
            audio_memory=audio_memory,
            color_memory=color_memory,
            layer_cache=layer_cache,
            pyramid_policy=pyramid_policy,
            layer_idx=i,
            video_frame_seqlen=frame_seqlen,
            audio_frame_seqlen=1,
            video_real_len=V_total_real,
            audio_real_len=A_total_real,
        )

        # ── Profiler: per-block ──
        if _PROFILER is not None:
            _bs = _PROFILER._new_event()
            _bs.record()
            # Mark end of pre-block phase on first block
            if i == 0:
                _prof_pre_e = _PROFILER._new_event()
                _prof_pre_e.record()
                _PROFILER.record_phase("pre_blocks", _prof_pre_s, _prof_pre_e)

        if use_ckpt:
            video_x, audio_x, new_cache = torch.utils.checkpoint.checkpoint(
                _run_block_with_kv_cache,
                tblock, kv_kwargs,
                use_reentrant=False,
            )
        else:
            video_x, audio_x, new_cache = tblock(_kv_cache_kwargs=kv_kwargs)

        if kv_cache.layerwise_cpu_offload:
            # Cache tensors are outputs, not part of the current block's loss.
            # Detaching them preserves the activation graph through video_x /
            # audio_x while preventing cache outputs from retaining it.
            new_cache = _move_layer_cache(new_cache, "cpu", detach=True)
        kv_cache.layers[i] = new_cache

        if _PROFILER is not None:
            _be = _PROFILER._new_event()
            _be.record()
            _PROFILER.record_block(i, _bs, _be)

    if kv_cache_only:
        if _PROFILER is not None:
            _prof_post_s = _PROFILER._new_event()
            _prof_post_s.record()
            _prof_post_e = _PROFILER._new_event()
            _prof_post_e.record()
            _PROFILER.record_phase("post_blocks_skipped_kv_only", _prof_post_s, _prof_post_e)
        return None, None, kv_cache

    # ================================================================
    # 11.5 [Ulysses SP] Gather full sequence before output projection
    # ================================================================
    # Output norm/scale/shift/proj_out + unpatchify operate on the FULL
    # sequence; embedded_ts is full-replicated on every rank already.
    if sp_size > 1:
        video_x = _gather_sequence(video_x, dim=1, sp_size=sp_size)
        audio_x = _gather_sequence(audio_x, dim=1, sp_size=sp_size)

    # ── SP-pad strip (NEW) ────────────────────────────────────────────────
    # Drop the trailing pad rows appended at the SP entry. Their K/V were
    # narrowed in ``attention_with_cache`` and their query outputs are not part
    # of the visible sequence.
    if video_pad_extra > 0:
        video_x = video_x[:, :V_total_real]
    if audio_pad_extra > 0:
        audio_x = audio_x[:, :A_total_real]

    # ================================================================
    # 12. Strip Sink Tokens before output projection
    # ================================================================
    audio_out_x = audio_x
    if num_sink > 0 and include_audio_sinks:
        audio_out_x = audio_out_x[:, num_sink:]

    # ── Profiler: post-blocks phase start ──
    if _PROFILER is not None:
        _prof_post_s = _PROFILER._new_event()
        _prof_post_s.record()

    # ================================================================
    # 13. Output Layer (timestep-conditioned projection)
    # ================================================================
    video_out = model._process_output(
        model.scale_shift_table, model.norm_out, model.proj_out,
        video_x, video_embedded_ts,
    )
    audio_out = model._process_output(
        model.audio_scale_shift_table, model.audio_norm_out, model.audio_proj_out,
        audio_out_x, audio_embedded_ts,
    )

    # ================================================================
    # 14. Unpatchify Video: [B, T, C] → [B, F, C, H, W]
    # ================================================================
    C_out = config.out_channels
    video_out = video_out.reshape(B, F_v, H_v, W_v, C_out).permute(0, 1, 4, 2, 3)

    # ── Profiler: post-blocks phase end ──
    if _PROFILER is not None:
        _prof_post_e = _PROFILER._new_event()
        _prof_post_e.record()
        _PROFILER.record_phase("post_blocks", _prof_post_s, _prof_post_e)

    return video_out, audio_out, kv_cache
