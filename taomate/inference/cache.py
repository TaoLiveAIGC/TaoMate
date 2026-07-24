from typing import Dict, List, Tuple

import torch

from ltx_causal.transformer.kv_cache import KVCache, LayerKVCache

def cache_segments(cache: KVCache) -> List[Dict[str, int]]:
    return [dict(item) for item in cache.segments]

def attach_segments(cache: KVCache, segments: List[Dict[str, int]]) -> KVCache:
    cache.segments = [dict(item) for item in segments]
    return cache

def _segment_offsets(
    segments: List[Dict[str, int]],
) -> List[Tuple[int, int, int, int]]:
    video_offset = 0
    audio_offset = 0
    offsets = []
    for segment in segments:
        video_end = video_offset + int(segment["video_tokens"])
        audio_end = audio_offset + int(segment["audio_tokens"])
        offsets.append((video_offset, video_end, audio_offset, audio_end))
        video_offset = video_end
        audio_offset = audio_end
    return offsets

def _slice_tensor(
    tensor: torch.Tensor, spans: List[Tuple[int, int]]
) -> torch.Tensor:
    pieces = [tensor[:, start:end] for start, end in spans if end > start]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=1)

def _slice_cache(cache: KVCache, indices: List[int]) -> KVCache:
    segments = cache_segments(cache)
    offsets = _segment_offsets(segments)
    video_spans = [(offsets[index][0], offsets[index][1]) for index in indices]
    audio_spans = [(offsets[index][2], offsets[index][3]) for index in indices]
    layers = []
    for layer in cache.layers:
        layers.append(
            LayerKVCache(
                video_self_k=_slice_tensor(layer.video_self_k, video_spans),
                video_self_v=_slice_tensor(layer.video_self_v, video_spans),
                audio_self_k=_slice_tensor(layer.audio_self_k, audio_spans),
                audio_self_v=_slice_tensor(layer.audio_self_v, audio_spans),
                a2v_k=_slice_tensor(layer.a2v_k, audio_spans),
                a2v_v=_slice_tensor(layer.a2v_v, audio_spans),
                v2a_k=_slice_tensor(layer.v2a_k, video_spans),
                v2a_v=_slice_tensor(layer.v2a_v, video_spans),
            )
        )
    return attach_segments(
        KVCache(layers=layers),
        [segments[index] for index in indices],
    )

def retain_streaming_context(cache: KVCache) -> KVCache:
    segments = cache_segments(cache)
    if not segments:
        raise RuntimeError("Cached inference did not attach block metadata.")
    sink = next(
        index for index, item in enumerate(segments) if int(item["block_idx"]) == 0
    )
    non_sink = [index for index in range(len(segments)) if index != sink]
    keep = sorted([sink, *non_sink[-2:]])
    return cache if keep == list(range(len(segments))) else _slice_cache(cache, keep)
