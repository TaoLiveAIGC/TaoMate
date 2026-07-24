"""
Pyramid Forcing head-aware KV cache policy (LTX-2 minimal port).

Reference: arXiv 2605.13111 (Pyramid Forcing). This module implements the *core*
head-aware cache reorganization: each (layer, head) carries a pre-computed
label that picks one of three frame-selection strategies. We keep the existing
KV cache append-only (full_k / full_v) and only re-select on each call, so the
cache itself never becomes ragged.

Scope (v1):
    - Self-attention only: ``video_self`` (attn1) and ``audio_self`` (audio_attn1).
    - Cross-modal (A2V / V2A) and text-cross attention go through the original path.
    - Padding + per-head bool mask (no ragged kernel). Padding K/V positions are
      gathered from index 0 then masked out via SDPA's native ``attn_mask``.
    - Audio prompt-side sink tokens (``num_audio_sink_tokens``) are kept verbatim
      and never enter the strategy selection.

Label semantics:
    label =  1  Anchor (sta+) → sink + middle-stride + recent
    label = -1  Wave   (osc)  → sink + cyclic buckets   + recent
    label =  2  Veil   (sta-) → sink + middle-merge     + recent
    other       → fallback to Anchor strategy.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ============================================================================
# Thread-local calibration capture hook
# ============================================================================
#
# Lives here (leaf module) so that ``kv_cache.attention_with_cache`` can poll
# it without introducing a circular import via ``head_calibration``.

_TLS = threading.local()


def get_active_capture_hook():
    """Return the currently-installed calibration capture (or ``None``)."""
    return getattr(_TLS, "capture", None)


def set_active_capture_hook(hook) -> None:
    """Install/uninstall a calibration capture hook (thread-local)."""
    _TLS.capture = hook


# ============================================================================
# Frame-index selection helpers (pure Python, called once per (layer, head))
# ============================================================================

def _strided_select(
    n_frames: int, sink: int, recent: int, interval: int, cap: int,
) -> List[int]:
    """Anchor strategy: sink + middle stride + recent."""
    sel = list(range(sink))
    middle_start = sink
    middle_end = max(n_frames - recent, sink)
    f = middle_start
    chosen_middle: List[int] = []
    while f < middle_end and len(chosen_middle) < cap:
        chosen_middle.append(f)
        f += max(interval, 1)
    sel.extend(chosen_middle)
    sel.extend(range(middle_end, n_frames))
    return sorted(set(sel))


def _cyclic_select(
    n_frames: int, sink: int, recent: int, period: int, bucket_cap: int,
) -> List[int]:
    """Wave strategy: sink + buckets of size ``bucket_cap`` every ``period`` + recent."""
    sel = list(range(sink))
    middle_start = sink
    middle_end = max(n_frames - recent, sink)
    f = middle_start
    while f < middle_end:
        bucket_end = min(f + max(bucket_cap, 1), middle_end)
        sel.extend(range(f, bucket_end))
        f += max(period, 1)
    sel.extend(range(middle_end, n_frames))
    return sorted(set(sel))


def _merge_select(
    n_frames: int, sink: int, recent: int, patch_size: int, cap: int,
) -> List[int]:
    """Veil strategy: sink + 1 frame per ``patch_size`` (capped) + recent.

    Note: real Pyramid Forcing mean-pools ``patch_size`` frames; v1 uses sub-
    sampling (first frame of each patch) for simplicity. Plan v2 will swap in
    mean-pooling once the wiring is verified.
    """
    sel = list(range(sink))
    middle_start = sink
    middle_end = max(n_frames - recent, sink)
    f = middle_start
    chosen: List[int] = []
    while f < middle_end and len(chosen) < cap:
        chosen.append(f)
        f += max(patch_size, 1)
    sel.extend(chosen)
    sel.extend(range(middle_end, n_frames))
    return sorted(set(sel))


_STRATEGY_BY_LABEL = {
    1: "anchor",
    -1: "wave",
    2: "veil",
}


# ============================================================================
# CSV path resolution
# ============================================================================

def resolve_label_csv_paths(
    ckpt_path: Optional[str],
    video_csv_arg: Optional[str] = None,
    audio_csv_arg: Optional[str] = None,
) -> Tuple[str, str]:
    """Resolve (video_csv, audio_csv) by priority: explicit arg > ckpt_dir default.

    Search order per modality:
        1. explicit ``*_csv_arg`` if non-empty and existing
        2. ``<ckpt_dir>/head_configs/{video_self,audio_self}_labels.csv``

    Raises:
        FileNotFoundError: when no candidate exists. Error message lists every
            tried path so the user can pick the right one.
    """

    def _build_candidates(arg: Optional[str], default_name: str) -> List[str]:
        cands: List[str] = []
        if arg:
            cands.append(arg)
        if ckpt_path:
            ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
            cands.append(os.path.join(ckpt_dir, "head_configs", default_name))
        return cands

    def _pick(cands: List[str], name: str) -> str:
        for p in cands:
            if p and os.path.isfile(p):
                return p
        raise FileNotFoundError(
            f"Pyramid KV: cannot locate {name}. Tried (in priority order): "
            + ", ".join(cands or ["<no candidate>"])
        )

    v_cands = _build_candidates(video_csv_arg, "video_self_labels.csv")
    a_cands = _build_candidates(audio_csv_arg, "audio_self_labels.csv")
    return _pick(v_cands, "video_self_labels.csv"), _pick(a_cands, "audio_self_labels.csv")


# ============================================================================
# PyramidKVPolicy
# ============================================================================

@dataclass
class _SelectionPlan:
    """Per (layer, modality, n_frames) cached selection plan."""
    gather_idx: Tensor   # [H, max_kept_tokens] long
    mask: Tensor         # [H, max_kept_tokens] bool


class PyramidKVPolicy:
    """Head-aware KV cache selector for video_self / audio_self attention.

    Usage::

        policy = PyramidKVPolicy(
            video_csv="…/video_self_labels.csv",
            audio_csv="…/audio_self_labels.csv",
            video_frame_seqlen=384,
            audio_frame_seqlen=1,
            audio_prompt_sink_tokens=num_audio_sink_tokens,
            sink_frames=3, recent_frames=4,
            stride_interval=6, stride_cap=4,
            cyclic_period=6, cyclic_bucket_cap=4,
            merge_patch_size=2, merge_cap=4,
        )

        sel_k, sel_v, attn_mask = policy.select_for_layer(
            layer_idx=L, modality="video",
            cached_k=full_k, cached_v=full_v,
            frame_seqlen=384,
        )
    """

    def __init__(
        self,
        video_csv: str,
        audio_csv: str,
        *,
        video_frame_seqlen: int,
        audio_frame_seqlen: int,
        audio_prompt_sink_tokens: int = 0,
        sink_frames: int = 3,
        recent_frames: int = 4,
        stride_interval: int = 6,
        stride_cap: int = 4,
        cyclic_period: int = 6,
        cyclic_bucket_cap: int = 4,
        merge_patch_size: int = 2,
        merge_cap: int = 4,
    ) -> None:
        self.video_labels = self._load_labels_csv(video_csv)
        self.audio_labels = self._load_labels_csv(audio_csv)
        self.video_frame_seqlen = int(video_frame_seqlen)
        self.audio_frame_seqlen = int(audio_frame_seqlen)
        self.audio_prompt_sink_tokens = int(audio_prompt_sink_tokens)
        self.sink_frames = int(sink_frames)
        self.recent_frames = int(recent_frames)
        self.stride_interval = int(stride_interval)
        self.stride_cap = int(stride_cap)
        self.cyclic_period = int(cyclic_period)
        self.cyclic_bucket_cap = int(cyclic_bucket_cap)
        self.merge_patch_size = int(merge_patch_size)
        self.merge_cap = int(merge_cap)

        # Cache compiled per-head plans by (layer, modality, n_frames)
        self._plan_cache: dict[Tuple[int, str, int], _SelectionPlan] = {}

    # -------------------------------------------------------------------- I/O

    @staticmethod
    def _load_labels_csv(path: str) -> np.ndarray:
        labels = np.loadtxt(path, delimiter=",", dtype=np.int64)
        if labels.ndim != 2:
            raise ValueError(
                f"Pyramid KV label CSV must be 2-D [num_layers, num_heads], "
                f"got shape {labels.shape} from {path}"
            )
        return labels

    # ----------------------------------------------------------- Plan builder

    def _build_plan(
        self, layer_idx: int, modality: str, n_frames: int, frame_seqlen: int,
        device: torch.device,
    ) -> _SelectionPlan:
        labels = self.video_labels if modality == "video" else self.audio_labels
        if layer_idx >= labels.shape[0]:
            raise IndexError(
                f"Pyramid KV: layer_idx={layer_idx} out of range for "
                f"{modality} labels shape {labels.shape}"
            )
        H = labels.shape[1]
        head_labels = labels[layer_idx]

        per_head_frames: List[List[int]] = []
        for h in range(H):
            lbl = int(head_labels[h])
            strat = _STRATEGY_BY_LABEL.get(lbl, "anchor")
            if strat == "wave":
                fr = _cyclic_select(
                    n_frames, self.sink_frames, self.recent_frames,
                    self.cyclic_period, self.cyclic_bucket_cap,
                )
            elif strat == "veil":
                fr = _merge_select(
                    n_frames, self.sink_frames, self.recent_frames,
                    self.merge_patch_size, self.merge_cap,
                )
            else:
                fr = _strided_select(
                    n_frames, self.sink_frames, self.recent_frames,
                    self.stride_interval, self.stride_cap,
                )
            per_head_frames.append(fr)

        max_kept_frames = max(len(fr) for fr in per_head_frames)
        max_kept_tokens = max_kept_frames * frame_seqlen

        gather_idx = torch.zeros(H, max_kept_tokens, dtype=torch.long, device=device)
        mask = torch.zeros(H, max_kept_tokens, dtype=torch.bool, device=device)
        for h, frames in enumerate(per_head_frames):
            token_ids: List[int] = []
            for f in frames:
                token_ids.extend(range(f * frame_seqlen, (f + 1) * frame_seqlen))
            n_tok = len(token_ids)
            if n_tok > 0:
                gather_idx[h, :n_tok] = torch.tensor(
                    token_ids, dtype=torch.long, device=device,
                )
                mask[h, :n_tok] = True
        return _SelectionPlan(gather_idx=gather_idx, mask=mask)

    def _get_plan(
        self, layer_idx: int, modality: str, n_frames: int, frame_seqlen: int,
        device: torch.device,
    ) -> _SelectionPlan:
        key = (layer_idx, modality, n_frames)
        plan = self._plan_cache.get(key)
        if plan is None or plan.gather_idx.device != device:
            plan = self._build_plan(layer_idx, modality, n_frames, frame_seqlen, device)
            self._plan_cache[key] = plan
        return plan

    # ----------------------------------------------------------- Public entry

    def select_for_layer(
        self,
        layer_idx: int,
        modality: str,
        cached_k: Tensor,
        cached_v: Tensor,
        frame_seqlen: int,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """Select per-head K/V slices and build attention mask.

        Args:
            layer_idx: 0-based transformer block index.
            modality: ``"video"`` or ``"audio"``.
            cached_k: Full key cache ``[B, L_kv, H, D_h]``.
            cached_v: Full value cache, same shape.
            frame_seqlen: Tokens per frame for this modality (video=384, audio=1).

        Returns:
            ``(sel_k, sel_v, attn_mask)`` where:
                - ``sel_k/sel_v``: ``[B, L_sel, H, D_h]`` (L_sel = max kept tokens
                  + audio prompt-sink prefix when applicable).
                - ``attn_mask``: ``[1, H, 1, L_sel]`` bool (True = keep). When
                  selection has no effect (short context), returns ``None`` and
                  the original cached_k/cached_v.
        """
        if modality not in ("video", "audio"):
            raise ValueError(f"modality must be 'video' or 'audio', got {modality!r}")

        B, L_kv, H, D_h = cached_k.shape
        device = cached_k.device

        # Carve out audio prompt-side sink (these tokens never participate)
        if modality == "audio" and self.audio_prompt_sink_tokens > 0:
            n_pre = min(self.audio_prompt_sink_tokens, L_kv)
            pre_k = cached_k[:, :n_pre]
            pre_v = cached_v[:, :n_pre]
            body_k = cached_k[:, n_pre:]
            body_v = cached_v[:, n_pre:]
        else:
            n_pre = 0
            pre_k = pre_v = None
            body_k, body_v = cached_k, cached_v

        L_body = body_k.shape[1]
        if L_body == 0:
            return cached_k, cached_v, None

        if L_body % frame_seqlen != 0:
            raise AssertionError(
                f"Pyramid KV: body L_body={L_body} not divisible by "
                f"frame_seqlen={frame_seqlen} (modality={modality}, layer={layer_idx})"
            )
        n_frames = L_body // frame_seqlen

        # Short context → keep full cache, no mask
        if n_frames <= self.sink_frames + self.recent_frames:
            return cached_k, cached_v, None

        plan = self._get_plan(layer_idx, modality, n_frames, frame_seqlen, device)
        gather_idx = plan.gather_idx              # [H, max_kept_tokens]
        body_mask = plan.mask                     # [H, max_kept_tokens]
        max_kept = gather_idx.shape[1]

        # Per-head gather: body_k [B, L_body, H, D_h] → [B, max_kept, H, D_h]
        body_k_perm = body_k.permute(0, 2, 1, 3).contiguous()  # [B, H, L_body, D_h]
        body_v_perm = body_v.permute(0, 2, 1, 3).contiguous()
        idx = gather_idx[None, :, :, None].expand(B, H, max_kept, D_h)
        sel_k = body_k_perm.gather(2, idx).permute(0, 2, 1, 3)
        sel_v = body_v_perm.gather(2, idx).permute(0, 2, 1, 3)

        # Concat audio prompt-sink prefix (mask = all True for those positions)
        if pre_k is not None:
            sel_k = torch.cat([pre_k, sel_k], dim=1)
            sel_v = torch.cat([pre_v, sel_v], dim=1)
            pre_mask = torch.ones(H, n_pre, dtype=torch.bool, device=device)
            full_mask = torch.cat([pre_mask, body_mask], dim=1)  # [H, n_pre+max_kept]
        else:
            full_mask = body_mask

        # Build [1, H, 1, L_sel] mask → SDPA broadcasts over B and L_q
        attn_mask = full_mask[None, :, None, :]
        return sel_k, sel_v, attn_mask
