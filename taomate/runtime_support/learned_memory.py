"""Utilities for learned long-memory conditioning.

The learned-memory experiments pass compact video/audio history tokens through
``conditional_dict``.  The transformer owns the learnable adapters; this module
only builds the detached EMA memory state from generated latents so training and
inference can share the same update rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize_spatial(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def _normalize_temporal(x: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def _channel_stats(
    latent: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x = latent.detach().to(torch.float32)
    mean = x.mean(dim=(1, 3, 4))
    std = x.std(dim=(1, 3, 4), unbiased=False).clamp_min(eps)
    return mean, std


def _apply_channel_renorm(
    latent: torch.Tensor,
    ref_stats: Tuple[torch.Tensor, torch.Tensor],
    alpha: float,
    eps: float,
) -> torch.Tensor:
    alpha = _clamp01(alpha)
    if alpha <= 0.0:
        return latent
    cur_mean, cur_std = _channel_stats(latent, eps)
    ref_mean, ref_std = ref_stats
    ref_mean = ref_mean.to(device=latent.device, dtype=torch.float32)
    ref_std = ref_std.to(device=latent.device, dtype=torch.float32).clamp_min(eps)
    y = (
        (latent.to(torch.float32) - cur_mean[:, None, :, None, None])
        / cur_std[:, None, :, None, None].clamp_min(eps)
        * ref_std[:, None, :, None, None]
        + ref_mean[:, None, :, None, None]
    )
    out = latent.to(torch.float32) * (1.0 - alpha) + y * alpha
    return out.to(dtype=latent.dtype)


def _lowfreq_proto(latent: torch.Tensor, grid: int) -> torch.Tensor:
    grid = max(1, int(grid))
    x = latent.detach().to(torch.float32).mean(dim=1)
    return F.adaptive_avg_pool2d(x, output_size=(grid, grid))


def _apply_lowfreq_proto(
    latent: torch.Tensor,
    proto: Optional[torch.Tensor],
    alpha: float,
    max_correction: float,
    eps: float,
) -> torch.Tensor:
    alpha = _clamp01(alpha)
    if alpha <= 0.0 or proto is None:
        return latent
    cur_proto = _lowfreq_proto(latent, int(proto.shape[-1]))
    target = proto.to(device=latent.device, dtype=torch.float32)
    delta = target - cur_proto
    if max_correction > 0.0:
        _, cur_std = _channel_stats(latent, eps)
        clamp = (
            cur_std.to(device=latent.device, dtype=torch.float32)
            .mean(dim=0, keepdim=True)
            .view(1, -1, 1, 1)
            * float(max_correction)
        )
        delta = torch.max(torch.min(delta, clamp), -clamp)
    delta = F.interpolate(
        delta,
        size=latent.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    out = latent.to(torch.float32) + alpha * delta[:, None]
    return out.to(dtype=latent.dtype)


def _stats_drift_score(
    stats: Tuple[torch.Tensor, torch.Tensor],
    anchor_stats: Tuple[torch.Tensor, torch.Tensor],
    eps: float,
) -> float:
    cur_m, cur_s = stats
    ref_m, ref_s = anchor_stats
    ref_m = ref_m.to(cur_m.device)
    ref_s = ref_s.to(cur_s.device).clamp_min(eps)
    mean_score = ((cur_m.float() - ref_m.float()).abs() / ref_s.float()).mean()
    std_score = torch.log(cur_s.float().clamp_min(eps) / ref_s.float()).abs().mean()
    return float((mean_score + 0.1 * std_score).detach().cpu())


def _pooled_cosine_score(
    current: torch.Tensor,
    anchor: torch.Tensor,
    eps: float,
) -> float:
    """Cosine similarity over aligned memory tokens.

    Video memory is spatially normalized before it reaches this function, so
    averaging tokens first produces an almost-zero vector and makes the old
    cosine score hover around zero even for related clips.  Flattening the
    aligned token grid preserves the spatial/channel signal used by the memory
    branch and gives the drift gate a meaningful dynamic range.
    """
    cur = current.detach().to(torch.float32)
    anc = anchor.detach().to(device=cur.device, dtype=torch.float32)
    if cur.ndim != 3 or anc.ndim != 3:
        return 1.0
    if cur.shape[1] != anc.shape[1]:
        n = min(int(cur.shape[1]), int(anc.shape[1]))
        if n <= 0:
            return 1.0
        cur = cur[:, :n]
        anc = anc[:, :n]
    cur_vec = F.normalize(cur.flatten(start_dim=1), dim=-1, eps=eps)
    anc_vec = F.normalize(anc.flatten(start_dim=1), dim=-1, eps=eps)
    score = (cur_vec * anc_vec).sum(dim=-1).mean()
    return float(score.detach().cpu())


def _smooth_gate_from_score(
    score: float,
    *,
    threshold: float,
    temperature: float,
    min_gate: float,
) -> float:
    min_gate = _clamp01(min_gate)
    if temperature <= 0.0:
        return 1.0 if score >= threshold else min_gate
    x = (float(score) - float(threshold)) / max(float(temperature), 1e-6)
    if x >= 50.0:
        smooth = 1.0
    elif x <= -50.0:
        smooth = 0.0
    else:
        smooth = 1.0 / (1.0 + math.exp(-x))
    return min_gate + (1.0 - min_gate) * smooth


def apply_color_memory_snapshot(
    video_latent: Optional[torch.Tensor],
    snapshot: Optional[Dict[str, Any]],
) -> Optional[torch.Tensor]:
    if video_latent is None or not snapshot:
        return video_latent
    out = video_latent
    eps = float(snapshot.get("eps", 1e-6))
    alpha = float(snapshot.get("color_alpha", 0.0))
    proto_alpha = float(snapshot.get("color_proto_alpha", 0.0))
    if alpha > 0.0 and snapshot.get("color_stats") is not None:
        out = _apply_channel_renorm(out, snapshot["color_stats"], alpha, eps)
    if proto_alpha > 0.0:
        out = _apply_lowfreq_proto(
            out,
            snapshot.get("color_proto"),
            proto_alpha,
            float(snapshot.get("color_max_correction", 0.5)),
            eps,
        )
    return out


def summarize_video_memory(
    video_latent: Optional[torch.Tensor],
    *,
    downsample: int = 4,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Return video memory tokens ``[B, N_v, 128]`` from ``[B, F, 128, H, W]``."""
    if video_latent is None or video_latent.numel() == 0:
        return None
    x = video_latent.detach().to(torch.float32)
    if x.ndim != 5:
        raise ValueError(f"video_latent must be [B,F,C,H,W], got {tuple(x.shape)}")
    x = x.mean(dim=1)
    x = _normalize_spatial(x, eps)
    h = max(1, int(x.shape[-2]) // max(1, int(downsample)))
    w = max(1, int(x.shape[-1]) // max(1, int(downsample)))
    x = F.adaptive_avg_pool2d(x, (h, w))
    return x.permute(0, 2, 3, 1).reshape(x.shape[0], h * w, x.shape[1]).contiguous()


def summarize_audio_memory(
    audio_latent: Optional[torch.Tensor],
    *,
    num_tokens: int = 64,
    eps: float = 1e-6,
) -> Optional[torch.Tensor]:
    """Return audio memory tokens ``[B, N_a, 128]`` from ``[B, F_a, 128]``."""
    if audio_latent is None or audio_latent.numel() == 0:
        return None
    x = audio_latent.detach().to(torch.float32)
    if x.ndim != 3:
        raise ValueError(f"audio_latent must be [B,F,C], got {tuple(x.shape)}")
    x = x.transpose(1, 2)
    x = _normalize_temporal(x, eps)
    n = max(1, int(num_tokens))
    x = F.adaptive_avg_pool1d(x, n)
    return x.transpose(1, 2).contiguous()


def _ema_update(
    prev: Optional[torch.Tensor],
    anchor: Optional[torch.Tensor],
    current: torch.Tensor,
    *,
    beta: float,
    tether: float,
) -> torch.Tensor:
    beta = _clamp01(beta)
    tether = _clamp01(tether)
    cur = current.detach().contiguous()
    if prev is None:
        return cur.clone()
    ema = prev.detach().to(device=cur.device, dtype=cur.dtype) * (1.0 - beta) + cur * beta
    if anchor is not None and tether > 0.0:
        anc = anchor.detach().to(device=cur.device, dtype=cur.dtype)
        ema = ema * (1.0 - tether) + anc * tether
    return ema.detach().contiguous()


@dataclass
class LearnedMemoryState:
    enabled: bool = False
    video_downsample: int = 4
    audio_tokens: int = 64
    video_beta: float = 0.15
    audio_beta: float = 0.10
    video_anchor_tether: float = 0.20
    audio_anchor_tether: float = 0.10
    identity_anchor_enabled: bool = False
    identity_anchor_scale: float = 1.0
    drift_gate_enabled: bool = False
    drift_gate_threshold: float = 0.05
    drift_gate_temperature: float = 0.10
    drift_gate_min: float = 0.10
    drift_gate_apply_to_color: bool = True
    color_enabled: bool = False
    color_alpha: float = 0.0
    color_proto_alpha: float = 0.0
    color_update_beta: float = 0.05
    color_anchor_tether: float = 0.40
    color_proto_grid: int = 4
    color_drift_threshold: float = 2.5
    color_max_correction: float = 0.5
    color_film_enabled: bool = False
    reference_anchor_enabled: bool = False
    eps: float = 1e-6

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        self.video_downsample = max(1, int(self.video_downsample))
        self.audio_tokens = max(1, int(self.audio_tokens))
        self.video_beta = _clamp01(self.video_beta)
        self.audio_beta = _clamp01(self.audio_beta)
        self.video_anchor_tether = _clamp01(self.video_anchor_tether)
        self.audio_anchor_tether = _clamp01(self.audio_anchor_tether)
        self.identity_anchor_enabled = bool(self.identity_anchor_enabled)
        self.identity_anchor_scale = max(0.0, float(self.identity_anchor_scale))
        self.drift_gate_enabled = bool(self.drift_gate_enabled)
        self.drift_gate_threshold = float(self.drift_gate_threshold)
        self.drift_gate_temperature = max(0.0, float(self.drift_gate_temperature))
        self.drift_gate_min = _clamp01(self.drift_gate_min)
        self.drift_gate_apply_to_color = bool(self.drift_gate_apply_to_color)
        self.color_enabled = bool(self.color_enabled or self.color_film_enabled)
        self.color_alpha = _clamp01(self.color_alpha)
        self.color_proto_alpha = _clamp01(self.color_proto_alpha)
        self.color_update_beta = _clamp01(self.color_update_beta)
        self.color_anchor_tether = _clamp01(self.color_anchor_tether)
        self.color_proto_grid = max(1, int(self.color_proto_grid))
        self.color_max_correction = max(0.0, float(self.color_max_correction))
        self.color_film_enabled = bool(self.color_film_enabled)
        self.reference_anchor_enabled = bool(self.reference_anchor_enabled)
        self.video: Optional[torch.Tensor] = None
        self.audio: Optional[torch.Tensor] = None
        self.video_anchor: Optional[torch.Tensor] = None
        self.audio_anchor: Optional[torch.Tensor] = None
        self.color_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.color_anchor_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.color_proto: Optional[torch.Tensor] = None
        self.color_anchor_proto: Optional[torch.Tensor] = None
        self.color_last_drift: Optional[float] = None
        self.color_last_updated: bool = False
        self.video_last_anchor_cosine: Optional[float] = None
        self.video_last_update_gate: Optional[float] = None
        self.reference_initialized: bool = False

    def reset(self) -> None:
        self.video = None
        self.audio = None
        self.video_anchor = None
        self.audio_anchor = None
        self.color_stats = None
        self.color_anchor_stats = None
        self.color_proto = None
        self.color_anchor_proto = None
        self.color_last_drift = None
        self.color_last_updated = False
        self.video_last_anchor_cosine = None
        self.video_last_update_gate = None
        self.reference_initialized = False

    def apply_color_memory(
        self,
        video_latent: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        return apply_color_memory_snapshot(video_latent, self.color_snapshot())

    def color_snapshot(self) -> Optional[Dict[str, Any]]:
        if (
            not self.color_enabled
            or self.color_stats is None
            or (self.color_alpha <= 0.0 and self.color_proto_alpha <= 0.0)
        ):
            return None
        return {
            "color_stats": tuple(t.detach() for t in self.color_stats),
            "color_proto": (
                self.color_proto.detach() if self.color_proto is not None else None
            ),
            "color_alpha": self.color_alpha,
            "color_proto_alpha": self.color_proto_alpha,
            "color_max_correction": self.color_max_correction,
            "eps": self.eps,
        }

    def set_reference(
        self,
        video_latent: Optional[torch.Tensor],
        audio_latent: Optional[torch.Tensor] = None,
    ) -> None:
        """Freeze the first visible block as the long-memory reference anchor.

        ``update()`` already falls back to the first seen latent when this method
        is never called.  The explicit hook is used by train/inference code to
        guarantee that the anchor is block 0 before later generated prefix blocks
        can shift the EMA state.
        """
        if not self.enabled or not self.reference_anchor_enabled:
            return
        if self.reference_initialized:
            return

        v = summarize_video_memory(
            video_latent, downsample=self.video_downsample, eps=self.eps
        )
        if v is not None:
            self.video_anchor = v.detach().contiguous().clone()
            if self.video is None:
                self.video = self.video_anchor.clone()
            self.video_last_anchor_cosine = 1.0
            self.video_last_update_gate = 1.0

        a = summarize_audio_memory(
            audio_latent, num_tokens=self.audio_tokens, eps=self.eps
        )
        if a is not None:
            self.audio_anchor = a.detach().contiguous().clone()
            if self.audio is None:
                self.audio = self.audio_anchor.clone()

        if self.color_enabled and video_latent is not None and video_latent.numel() > 0:
            stats = tuple(
                t.detach().contiguous().clone()
                for t in _channel_stats(video_latent, self.eps)
            )
            proto = (
                _lowfreq_proto(video_latent, self.color_proto_grid)
                .detach()
                .contiguous()
                .clone()
            )
            self.color_anchor_stats = stats
            self.color_anchor_proto = proto
            if self.color_stats is None:
                self.color_stats = tuple(t.clone() for t in stats)
            if self.color_proto is None:
                self.color_proto = proto.clone()
            self.color_last_drift = 0.0
            self.color_last_updated = True

        if v is not None or a is not None:
            self.reference_initialized = True

    def _video_update_gate(self, current: torch.Tensor) -> float:
        if not self.drift_gate_enabled or self.video_anchor is None:
            self.video_last_anchor_cosine = None
            self.video_last_update_gate = 1.0
            return 1.0
        score = _pooled_cosine_score(current, self.video_anchor, self.eps)
        gate = _smooth_gate_from_score(
            score,
            threshold=self.drift_gate_threshold,
            temperature=self.drift_gate_temperature,
            min_gate=self.drift_gate_min,
        )
        self.video_last_anchor_cosine = score
        self.video_last_update_gate = gate
        return gate

    def _update_color_memory(
        self,
        video_latent: Optional[torch.Tensor],
        *,
        update_scale: float = 1.0,
    ) -> None:
        if not self.color_enabled or video_latent is None or video_latent.numel() == 0:
            return
        update_scale = _clamp01(update_scale)
        stats = tuple(
            t.detach().contiguous().clone()
            for t in _channel_stats(video_latent, self.eps)
        )
        proto = (
            _lowfreq_proto(video_latent, self.color_proto_grid)
            .detach()
            .contiguous()
            .clone()
        )
        if self.color_anchor_stats is None:
            self.color_anchor_stats = stats
            self.color_stats = tuple(t.clone() for t in stats)
            self.color_anchor_proto = proto
            self.color_proto = proto.clone()
            self.color_last_drift = 0.0
            self.color_last_updated = True
            return
        drift = _stats_drift_score(stats, self.color_anchor_stats, self.eps)
        self.color_last_drift = drift
        if self.color_drift_threshold > 0.0 and drift > self.color_drift_threshold:
            self.color_last_updated = False
            return
        beta = self.color_update_beta * update_scale
        if beta <= 0.0:
            self.color_last_updated = False
            return
        tether = self.color_anchor_tether
        prev_stats = self.color_stats or stats
        mixed_stats = tuple(
            (1.0 - beta) * old.to(new.device) + beta * new
            for old, new in zip(prev_stats, stats)
        )
        if tether > 0.0 and self.color_anchor_stats is not None:
            mixed_stats = tuple(
                (1.0 - tether) * cur + tether * anchor.to(cur.device)
                for cur, anchor in zip(mixed_stats, self.color_anchor_stats)
            )
        self.color_stats = tuple(t.detach().contiguous() for t in mixed_stats)

        prev_proto = self.color_proto if self.color_proto is not None else proto
        mixed_proto = (1.0 - beta) * prev_proto.to(proto.device) + beta * proto
        if tether > 0.0 and self.color_anchor_proto is not None:
            mixed_proto = (
                (1.0 - tether) * mixed_proto
                + tether * self.color_anchor_proto.to(proto.device)
            )
        self.color_proto = mixed_proto.detach().contiguous()
        self.color_last_updated = True

    def _color_condition_vector(self) -> Optional[torch.Tensor]:
        if not self.color_film_enabled or self.color_stats is None:
            return None
        mean, std = self.color_stats
        if self.color_anchor_stats is None:
            anchor_mean, anchor_std = mean, std
        else:
            anchor_mean, anchor_std = self.color_anchor_stats
        return torch.cat(
            [
                mean.detach(),
                std.detach().clamp_min(self.eps).log(),
                anchor_mean.detach().to(mean.device),
                anchor_std.detach().to(std.device).clamp_min(self.eps).log(),
            ],
            dim=-1,
        )

    def update(
        self,
        video_latent: Optional[torch.Tensor],
        audio_latent: Optional[torch.Tensor] = None,
    ) -> None:
        if not self.enabled:
            return
        v = summarize_video_memory(
            video_latent, downsample=self.video_downsample, eps=self.eps
        )
        video_update_gate = 1.0
        if v is not None:
            if self.video_anchor is None:
                self.video_anchor = v.detach().contiguous().clone()
                self.video_last_anchor_cosine = 1.0
                self.video_last_update_gate = 1.0
            else:
                video_update_gate = self._video_update_gate(v)
            self.video = _ema_update(
                self.video,
                self.video_anchor,
                v,
                beta=self.video_beta * video_update_gate,
                tether=self.video_anchor_tether,
            )
        a = summarize_audio_memory(
            audio_latent, num_tokens=self.audio_tokens, eps=self.eps
        )
        if a is not None:
            if self.audio_anchor is None:
                self.audio_anchor = a.detach().contiguous().clone()
            self.audio = _ema_update(
                self.audio,
                self.audio_anchor,
                a,
                beta=self.audio_beta,
                tether=self.audio_anchor_tether,
            )
        color_update_scale = (
            video_update_gate if self.drift_gate_apply_to_color else 1.0
        )
        self._update_color_memory(video_latent, update_scale=color_update_scale)

    def _video_condition_memory(self) -> Optional[torch.Tensor]:
        memories = []
        if self.identity_anchor_enabled and self.video_anchor is not None:
            anchor = self.video_anchor.detach()
            if self.identity_anchor_scale != 1.0:
                anchor = anchor * self.identity_anchor_scale
            memories.append(anchor)
        if self.video is not None:
            memories.append(self.video)
        if not memories:
            return None
        if len(memories) == 1:
            return memories[0]
        return torch.cat(memories, dim=1)

    def with_conditional_memory(
        self,
        conditional_dict: Dict[str, Any],
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        video_memory = self._video_condition_memory()
        if not self.enabled or (video_memory is None and self.audio is None):
            return conditional_dict
        out = dict(conditional_dict)
        if video_memory is not None:
            v = video_memory
            if device is not None or dtype is not None:
                v = v.to(device=device or v.device, dtype=dtype or v.dtype)
            out["learned_memory_video"] = v
        if self.audio is not None:
            a = self.audio
            if device is not None or dtype is not None:
                a = a.to(device=device or a.device, dtype=dtype or a.dtype)
            out["learned_memory_audio"] = a
        color = self._color_condition_vector()
        if color is not None:
            if device is not None or dtype is not None:
                color = color.to(device=device or color.device, dtype=dtype or color.dtype)
            out["learned_memory_color"] = color
        return out
