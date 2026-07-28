#!/usr/bin/env python3
"""Model loading and streaming KV-cache inference for TaoMate."""

from dataclasses import fields as dataclass_fields
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchaudio

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.types import Audio
from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
from ltx_causal.rope.causal_rope import CausalRopeType
from ltx_causal.attention.mask_builder import compute_av_blocks, compute_aligned_audio_frames
from taomate.runtime_support.learned_memory import LearnedMemoryState

# Runtime constants
VIDEO_FPS = 24


def _channel_stats(
    latent: torch.Tensor,
    modality: str = "video",
) -> Tuple[torch.Tensor, torch.Tensor]:
    dims = (1, 3, 4) if modality == "video" else (1,)
    mean = latent.mean(dim=dims, keepdim=True)
    std = latent.std(dim=dims, keepdim=True).clamp_min(1e-6)
    return mean, std


def _apply_renorm(
    latent: torch.Tensor,
    anchor_stats: Optional[Tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
    modality: str = "video",
) -> torch.Tensor:
    if latent is None or latent.numel() == 0 or anchor_stats is None:
        return latent
    current_mean, current_std = _channel_stats(latent, modality)
    anchor_mean, anchor_std = anchor_stats
    target_mean = (1.0 - alpha) * current_mean + alpha * anchor_mean.to(current_mean.dtype)
    target_std = (1.0 - alpha) * current_std + alpha * anchor_std.to(current_std.dtype)
    return (((latent - current_mean) / current_std) * target_std + target_mean).to(latent.dtype)


def _format_block_stats(
    block_idx: int,
    video: Optional[torch.Tensor],
    audio: Optional[torch.Tensor],
    suffix: str = "",
) -> str:
    parts = [f"[PrefixRenorm] blk={block_idx:02d}"]
    if suffix:
        parts.append(suffix)
    if video is not None and video.numel() > 0:
        parts.append(
            f"v(mean={video.float().mean().item():+.4f},std={video.float().std().item():.4f})"
        )
    if audio is not None and audio.numel() > 0:
        parts.append(
            f"a(mean={audio.float().mean().item():+.4f},std={audio.float().std().item():.4f})"
        )
    return " ".join(parts)


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


def compute_denoising_sigmas(
    denoising_step_list: List[int],
    num_inference_steps: int = 40,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Compute sigma sequence EXACTLY matching training (dmd.py L209-217).

    Uses LTX2Scheduler's shifted+stretched sigmoid schedule, then finds the
    closest sigma for each timestep in denoising_step_list via argmin.
    """
    full_sigmas = LTX2Scheduler().execute(steps=num_inference_steps)
    sigmas = []
    for t in denoising_step_list:
        target_sigma = t / 1000.0
        idx = (full_sigmas - target_sigma).abs().argmin().item()
        sigmas.append(full_sigmas[idx])
    return torch.stack(sigmas).to(device)


def add_noise(
    original: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Flow matching noise: x_t = (1 - sigma) * x_0 + sigma * eps  (dmd.py L381-408)."""
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


def _remap_state_dict_keys(state_dict: dict) -> dict:
    """Remap original checkpoint keys to wrapper format (mirrors dmd.py L465-510)."""
    non_transformer_prefixes = (
        "vae.", "audio_vae.", "vocoder.",
        "model.vae.", "model.audio_vae.", "model.vocoder.",
    )
    remapped_connector_prefixes = (
        "model.audio_embeddings_connector.",
        "model.video_embeddings_connector.",
    )

    has_diffusion_model = any(
        k.startswith("model.diffusion_model.") for k in state_dict
    )
    if has_diffusion_model:
        remapped = {}
        for k, v in state_dict.items():
            if not k.startswith("model.diffusion_model."):
                continue
            new_key = "model." + k[len("model.diffusion_model."):]
            if any(new_key.startswith(p) for p in remapped_connector_prefixes):
                continue
            remapped[new_key] = v
        return remapped

    first_key = next(iter(state_dict))
    if first_key.startswith("model.velocity_model."):
        return {
            "model." + k[len("model.velocity_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.velocity_model.")
        }
    if first_key.startswith("model."):
        return {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in non_transformer_prefixes)
        }
    return {
        "model." + k: v
        for k, v in state_dict.items()
        if not any(k.startswith(p) for p in non_transformer_prefixes)
    }


# Model loading
def load_model_generator(
    model_ckpt_path: str,
    original_ckpt_path: str,
    video_height: int = 512,
    video_width: int = 768,
    num_frame_per_block: int = 3,
    num_frame_per_block_first: int = 0,
    num_audio_sink_tokens: int = 16,
    use_flex_attention: bool = True,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    causal_rope_type: str = "split",
    rope_extrapolation: str = "off",
    rope_train_max_seconds: float = 8.0,
    learned_memory_enabled: bool = False,
    learned_memory_mode: str = "cross_attn_adapter",
    learned_memory_layer_interval: int = 4,
    learned_memory_video_dim: int = 512,
    learned_memory_audio_dim: int = 256,
    learned_memory_heads: int = 8,
    learned_memory_color_film: bool = False,
    learned_memory_color_film_hidden_dim: int = 256,
) -> CausalLTX2DiffusionWrapper:
    """
    Load TaoMate causal generator.

    TaoMate checkpoint structure: {"generator_ema": state_dict}
    Generator keys are in ``model.*`` format (CausalLTX2DiffusionWrapper).

    Missing from TaoMate checkpoint (loaded from original_ckpt_path):
      - Text encoder, Video VAE, Audio VAE, Vocoder

    Args:
    Returns:
        generator_wrapper
    """
    import gc

    print(f"[Load] TaoMate checkpoint: {model_ckpt_path}")
    print("[Load] Using memory-mapped EMA weights")
    ckpt = torch.load(
        model_ckpt_path,
        map_location="cpu",
        mmap=True,
        weights_only=True,
    )
    if not isinstance(ckpt, dict) or "generator_ema" not in ckpt:
        raise ValueError("Expected a TaoMate EMA checkpoint.")
    gen_sd = ckpt["generator_ema"]

    del ckpt
    gc.collect()

    # ── Analyze checkpoint to auto-detect model features ──
    has_audio_sink = any("audio_sink_tokens" in k for k in gen_sd)
    has_gated_attn = any("to_gate_logits" in k for k in gen_sd)
    has_learned_memory = any(
        "learned_memory" in k or "memory_attn" in k for k in gen_sd
    )
    has_color_film = any("color_film" in k for k in gen_sd)
    if learned_memory_color_film and not has_color_film:
        raise ValueError(
            "--learned_memory_color_film was requested, but this checkpoint "
            "does not contain FiLM weights. Train the FiLM variant first; "
            "external meta-device inference cannot safely materialize missing "
            "22B-side parameters."
        )
    if learned_memory_enabled and not has_learned_memory:
        raise ValueError(
            "--learned_memory was requested, but this checkpoint does not "
            "contain learned-memory weights. Train D1/D2 first, or disable "
            "the inference learned-memory flag."
        )

    adaln_key = "model.adaln_single.linear.weight"
    cross_attention_adaln = False
    if adaln_key in gen_sd:
        D = gen_sd[adaln_key].shape[1]
        coeff = gen_sd[adaln_key].shape[0] // D
        cross_attention_adaln = (coeff == 9)

    caption_proj_before = not any(
        k.startswith("model.caption_projection.") for k in gen_sd
    )

    # Auto-detect condition_sink_on_text: the checkpoint contains
    # sink_text_condition.* keys if the model was trained with text-conditioned
    # audio sink tokens.  Without this, the trained MLP weights are silently
    # dropped and sink tokens revert to static — a train-test mismatch.
    has_condition_sink = any("sink_text_condition" in k for k in gen_sd)

    print(f"[Load] Auto-detected config:")
    print(f"  audio_sink_tokens={has_audio_sink}  gated_attention={has_gated_attn}")
    print(f"  cross_attention_adaln={cross_attention_adaln}  "
          f"caption_proj_before={caption_proj_before}")
    print(f"  condition_sink_on_text={has_condition_sink}")
    if learned_memory_enabled or has_learned_memory:
        print(
            f"  learned_memory=ON mode={learned_memory_mode} "
            f"interval={learned_memory_layer_interval}"
        )
    if learned_memory_color_film or has_color_film:
        print(
            f"  learned_memory_color_film=ON hidden={learned_memory_color_film_hidden_dim}"
        )
    print(f"  total keys: {len(gen_sd)}")

    # ── Fill missing keys from original checkpoint BEFORE building model ──
    # This avoids holding both the model and gen_sd simultaneously.
    if original_ckpt_path:
        # Quick scan: check if gen_sd has all expected keys by building a
        # temporary model config and listing expected keys.  For TaoMate we
        # typically don't miss keys, but handle gracefully.
        pass  # Filling happens after load_state_dict below.

    # ── Build model architecture (meta device → zero memory allocation) ──
    # The 22B model in bf16 is ~44GB. Building on meta avoids allocating RAM
    # that would double peak to 88GB (gen_sd 44GB + model 44GB).
    # With assign=True in load_state_dict, gen_sd tensors become model params
    # directly (zero-copy), keeping peak at ~44GB.
    try:
        resolved_rope_type = CausalRopeType(str(causal_rope_type).lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in CausalRopeType)
        raise ValueError(
            f"Unsupported causal_rope_type={causal_rope_type!r}; choose one of: {valid}"
        ) from exc
    print(f"  causal_rope_type={resolved_rope_type.value}")

    config = CausalLTXModelConfig(
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
        num_audio_sink_tokens=num_audio_sink_tokens,
        apply_gated_attention=has_gated_attn,
        cross_attention_adaln=cross_attention_adaln,
        caption_proj_before_connector=caption_proj_before,
        condition_sink_on_text=has_condition_sink,
        rope_type=resolved_rope_type,
        rope_extrapolation=rope_extrapolation,
        rope_train_max_seconds=rope_train_max_seconds,
        learned_memory_enabled=bool(learned_memory_enabled or has_learned_memory),
        learned_memory_mode=learned_memory_mode,
        learned_memory_layer_interval=learned_memory_layer_interval,
        learned_memory_video_dim=learned_memory_video_dim,
        learned_memory_audio_dim=learned_memory_audio_dim,
        learned_memory_heads=learned_memory_heads,
        learned_memory_color_film_enabled=bool(
            learned_memory_color_film or has_color_film
        ),
        learned_memory_color_condition_dim=4 * 128,
        learned_memory_color_film_hidden_dim=learned_memory_color_film_hidden_dim,
    )
    with torch.device("meta"):
        model = CausalLTXModel(config)
    model = model.to(dtype=dtype)
    wrapper = CausalLTX2DiffusionWrapper(
        model=model,
        video_height=video_height,
        video_width=video_width,
        num_frame_per_block=num_frame_per_block,
        num_frame_per_block_first=num_frame_per_block_first,
        num_audio_sink_tokens=num_audio_sink_tokens,
        use_flex_attention=use_flex_attention,
    )
    print(f"[Load] Model built on meta device (zero alloc), loading weights with assign=True...")

    # ── Load weights (assign=True: zero-copy, gen_sd tensors become params) ──
    # Cast gen_sd tensors to target dtype if needed (checkpoint might be fp32).
    for k in gen_sd:
        if gen_sd[k].dtype != dtype:
            gen_sd[k] = gen_sd[k].to(dtype=dtype)

    missing, unexpected = wrapper.load_state_dict(gen_sd, strict=False, assign=True)
    expected_missing_pats = [
        "mask_builder",
        "learned_memory",
        "memory_attn",
        "color_film",
    ]
    real_missing = [
        k for k in missing
        if not any(p in k for p in expected_missing_pats)
    ]

    if real_missing:
        print(f"[Load] WARNING: {len(real_missing)} missing keys in TaoMate ckpt:")
        for k in real_missing[:10]:
            print(f"  - {k}")

        # Attempt to fill from original checkpoint
        if original_ckpt_path:
            print(f"[Load] Filling missing keys from: {original_ckpt_path}")
            if original_ckpt_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                orig_sd = load_file(original_ckpt_path)
            else:
                orig_sd = torch.load(original_ckpt_path, **_load_kw)
            orig_remapped = _remap_state_dict_keys(orig_sd)
            del orig_sd
            gc.collect()

            fill_count = 0
            for k in real_missing:
                if k in orig_remapped:
                    gen_sd[k] = orig_remapped[k]
                    fill_count += 1
            if fill_count > 0:
                print(f"[Load] Filled {fill_count}/{len(real_missing)} keys from original ckpt")
                wrapper.load_state_dict(gen_sd, strict=False, assign=True)
            else:
                print("[Load] WARNING: could not fill any missing keys")
            del orig_remapped
    else:
        print("[Load] All generator weights loaded (expected missing: mask_builder)")

    # Move model to target device AFTER loading (for non-meta params like buffers).
    # For FSDP mode (device="cpu") this is a no-op; for single GPU it moves to CUDA.
    # Safety: materialize any remaining meta parameters (from missing keys) to zeros.
    meta_params = [
        (name, param) for name, param in wrapper.named_parameters()
        if param.device == torch.device("meta")
    ]
    if meta_params:
        print(f"[Load] Materializing {len(meta_params)} remaining meta parameters to zeros")
        for name, param in meta_params:
            parts = name.split(".")
            mod = wrapper
            for p in parts[:-1]:
                mod = getattr(mod, p)
            setattr(mod, parts[-1], torch.nn.Parameter(
                torch.zeros(param.shape, dtype=dtype, device="cpu"),
                requires_grad=param.requires_grad,
            ))
    wrapper = wrapper.to(device=device)

    # Free gen_sd immediately — model params now own these tensors.
    del gen_sd
    gc.collect()

    if unexpected:
        print(f"[Load] Unexpected keys: {len(unexpected)}")

    wrapper.eval()
    return wrapper


# ═══════════════════════════════════════════════════════════════════════════
# KV-Cache Accelerated Pipeline
# ═══════════════════════════════════════════════════════════════════════════


def _resolve_prefix_ctx_sigma(
    k: int,
    window: int,
    base: float,
    s_max: float,
    schedule: str,
) -> float:
    """Per-block context-noise sigma with distance coupling.

    ``k`` is the index into the kept prefix-block list, ordered from oldest
    (k=0, farthest from current block) to newest (k=window-1, closest).
    The *farthest* block carries the largest accumulated self-forcing
    error variance, so it gets the largest sigma.

    Schedules:
        constant:   sigma = base                           (legacy)
        linear:     sigma = base + (s_max - base) * r
        sqrt:       sigma = base + (s_max - base) * sqrt(r)
    where r = 1 - k / (W - 1) peaks at r=1 for the farthest block.
    """
    if schedule == "constant" or window <= 1:
        return base
    t = float(k) / float(max(window - 1, 1))
    r = 1.0 - t
    factor = r if schedule == "linear" else r ** 0.5
    sigma = base + (s_max - base) * factor
    lo, hi = min(base, s_max), max(base, s_max)
    return max(lo, min(hi, sigma))


class KVCacheCausalPipeline:
    """
    KV-cache accelerated causal inference pipeline.

    Key insight: With causal masking, prefix tokens CANNOT attend to future
    block tokens. Their KV states are independent of the current block's
    content, so caching across denoising steps is **mathematically exact**.

    Required model interface (CausalLTXModel.forward_inference):
        forward_inference(
            video_latent, audio_latent, timesteps, audio_timesteps,
            video_context, audio_context, video_context_mask, audio_context_mask,
            kv_cache=None, video_start_frame=0, audio_start_frame=0,
            include_audio_sinks=True,
        ) -> (video_velocity, audio_velocity, kv_cache)

    Raises:
        NotImplementedError if the model does not implement forward_inference().
    """

    def __init__(
        self,
        generator: CausalLTX2DiffusionWrapper,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        num_frame_per_block_first: int = 0,
        num_audio_sink_tokens: int = 0,
        context_noise: float = 0.0,
        context_noise_max: Optional[float] = None,
        context_noise_schedule: str = "constant",
        max_prefix_blocks: Optional[int] = 5,
        block0_sink_enabled: bool = False,
        prefix_renorm: bool = False,
        prefix_renorm_alpha: float = 1.0,
        prefix_renorm_anchor_block: int = 0,
        prefix_renorm_debug: bool = False,
        learned_memory: bool = False,
        learned_memory_video_downsample: int = 4,
        learned_memory_audio_tokens: int = 64,
        learned_memory_video_beta: float = 0.15,
        learned_memory_audio_beta: float = 0.10,
        learned_memory_video_anchor_tether: float = 0.20,
        learned_memory_audio_anchor_tether: float = 0.10,
        learned_memory_identity_anchor: bool = False,
        learned_memory_identity_anchor_scale: float = 1.0,
        learned_memory_ref_video_anchor: bool = False,
        learned_memory_drift_gate: bool = False,
        learned_memory_drift_gate_threshold: float = 0.05,
        learned_memory_drift_gate_temperature: float = 0.10,
        learned_memory_drift_gate_min: float = 0.10,
        learned_memory_drift_gate_apply_to_color: bool = True,
        learned_memory_color_alpha: float = 0.0,
        learned_memory_color_proto_alpha: float = 0.0,
        learned_memory_color_update_beta: float = 0.05,
        learned_memory_color_anchor_tether: float = 0.40,
        learned_memory_color_proto_grid: int = 4,
        learned_memory_color_drift_threshold: float = 2.5,
        learned_memory_color_max_correction: float = 0.5,
        learned_memory_color_film: bool = False,
        pyramid_policy=None,
        profile_callback: Optional[Any] = None,
    ):
        self.generator = generator
        self.denoising_sigmas = denoising_sigmas
        self.profile_callback = profile_callback
        self.num_frame_per_block = num_frame_per_block
        self.num_frame_per_block_first = num_frame_per_block_first
        # Must match training config — controls include_audio_sinks on prefix
        # rebuild (training refresh uses False when num_audio_sink_tokens==0).
        self.num_audio_sink_tokens = num_audio_sink_tokens
        # Block 0 uses the same causal forward_inference path as block 1..N,
        # with an empty KV cache and no prefix. This matches TaoMate KV-cache
        # training and keeps i2v frame-conditioning inside the denoising loop.
        # Context noise sigma injected onto prefix latents during KV-cache
        # pre-fill, to align with training self_forcing_context_noise
        # distribution. Timesteps remain 0 — only the latent is noised,
        # matching dmd.py KV refresh ("noisy video + timestep=0" pair).
        # ``context_noise`` is the BASE sigma (nearest prefix block);
        # ``context_noise_max`` is the upper bound for the farthest kept
        # prefix block. Schedule in {"constant", "linear", "sqrt"}.
        self.context_noise = float(context_noise)
        self.context_noise_max = (
            float(context_noise_max) if context_noise_max is not None
            else float(context_noise)
        )
        schedule = str(context_noise_schedule or "constant").lower()
        if schedule not in ("constant", "linear", "sqrt"):
            raise ValueError(
                f"context_noise_schedule must be one of constant|linear|sqrt, "
                f"got {context_noise_schedule!r}"
            )
        self.context_noise_schedule = schedule
        # Sliding window: keep only the most-recent N prefix blocks in the
        # pre-fill pass. ``None`` / ``<=0`` disables windowing (full prefix).
        # RoPE positions remain absolute (pb.video_start) — we just drop
        # out-of-window blocks from the KV cache.
        self._max_prefix_blocks = (
            int(max_prefix_blocks)
            if max_prefix_blocks is not None and int(max_prefix_blocks) > 0
            else None
        )
        # Block 0 Sink: when True, Block 0 (clip-start anchor) is ALWAYS kept
        # in the prefix KV cache regardless of the retention cutoff.
        # Mirrors training-side ``block0_sink_enabled``; combined with
        # ``max_prefix_blocks`` keeps the sink and recent prefix blocks.
        # Semantics:
        #   - sink disabled: max_prefix_blocks = total prefix block count
        #   - sink enabled:  max_prefix_blocks = NON-SINK warm window size
        #                    (sink is an additional +1, total = 1 + max_prefix_blocks).
        #     This matches training where ``self_forcing_prefix_blocks`` counts
        #     warm prefix blocks WITHOUT the sink.
        # Layout after windowing (sink enabled):
        #   [Block 0 sink] + [most-recent max_prefix_blocks blocks]
        # Middle blocks between sink and warm prefix are dropped from KV
        # (same semantics as training ``sink_prefix_gap_skip=True``).
        self.block0_sink_enabled = bool(block0_sink_enabled)

        # ── Prefix latent renormalization (mitigates long-horizon brightness drift) ──
        # See ltx_distillation/inference/causal_pipeline.py for the rationale.
        # Hook is applied after every block's denoising completes — written-back
        # latent feeds the *next* block's prefix KV pre-fill, so anchoring here
        # propagates automatically.
        self.prefix_renorm = bool(prefix_renorm)
        self.prefix_renorm_alpha = max(0.0, min(1.0, float(prefix_renorm_alpha)))
        self.prefix_renorm_anchor_block = max(0, int(prefix_renorm_anchor_block))
        self.prefix_renorm_debug = bool(prefix_renorm_debug)
        self._anchor_v: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self._anchor_a: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.learned_memory = bool(learned_memory)
        self.learned_memory_video_downsample = max(1, int(learned_memory_video_downsample))
        self.learned_memory_audio_tokens = max(1, int(learned_memory_audio_tokens))
        self.learned_memory_video_beta = max(0.0, min(1.0, float(learned_memory_video_beta)))
        self.learned_memory_audio_beta = max(0.0, min(1.0, float(learned_memory_audio_beta)))
        self.learned_memory_video_anchor_tether = max(
            0.0, min(1.0, float(learned_memory_video_anchor_tether))
        )
        self.learned_memory_audio_anchor_tether = max(
            0.0, min(1.0, float(learned_memory_audio_anchor_tether))
        )
        self.learned_memory_identity_anchor = bool(learned_memory_identity_anchor)
        self.learned_memory_identity_anchor_scale = max(
            0.0, float(learned_memory_identity_anchor_scale)
        )
        self.learned_memory_ref_video_anchor = bool(learned_memory_ref_video_anchor)
        self.learned_memory_drift_gate = bool(learned_memory_drift_gate)
        self.learned_memory_drift_gate_threshold = float(
            learned_memory_drift_gate_threshold
        )
        self.learned_memory_drift_gate_temperature = max(
            0.0, float(learned_memory_drift_gate_temperature)
        )
        self.learned_memory_drift_gate_min = max(
            0.0, min(1.0, float(learned_memory_drift_gate_min))
        )
        self.learned_memory_drift_gate_apply_to_color = bool(
            learned_memory_drift_gate_apply_to_color
        )
        self.learned_memory_color_alpha = max(
            0.0, min(1.0, float(learned_memory_color_alpha))
        )
        self.learned_memory_color_proto_alpha = max(
            0.0, min(1.0, float(learned_memory_color_proto_alpha))
        )
        self.learned_memory_color_update_beta = max(
            0.0, min(1.0, float(learned_memory_color_update_beta))
        )
        self.learned_memory_color_anchor_tether = max(
            0.0, min(1.0, float(learned_memory_color_anchor_tether))
        )
        self.learned_memory_color_proto_grid = max(1, int(learned_memory_color_proto_grid))
        self.learned_memory_color_drift_threshold = float(
            learned_memory_color_drift_threshold
        )
        self.learned_memory_color_max_correction = max(
            0.0, float(learned_memory_color_max_correction)
        )
        self.learned_memory_color_film = bool(learned_memory_color_film)
        self.learned_memory_color_enabled = bool(
            self.learned_memory
            and (
                self.learned_memory_color_alpha > 0.0
                or self.learned_memory_color_proto_alpha > 0.0
                or self.learned_memory_color_film
            )
        )

        if not hasattr(generator.model, "forward_inference"):
            raise NotImplementedError(
                "KV-cache runtime requires CausalLTXModel.forward_inference(), "
                "which is not yet implemented. Either:\n"
                "  1. Implement forward_inference() in CausalLTXModel, or\n"
                "  2. Use --runtime prefix_rerun instead."
            )
        # Pyramid Forcing head-aware KV policy (None ⇒ default cache path).
        self.pyramid_policy = pyramid_policy
        if self.pyramid_policy is not None:
            print("[KVCache] Pyramid head-aware KV policy ENABLED")
        print("[KVCache] Model supports forward_inference — KV-cache mode ACTIVE")

    def _profile_sync(self, device: torch.device) -> None:
        if self.profile_callback is None:
            return
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)

    def _profile(self, stage: str, **fields: Any) -> None:
        if self.profile_callback is None:
            return
        self.profile_callback(stage, **fields)

    @staticmethod
    def _reshape_sigma_broadcast(sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 1:
            return sigma.view(-1, *([1] * (target.ndim - 1)))
        elif sigma.ndim == 2:
            return sigma.view(sigma.shape[0], sigma.shape[1], *([1] * (target.ndim - 2)))
        return sigma

    def _velocity_to_x0(self, sample, velocity, sigma):
        """velocity → x0: x0 = sample - velocity * sigma  (matches wrapper.forward)."""
        s = self._reshape_sigma_broadcast(sigma, sample)
        return (
            sample.to(torch.float32)
            - velocity.to(torch.float32) * s.to(torch.float32)
        ).to(sample.dtype)

    # ------------------------------------------------------------------
    # Prefix renormalization helpers — duplicated logic of the same
    # functionality in CausalAVInferencePipeline so both runtimes share
    # behaviour without coupling their class hierarchies.
    # ------------------------------------------------------------------
    def _reset_renorm_anchor(self) -> None:
        self._anchor_v = None
        self._anchor_a = None
        self._abstract_v2_anchor_stats = None
        self._abstract_v2_anchor_proto = None
        self._abstract_v2_stats = None
        self._abstract_v2_proto = None

    def _new_learned_memory_state(self) -> Optional[LearnedMemoryState]:
        if not self.learned_memory:
            return None
        return LearnedMemoryState(
            enabled=True,
            video_downsample=self.learned_memory_video_downsample,
            audio_tokens=self.learned_memory_audio_tokens,
            video_beta=self.learned_memory_video_beta,
            audio_beta=self.learned_memory_audio_beta,
            video_anchor_tether=self.learned_memory_video_anchor_tether,
            audio_anchor_tether=self.learned_memory_audio_anchor_tether,
            identity_anchor_enabled=self.learned_memory_identity_anchor,
            identity_anchor_scale=self.learned_memory_identity_anchor_scale,
            reference_anchor_enabled=self.learned_memory_ref_video_anchor,
            drift_gate_enabled=self.learned_memory_drift_gate,
            drift_gate_threshold=self.learned_memory_drift_gate_threshold,
            drift_gate_temperature=self.learned_memory_drift_gate_temperature,
            drift_gate_min=self.learned_memory_drift_gate_min,
            drift_gate_apply_to_color=self.learned_memory_drift_gate_apply_to_color,
            color_enabled=self.learned_memory_color_enabled,
            color_alpha=self.learned_memory_color_alpha,
            color_proto_alpha=self.learned_memory_color_proto_alpha,
            color_update_beta=self.learned_memory_color_update_beta,
            color_anchor_tether=self.learned_memory_color_anchor_tether,
            color_proto_grid=self.learned_memory_color_proto_grid,
            color_drift_threshold=self.learned_memory_color_drift_threshold,
            color_max_correction=self.learned_memory_color_max_correction,
            color_film_enabled=self.learned_memory_color_film,
        )

    def _maybe_renorm_block(
        self,
        block_idx: int,
        current_video: torch.Tensor,
        current_audio: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Capture anchor at ``anchor_block`` and renorm subsequent blocks.

        NOTE: Only applied to VIDEO latents. Audio latents are passed through
        unchanged — audio's low token count (1 per frame) makes per-channel
        renorm destabilising (introduces noise artefacts over long horizons).
        """
        if not self.prefix_renorm:
            return current_video, current_audio
        if block_idx < self.prefix_renorm_anchor_block:
            return current_video, current_audio
        if block_idx == self.prefix_renorm_anchor_block:
            if current_video is not None and current_video.numel() > 0:
                m, s = _channel_stats(current_video, "video")
                self._anchor_v = (m.detach(), s.detach())
            if self.prefix_renorm_debug:
                print(_format_block_stats(
                    block_idx, current_video, current_audio, suffix="ANCHOR"
                ), flush=True)
            return current_video, current_audio
        # block_idx > anchor_block: apply renorm to video only
        new_v = current_video
        if current_video is not None and self._anchor_v is not None:
            new_v = _apply_renorm(
                current_video, self._anchor_v, self.prefix_renorm_alpha, "video"
            )
        if self.prefix_renorm_debug:
            print(_format_block_stats(
                block_idx, current_video, current_audio, suffix="before"
            ), flush=True)
            print(_format_block_stats(
                block_idx, new_v, current_audio, suffix="after "
            ), flush=True)
        return new_v, current_audio

    def _denoise_block_with_kv(
        self,
        *,
        block,
        B: int,
        video_tail_shape: Tuple[int, ...],
        audio_channels: Optional[int],
        conditional_dict: Dict[str, Any],
        kv_cache,
        final_step_kv_cache=None,
        final_step_kv_cache_builder=None,
        device: torch.device,
        dtype: torch.dtype,
        local_cond_pairs: Optional[List[Tuple[int, torch.Tensor]]] = None,
        audio_condition_latent: Optional[torch.Tensor] = None,
        initial_video: Optional[torch.Tensor] = None,
        initial_audio: Optional[torch.Tensor] = None,
        start_step: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Multi-step denoising of one block with pre-built prefix KV cache.

        Mirrors the training KV-cache loop
        ([dmd.py L1463-L1626](ltx_distillation/dmd.py)) exactly:
          - pure noise init
          - Save ``pre_block_cache_layers`` BEFORE entering the denoising loop
          - Per-step: **restore** ``kv_cache.layers = list(pre_block_cache_layers)``
            so that each step reads ONLY the prefix (blocks 0..N-1) and never
            sees the current block's in-progress K/V from previous steps.
            ``forward_inference`` mutates ``kv_cache.layers[i]`` in-place by
            appending the current block's K/V via ``torch.cat`` (see
            ``kv_cache.py::_cat_cache``), so without this restoration the
            cache accumulates duplicate in-progress K/V at the same RoPE
            positions across steps → catastrophic attention corruption.
          - velocity → x0
          - re-noise via add_noise(x0, randn_like, next_sigma)

        ``local_cond_pairs`` is the multi-modal conditioning frame map for
        this block: list of (local_idx, latent[B,1,C,H,W]).  Each entry is
        re-injected (clean latent + sigma=0) at every denoising step, mirroring
        the training-side ``_run_self_forcing_kv_all_grad_rollout`` cond path.
        ``None`` or empty list = pure t2v (default).
        """
        start_step = int(start_step)
        if start_step < 0 or start_step >= max(1, len(self.denoising_sigmas[:-1])):
            raise ValueError(
                f"start_step must be in [0, {len(self.denoising_sigmas[:-1]) - 1}], "
                f"got {start_step}"
            )
        if initial_video is None:
            cur_video = torch.randn(
                (B, block.video_frames, *video_tail_shape), device=device, dtype=dtype,
            )
        else:
            cur_video = initial_video.to(device=device, dtype=dtype)
        cur_audio = None
        if audio_channels is not None and block.audio_frames > 0:
            if audio_condition_latent is not None:
                cur_audio = audio_condition_latent.to(device=device, dtype=dtype)
                expected = (B, block.audio_frames, audio_channels)
                if tuple(cur_audio.shape) != expected:
                    raise ValueError(
                        "Audio condition block has the wrong shape: "
                        f"got {tuple(cur_audio.shape)}, expected {expected}"
                    )
            elif initial_audio is None:
                cur_audio = torch.randn(
                    (B, block.audio_frames, audio_channels), device=device, dtype=dtype,
                )
            else:
                cur_audio = initial_audio.to(device=device, dtype=dtype)
        # Inject condition frames into initial noise/state.
        if local_cond_pairs:
            for _li, _lat in local_cond_pairs:
                cur_video[:, _li:_li + 1] = _lat.to(device=device, dtype=dtype)

        # Snapshot prefix cache state (list of LayerKVCache references).
        # Training saves this BEFORE the denoising loop and restores it at
        # the start of every step — see dmd.py L1467 / L1485 / L1592.
        # NOTE: shallow copy of the list is sufficient because
        # ``forward_inference`` replaces ``kv_cache.layers[i]`` with a NEW
        # ``LayerKVCache`` object (it does not mutate the existing dataclass
        # fields), so the original LayerKVCache references remain intact.
        pre_block_cache_layers = list(kv_cache.layers) if kv_cache is not None else None
        final_step_cache_layers = (
            list(final_step_kv_cache.layers)
            if final_step_kv_cache is not None else None
        )
        last_denoise_step = len(self.denoising_sigmas[:-1]) - 1

        for si, sigma in enumerate(self.denoising_sigmas[:-1]):
            if si < start_step:
                continue
            if (
                final_step_cache_layers is None
                and final_step_kv_cache_builder is not None
                and si == last_denoise_step
            ):
                # Low-memory twin-cache path: keep only the noisy prefix cache
                # for intermediate denoise steps, then drop it before lazily
                # constructing the clean prefix cache for the final step.
                if kv_cache is not None:
                    kv_cache.layers = []
                pre_block_cache_layers = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                final_step_kv_cache = final_step_kv_cache_builder()
                final_step_cache_layers = (
                    list(final_step_kv_cache.layers)
                    if final_step_kv_cache is not None else None
                )
            use_final_step_cache = (
                final_step_cache_layers is not None
                and si == last_denoise_step
            )
            active_kv_cache = final_step_kv_cache if use_final_step_cache else kv_cache
            active_cache_layers = (
                final_step_cache_layers if use_final_step_cache
                else pre_block_cache_layers
            )
            # Restore prefix-only cache at the start of each step (matches
            # training exactly). Without this, step k sees block N's K/V
            # from steps 0..k-1 appended to the prefix → broken attention.
            if active_cache_layers is not None and active_kv_cache is not None:
                active_kv_cache.layers = list(active_cache_layers)

            vs = sigma.expand(B, cur_video.shape[1]).to(device)
            a_s = sigma.expand(B, cur_audio.shape[1]).to(device) if cur_audio is not None else None
            if audio_condition_latent is not None and cur_audio is not None:
                cur_audio = audio_condition_latent.to(device=device, dtype=dtype)
                a_s = torch.zeros_like(a_s)
            # Re-inject cond frames + zero sigma at cond positions.  Mirrors
            # the training rollout: cond positions are always "clean + 0"
            # at every step regardless of velocity update history.
            if local_cond_pairs:
                cur_video = cur_video.clone()
                vs = vs.clone()
                for _li, _lat in local_cond_pairs:
                    cur_video[:, _li:_li + 1] = _lat.to(device=device, dtype=dtype)
                    vs[:, _li] = 0

            self._profile_sync(device)
            _t_forward = time.perf_counter() if self.profile_callback is not None else None
            pred_v_vel, pred_a_vel, _ = self.generator.model.forward_inference(
                video_latent=cur_video, audio_latent=cur_audio,
                timesteps=vs, audio_timesteps=a_s,
                video_context=conditional_dict["video_context"],
                audio_context=conditional_dict["audio_context"],
                video_context_mask=conditional_dict.get("video_context_mask"),
                audio_context_mask=conditional_dict.get("audio_context_mask"),
                learned_memory_video=conditional_dict.get("learned_memory_video"),
                learned_memory_audio=conditional_dict.get("learned_memory_audio"),
                learned_memory_color=conditional_dict.get("learned_memory_color"),
                kv_cache=active_kv_cache,
                video_start_frame=block.video_start,
                audio_start_frame=block.audio_start,
                # Block 0 may need sink tokens when configured. For block > 0
                # sinks are never injected — they live only at the start of
                # the audio sequence.
                include_audio_sinks=(block.block_idx == 0 and self.num_audio_sink_tokens > 0),
                pyramid_policy=self.pyramid_policy,
            )
            if _t_forward is not None:
                self._profile_sync(device)
                self._profile(
                    "kv_denoise_forward_done",
                    block=block.block_idx,
                    step=si,
                    sigma=f"{float(sigma.item()):.4f}",
                    elapsed=f"{time.perf_counter() - _t_forward:.3f}s",
                )

            cur_video = self._velocity_to_x0(cur_video, pred_v_vel, vs)
            if cur_audio is not None:
                cur_audio = self._velocity_to_x0(cur_audio, pred_a_vel, a_s)
                if audio_condition_latent is not None:
                    cur_audio = audio_condition_latent.to(device=device, dtype=dtype)
            # Overwrite cond positions in the denoised result so next-step
            # add_noise doesn't drift them away from clean.
            if local_cond_pairs:
                for _li, _lat in local_cond_pairs:
                    cur_video[:, _li:_li + 1] = _lat.to(device=device, dtype=cur_video.dtype)

            next_sigma = self.denoising_sigmas[si + 1]
            if float(next_sigma.item()) > 0.0:
                v_ns = next_sigma.expand(B, cur_video.shape[1]).to(device)
                cur_video = add_noise(cur_video, torch.randn_like(cur_video), v_ns)
                if cur_audio is not None:
                    a_ns = next_sigma.expand(B, cur_audio.shape[1]).to(device)
                    cur_audio = add_noise(cur_audio, torch.randn_like(cur_audio), a_ns)
                    if audio_condition_latent is not None:
                        cur_audio = audio_condition_latent.to(device=device, dtype=dtype)

        return cur_video, cur_audio

    @torch.no_grad()
    def generate(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Optional[Tuple[int, ...]],
        conditional_dict: Dict[str, Any],
        seed: Optional[int] = None,
        conditioning_mode: str = "t2v",
        first_frame_latent: Optional[torch.Tensor] = None,
        end_frame_latent: Optional[torch.Tensor] = None,
        audio_condition_latent: Optional[torch.Tensor] = None,
        block_callback: Optional[Any] = None,
        learned_memory_state: Optional[LearnedMemoryState] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate video (+ audio) latents with KV-cache acceleration.

        Multi-modal conditioning (i2v / ii2v):
          - ``conditioning_mode='i2v'`` requires ``first_frame_latent`` [B,1,C,H,W].
          - ``conditioning_mode='ii2v'`` requires ``first_frame_latent`` and
            (optionally) ``end_frame_latent`` [B,1,C,H,W]; if omitted, the
            model degrades to i2v-style behavior at the last frame.
          - ``conditioning_mode='t2v'`` (default) ignores both latents and
            preserves legacy behavior.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Reset prefix-renorm anchor at the start of each video.
        self._reset_renorm_anchor()
        if learned_memory_state is None:
            learned_memory_state = self._new_learned_memory_state()

        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        B = video_shape[0]
        F_v = video_shape[1]

        # ---- Build global condition frame map (frame_idx -> latent) ----
        cond_mode_eff = (conditioning_mode or "t2v").lower()
        audio_conditioned = cond_mode_eff in ("ta2v", "tia2v")
        if audio_conditioned:
            if audio_shape is None or audio_condition_latent is None:
                raise ValueError(
                    f"conditioning_mode={cond_mode_eff!r} requires an audio condition latent"
                )
            if tuple(audio_condition_latent.shape) != tuple(audio_shape):
                raise ValueError(
                    "Audio condition shape must match audio_shape: "
                    f"got {tuple(audio_condition_latent.shape)}, expected {tuple(audio_shape)}"
                )
        cond_global_frames: Dict[int, torch.Tensor] = {}
        if cond_mode_eff in ("i2v", "ii2v", "tia2v"):
            if first_frame_latent is None:
                raise ValueError(
                    f"conditioning_mode={cond_mode_eff!r} requires first_frame_latent "
                    f"of shape [B, 1, C, H, W]"
                )
            cond_global_frames[0] = first_frame_latent.to(device=device, dtype=dtype)
        if cond_mode_eff == "ii2v":
            _last = F_v - 1
            if end_frame_latent is not None:
                cond_global_frames[_last] = end_frame_latent.to(device=device, dtype=dtype)
            # else: silently fall back to no explicit end anchor (i2v-like)

        blocks = compute_av_blocks(
            total_video_latent_frames=F_v,
            num_frame_per_block=self.num_frame_per_block,
            num_frame_per_block_first=self.num_frame_per_block_first,
        )
        # Legacy block layout: merge block-0 (1 frame) + block-1.
        # OmniForcing mode: Block 0 already has num_frame_per_block_first frames, no merge.
        if self.num_frame_per_block_first == 0 and len(blocks) >= 2 and blocks[0].video_frames == 1:
            merged = type(blocks[0])(
                block_idx=0,
                video_start=blocks[0].video_start,
                video_end=blocks[1].video_end,
                audio_start=blocks[0].audio_start,
                audio_end=blocks[1].audio_end,
            )
            blocks = [merged, *blocks[2:]]

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = None
        if audio_shape is not None:
            audio = torch.zeros(audio_shape, device=device, dtype=dtype)
        cache_segments = []
        video_hw = (int(video_shape[-2]), int(video_shape[-1]))

        def _cache_segment_for_block(_block, *, include_audio_sinks: bool, window_slot: int):
            video_frames = max(0, int(_block.video_end) - int(_block.video_start))
            audio_frames = max(0, int(_block.audio_end) - int(_block.audio_start))
            audio_tokens = audio_frames
            if include_audio_sinks:
                audio_tokens += max(0, int(self.num_audio_sink_tokens))
            return {
                "kind": "generated",
                "block_idx": int(_block.block_idx),
                "window_slot": int(window_slot),
                "video_tokens": int(video_frames * video_hw[0] * video_hw[1]),
                "audio_tokens": int(audio_tokens),
                "include_audio_sinks": bool(include_audio_sinks),
            }

        def _pairs_for_block(_blk) -> List[Tuple[int, torch.Tensor]]:
            """Return (local_idx, latent) for cond frames falling inside this block."""
            _out: List[Tuple[int, torch.Tensor]] = []
            for _g, _l in cond_global_frames.items():
                if _blk.video_start <= _g < _blk.video_end:
                    _out.append((_g - _blk.video_start, _l))
            return _out

        def _callback_requests_stop(_result: Any) -> bool:
            if isinstance(_result, dict):
                return bool(
                    _result.get("stop")
                    or _result.get("stop_after_block")
                    or _result.get("stop_after_current_block")
                )
            return bool(_result)

        def _commit_clean_kv(
            *,
            _kv_cache,
            _block,
            _block_conditional_dict: Dict[str, Any],
            _video_latent: torch.Tensor,
            _audio_latent: Optional[torch.Tensor],
            _pre_block_cache_layers: Optional[List[Any]],
            _include_audio_sinks: bool,
            _window_slot: int,
            _profile_label: str,
        ):
            if _pre_block_cache_layers is None:
                _kv_cache = self.generator.init_kv_cache()
            else:
                _kv_cache.layers = list(_pre_block_cache_layers)
            _vs = torch.zeros(
                (B, _video_latent.shape[1]),
                device=device,
                dtype=dtype,
            )
            _as = (
                torch.zeros(
                    (B, _audio_latent.shape[1]),
                    device=device,
                    dtype=dtype,
                )
                if _audio_latent is not None else None
            )
            self._profile_sync(device)
            _t_refresh = time.perf_counter() if self.profile_callback is not None else None
            _, _, _kv_cache = self.generator.model.forward_inference(
                video_latent=_video_latent,
                audio_latent=_audio_latent,
                timesteps=_vs,
                audio_timesteps=_as,
                video_context=_block_conditional_dict["video_context"],
                audio_context=_block_conditional_dict["audio_context"],
                video_context_mask=_block_conditional_dict.get("video_context_mask"),
                audio_context_mask=_block_conditional_dict.get("audio_context_mask"),
                learned_memory_video=_block_conditional_dict.get("learned_memory_video"),
                learned_memory_audio=_block_conditional_dict.get("learned_memory_audio"),
                learned_memory_color=_block_conditional_dict.get("learned_memory_color"),
                kv_cache=_kv_cache,
                video_start_frame=_block.video_start,
                audio_start_frame=_block.audio_start,
                include_audio_sinks=_include_audio_sinks,
                pyramid_policy=self.pyramid_policy,
                kv_cache_only=True,
            )
            if _t_refresh is not None:
                self._profile_sync(device)
                self._profile(
                    _profile_label,
                    block=_block.block_idx,
                    video_frames=_video_latent.shape[1],
                    audio_frames=_audio_latent.shape[1] if _audio_latent is not None else 0,
                    elapsed=f"{time.perf_counter() - _t_refresh:.3f}s",
                )
            cache_segments.append(
                _cache_segment_for_block(
                    _block,
                    include_audio_sinks=_include_audio_sinks,
                    window_slot=_window_slot,
                )
            )
            return _kv_cache

        # ── Persistent incremental KV cache (matches training accumulation) ──
        # Training accumulates KV across blocks: after denoising block N, a
        # sigma=0 "refresh" forward appends block N's clean K/V to the cache
        # for use by block N+1.  The previous per-block-rebuild approach
        # re-created the full prefix from scratch which subtly differs (each
        # forward with previously-denoised latent is slightly different from
        # the forward that produced the latent) and is 2x slower.
        kv_cache = None
        audio_channels = audio_shape[2] if audio_shape is not None else None
        stop_video_end: Optional[int] = None
        stop_audio_end: Optional[int] = None
        for block_i, block in enumerate(blocks):
            is_final_block = block_i == len(blocks) - 1
            block_conditional_dict = (
                learned_memory_state.with_conditional_memory(
                    conditional_dict, device=device, dtype=dtype
                )
                if learned_memory_state is not None else conditional_dict
            )
            # ── Block 0 ──
            if block.block_idx == 0:
                # Causal KV-cache path — matches OmniForcing training.
                kv_cache = self.generator.init_kv_cache()
                _b0_pairs = _pairs_for_block(block)
                bv, ba = self._denoise_block_with_kv(
                    block=block,
                    B=B,
                    video_tail_shape=video_shape[2:],
                    audio_channels=audio_channels,
                    conditional_dict=block_conditional_dict,
                    kv_cache=kv_cache,
                    device=device,
                    dtype=dtype,
                    local_cond_pairs=_b0_pairs,
                    audio_condition_latent=(
                        audio_condition_latent[:, block.audio_start:block.audio_end]
                        if audio_conditioned else None
                    ),
                )
                bv, ba = self._maybe_renorm_block(block.block_idx, bv, ba)
                if audio_conditioned:
                    ba = audio_condition_latent[:, block.audio_start:block.audio_end].to(
                        device=device, dtype=dtype,
                    )
                if learned_memory_state is not None:
                    bv = learned_memory_state.apply_color_memory(bv)
                if _b0_pairs:
                    for _li, _lat in _b0_pairs:
                        bv[:, _li:_li + 1] = _lat.to(device=device, dtype=bv.dtype)
                video[:, block.video_start:block.video_end] = bv
                if audio is not None and ba is not None:
                    audio[:, block.audio_start:block.audio_end] = ba
                # Publish realtime preview as soon as the clean block latent is
                # available.  The sigma=0 refresh below is still required for
                # exact KV semantics, but it only prepares the next block.
                stop_after_block = False
                if block_callback is not None:
                    callback_result = block_callback(bv, ba, block)
                    stop_after_block = _callback_requests_stop(callback_result)
                    if stop_after_block:
                        stop_video_end = int(block.video_end)
                        stop_audio_end = int(block.audio_end)
                commit_conditional_dict = block_conditional_dict
                if is_final_block:
                    kv_cache = _commit_clean_kv(
                        _kv_cache=kv_cache,
                        _block=block,
                        _block_conditional_dict=commit_conditional_dict,
                        _video_latent=bv,
                        _audio_latent=ba,
                        _pre_block_cache_layers=None,
                        _include_audio_sinks=(self.num_audio_sink_tokens > 0),
                        _window_slot=block_i,
                        _profile_label="window_final_kv_clean_commit_done",
                    )
                    if learned_memory_state is not None:
                        learned_memory_state.set_reference(bv, ba)
                        learned_memory_state.update(bv, ba)
                    if stop_after_block:
                        break
                    continue

                # Clean cache commit
                kv_cache = _commit_clean_kv(
                    _kv_cache=kv_cache,
                    _block=block,
                    _block_conditional_dict=commit_conditional_dict,
                    _video_latent=bv,
                    _audio_latent=ba,
                    _pre_block_cache_layers=None,
                    _include_audio_sinks=(self.num_audio_sink_tokens > 0),
                    _window_slot=block_i,
                    _profile_label="kv_clean_refresh_done",
                )
                if learned_memory_state is not None:
                    learned_memory_state.set_reference(bv, ba)
                    learned_memory_state.update(bv, ba)
                if stop_after_block:
                    break
                continue

            # ── Block > 0: incremental KV cache (no prefix rebuild) ──
            # kv_cache already holds blocks 0..N-1 from prior refreshes.
            # Save pre-block state so we can restore after denoising and
            # do a clean refresh (matches training's pre_block_cache_layers).
            pre_block_cache_layers = list(kv_cache.layers)

            _blk_pairs = _pairs_for_block(block)
            cur_video, cur_audio = self._denoise_block_with_kv(
                block=block,
                B=B,
                video_tail_shape=video_shape[2:],
                audio_channels=audio_channels,
                conditional_dict=block_conditional_dict,
                kv_cache=kv_cache,
                device=device,
                dtype=dtype,
                local_cond_pairs=_blk_pairs,
                audio_condition_latent=(
                    audio_condition_latent[:, block.audio_start:block.audio_end]
                    if audio_conditioned else None
                ),
            )

            cur_video, cur_audio = self._maybe_renorm_block(
                block.block_idx, cur_video, cur_audio
            )
            if audio_conditioned:
                cur_audio = audio_condition_latent[:, block.audio_start:block.audio_end].to(
                    device=device, dtype=dtype,
                )
            if learned_memory_state is not None:
                cur_video = learned_memory_state.apply_color_memory(cur_video)
            if _blk_pairs:
                for _li, _lat in _blk_pairs:
                    cur_video[:, _li:_li + 1] = _lat.to(
                        device=device, dtype=cur_video.dtype
                    )
            video[:, block.video_start:block.video_end] = cur_video
            if audio is not None and cur_audio is not None:
                audio[:, block.audio_start:block.audio_end] = cur_audio
            # Publish realtime preview before the sigma=0 clean-KV refresh.
            # This does not change the final generated latents; it just avoids
            # hiding a full refresh forward behind the first visible chunk.
            stop_after_block = False
            if block_callback is not None:
                callback_result = block_callback(cur_video, cur_audio, block)
                stop_after_block = _callback_requests_stop(callback_result)
                if stop_after_block:
                    stop_video_end = int(block.video_end)
                    stop_audio_end = int(block.audio_end)
            commit_conditional_dict = block_conditional_dict

            if is_final_block:
                kv_cache = _commit_clean_kv(
                    _kv_cache=kv_cache,
                    _block=block,
                    _block_conditional_dict=commit_conditional_dict,
                    _video_latent=cur_video,
                    _audio_latent=cur_audio,
                    _pre_block_cache_layers=pre_block_cache_layers,
                    _include_audio_sinks=False,
                    _window_slot=block_i,
                    _profile_label="window_final_kv_clean_commit_done",
                )
                if learned_memory_state is not None:
                    learned_memory_state.update(cur_video, cur_audio)
                if stop_after_block:
                    break
                continue

            # Clean cache commit
            kv_cache = _commit_clean_kv(
                _kv_cache=kv_cache,
                _block=block,
                _block_conditional_dict=commit_conditional_dict,
                _video_latent=cur_video,
                _audio_latent=cur_audio,
                _pre_block_cache_layers=pre_block_cache_layers,
                _include_audio_sinks=False,
                _window_slot=block_i,
                _profile_label="kv_clean_refresh_done",
            )
            if learned_memory_state is not None:
                learned_memory_state.update(cur_video, cur_audio)
            if stop_after_block:
                break

        # Keep the final incremental KV cache available to persistent callers
        # that want to reuse the exact next-prefix layout in memory.  This is
        # intentionally process-local and is never serialized by the normal CLI.
        self.last_kv_cache = kv_cache
        if kv_cache is not None:
            kv_cache._interactive_segments = [dict(s) for s in cache_segments]
        self.last_kv_cache_segments = [dict(s) for s in cache_segments]
        if stop_video_end is not None:
            video = video[:, :stop_video_end].contiguous()
            if audio is not None and stop_audio_end is not None:
                audio = audio[:, :stop_audio_end].contiguous()
        return video, audio

def _load_image_pixels(
    image_path: str,
    video_height: int,
    video_width: int,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Load an image file → [B=1, C=3, F=1, H, W] tensor in [-1, 1].

    Mirrors the pixel range expected by `ltx_distillation.dmd.LTX2DMD.
    encode_end_frame_pixels` (which assumes inputs are already in [-1, 1]).
    """
    from PIL import Image, ImageOps
    img = ImageOps.fit(
        Image.open(image_path).convert("RGB"),
        (video_width, video_height),
        method=Image.BICUBIC,
        centering=(0.5, 0.5),
    )
    raw = torch.frombuffer(img.tobytes(), dtype=torch.uint8).clone()
    arr = raw.view(video_height, video_width, 3).to(torch.float32)
    arr = arr.mul_(1.0 / 127.5).sub_(1.0)          # [H, W, 3] in [-1, 1]
    pix = arr.permute(2, 0, 1).contiguous()        # [3, H, W]
    pix = pix.unsqueeze(0).unsqueeze(2)            # [1, 3, 1, H, W]
    return pix.to(device=device, dtype=dtype)


def _ensure_video_encoder(video_vae, checkpoint_path: str, dtype: torch.dtype):
    """Lazy-load the video VAE encoder onto the wrapper if missing.

    ``create_vae_wrappers`` only loads the decoder by default (encoder is
    `None`).  When multi-conditioning (i2v / ii2v) is requested we materialise
    the encoder on demand and cache it on the wrapper so subsequent tasks
    re-use the same module.  The encoder is kept on CPU and moved to GPU only
    during the actual `encode()` call.
    """
    if getattr(video_vae, "encoder", None) is not None:
        return
    from ltx_pipelines.utils.model_ledger import ModelLedger
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )
    encoder = ledger.video_encoder().to(dtype=dtype)
    encoder.eval()
    video_vae.encoder = encoder


def _ensure_audio_encoder(audio_vae, checkpoint_path: str, dtype: torch.dtype):
    """Lazy-load the audio VAE encoder and keep it off GPU between calls."""
    if getattr(audio_vae, "encoder", None) is not None:
        return
    from ltx_pipelines.utils.model_ledger import ModelLedger

    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
    )
    encoder = ledger.audio_encoder().to(dtype=dtype)
    encoder.eval()
    audio_vae.encoder = encoder


def _extract_audio_latent_tensor(value: Any) -> torch.Tensor:
    """Extract an audio latent tensor from common ODE/checkpoint containers."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in (
            "audio_condition_latent",
            "audio_latent",
            "clean_audio",
            "audio",
        ):
            candidate = value.get(key)
            if torch.is_tensor(candidate):
                return candidate
        trajectory = value.get("audio_trajectory")
        if torch.is_tensor(trajectory):
            return trajectory[-1] if trajectory.ndim >= 4 else trajectory
        if isinstance(trajectory, (list, tuple)) and trajectory:
            candidate = trajectory[-1]
            if torch.is_tensor(candidate):
                return candidate
    raise ValueError(
        "Could not find an audio latent tensor. Expected a tensor or one of "
        "audio_condition_latent/audio_latent/clean_audio/audio/audio_trajectory."
    )


def _normalize_audio_condition_latent(latent: torch.Tensor) -> torch.Tensor:
    """Normalize audio VAE or saved latents to transformer layout [B,T,128]."""
    latent = latent.detach()
    if latent.ndim == 4:
        # Audio VAE layout: [B, C, T, F] -> [B, T, C*F].
        latent = latent.permute(0, 2, 1, 3).reshape(
            latent.shape[0], latent.shape[2], -1
        )
    elif latent.ndim == 3:
        if latent.shape[-1] == 128:
            pass
        elif latent.shape[0] * latent.shape[-1] == 128:
            # Unbatched audio VAE layout: [C, T, F].
            latent = latent.permute(1, 0, 2).reshape(1, latent.shape[1], -1)
        else:
            raise ValueError(
                "Ambiguous 3D audio latent shape. Expected [B,T,128] or "
                f"[C,T,F] with C*F=128, got {tuple(latent.shape)}."
            )
    elif latent.ndim == 2 and latent.shape[-1] == 128:
        latent = latent.unsqueeze(0)
    else:
        raise ValueError(
            "Unsupported audio latent shape. Expected [B,C,T,F], [C,T,F], "
            f"[B,T,128], or [T,128], got {tuple(latent.shape)}."
        )
    if latent.shape[-1] != 128:
        raise ValueError(
            f"Audio latent feature width must be 128, got {latent.shape[-1]}."
        )
    return latent.contiguous()


def _fit_audio_condition_latent(
    latent: torch.Tensor,
    *,
    target_frames: int,
    batch_size: int,
) -> torch.Tensor:
    """Fit [B,T,128] audio conditioning to the exact causal AV layout."""
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    latent = _normalize_audio_condition_latent(latent)
    if latent.shape[0] == 1 and batch_size > 1:
        latent = latent.expand(batch_size, -1, -1)
    elif latent.shape[0] != batch_size:
        raise ValueError(
            f"Audio condition batch mismatch: got {latent.shape[0]}, "
            f"expected {batch_size}."
        )
    if latent.shape[1] > target_frames:
        latent = latent[:, :target_frames]
    elif latent.shape[1] < target_frames:
        # Zero is the mean point in the normalized audio latent space and is a
        # safer silence fallback than repeating the final speech token.
        pad = latent.new_zeros(
            latent.shape[0], target_frames - latent.shape[1], latent.shape[2]
        )
        latent = torch.cat([latent, pad], dim=1)
    return latent.contiguous()


def _load_or_encode_audio_condition(
    *,
    audio_vae,
    checkpoint_path: str,
    dtype: torch.dtype,
    device: str,
    target_frames: int,
    batch_size: int = 1,
    audio_path: Optional[str] = None,
    audio_latent_path: Optional[str] = None,
    audio_start_time: float = 0.0,
) -> torch.Tensor:
    """Load or VAE-encode clean audio for ta2v/tia2v conditioning."""
    if bool(audio_path) == bool(audio_latent_path):
        raise ValueError(
            "Provide exactly one of audio_path or audio_latent_path for "
            "ta2v/tia2v conditioning."
        )

    if audio_latent_path:
        suffix = os.path.splitext(audio_latent_path)[1].lower()
        if suffix in (".npy", ".npz"):
            import numpy as np

            loaded = np.load(audio_latent_path)
            if isinstance(loaded, np.lib.npyio.NpzFile):
                try:
                    if "latents" in loaded:
                        array = loaded["latents"]
                    elif "latent" in loaded:
                        array = loaded["latent"]
                    elif loaded.files:
                        array = loaded[loaded.files[0]]
                    else:
                        raise ValueError(f"Empty npz audio latent: {audio_latent_path}")
                    raw = torch.from_numpy(array.copy())
                finally:
                    loaded.close()
            else:
                raw = torch.from_numpy(loaded.copy())
        else:
            raw = torch.load(audio_latent_path, map_location="cpu", weights_only=False)
        latent = _extract_audio_latent_tensor(raw)
    else:
        from ltx_core.model.audio_vae import encode_audio
        from ltx_pipelines.utils.media_io import decode_audio_from_file

        # LTX audio latents run at 25 Hz. Pad the waveform before VAE encoding
        # so short files become actual encoded silence rather than repeated
        # speech latents.
        target_duration = float(target_frames) / 25.0
        audio = decode_audio_from_file(
            audio_path,
            torch.device("cpu"),
            start_time=float(audio_start_time),
            max_duration=target_duration,
        )
        if audio is None:
            raise ValueError(f"No audio stream found in {audio_path!r}.")
        target_samples = int(round(target_duration * audio.sampling_rate))
        waveform = audio.waveform
        if waveform.shape[-1] < target_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, target_samples - waveform.shape[-1])
            )
        elif waveform.shape[-1] > target_samples:
            waveform = waveform[..., :target_samples]
        audio = Audio(waveform=waveform, sampling_rate=audio.sampling_rate)

        _ensure_audio_encoder(audio_vae, checkpoint_path, dtype)
        encoder = audio_vae.encoder
        encoder.to(device)
        try:
            with torch.no_grad():
                latent = encode_audio(audio, encoder, None)
        finally:
            encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    latent = _fit_audio_condition_latent(
        latent,
        target_frames=target_frames,
        batch_size=batch_size,
    )
    return latent.to(device=device, dtype=dtype)


def _encode_image_to_latent(
    image_path: str,
    video_vae,
    video_height: int,
    video_width: int,
    device: str,
    dtype: torch.dtype,
    checkpoint_path: Optional[str] = None,
) -> torch.Tensor:
    """VAE-encode a single image to a one-frame latent [B=1, 1, C, H', W'].

    Returned shape matches the pipeline convention used throughout this script
    (`[B, F, C, H, W]`) so it can be sliced directly into `current_video`.
    Moves the VAE encoder to `device` for encoding, then back to CPU to keep
    GPU memory budget aligned with the existing decoder pattern.
    """
    if getattr(video_vae, "encoder", None) is None:
        if checkpoint_path is None:
            raise RuntimeError(
                "video_vae.encoder is None and no checkpoint_path was provided "
                "for lazy loading; pass --original_ckpt through to "
                "_encode_image_to_latent."
            )
        _ensure_video_encoder(video_vae, checkpoint_path, dtype)
    pixels = _load_image_pixels(
        image_path, video_height, video_width, device, dtype,
    )
    enc = video_vae.encoder
    enc.to(device)
    try:
        with torch.no_grad():
            lat = enc(pixels)                       # [1, C, 1, H', W']
    finally:
        enc.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    lat = lat.permute(0, 2, 1, 3, 4).contiguous()   # [1, 1, C, H', W']
    return lat.to(dtype=dtype)


def _parse_task_image_entries(task: Dict[str, Any]) -> List[Tuple[str, int, float]]:
    """Parse benchmark `image` / `images` entries.

    Supported forms mirror unified_inference task JSON:
      - "image": [["/path/frame.png", 0, 0.9], ...]
      - "images": [["/path/frame.png", 0, 0.9], ...]
      - "image": ["/path/frame.png", 0, 0.9]
      - "image": "/path/frame.png"

    The strength value is parsed for compatibility but Model frame
    conditioning is hard-pinned to the encoded latent, matching training
    `allowed_ode_modes=[i2v]` semantics.
    """
    raw = task.get("image")
    if raw is None:
        raw = task.get("images")
    if raw is None:
        return []

    if isinstance(raw, str):
        raw_entries = [raw]
    elif isinstance(raw, (list, tuple)):
        if raw and isinstance(raw[0], str):
            raw_entries = [raw]
        else:
            raw_entries = list(raw)
    else:
        return []

    entries: List[Tuple[str, int, float]] = []
    for item in raw_entries:
        if isinstance(item, str):
            entries.append((item, 0, 1.0))
            continue
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            continue
        path = str(item[0])
        frame_idx = int(item[1]) if len(item) > 1 else 0
        strength = float(item[2]) if len(item) > 2 else 1.0
        entries.append((path, frame_idx, strength))
    return entries


def _resolve_cond_inputs(
    task: Dict[str, Any],
    args,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Resolve image/audio conditioning inputs for a task.

    Priority order:
      1. Per-task JSON fields: `conditioning_mode`, `first_frame_path`,
         `end_frame_path`.
      2. Per-task benchmark `image` / `images` entries. A frame_idx=0 entry
         becomes the i2v first frame; the largest non-zero frame_idx can become
         the ii2v end frame.
      3. Per-task `audio_path`/`wav_path` or `audio_latent_path`.
      4. Matching CLI flags.
    Returns ``(mode, first_path, end_path, audio_path, audio_latent_path)``.
    """
    image_entries = _parse_task_image_entries(task)
    first_from_image = None
    end_from_image = None
    if image_entries:
        first_entry = next(
            (entry for entry in image_entries if int(entry[1]) == 0),
            min(image_entries, key=lambda entry: int(entry[1])),
        )
        first_from_image = first_entry[0]
        nonzero_entries = [entry for entry in image_entries if int(entry[1]) != 0]
        if nonzero_entries:
            end_from_image = max(nonzero_entries, key=lambda entry: int(entry[1]))[0]

    mode = (task.get("conditioning_mode")
            or args.conditioning_mode
            or "t2v").lower()
    first_p = task.get("first_frame_path") or first_from_image or args.first_frame_path
    end_p = task.get("end_frame_path") or end_from_image or args.end_frame_path
    audio_p = (
        task.get("audio_path")
        or task.get("wav_path")
        or getattr(args, "audio_path", None)
    )
    audio_latent_p = (
        task.get("audio_latent_path")
        or getattr(args, "audio_latent_path", None)
    )
    has_audio = bool(audio_p or audio_latent_p)
    if audio_p and audio_latent_p:
        raise ValueError(
            "Specify only one audio source: audio_path/wav_path or audio_latent_path."
        )

    if mode == "t2v":
        if first_from_image or first_p:
            mode = "tia2v" if has_audio else ("ii2v" if end_from_image else "i2v")
        elif has_audio:
            mode = "ta2v"
    elif mode == "i2v" and has_audio:
        mode = "tia2v"
    elif mode == "ta2v" and first_p:
        mode = "tia2v"

    supported = {"t2v", "i2v", "ii2v", "ta2v", "tia2v"}
    if mode not in supported:
        raise ValueError(
            f"Unsupported conditioning_mode={mode!r}; expected one of {sorted(supported)}."
        )
    if mode == "ii2v" and has_audio:
        raise ValueError(
            "ii2v plus clean-audio conditioning has no matching training mode. "
            "Use tia2v (first frame + audio) or remove the audio input."
        )
    if mode in ("i2v", "ii2v", "tia2v") and not first_p:
        raise ValueError(
            f"conditioning_mode={mode!r} requires first_frame_path "
            "(set --first_frame_path, task['first_frame_path'], or task['image'])."
        )
    if mode in ("ta2v", "tia2v") and not has_audio:
        raise ValueError(
            f"conditioning_mode={mode!r} requires --audio_path, "
            "--audio_latent_path, or the matching task JSON field."
        )
    if mode in ("t2v", "ta2v"):
        first_p, end_p = None, None
    if mode not in ("ta2v", "tia2v"):
        audio_p, audio_latent_p = None, None
    return mode, first_p, end_p, audio_p, audio_latent_p
