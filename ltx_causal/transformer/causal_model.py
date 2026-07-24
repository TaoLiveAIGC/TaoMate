"""
CausalLTXModel: Full causal LTX-2 model for ODE causal training.

This module implements:
- Complete causal transformer with 48 layers
- Training mode: Flexattention with BlockMask for causal masking
- Weight loading from original LTX-2 checkpoints

Weight compatibility:
    All module names and state_dict keys match the original LTXModel exactly.
    The causal model adds block masks at runtime, which do not
    affect the checkpoint structure.

Based on LTX-2's LTXModel architecture (ltx_core/model/transformer/model.py).
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict, Any
import logging
import os

import torch
import torch.nn as nn

_diag_logger = logging.getLogger(__name__)

from ltx_causal.config import (
    CausalMaskConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
)
from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    build_all_causal_masks,
    compute_aligned_audio_frames,
    compute_av_blocks,
    compute_causal_log_scales,
)
from ltx_causal.transformer.causal_block import (
    CausalAVTransformerBlock,
    CausalTransformerArgs,
    TransformerConfig,
    rms_norm,
)
from ltx_causal.transformer.compat import (
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    causal_precompute_freqs_cis,
)

# === Optional Sequence Parallel support ===
# Same lazy-import pattern as causal_attention.py: taomate.runtime_support.parallel
# is the canonical home for SP infra, but ltx_distillation/__init__.py
# pulls in heavy training-only deps (lmdb, etc.). We import lazily and
# fall back to no-op stubs so the model file remains importable in
# inference-only / lightweight environments.
try:
    from taomate.runtime_support.parallel import (  # type: ignore[import-not-found]
        is_sp_enabled as _is_sp_enabled,
        get_sp_world_size as _get_sp_world_size,
        get_sp_group as _get_sp_group,
        split_sequence as _split_sequence,
        gather_sequence as _gather_sequence,
        pad_sequence_to_multiple as _pad_sequence_to_multiple,
        unpad_sequence as _unpad_sequence,
    )
except Exception:  # ImportError or transitive dep missing
    def _is_sp_enabled() -> bool:  # type: ignore[misc]
        return False

    def _get_sp_world_size() -> int:  # type: ignore[misc]
        return 1

    def _get_sp_group():  # type: ignore[misc]
        return None

    def _split_sequence(x, dim=1, group=None, sp_size=None):  # type: ignore[misc]
        return x

    def _gather_sequence(x, dim=1, group=None, sp_size=None):  # type: ignore[misc]
        return x

    def _pad_sequence_to_multiple(x, multiple, dim=1, pad_value=0.0):  # type: ignore[misc]
        return x, x.shape[dim], x.shape[dim]

    def _unpad_sequence(x, original_length, dim=1):  # type: ignore[misc]
        return x


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class CausalLTXModelConfig:
    """Configuration for CausalLTXModel."""

    # Model dimensions
    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_heads: int = 32
    audio_heads: int = 32
    video_d_head: int = 128
    audio_d_head: int = 64

    # Cross-attention context dimension
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048  # Also used as inner_dim for cross-modal RoPE

    # Patch embedding (LTX-2 uses patch_size=1 with nn.Linear)
    in_channels: int = 128
    out_channels: int = 128
    patch_size: Tuple[int, int, int] = (1, 1, 1)

    # Caption (text) projection
    caption_channels: int = 3840  # Gemma text encoder output dim

    # Position embedding
    pe_theta: float = 10000.0
    pe_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    audio_pe_max_pos: Tuple[int] = (20,)

    # RoPE long-horizon extrapolation. When the target video is longer than
    # ``rope_train_max_seconds`` (the longest clip seen during training),
    # applying plain RoPE with absolute positions pushes into untrained
    # frequencies and produces high-frequency aliasing. ``rope_extrapolation``
    # chooses how to handle this:
    #   "off"  — no scaling (legacy). Fine when target length ≤ train length.
    #   "ntk"  — NTK-aware scaling (bloc97): theta' = theta * r^(dim/(dim-2))
    #            where r = max(1, t_max / rope_train_max_seconds). Preserves
    #            low-frequency components exactly; stretches high frequencies.
    # ``rope_train_max_seconds`` should be set to the longest clip length
    # observed during Self-Forcing training (4-3-3-3 @ 24fps ≈ 5.4 s, so a
    # safe default is 8 s; for V2V fine-tuning with 5 s clips keep it at 5).
    rope_extrapolation: str = "off"
    rope_train_max_seconds: float = 8.0

    # Timestep embedding
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1

    # Normalization
    norm_eps: float = 1e-6

    # Causal generation
    num_frame_per_block: int = 3
    # Video frames in Block 0. 0 = legacy Global Prefix (1-3-3-3).
    # >0 = OmniForcing layout (e.g. 4 → 4-3-3-3).
    num_frame_per_block_first: int = 0

    # Audio sink tokens
    num_audio_sink_tokens: int = 0

    # When True, audio sink tokens are modulated by text-derived audio context
    # via AdaLN (scale + shift), making them sample-dependent instead of static.
    # The conditioning MLP is zero-initialized so it starts as identity.
    condition_sink_on_text: bool = False

    # Per-head gated attention
    apply_gated_attention: bool = False

    # RoPE
    rope_type: CausalRopeType = CausalRopeType.SPLIT

    # Token sizes
    video_frame_seqlen: int = 384  # For 512x768: (512/32)*(768/32)
    audio_frame_seqlen: int = 1

    # Log-ratio entropy-aligned rescaling for causal attention outputs.
    # When True, each token's causal attention output is scaled by
    # log(1 + visible_tokens) / log(1 + total_tokens), compensating for
    # the information deficit caused by causal masking vs bidirectional.
    # No learnable parameters — purely structural rescaling.
    enable_causal_log_rescale: bool = False

    # Text cross-attention AdaLN modulation (LTX-2.3 feature).
    # When True, text cross-attention uses AdaLN (shift/scale/gate) from
    # timestep embeddings, increasing adaln coefficient from 6 to 9.
    # When False, text cross-attention uses simple RMS norm (LTX-2 default).
    cross_attention_adaln: bool = False

    # V2/22B models: caption projection is done inside the text encoder
    # (Embeddings1DConnector), so the transformer should NOT create its own
    # caption_projection layers. When True, caption_projection is set to None
    # and _prepare_context becomes a simple reshape.
    caption_proj_before_connector: bool = False

    # Learned long-memory adapters (D1/D2). Default OFF to preserve exact
    # checkpoint compatibility and inference behaviour.
    learned_memory_enabled: bool = False
    # D1: "cross_attn_adapter"; D2: "memory_kv_side_branch".
    learned_memory_mode: str = "cross_attn_adapter"
    learned_memory_layer_interval: int = 4
    learned_memory_video_dim: int = 512
    learned_memory_audio_dim: int = 256
    learned_memory_heads: int = 8
    learned_memory_color_film_enabled: bool = False
    learned_memory_color_condition_dim: int = 512
    learned_memory_color_film_hidden_dim: int = 256


# ============================================================================
# CausalLTXModel
# ============================================================================

class CausalLTXModel(nn.Module):
    """
    Causal LTX-2 Model for ODE causal training via masking.

    This model is weight-compatible with the original LTX-2 checkpoint.
    All module names match the original LTXModel to enable direct checkpoint loading.

    Module name mapping to original LTXModel:
        patchify_proj           → nn.Linear(128, 4096)
        audio_patchify_proj     → nn.Linear(128, 2048)
        adaln_single            → AdaLayerNormSingle(4096, coefficient=6 or 9)
        audio_adaln_single      → AdaLayerNormSingle(2048, coefficient=6 or 9)
        caption_projection      → PixArtAlphaTextProjection(3840, 4096)
        audio_caption_projection → PixArtAlphaTextProjection(3840, 2048)
        av_ca_video_scale_shift_adaln_single → AdaLayerNormSingle(4096, coefficient=4)
        av_ca_audio_scale_shift_adaln_single → AdaLayerNormSingle(2048, coefficient=4)
        av_ca_a2v_gate_adaln_single          → AdaLayerNormSingle(4096, coefficient=1)
        av_ca_v2a_gate_adaln_single          → AdaLayerNormSingle(2048, coefficient=1)
        transformer_blocks      → ModuleList of CausalAVTransformerBlock
        scale_shift_table       → Parameter([2, 4096])
        norm_out                → LayerNorm(4096, elementwise_affine=False)
        proj_out                → nn.Linear(4096, 128)
        audio_scale_shift_table → Parameter([2, 2048])
        audio_norm_out          → LayerNorm(2048, elementwise_affine=False)
        audio_proj_out          → nn.Linear(2048, 128)
    """

    def __init__(self, config: CausalLTXModelConfig):
        super().__init__()
        self.config = config

        # Store key parameters for easy access
        self.timestep_scale_multiplier = config.timestep_scale_multiplier
        self.av_ca_timestep_scale_multiplier = config.av_ca_timestep_scale_multiplier
        self.cross_attention_adaln = config.cross_attention_adaln

        # Compute AdaLN coefficient: 6 (base) or 9 (with text cross-attention AdaLN)
        adaln_coeff = 9 if config.cross_attention_adaln else 6

        # === Patch Embedding (Linear, matching original) ===
        self.patchify_proj = nn.Linear(config.in_channels, config.video_dim, bias=True)
        self.audio_patchify_proj = nn.Linear(config.in_channels, config.audio_dim, bias=True)

        if config.learned_memory_enabled:
            mode = str(config.learned_memory_mode)
            if mode not in {"cross_attn_adapter", "memory_kv_side_branch"}:
                raise ValueError(
                    "learned_memory_mode must be cross_attn_adapter or "
                    f"memory_kv_side_branch, got {mode!r}"
                )
            self.learned_memory_video_encoder = nn.Linear(
                config.in_channels, config.learned_memory_video_dim, bias=True
            )
            self.learned_memory_audio_encoder = nn.Linear(
                config.in_channels, config.learned_memory_audio_dim, bias=True
            )
        else:
            self.learned_memory_video_encoder = None
            self.learned_memory_audio_encoder = None

        # === AdaLN Timestep Embedding ===
        self.adaln_single = AdaLayerNormSingle(config.video_dim, embedding_coefficient=adaln_coeff)
        self.audio_adaln_single = AdaLayerNormSingle(config.audio_dim, embedding_coefficient=adaln_coeff)

        # === Text Cross-Attention AdaLN (LTX-2.3) ===
        # Separate prompt_adaln modules for text key/value modulation
        self.prompt_adaln_single = (
            AdaLayerNormSingle(config.video_dim, embedding_coefficient=2)
            if config.cross_attention_adaln else None
        )
        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(config.audio_dim, embedding_coefficient=2)
            if config.cross_attention_adaln else None
        )

        # === Caption (Text) Projection ===
        # V2/22B: projection is already done in the text encoder connector,
        # so we skip creating projection layers here.
        if config.caption_proj_before_connector:
            self.caption_projection = None
            self.audio_caption_projection = None
        else:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels,
                hidden_size=config.video_dim,
            )
            self.audio_caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels,
                hidden_size=config.audio_dim,
            )

        # === Cross-Attention AdaLN (for A2V / V2A) ===
        # 4 additional AdaLN modules for cross-modal timestep conditioning
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=4,
        )
        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=4,
        )
        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            config.video_dim, embedding_coefficient=1,
        )
        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            config.audio_dim, embedding_coefficient=1,
        )

        # === Transformer Blocks ===
        base_video_config = TransformerConfig(
            dim=config.video_dim,
            heads=config.video_heads,
            d_head=config.video_d_head,
            context_dim=config.cross_attention_dim,
            cross_attention_adaln=config.cross_attention_adaln,
            apply_gated_attention=config.apply_gated_attention,
        )

        base_audio_config = TransformerConfig(
            dim=config.audio_dim,
            heads=config.audio_heads,
            d_head=config.audio_d_head,
            context_dim=config.audio_cross_attention_dim,
            cross_attention_adaln=config.cross_attention_adaln,
            apply_gated_attention=config.apply_gated_attention,
        )

        # Name must be `transformer_blocks` to match original state_dict
        self.transformer_blocks = nn.ModuleList([
            CausalAVTransformerBlock(
                idx=i,
                video=replace(
                    base_video_config,
                    learned_memory_enabled=(
                        config.learned_memory_enabled
                        and config.learned_memory_layer_interval > 0
                        and (i + 1) % config.learned_memory_layer_interval == 0
                    ),
                    learned_memory_mode=config.learned_memory_mode,
                    learned_memory_dim=config.learned_memory_video_dim,
                    learned_memory_heads=config.learned_memory_heads,
                    learned_memory_color_film_enabled=(
                        config.learned_memory_color_film_enabled
                        and config.learned_memory_layer_interval > 0
                        and (i + 1) % config.learned_memory_layer_interval == 0
                    ),
                    learned_memory_color_condition_dim=(
                        config.learned_memory_color_condition_dim
                    ),
                    learned_memory_color_film_hidden_dim=(
                        config.learned_memory_color_film_hidden_dim
                    ),
                ),
                audio=replace(
                    base_audio_config,
                    learned_memory_enabled=(
                        config.learned_memory_enabled
                        and config.learned_memory_layer_interval > 0
                        and (i + 1) % config.learned_memory_layer_interval == 0
                    ),
                    learned_memory_mode=config.learned_memory_mode,
                    learned_memory_dim=config.learned_memory_audio_dim,
                    learned_memory_heads=config.learned_memory_heads,
                ),
                rope_type=config.rope_type,
                norm_eps=config.norm_eps,
            )
            for i in range(config.num_layers)
        ])

        # === Output Layers (matching original names exactly) ===
        # Video output
        self.scale_shift_table = nn.Parameter(torch.empty(2, config.video_dim))
        self.norm_out = nn.LayerNorm(
            config.video_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.proj_out = nn.Linear(config.video_dim, config.out_channels)

        # Audio output
        self.audio_scale_shift_table = nn.Parameter(torch.empty(2, config.audio_dim))
        self.audio_norm_out = nn.LayerNorm(
            config.audio_dim, elementwise_affine=False, eps=config.norm_eps
        )
        self.audio_proj_out = nn.Linear(config.audio_dim, config.out_channels)

        # === Audio Sink Tokens ===
        if config.num_audio_sink_tokens > 0:
            self.audio_sink_tokens = nn.Parameter(
                torch.zeros(1, config.num_audio_sink_tokens, config.audio_dim)
            )
            nn.init.normal_(self.audio_sink_tokens, std=0.02)

            # Text-conditioned modulation: audio_ctx → (scale, shift) for sink tokens
            if config.condition_sink_on_text:
                self.sink_text_condition = nn.Sequential(
                    nn.Linear(config.audio_dim, config.audio_dim),
                    nn.SiLU(),
                    nn.Linear(config.audio_dim, 2 * config.audio_dim),
                )
                # Zero-init last layer → identity at start (scale=0, shift=0)
                nn.init.zeros_(self.sink_text_condition[-1].weight)
                nn.init.zeros_(self.sink_text_condition[-1].bias)

        # === Mask Builder ===
        self.mask_builder = AVCausalMaskBuilder(
            video_frame_seqlen=config.video_frame_seqlen,
            audio_frame_seqlen=config.audio_frame_seqlen,
            num_frame_per_block=config.num_frame_per_block,
            num_audio_sink_tokens=config.num_audio_sink_tokens,
        )

        # Gradient checkpointing
        self.gradient_checkpointing = False

    # ================================================================
    # Timestep Processing
    # ================================================================

    def _prepare_timestep(
        self,
        timestep: torch.Tensor,
        adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings via AdaLayerNormSingle.

        Matches TransformerArgsPreprocessor._prepare_timestep exactly.

        Args:
            timestep: [B] or [B, F] raw sigma values
            adaln: AdaLayerNormSingle module
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (timestep_6d, embedded_timestep):
                timestep_6d: [B, L, coeff*D] for block AdaLN (coeff=6 or 9)
                embedded_timestep: [B, L, D] for output layer
        """
        timestep = timestep * self.timestep_scale_multiplier
        timestep_6d, embedded_timestep = adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        timestep_6d = timestep_6d.view(batch_size, -1, timestep_6d.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])
        return timestep_6d, embedded_timestep

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross-attention timestep embeddings.

        Matches MultiModalTransformerArgsPreprocessor._prepare_cross_attention_timestep.

        Args:
            timestep: [B] or [B, F] raw sigma values
            cross_scale_shift_adaln: AdaLN for scale/shift (coefficient=4)
            cross_gate_adaln: AdaLN for gate (coefficient=1)
            batch_size: Batch size
            hidden_dtype: Target dtype

        Returns:
            (scale_shift_timestep, gate_timestep):
                scale_shift_timestep: [B, L, 4*D]
                gate_timestep: [B, L, D]
        """
        timestep_scaled = timestep * self.timestep_scale_multiplier
        av_ca_factor = self.av_ca_timestep_scale_multiplier / self.timestep_scale_multiplier

        scale_shift_timestep, _ = cross_scale_shift_adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(
            batch_size, -1, scale_shift_timestep.shape[-1]
        )

        gate_timestep, _ = cross_gate_adaln(
            timestep_scaled.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_timestep = gate_timestep.view(
            batch_size, -1, gate_timestep.shape[-1]
        )

        return scale_shift_timestep, gate_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        projection: Optional[PixArtAlphaTextProjection],
        target_dim: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Project and reshape text context.

        Args:
            context: [B, L_ctx, caption_channels] raw text embeddings
            projection: PixArtAlphaTextProjection module (None if caption_proj_before_connector)
            target_dim: Target hidden dimension
            batch_size: Batch size

        Returns:
            [B, L_ctx, target_dim] projected context
        """
        if projection is not None:
            context = projection(context)
        context = context.view(batch_size, -1, target_dim)
        return context

    def _prepare_learned_memory(
        self,
        video_memory: Optional[torch.Tensor],
        audio_memory: Optional[torch.Tensor],
        color_memory: Optional[torch.Tensor] = None,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.learned_memory_enabled:
            return None, None, None
        vm = None
        am = None
        cm = None
        if video_memory is not None and self.learned_memory_video_encoder is not None:
            vm = self.learned_memory_video_encoder(
                video_memory.to(device=device, dtype=dtype)
            )
        if audio_memory is not None and self.learned_memory_audio_encoder is not None:
            am = self.learned_memory_audio_encoder(
                audio_memory.to(device=device, dtype=dtype)
            )
        if color_memory is not None and self.config.learned_memory_color_film_enabled:
            cm = color_memory.to(device=device, dtype=dtype)
        return vm, am, cm

    def _prepare_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        x_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Prepare attention mask (convert bool/int to float mask)."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    # ================================================================
    # Output Processing
    # ================================================================

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: nn.LayerNorm,
        proj_out: nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output with timestep-conditioned modulation.

        Matches LTXModel._process_output exactly.

        Args:
            scale_shift_table: [2, D] learnable parameters
            norm_out: LayerNorm (elementwise_affine=False)
            proj_out: Linear projection to output channels
            x: [B, T, D] transformer output
            embedded_timestep: [B, L, D] from AdaLN

        Returns:
            [B, T, out_channels] projected output
        """
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
            + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    # ================================================================
    # Forward Pass
    # ================================================================

    def forward(
        self,
        video_latent: torch.Tensor = None,
        audio_latent: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        video_context: torch.Tensor = None,
        audio_context: torch.Tensor = None,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        audio_timesteps: Optional[torch.Tensor] = None,
        learned_memory_video: Optional[torch.Tensor] = None,
        learned_memory_audio: Optional[torch.Tensor] = None,
        learned_memory_color: Optional[torch.Tensor] = None,
        masks: Optional[Dict[str, Any]] = None,
        *,
        _kv_cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (training only).

        Args:
            video_latent: [B, F_v, C, H, W] video latent
            audio_latent: [B, F_a, C] audio latent
            timesteps: [B] or [B, F_v] diffusion timesteps (sigma values)
            video_context: [B, L_ctx, caption_channels] video text context
            audio_context: [B, L_ctx, caption_channels] audio text context
            video_context_mask: Optional attention mask for video context
            audio_context_mask: Optional attention mask for audio context
            audio_timesteps: [B] or [B, F_a] audio timesteps (optional, defaults to video)
            masks: Pre-computed causal masks (from build_all_causal_masks)

        Returns:
            (video_velocity, audio_velocity): Velocity predictions
        """
        # KV-cache routing: delegate to forward_inference when called with
        # _kv_cache_kwargs.  This ensures nested FSDP hooks are triggered
        # properly through __call__.
        if _kv_cache_kwargs is not None:
            return self.forward_inference(**_kv_cache_kwargs)

        B = video_latent.shape[0]
        device = video_latent.device
        hidden_dtype = video_latent.dtype

        # === Patch Embedding ===
        # Video: [B, F, C, H, W] → patchify → [B, T, C] → project → [B, T, D]
        B_v, F_v, C_v, H_v, W_v = video_latent.shape
        video_flat = video_latent.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        video_flat = video_flat.reshape(B_v, C_v, -1).permute(0, 2, 1)  # [B, F*H*W, C]
        video_x = self.patchify_proj(video_flat)  # [B, T, D]
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        # Audio: [B, F_a, C] -> [B, F_a, audio_dim]
        audio_x = self.audio_patchify_proj(audio_latent)
        F_a_original = audio_x.shape[1]
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # === Context Projection (moved before sink tokens for text conditioning) ===
        video_ctx = self._prepare_context(
            video_context, self.caption_projection, self.config.video_dim, B
        )
        audio_ctx = self._prepare_context(
            audio_context, self.audio_caption_projection, self.config.audio_dim, B
        )
        video_memory, audio_memory, color_memory = self._prepare_learned_memory(
            learned_memory_video,
            learned_memory_audio,
            learned_memory_color,
            device=device,
            dtype=hidden_dtype,
        )

        # === Prepend Audio Sink Tokens ===
        num_sink = self.config.num_audio_sink_tokens
        if num_sink > 0:
            sink_expanded = self.audio_sink_tokens.expand(B, -1, -1).to(audio_x.dtype)

            # Text-conditioned modulation: make sink tokens sample-dependent
            if self.config.condition_sink_on_text and hasattr(self, 'sink_text_condition'):
                # Pool audio context: [B, L_ctx, audio_dim] -> [B, audio_dim]
                ctx_pooled = audio_ctx.mean(dim=1)
                # Generate scale + shift: [B, 2 * audio_dim]
                scale_shift = self.sink_text_condition(ctx_pooled)
                scale, shift = scale_shift.chunk(2, dim=-1)  # each [B, audio_dim]
                # Modulate: broadcast [B, 1, audio_dim] over [B, num_sink, audio_dim]
                sink_expanded = sink_expanded * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

            audio_x = torch.cat([sink_expanded, audio_x], dim=1)

        # Prepare attention masks
        video_context_mask = self._prepare_attention_mask(video_context_mask, hidden_dtype)
        audio_context_mask = self._prepare_attention_mask(audio_context_mask, hidden_dtype)

        # === Timestep Embedding via AdaLayerNormSingle ===
        # Video timestep
        video_ts = timesteps  # [B] or [B, F_v]
        video_timestep_6d, video_embedded_ts = self._prepare_timestep(
            video_ts, self.adaln_single, B, hidden_dtype
        )

        # Audio timestep (defaults to video timestep if not provided)
        audio_ts = audio_timesteps if audio_timesteps is not None else timesteps

        # Expand audio timestep with sink entries (same as first frame's timestep)
        if num_sink > 0 and audio_ts.ndim == 2:
            # audio_ts is [B, F_a], prepend num_sink copies of first frame's value
            sink_ts = audio_ts[:, :1].expand(-1, num_sink)  # [B, num_sink]
            audio_ts_expanded = torch.cat([sink_ts, audio_ts], dim=1)  # [B, num_sink + F_a]
        else:
            audio_ts_expanded = audio_ts

        # Save original audio embedded timestep (without sinks) for output
        audio_timestep_6d, audio_embedded_ts_full = self._prepare_timestep(
            audio_ts_expanded, self.audio_adaln_single, B, hidden_dtype
        )
        # Strip sink entries from embedded timestep for output processing
        if num_sink > 0 and audio_embedded_ts_full.shape[1] > 1:
            audio_embedded_ts = audio_embedded_ts_full[:, num_sink:]
        else:
            audio_embedded_ts = audio_embedded_ts_full

        # === Cross-Attention Timesteps ===
        video_cross_ss, video_cross_gate = self._prepare_cross_attention_timestep(
            video_ts,
            self.av_ca_video_scale_shift_adaln_single,
            self.av_ca_a2v_gate_adaln_single,
            B, hidden_dtype,
        )
        audio_cross_ss, audio_cross_gate = self._prepare_cross_attention_timestep(
            audio_ts_expanded,
            self.av_ca_audio_scale_shift_adaln_single,
            self.av_ca_v2a_gate_adaln_single,
            B, hidden_dtype,
        )

        # === Prompt Timestep for Text Cross-Attention AdaLN (LTX-2.3) ===
        # prompt_timestep modulates the TEXT context (key/value), NOT video tokens.
        # It must be [B, 1, 2*D] to broadcast against [B, L_ctx, D].
        # When timestep is per-frame [B, F_v], we take the mean across frames
        # so that all text tokens share a single global modulation.
        video_prompt_ts = None
        audio_prompt_ts = None
        if self.cross_attention_adaln:
            # Reduce per-frame timestep to scalar for text modulation
            video_ts_scalar = video_ts.mean(dim=-1, keepdim=True) if video_ts.ndim == 2 else video_ts
            audio_ts_scalar = audio_ts_expanded.mean(dim=-1, keepdim=True) if audio_ts_expanded.ndim == 2 else audio_ts_expanded
            video_prompt_ts, _ = self._prepare_timestep(
                video_ts_scalar, self.prompt_adaln_single, B, hidden_dtype
            )
            audio_prompt_ts, _ = self._prepare_timestep(
                audio_ts_scalar, self.audio_prompt_adaln_single, B, hidden_dtype
            )

        # === Expand per-frame timestep embeddings to per-token ===
        # When timesteps is [B, F_v] (per-frame), AdaLN output is [B, F_v, *]
        # but transformer blocks need [B, F_v*H*W, *] (per-token).
        # When timesteps is [B] (scalar), output is [B, 1, *] which broadcasts.
        # Audio has 1 token/frame so no expansion needed.
        frame_seqlen = H_v * W_v
        video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)
        # NOTE: video_prompt_ts and audio_prompt_ts are intentionally NOT expanded
        # per-token. They modulate text context [B, L_ctx, D], not video tokens,
        # so they stay as [B, 1, 2*D] to broadcast correctly.

        # Reuse wrapper-provided masks whenever available. The wrapper already
        # builds them with num_audio_sink_tokens=num_sink, so sink tokens do not
        # require an unconditional rebuild here.
        if masks is None:
            num_video_frames = video_grid_sizes[0, 0].item()
            num_audio_frames = audio_grid_sizes[0, 0].item()  # Original count (without sinks)
            # Use ACTUAL spatial dims of the current batch (H_v * W_v) instead
            # of self.config.video_frame_seqlen (frozen at __init__ to the
            # yaml-default 384). Required to support multi-resolution training/
            # inference; otherwise causal mask shape mismatches the token grid.
            mask_config = CausalMaskConfig(
                video_frame_seqlen=frame_seqlen,
                num_frame_per_block=self.config.num_frame_per_block,
                num_audio_sink_tokens=num_sink,
            )
            masks = build_all_causal_masks(
                num_video_frames, num_audio_frames,
                config=mask_config,
                device=device,
            )

        # === Compute log-ratio scales for causal attention rescaling ===
        log_scales = None
        if self.config.enable_causal_log_rescale:
            blocks = compute_av_blocks(
                F_v, self.config.num_frame_per_block,
                num_frame_per_block_first=self.config.num_frame_per_block_first,
            )
            log_scales = compute_causal_log_scales(
                blocks,
                video_frame_seqlen=frame_seqlen,
                audio_frame_seqlen=self.config.audio_frame_seqlen,
                device=device,
                num_audio_sink_tokens=num_sink,
            )

        # === Precompute RoPE ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            rope_extrapolation=getattr(self.config, "rope_extrapolation", "off"),
            rope_train_max_seconds=getattr(self.config, "rope_train_max_seconds", 8.0),
            device=device, dtype=video_x.dtype,
            num_attention_heads=self.config.video_heads,
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            rope_extrapolation=getattr(self.config, "rope_extrapolation", "off"),
            rope_train_max_seconds=getattr(self.config, "rope_train_max_seconds", 8.0),
            device=device, dtype=audio_x.dtype,
            is_audio=True,
            num_attention_heads=self.config.audio_heads,
        )

        # Prepend identity RoPE for sink tokens (cos=1, sin=0 → no rotation)
        if num_sink > 0:
            if self.config.rope_type == CausalRopeType.SPLIT:
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
                audio_rope_dim = audio_pe[0].shape[-1]
                sink_cos = torch.ones(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[0].dtype)
                sink_sin = torch.zeros(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[1].dtype)
                audio_pe = (
                    torch.cat([sink_cos, audio_pe[0]], dim=1),
                    torch.cat([sink_sin, audio_pe[1]], dim=1),
                )

        # === Cross-attention RoPE ===
        # Original uses 1D temporal-only positions at audio_cross_attention_dim (2048).
        # cross_pe_max_pos = max(pe_max_pos[0], audio_pe_max_pos[0]) = max(20, 20) = 20
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video cross-PE: temporal positions from video grid (1D, video temporal)
        video_temporal_grid = torch.tensor(
            [[F_v]], device=device, dtype=torch.long
        )  # [1, 1]
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            rope_extrapolation=getattr(self.config, "rope_extrapolation", "off"),
            rope_train_max_seconds=getattr(self.config, "rope_train_max_seconds", 8.0),
            device=device, dtype=video_x.dtype,
            is_audio=False,  # Video temporal conversion
            num_attention_heads=self.config.audio_heads,
        )
        # Expand temporal PE to full video sequence: each frame's tokens share same temporal PE
        if self.config.rope_type == CausalRopeType.SPLIT:
            # SPLIT: video_cross_pe shape is (B, H, F_v, D_half) → need (B, H, F_v*seqlen, D_half)
            b, h, f_v, d_half = video_cross_pe[0].shape
            video_cross_pe = (
                video_cross_pe[0].unsqueeze(3).expand(-1, -1, -1, self.config.video_frame_seqlen, -1)
                .reshape(b, h, -1, d_half),
                video_cross_pe[1].unsqueeze(3).expand(-1, -1, -1, self.config.video_frame_seqlen, -1)
                .reshape(b, h, -1, d_half),
            )
        else:
            # INTERLEAVED: video_cross_pe shape is (B, F_v, D) → need (B, F_v*seqlen, D)
            video_cross_pe = (
                video_cross_pe[0].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
                .reshape(1, -1, video_cross_pe[0].shape[-1]),
                video_cross_pe[1].unsqueeze(2).expand(-1, -1, self.config.video_frame_seqlen, -1)
                .reshape(1, -1, video_cross_pe[1].shape[-1]),
            )

        # Audio cross-PE: temporal positions from audio grid (1D, audio temporal)
        # Use original audio frame count (without sinks) for cross-PE computation
        audio_temporal_grid = torch.tensor(
            [[F_a_original]], device=device, dtype=torch.long
        )  # [1, 1]
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            rope_extrapolation=getattr(self.config, "rope_extrapolation", "off"),
            rope_train_max_seconds=getattr(self.config, "rope_train_max_seconds", 8.0),
            device=device, dtype=audio_x.dtype,
            is_audio=True,  # Audio temporal conversion
            num_attention_heads=self.config.audio_heads,
        )

        # Prepend identity RoPE for sink tokens in cross-PE
        if num_sink > 0:
            if self.config.rope_type == CausalRopeType.SPLIT:
                # SPLIT: shape (B, H, T, D_half)
                b, h, _, d_half = audio_cross_pe[0].shape
                sink_cross_cos = torch.ones(b, h, num_sink, d_half, device=device, dtype=audio_cross_pe[0].dtype)
                sink_cross_sin = torch.zeros(b, h, num_sink, d_half, device=device, dtype=audio_cross_pe[1].dtype)
                audio_cross_pe = (
                    torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=2),
                    torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=2),
                )
            else:
                cross_rope_dim = audio_cross_pe[0].shape[-1]
                sink_cross_cos = torch.ones(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[0].dtype)
                sink_cross_sin = torch.zeros(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[1].dtype)
                audio_cross_pe = (
                    torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=1),
                    torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=1),
                )

        # === [Ulysses SP] Sequence-Parallel sharding (entry) ===
        # Slice every per-token tensor along the sequence dim into the local
        # SP shard. Tensors that stay full on every rank (block_mask,
        # cross_causal_mask, log_scales, text context, video/audio_context_mask,
        # and the *_embedded_ts used at the output stage) are intentionally
        # NOT split: attention layers internally all-to-all back to the full
        # sequence before applying the full-shape ones, and embedded_ts is
        # only consumed after gather() at the output stage.
        #
        # Numerics guarantee: when sp_size == 1 every helper is a no-op so
        # this branch is bit-equal with the original model on a single rank.
        #
        # Padding policy (causal training path):
        #   - VIDEO: still requires V_total % sp_size == 0. Tuning H/W on a
        #     high-resolution latent grid is the natural fix; padding video
        #     would either break the FlexAttention BlockMask structure or
        #     shift frame-id RoPE for downstream blocks.
        #   - AUDIO: end-padded automatically. Audio sequences are short and
        #     rarely divisible by sp_size. We rely on the BlockMask's
        #     internal mod-128 padded_length having enough headroom (sp_size
        #     << 128) so that A_padded fits without rebuilding the mask;
        #     ``flex_attention_forward`` already pads q/k/v to mod 128 and
        #     unpads back, so the mask Just Works as long as A_padded ≤
        #     padded_length. The dense a2v / v2a masks and the per-Q
        #     log-scales are extended to A_padded with neutral values
        #     (False rows/cols, log-scale = 1.0). PE pad uses identity
        #     rotation (cos=1, sin=0). Padded query rows are dropped by the
        #     post-gather unpad below.
        sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
        audio_pad_extra = 0
        A_total_real = audio_x.shape[1]
        if sp_size > 1:
            V_total = video_x.shape[1]
            A_total = audio_x.shape[1]
            if V_total % sp_size != 0:
                raise RuntimeError(
                    f"[CausalLTXModel.forward] video token count must be "
                    f"divisible by sp_size={sp_size}, got V_total={V_total} "
                    f"(F*H*W={F_v}*{H_v}*{W_v}). Recommended fix: pick H or "
                    f"W so that F*H*W % sp_size == 0. Auto-padding for "
                    f"video is intentionally disabled because the "
                    f"FlexAttention BlockMask and frame-id RoPE are "
                    f"layout-sensitive; use the bidirectional model path if "
                    f"you need automatic length alignment."
                )

            # ── Audio end-pad (NEW) ───────────────────────────────────────
            if A_total % sp_size != 0:
                audio_pad_extra = sp_size - (A_total % sp_size)
                extra = audio_pad_extra

                # 1) audio_x — zero pad (padded query rows are discarded
                #    post-gather; their attention output is never read).
                B_a, _, D_a = audio_x.shape
                audio_x = torch.cat([
                    audio_x,
                    torch.zeros((B_a, extra, D_a), device=audio_x.device, dtype=audio_x.dtype),
                ], dim=1)

                # 2) audio_pe / audio_cross_pe — identity rotation pad.
                def _pad_pe(pe):
                    cos, sin = pe
                    cos_pad = torch.ones(
                        (cos.shape[0], extra, cos.shape[2]),
                        device=cos.device, dtype=cos.dtype,
                    )
                    sin_pad = torch.zeros(
                        (sin.shape[0], extra, sin.shape[2]),
                        device=sin.device, dtype=sin.dtype,
                    )
                    return (
                        torch.cat([cos, cos_pad], dim=1),
                        torch.cat([sin, sin_pad], dim=1),
                    )

                audio_pe = _pad_pe(audio_pe)
                audio_cross_pe = _pad_pe(audio_cross_pe)

                # 3) Per-token AdaLN tensors — replicate last frame to avoid
                #    NaN propagation through scale/shift; padded rows are
                #    discarded post-gather anyway.
                def _pad_repeat_last(t):
                    if t.shape[1] <= 1:
                        return t  # broadcast tensor stays as-is
                    last = t[:, -1:].expand(-1, extra, *([-1] * (t.ndim - 2)))
                    return torch.cat([t, last.contiguous()], dim=1)

                audio_timestep_6d = _pad_repeat_last(audio_timestep_6d)
                audio_cross_ss = _pad_repeat_last(audio_cross_ss)

                # 4) audio_cross_gate — zero gate so the cross-attn output
                #    contribution at padded query rows is exactly zero.
                if audio_cross_gate.shape[1] > 1:
                    B_g, _, D_g = audio_cross_gate.shape
                    audio_cross_gate = torch.cat([
                        audio_cross_gate,
                        torch.zeros((B_g, extra, D_g),
                                    device=audio_cross_gate.device,
                                    dtype=audio_cross_gate.dtype),
                    ], dim=1)

                # 5) Per-Q log-scales — 1.0 is the neutral scale (no
                #    rescaling on padded queries; their outputs are
                #    discarded). audio_self_scale & v2a_scale are indexed
                #    by audio Q position; a2v_scale is indexed by video Q
                #    and stays untouched.
                if log_scales is not None:
                    def _pad_log_scale(s):
                        # shape [1, T_a, 1]
                        pad = torch.ones(
                            (s.shape[0], extra, s.shape[2]),
                            device=s.device, dtype=s.dtype,
                        )
                        return torch.cat([s, pad], dim=1)

                    log_scales = dict(log_scales)
                    log_scales['audio_self_scale'] = _pad_log_scale(
                        log_scales['audio_self_scale']
                    )
                    log_scales['v2a_scale'] = _pad_log_scale(
                        log_scales['v2a_scale']
                    )

                # 6) Cross-attention dense masks — extend the audio dim
                #    with all-False (padded slots cannot attend / be
                #    attended to). a2v mask shape: [T_v, T_a]; v2a shape:
                #    [T_a, T_v]. ``masks`` may be ``None`` when masks were
                #    not requested.
                if masks is not None:
                    a2v_mask = masks.get('a2v')
                    if a2v_mask is not None and a2v_mask.dim() >= 2:
                        # Pad the LAST dim (audio K dim).
                        false_pad = torch.zeros(
                            (*a2v_mask.shape[:-1], extra),
                            device=a2v_mask.device, dtype=a2v_mask.dtype,
                        )
                        masks = dict(masks)
                        masks['a2v'] = torch.cat([a2v_mask, false_pad], dim=-1)
                    v2a_mask = masks.get('v2a') if isinstance(masks, dict) else None
                    if v2a_mask is not None and v2a_mask.dim() >= 2:
                        # Pad the SECOND-TO-LAST dim (audio Q dim).
                        # v2a shape: [..., T_a, T_v]. We append along dim=-2.
                        pad_shape = list(v2a_mask.shape)
                        pad_shape[-2] = extra
                        false_pad = torch.zeros(
                            pad_shape,
                            device=v2a_mask.device, dtype=v2a_mask.dtype,
                        )
                        if not isinstance(masks, dict):
                            masks = dict(masks)
                        masks['v2a'] = torch.cat([v2a_mask, false_pad], dim=-2)

                # 7) audio_self BlockMask — NOT rebuilt. We rely on the
                #    mask's internal mod-128 padded_length having enough
                #    headroom (sp_size << 128). ``flex_attention_forward``
                #    pads q/k/v to mod 128 and unpads back, so as long as
                #    A_padded ≤ padded_length the mask Just Works. The
                #    pre-existing mask rule ``ends[total_tokens:] =
                #    total_tokens`` keeps the padded query rows benign
                #    (they may attend real tokens, output is discarded).

            # Token streams
            video_x = _split_sequence(video_x, dim=1, sp_size=sp_size)
            audio_x = _split_sequence(audio_x, dim=1, sp_size=sp_size)

            # RoPE (cos, sin) for self- and cross-attention
            video_pe = (
                _split_sequence(video_pe[0], dim=1, sp_size=sp_size),
                _split_sequence(video_pe[1], dim=1, sp_size=sp_size),
            )
            audio_pe = (
                _split_sequence(audio_pe[0], dim=1, sp_size=sp_size),
                _split_sequence(audio_pe[1], dim=1, sp_size=sp_size),
            )
            video_cross_pe = (
                _split_sequence(video_cross_pe[0], dim=1, sp_size=sp_size),
                _split_sequence(video_cross_pe[1], dim=1, sp_size=sp_size),
            )
            audio_cross_pe = (
                _split_sequence(audio_cross_pe[0], dim=1, sp_size=sp_size),
                _split_sequence(audio_cross_pe[1], dim=1, sp_size=sp_size),
            )

            # Per-token AdaLN tensors. Scalar/broadcast variants ([B, 1, *])
            # stay untouched and broadcast naturally over the local shard.
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

        # === Prepare Transformer Args ===
        video_args = CausalTransformerArgs(
            x=video_x,
            timesteps=video_timestep_6d,
            positional_embeddings=video_pe,
            context=video_ctx,
            context_mask=video_context_mask,
            memory=video_memory,
            memory_color=color_memory,
            block_mask=masks.get('video_self'),
            cross_causal_mask=masks.get('a2v'),
            cross_positional_embeddings=video_cross_pe,
            cross_scale_shift_timestep=video_cross_ss,
            cross_gate_timestep=video_cross_gate,
            prompt_timestep=video_prompt_ts,
            self_attn_log_scale=log_scales['video_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['a2v_scale'].to(hidden_dtype) if log_scales else None,
        )

        audio_args = CausalTransformerArgs(
            x=audio_x,
            timesteps=audio_timestep_6d,
            positional_embeddings=audio_pe,
            context=audio_ctx,
            context_mask=audio_context_mask,
            memory=audio_memory,
            block_mask=masks.get('audio_self'),
            cross_causal_mask=masks.get('v2a'),
            cross_positional_embeddings=audio_cross_pe,
            cross_scale_shift_timestep=audio_cross_ss,
            cross_gate_timestep=audio_cross_gate,
            prompt_timestep=audio_prompt_ts,
            self_attn_log_scale=log_scales['audio_self_scale'].to(hidden_dtype) if log_scales else None,
            cross_attn_log_scale=log_scales['v2a_scale'].to(hidden_dtype) if log_scales else None,
        )

        # === Transformer Blocks ===
        _grad_diag = (
            self.training
            and os.environ.get("LTX_GRAD_DIAG", "0") == "1"
            and int(os.environ.get("LOCAL_RANK", "0")) == 0
        )

        def _make_block_hook(tag):
            """Register on hidden_states between blocks to catch NaN gradient."""
            def _bh(grad):
                if grad is None:
                    return
                g = grad.detach()
                total = g.numel()
                nan_c = int(torch.isnan(g).sum().item())
                inf_c = int(torch.isinf(g).sum().item())
                if nan_c or inf_c:
                    finite = g[torch.isfinite(g)]
                    fa = float(finite.abs().max().item()) if finite.numel() > 0 else float('nan')
                    _diag_logger.warning(
                        f"GradDiag block_bwd {tag} !! nan={nan_c}/{total} "
                        f"inf={inf_c}/{total} finite_absmax={fa:.4e}"
                    )
                else:
                    fa = float(g.abs().max().item())
                    _diag_logger.warning(
                        f"GradDiag block_bwd {tag} OK absmax={fa:.4e}"
                    )
            return _bh

        for i, block in enumerate(self.transformer_blocks):
            if self.gradient_checkpointing and self.training:
                video_args, audio_args = torch.utils.checkpoint.checkpoint(
                    block, video_args, audio_args,
                    use_reentrant=False,
                )
            else:
                video_args, audio_args = block(video_args, audio_args)

        # === [Ulysses SP] Gather full sequence before output projection ===
        # The output stage (norm/scale/shift/proj_out + unpatchify) operates on
        # the FULL sequence, and embedded_ts kept its full shape on every rank.
        # Gather is autograd-friendly (backward = local slice).
        if sp_size > 1:
            video_args.x = _gather_sequence(video_args.x, dim=1, sp_size=sp_size)
            audio_args.x = _gather_sequence(audio_args.x, dim=1, sp_size=sp_size)

        # ── Audio SP-pad strip (NEW) ──────────────────────────────────────
        # Drop the trailing pad rows we appended at the SP entry. The
        # padded rows received zero contribution from cross-attn (gate=0),
        # benign self-attn output (their queries attend real K with
        # identity-rotation RoPE), and downstream norm/proj would otherwise
        # see junk; sink-strip and unpatchify expect the original A_total.
        if audio_pad_extra > 0:
            audio_args.x = audio_args.x[:, :A_total_real]

        # === Strip sink tokens before output processing ===
        audio_out_x = audio_args.x
        if num_sink > 0:
            audio_out_x = audio_out_x[:, num_sink:]

        # === Output Layer (with timestep conditioning) ===
        video_out = self._process_output(
            self.scale_shift_table, self.norm_out, self.proj_out,
            video_args.x, video_embedded_ts,
        )
        audio_out = self._process_output(
            self.audio_scale_shift_table, self.audio_norm_out, self.audio_proj_out,
            audio_out_x, audio_embedded_ts,
        )

        # === Unpatchify Video ===
        # [B, T, C] → [B, F, H, W, C] → [B, F, C, H, W]
        F, H, W = F_v, H_v, W_v
        C_out = self.config.out_channels
        video_out = video_out.reshape(B, F, H, W, C_out)
        video_out = video_out.permute(0, 1, 4, 2, 3)  # [B, F, C, H, W]

        return video_out, audio_out

    # ================================================================
    # KV-Cache Inference (delegates to kv_cache.py — no training code modified)
    # ================================================================

    def forward_inference(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        timesteps: torch.Tensor,
        audio_timesteps: Optional[torch.Tensor] = None,
        video_context: Optional[torch.Tensor] = None,
        audio_context: Optional[torch.Tensor] = None,
        video_context_mask: Optional[torch.Tensor] = None,
        audio_context_mask: Optional[torch.Tensor] = None,
        learned_memory_video: Optional[torch.Tensor] = None,
        learned_memory_audio: Optional[torch.Tensor] = None,
        learned_memory_color: Optional[torch.Tensor] = None,
        kv_cache=None,
        video_start_frame: int = 0,
        audio_start_frame: int = 0,
        include_audio_sinks: bool = True,
        pyramid_policy=None,
        kv_cache_only: bool = False,
    ):
        """KV-cache inference entry point.

        All logic lives in ``ltx_causal.transformer.kv_cache`` to keep this
        file unchanged for training.  See :func:`model_forward_inference` for
        full documentation.
        """
        from ltx_causal.transformer.kv_cache import model_forward_inference

        return model_forward_inference(
            model=self,
            video_latent=video_latent,
            audio_latent=audio_latent,
            timesteps=timesteps,
            audio_timesteps=audio_timesteps,
            video_context=video_context,
            audio_context=audio_context,
            video_context_mask=video_context_mask,
            audio_context_mask=audio_context_mask,
            learned_memory_video=learned_memory_video,
            learned_memory_audio=learned_memory_audio,
            learned_memory_color=learned_memory_color,
            kv_cache=kv_cache,
            video_start_frame=video_start_frame,
            audio_start_frame=audio_start_frame,
            include_audio_sinks=include_audio_sinks,
            pyramid_policy=pyramid_policy,
            kv_cache_only=kv_cache_only,
        )

    # ================================================================
    # Model Loading
    # ================================================================

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[CausalLTXModelConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "CausalLTXModel":
        """
        Load model from pretrained LTX-2 checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config: Model configuration (uses defaults if None)
            device: Target device
            dtype: Model dtype

        Returns:
            Initialized CausalLTXModel with loaded weights
        """
        if config is None:
            config = CausalLTXModelConfig()

        model = cls(config)
        model = model.to(device=device, dtype=dtype)

        # Load checkpoint
        if checkpoint_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, map_location=device)

        # Load with strict=False to ignore mask_builder and optional sink-token keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        # Verify only expected keys are missing
        expected_missing = ['mask_builder']
        if config.num_audio_sink_tokens > 0:
            expected_missing.append('audio_sink_tokens')
        real_missing = [
            key for key in missing
            if not any(pat in key for pat in expected_missing)
        ]
        if real_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint: {real_missing[:10]}. "
                f"Total missing: {len(real_missing)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        if unexpected:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {unexpected[:10]}. "
                f"Total unexpected: {len(unexpected)}. "
                f"This likely means the checkpoint is incompatible with CausalLTXModel."
            )

        return model

    @staticmethod
    def _expand_per_frame_to_per_token(
        per_frame: torch.Tensor,
        frame_seqlen: int,
    ) -> torch.Tensor:
        """Expand per-frame tensor to per-token by repeating each frame's value.

        When timesteps is [B, F] (per-frame), AdaLN outputs [B, F, D].
        But transformer blocks expect [B, F*H*W, D] (per-token).
        Each frame's H*W tokens share the same timestep embedding.

        When timesteps is [B] (scalar), AdaLN outputs [B, 1, D] which
        broadcasts naturally, so no expansion is needed.

        Args:
            per_frame: [B, F, D] per-frame values
            frame_seqlen: Number of tokens per frame (H*W for video, 1 for audio)

        Returns:
            [B, F*frame_seqlen, D] per-token values
        """
        if per_frame.shape[1] <= 1 or frame_seqlen <= 1:
            return per_frame
        B, F, D = per_frame.shape
        return (
            per_frame.unsqueeze(2)
            .expand(-1, -1, frame_seqlen, -1)
            .reshape(B, F * frame_seqlen, D)
        )

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
