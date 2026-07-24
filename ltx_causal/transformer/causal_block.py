"""
CausalAVTransformerBlock: Transformer block with causal attention for LTX-2.

This module implements a single transformer block with 6 attention types,
properly adapted for causal generation:

1. attn1 (video_self): Causal block mask - same block + previous blocks
2. attn2 (video_text): No causality needed - text is fixed
3. audio_attn1 (audio_self): Causal block mask
4. audio_attn2 (audio_text): No causality needed
5. audio_to_video_attn (A2V): Timestamp-based causal mask
6. video_to_audio_attn (V2A): Timestamp-based causal mask

Weight-compatible with original BasicAVTransformerBlock.
"""

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ltx_causal.attention.causal_attention import CausalLTXAttention
from ltx_causal.rope.causal_rope import CausalRopeType
from ltx_causal.transformer.compat import FeedForward

# Try to import BlockMask type
try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TransformerConfig:
    """Configuration for a transformer branch (video or audio)."""
    dim: int
    heads: int
    d_head: int
    context_dim: int  # Text context dimension
    cross_attention_adaln: bool = False  # LTX-2.3: AdaLN for text cross-attention
    apply_gated_attention: bool = False  # Per-head gated attention
    learned_memory_enabled: bool = False
    learned_memory_mode: str = "cross_attn_adapter"
    learned_memory_dim: int = 512
    learned_memory_heads: int = 8
    learned_memory_color_film_enabled: bool = False
    learned_memory_color_condition_dim: int = 512
    learned_memory_color_film_hidden_dim: int = 256


@dataclass
class CausalTransformerArgs:
    """
    Arguments for causal transformer forward pass (training only).
    """
    x: torch.Tensor                           # Hidden states [B, L, D]
    timesteps: torch.Tensor                   # Timestep embeddings for AdaLN
    positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # RoPE
    context: Optional[torch.Tensor] = None    # Text context
    context_mask: Optional[torch.Tensor] = None
    memory: Optional[torch.Tensor] = None     # Learned long-memory tokens [B, N, Dm]
    memory_color: Optional[torch.Tensor] = None  # Color memory vector [B, Dc]
    enabled: bool = True

    # Cross-attention RoPE (for A2V/V2A)
    cross_positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    cross_scale_shift_timestep: Optional[torch.Tensor] = None
    cross_gate_timestep: Optional[torch.Tensor] = None

    # Text cross-attention AdaLN (LTX-2.3)
    prompt_timestep: Optional[torch.Tensor] = None

    # Causal masks (training only)
    block_mask: Optional["BlockMask"] = None  # For self-attention
    cross_causal_mask: Optional[torch.Tensor] = None  # For cross-attention

    # Log-ratio scales for causal attention output (entropy-aligned rescaling)
    self_attn_log_scale: Optional[torch.Tensor] = None   # [1, L, 1]
    cross_attn_log_scale: Optional[torch.Tensor] = None  # [1, L, 1]


# FeedForward imported from compat.py (uses GELUApprox for weight-compatible state_dict keys)

# ============================================================================
# RMS Normalization
# ============================================================================

def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMS normalization.

    Uses torch.nn.functional.rms_norm to match the original LTX-2 implementation
    (ltx_core.utils.rms_norm). The manual implementation (x * rsqrt(mean(x²) + eps))
    can produce different results under bf16 due to intermediate precision handling.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=None, eps=eps)


# ============================================================================
# Causal AV Transformer Block
# ============================================================================

class LearnedMemoryAttention(nn.Module):
    """Low-rank cross-attention adapter for detached long-memory tokens.

    The output projection is zero-initialized, so enabling the module is an
    exact no-op at initialization while still allowing the projection to learn
    on the first optimizer steps.
    """

    def __init__(
        self,
        query_dim: int,
        memory_dim: int,
        heads: int,
    ):
        super().__init__()
        heads = max(1, int(heads))
        if memory_dim % heads != 0:
            raise ValueError(
                f"learned memory dim ({memory_dim}) must be divisible by heads ({heads})"
            )
        self.heads = heads
        self.head_dim = memory_dim // heads
        self.to_q = nn.Linear(query_dim, memory_dim, bias=False)
        self.to_k = nn.Linear(memory_dim, memory_dim, bias=False)
        self.to_v = nn.Linear(memory_dim, memory_dim, bias=False)
        self.to_out = nn.Linear(memory_dim, query_dim, bias=True)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if memory is None or memory.numel() == 0:
            return x.new_zeros(x.shape)
        if memory.shape[0] == 1 and x.shape[0] > 1:
            memory = memory.expand(x.shape[0], -1, -1)
        memory = memory.to(device=x.device, dtype=x.dtype)
        q = self.to_q(x)
        k = self.to_k(memory)
        v = self.to_v(memory)

        def _split_heads(t: torch.Tensor) -> torch.Tensor:
            b, n, d = t.shape
            return t.view(b, n, self.heads, self.head_dim).transpose(1, 2)

        q = _split_heads(q)
        k = _split_heads(k)
        v = _split_heads(v)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        return self.to_out(out)


class LearnedMemoryColorFiLM(nn.Module):
    """Zero-initialized FiLM adapter driven by low-frequency color memory."""

    def __init__(
        self,
        condition_dim: int,
        query_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(condition_dim), hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, query_dim * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        color: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if color is None or color.numel() == 0:
            return x.new_zeros(x.shape)
        if color.shape[0] == 1 and x.shape[0] > 1:
            color = color.expand(x.shape[0], -1)
        color = color.to(device=x.device, dtype=x.dtype)
        scale, shift = self.net(color).chunk(2, dim=-1)
        return x * scale.unsqueeze(1) + shift.unsqueeze(1)


class CausalAVTransformerBlock(nn.Module):
    """
    Causal Audio-Video Transformer Block.

    Contains 6 attention mechanisms with proper causality:
    - Video self-attention (causal)
    - Video-text cross-attention (non-causal, text is fixed)
    - Audio self-attention (causal)
    - Audio-text cross-attention (non-causal, text is fixed)
    - Audio-to-Video cross-attention (timestamp causal)
    - Video-to-Audio cross-attention (timestamp causal)

    Weight-compatible with original BasicAVTransformerBlock.
    """

    def __init__(
        self,
        idx: int,
        video: Optional[TransformerConfig] = None,
        audio: Optional[TransformerConfig] = None,
        rope_type: CausalRopeType = CausalRopeType.SPLIT,
        norm_eps: float = 1e-6,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = 16,
        sink_size: int = 1,
    ):
        """
        Initialize transformer block.

        Args:
            idx: Block index
            video: Video branch configuration
            audio: Audio branch configuration
            rope_type: Type of RoPE
            norm_eps: Epsilon for normalization
        """
        super().__init__()
        self.idx = idx
        self.norm_eps = norm_eps

        # Diagnostic: when True, forward() collects gate/scale stats into _gate_stats
        self._store_gate_stats = True
        self._gate_stats = {}

        # Curriculum learning: skip A2V/V2A cross-modal attention
        self.skip_cross_modal_attention = False

        # Gradient stabilization: detach cross-modal context in backward to prevent
        # gradient concentration (6144 video tokens → 126 audio tokens causes 10^4-10^8× amplification)
        self.detach_cross_modal_context = False
        self.learned_memory_mode = None

        # Text cross-attention AdaLN flag
        self.cross_attention_adaln = (
            (video is not None and video.cross_attention_adaln)
            or (audio is not None and audio.cross_attention_adaln)
        )

        # Compute AdaLN coefficient: 6 (base) + 3 (cross-attn shift/scale/gate) if enabled
        adaln_coeff = 9 if self.cross_attention_adaln else 6

        # === Video Branch ===
        if video is not None:
            # Video self-attention (CAUSAL) — with temperature cooling
            self.attn1 = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=None,  # Self-attention
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Video-text cross-attention (NON-CAUSAL - text is fixed, no temperature)
            self.attn2 = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Video feed-forward
            self.ff = FeedForward(video.dim, dim_out=video.dim)

            # AdaLN parameters: 6 (base) or 9 (with cross-attention adaln)
            self.scale_shift_table = nn.Parameter(torch.empty(adaln_coeff, video.dim))
            if video.learned_memory_enabled:
                self.learned_memory_mode = str(video.learned_memory_mode)
                self.video_memory_attn = LearnedMemoryAttention(
                    query_dim=video.dim,
                    memory_dim=video.learned_memory_dim,
                    heads=video.learned_memory_heads,
                )
            else:
                self.video_memory_attn = None
            if video.learned_memory_color_film_enabled:
                self.video_color_film = LearnedMemoryColorFiLM(
                    condition_dim=video.learned_memory_color_condition_dim,
                    query_dim=video.dim,
                    hidden_dim=video.learned_memory_color_film_hidden_dim,
                )
            else:
                self.video_color_film = None

        # === Audio Branch ===
        if audio is not None:
            # Audio self-attention (CAUSAL) — with temperature cooling
            self.audio_attn1 = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=None,  # Self-attention
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )

            # Audio-text cross-attention (NON-CAUSAL, no temperature)
            self.audio_attn2 = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )

            # Audio feed-forward
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)

            # AdaLN parameters
            self.audio_scale_shift_table = nn.Parameter(torch.empty(adaln_coeff, audio.dim))
            if audio.learned_memory_enabled:
                self.learned_memory_mode = str(audio.learned_memory_mode)
                self.audio_memory_attn = LearnedMemoryAttention(
                    query_dim=audio.dim,
                    memory_dim=audio.learned_memory_dim,
                    heads=audio.learned_memory_heads,
                )
            else:
                self.audio_memory_attn = None

        # === Text Cross-Attention AdaLN Tables (LTX-2.3) ===
        if self.cross_attention_adaln:
            if video is not None:
                self.prompt_scale_shift_table = nn.Parameter(torch.empty(2, video.dim))
            if audio is not None:
                self.audio_prompt_scale_shift_table = nn.Parameter(torch.empty(2, audio.dim))

        # === Cross-Modal Attention (A2V and V2A) ===
        if audio is not None and video is not None:
            # Audio-to-Video: Q=Video, K/V=Audio (CAUSAL with timestamp mask + temperature)
            self.audio_to_video_attn = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,  # Uses audio head config
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Video-to-Audio: Q=Audio, K/V=Video (CAUSAL with timestamp mask + temperature)
            self.video_to_audio_attn = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,  # Uses audio head config
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                apply_gated_attention=audio.apply_gated_attention,
            )

            # AdaLN for cross-attention (5 values: 4 scale/shift + 1 gate)
            self.scale_shift_table_a2v_ca_audio = nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = nn.Parameter(torch.empty(5, video.dim))

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: Optional[torch.Tensor],
        timestep: torch.Tensor,
        prompt_timestep: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation (LTX-2.3).

        When cross_attention_adaln is enabled:
          - Query is modulated by shift/scale/gate from scale_shift_table[6:9]
          - Key/Value context is modulated by shift/scale from prompt_scale_shift_table
        When disabled:
          - Simple RMS norm + attention (LTX-2 behavior)

        Sequence-Parallel note:
          The text ``context`` is the same on every SP rank (text encoder
          output is broadcast to all ranks; not sequence-sharded). We pass
          ``sp_context_sharded=False`` so the attention layer narrows K/V
          along the head dim instead of running an all-to-all on context.
          When SP is disabled this argument is a no-op.
        """
        if self.cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(
                scale_shift_table, x.shape[0], timestep, slice(6, 9)
            )
            # Modulate query
            attn_input = rms_norm(x, eps=self.norm_eps) * (1 + scale_q) + shift_q
            # Modulate context key/value
            batch_size = x.shape[0]
            shift_kv, scale_kv = (
                prompt_scale_shift_table[None, None]
                .to(device=x.device, dtype=x.dtype)
                + prompt_timestep.reshape(
                    batch_size, prompt_timestep.shape[1], 2, -1
                )
            ).unbind(dim=2)
            encoder_hidden_states = context * (1 + scale_kv) + shift_kv
            return attn(
                attn_input,
                context=encoder_hidden_states,
                mask=context_mask,
                sp_context_sharded=False,
            ) * gate

        return attn(
            rms_norm(x, eps=self.norm_eps),
            context=context,
            mask=context_mask,
            sp_context_sharded=False,
        )

    def get_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        timestep: torch.Tensor,
        indices: slice,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get AdaLN values (shift, scale, gate) from scale_shift_table + timestep.
        """
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0)
            .to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)

        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        num_scale_shift_values: int = 4,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Get AdaLN values for cross-attention.
        """
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :],
            batch_size,
            scale_shift_timestep,
            slice(None, None),
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :],
            batch_size,
            gate_timestep,
            slice(None, None),
        )

        scale_shift_chunks = [t.squeeze(2) for t in scale_shift_ada_values]
        gate_ada_values = [t.squeeze(2) for t in gate_ada_values]

        return (*scale_shift_chunks, *gate_ada_values)

    def _record_grad(self, name: str, grad: torch.Tensor):
        """Record gradient norm for diagnostics (called by tensor hooks)."""
        with torch.no_grad():
            self._gate_stats[f'grad_{name}_norm'] = grad.detach().float().norm().item()
            self._gate_stats[f'grad_{name}_absmax'] = grad.detach().float().abs().max().item()

    def _apply_learned_memory(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
        adapter: Optional[LearnedMemoryAttention],
    ) -> torch.Tensor:
        if adapter is None or memory is None:
            return x
        return x + adapter(rms_norm(x, eps=self.norm_eps), memory)

    def _apply_learned_color_film(
        self,
        x: torch.Tensor,
        color: Optional[torch.Tensor],
        adapter: Optional[LearnedMemoryColorFiLM],
    ) -> torch.Tensor:
        if adapter is None or color is None:
            return x
        return x + adapter(rms_norm(x, eps=self.norm_eps), color)

    def forward(
        self,
        video: Optional[CausalTransformerArgs] = None,
        audio: Optional[CausalTransformerArgs] = None,
        *,
        _kv_cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[Optional[CausalTransformerArgs], Optional[CausalTransformerArgs]]:
        """
        Forward pass through the transformer block.

        Args:
            video: Video branch arguments
            audio: Audio branch arguments

        Returns:
            Updated (video, audio) arguments
        """
        # KV-cache routing: delegate to block_forward_with_cache when called
        # with _kv_cache_kwargs.  This ensures this block's FSDP hooks are
        # triggered through __call__.
        if _kv_cache_kwargs is not None:
            from ltx_causal.transformer.kv_cache import block_forward_with_cache
            return block_forward_with_cache(block=self, **_kv_cache_kwargs)

        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0) and not self.skip_cross_modal_attention
        run_v2a = run_ax and (video is not None and vx.numel() > 0) and not self.skip_cross_modal_attention

        # === Video Self-Attention (CAUSAL) ===
        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )

            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa

            # Causal self-attention with block_mask
            vx_attn = self.attn1(
                norm_vx,
                pe=video.positional_embeddings,
                block_mask=video.block_mask,
                logit_log_scale=video.self_attn_log_scale,
            )

            # Collect gate stats for diagnostics (detach to avoid affecting
            # gradient checkpointing saved-tensor count)
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_msa_mean'] = vgate_msa.detach().float().mean().item()
                    self._gate_stats['vgate_msa_std'] = vgate_msa.detach().float().std().item()
                    self._gate_stats['vscale_msa_mean'] = vscale_msa.detach().float().mean().item()
                    self._gate_stats['vscale_msa_std'] = vscale_msa.detach().float().std().item()
                    self._gate_stats['vshift_msa_mean'] = vshift_msa.detach().float().mean().item()
                    self._gate_stats['vx_attn_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_absmax'] = vx_attn.detach().float().abs().max().item()
                if vx_attn.requires_grad:
                    vx_attn.register_hook(lambda g, s=self: s._record_grad('vx_self_attn', g))

            vx = vx + vx_attn * vgate_msa
            if self.learned_memory_mode == "memory_kv_side_branch":
                vx = self._apply_learned_memory(
                    vx, video.memory, getattr(self, "video_memory_attn", None)
                )
                vx = self._apply_learned_color_film(
                    vx, video.memory_color, getattr(self, "video_color_film", None)
                )

            # Video-text cross-attention (non-causal, with optional AdaLN)
            vx_text_attn = self._apply_text_cross_attention(
                vx, video.context, self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps, video.prompt_timestep,
                video.context_mask,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vx_text_attn_out_norm'] = vx_text_attn.detach().float().norm().item()
                    self._gate_stats['vx_text_attn_out_absmax'] = vx_text_attn.detach().float().abs().max().item()
                if vx_text_attn.requires_grad:
                    vx_text_attn.register_hook(lambda g, s=self: s._record_grad('vx_text_attn', g))
            vx = vx + vx_text_attn
            if self.learned_memory_mode == "cross_attn_adapter":
                vx = self._apply_learned_memory(
                    vx, video.memory, getattr(self, "video_memory_attn", None)
                )
                vx = self._apply_learned_color_film(
                    vx, video.memory_color, getattr(self, "video_color_film", None)
                )

            del vshift_msa, vscale_msa, vgate_msa

        # === Audio Self-Attention (CAUSAL) ===
        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa

            # Causal self-attention
            ax_attn = self.audio_attn1(
                norm_ax,
                pe=audio.positional_embeddings,
                block_mask=audio.block_mask,
                logit_log_scale=audio.self_attn_log_scale,
            )

            # Collect audio gate stats
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_msa_mean'] = agate_msa.detach().float().mean().item()
                    self._gate_stats['agate_msa_std'] = agate_msa.detach().float().std().item()
                    self._gate_stats['ax_attn_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_absmax'] = ax_attn.detach().float().abs().max().item()
                if ax_attn.requires_grad:
                    ax_attn.register_hook(lambda g, s=self: s._record_grad('ax_self_attn', g))

            ax = ax + ax_attn * agate_msa
            if self.learned_memory_mode == "memory_kv_side_branch":
                ax = self._apply_learned_memory(
                    ax, audio.memory, getattr(self, "audio_memory_attn", None)
                )

            # Audio-text cross-attention (non-causal, with optional AdaLN)
            ax_text_attn = self._apply_text_cross_attention(
                ax, audio.context, self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps, audio.prompt_timestep,
                audio.context_mask,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['ax_text_attn_out_norm'] = ax_text_attn.detach().float().norm().item()
                    self._gate_stats['ax_text_attn_out_absmax'] = ax_text_attn.detach().float().abs().max().item()
                if ax_text_attn.requires_grad:
                    ax_text_attn.register_hook(lambda g, s=self: s._record_grad('ax_text_attn', g))
            ax = ax + ax_text_attn
            if self.learned_memory_mode == "cross_attn_adapter":
                ax = self._apply_learned_memory(
                    ax, audio.memory, getattr(self, "audio_memory_attn", None)
                )

            del ashift_msa, ascale_msa, agate_msa

        # === Cross-Modal Attention (A2V and V2A - CAUSAL) ===
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            # Get AdaLN values for cross-attention
            (
                scale_ca_audio_hidden_states_a2v,
                shift_ca_audio_hidden_states_a2v,
                scale_ca_audio_hidden_states_v2a,
                shift_ca_audio_hidden_states_v2a,
                gate_out_v2a,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_audio,
                ax.shape[0],
                audio.cross_scale_shift_timestep,
                audio.cross_gate_timestep,
            )

            (
                scale_ca_video_hidden_states_a2v,
                shift_ca_video_hidden_states_a2v,
                scale_ca_video_hidden_states_v2a,
                shift_ca_video_hidden_states_v2a,
                gate_out_a2v,
            ) = self.get_av_ca_ada_values(
                self.scale_shift_table_a2v_ca_video,
                vx.shape[0],
                video.cross_scale_shift_timestep,
                video.cross_gate_timestep,
            )

            # A2V: Video attends to Audio (with timestamp causal mask)
            if run_a2v:
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_a2v) + shift_ca_video_hidden_states_a2v
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_a2v) + shift_ca_audio_hidden_states_a2v

                # Detach context to prevent gradient concentration:
                # In A2V backward, 6144 video Q gradients would concentrate into 126 audio K/V tokens
                # causing 10^4-10^8× amplification per block. Detaching breaks this path.
                a2v_context = ax_scaled.detach() if self.detach_cross_modal_context else ax_scaled

                a2v_out = self.audio_to_video_attn(
                    vx_scaled,
                    context=a2v_context,
                    pe=video.cross_positional_embeddings,
                    k_pe=audio.cross_positional_embeddings,
                    cross_causal_mask=video.cross_causal_mask,  # A2V timestamp mask
                    logit_log_scale=video.cross_attn_log_scale,
                )

                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_a2v_mean'] = gate_out_a2v.detach().float().mean().item()
                        self._gate_stats['a2v_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_absmax'] = a2v_out.detach().float().abs().max().item()
                    if a2v_out.requires_grad:
                        a2v_out.register_hook(lambda g, s=self: s._record_grad('a2v_attn', g))

                vx = vx + a2v_out * gate_out_a2v

            # V2A: Audio attends to Video (with timestamp causal mask)
            if run_v2a:
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_hidden_states_v2a) + shift_ca_audio_hidden_states_v2a
                vx_scaled = vx_norm3 * (1 + scale_ca_video_hidden_states_v2a) + shift_ca_video_hidden_states_v2a

                # Detach context to prevent gradient feedback loop:
                # Without detach, large audio gradient → V2A backward → amplifies video gradient
                # → next block's A2V backward → further amplifies audio → exponential blowup.
                v2a_context = vx_scaled.detach() if self.detach_cross_modal_context else vx_scaled

                v2a_out = self.video_to_audio_attn(
                    ax_scaled,
                    context=v2a_context,
                    pe=audio.cross_positional_embeddings,
                    k_pe=video.cross_positional_embeddings,
                    cross_causal_mask=audio.cross_causal_mask,  # V2A timestamp mask
                    logit_log_scale=audio.cross_attn_log_scale,
                )

                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_v2a_mean'] = gate_out_v2a.detach().float().mean().item()
                        self._gate_stats['v2a_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_absmax'] = v2a_out.detach().float().abs().max().item()
                    if v2a_out.requires_grad:
                        v2a_out.register_hook(lambda g, s=self: s._record_grad('v2a_attn', g))

                ax = ax + v2a_out * gate_out_v2a

            del gate_out_a2v, gate_out_v2a

        # === Feed-Forward Networks ===
        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_mlp_mean'] = vgate_mlp.detach().float().mean().item()
                    self._gate_stats['vgate_mlp_std'] = vgate_mlp.detach().float().std().item()

            del vshift_mlp, vscale_mlp, vgate_mlp

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_mlp_mean'] = agate_mlp.detach().float().mean().item()
                    self._gate_stats['agate_mlp_std'] = agate_mlp.detach().float().std().item()

            del ashift_mlp, ascale_mlp, agate_mlp

        # Return updated arguments
        return (
            replace(video, x=vx) if video is not None else None,
            replace(audio, x=ax) if audio is not None else None,
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_causal_av_block(
    idx: int,
    video_dim: int = 4096,
    audio_dim: int = 2048,
    video_heads: int = 32,
    audio_heads: int = 32,
    video_d_head: int = 128,
    audio_d_head: int = 64,
    context_dim: int = 4096,
    **kwargs,
) -> CausalAVTransformerBlock:
    """
    Create a causal AV transformer block with LTX-2 19B defaults.

    """
    video_config = TransformerConfig(
        dim=video_dim,
        heads=video_heads,
        d_head=video_d_head,
        context_dim=context_dim,
    )

    audio_config = TransformerConfig(
        dim=audio_dim,
        heads=audio_heads,
        d_head=audio_d_head,
        context_dim=context_dim,
    )

    return CausalAVTransformerBlock(
        idx=idx,
        video=video_config,
        audio=audio_config,
        **kwargs,
    )
