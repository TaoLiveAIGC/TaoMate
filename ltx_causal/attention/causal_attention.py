"""
CausalLTXAttention: Causal attention module with Flexattention for training.

This module implements:
- Training mode: Flexattention with BlockMask for efficient block-wise causal attention
- Weight-compatible with original LTX-2 Attention module

Key Design Decisions:
1. Same projection layer structure as original Attention (to_q, to_k, to_v, to_out)
2. Same normalization (q_norm, k_norm with RMSNorm)
3. BlockMask for causal self-attention, dense mask for cross-attention
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ltx_causal.attention.flex_attention_utils import (
    FLEX_ATTENTION_AVAILABLE,
    flex_attention_forward,
    standard_attention_forward,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    apply_interleaved_rotary_emb,
    apply_split_rotary_emb,
)

from taomate.runtime_support.parallel import (
    get_sp_group as _get_sp_group,
    get_sp_rank as _get_sp_rank,
    get_sp_world_size as _get_sp_world_size,
    head_all_to_all_seq as _head_all_to_all_seq,
    is_sp_enabled as _is_sp_enabled,
    seq_all_to_all_head as _seq_all_to_all_head,
    seq_all_to_all_head_async as _seq_all_to_all_head_async,
    seq_all_to_all_head_many_async as _seq_all_to_all_head_many_async,
)


def _async_ulysses_enabled() -> bool:
    return False


def _async_ulysses_pack_qk_enabled() -> bool:
    return False

# Import BlockMask type for annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


class CausalLTXAttention(nn.Module):
    """
    Causal attention module for LTX-2.

    This module is weight-compatible with the original LTX-2 Attention:
    - Same linear projections (to_q, to_k, to_v, to_out)
    - Same RMSNorm for Q/K normalization
    - Supports both self-attention and cross-attention

    Causal Features:
    - Uses Flexattention with BlockMask for efficient causal attention
    - Dense mask for cross-modal causal attention (A2V, V2A)

    Args:
        query_dim: Dimension of query input
        context_dim: Dimension of context input (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: Type of RoPE (SPLIT or INTERLEAVED)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: CausalRopeType = CausalRopeType.SPLIT,
        apply_gated_attention: bool = False,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = -1,
        sink_size: int = 1,
    ):
        super().__init__()

        self.rope_type = rope_type
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.is_cross_attention = context_dim is not None
        context_dim = query_dim if context_dim is None else context_dim

        # === Projection Layers (Weight-Compatible with Original) ===
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=True)

        # Q/K Normalization
        self.q_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)

        # Optional per-head gating (weight-compatible with original Attention)
        if apply_gated_attention:
            self.to_gate_logits = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === Causal Training Parameters ===
        block_mask: Optional["BlockMask"] = None,
        cross_causal_mask: Optional[torch.Tensor] = None,
        logit_log_scale: Optional[torch.Tensor] = None,
        # === Ulysses Sequence Parallel Parameters ===
        sp_context_sharded: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass for training with causal masks.

        Args:
            x: Query input ``[B, L_local, D]`` (already SP-sharded along seq dim
                when SP is enabled; ``L_local == L`` when SP is off).
            context: Context for cross-attention ``[B, L_ctx_*, D_ctx]``.
                When ``sp_context_sharded=True`` this is the local SP shard
                of the context sequence; when ``False`` (e.g. text encoder
                output for video/audio<->text cross-attn) it must be the
                full sequence replicated on every SP rank.
            mask: Optional attention mask (for non-causal attention, e.g. text)
            pe: RoPE frequencies for Q (cos, sin)
            k_pe: RoPE frequencies for K (if different from Q)
            block_mask: BlockMask for flexattention (causal self-attention).
                Built against the **full** sequence length when SP is on.
            cross_causal_mask: Dense mask for cross-attention causality (A2V/V2A).
                Same full-sequence convention as ``block_mask``.
            logit_log_scale: Per-position log-ratio scale ``[1, L_full, 1]``
                applied to Q before attention. Acts as a position-dependent
                temperature.
            sp_context_sharded: Whether ``context`` is SP-sharded along the
                sequence dimension. ``True`` for self-attention and
                cross-modal A2V/V2A (both modalities are SP-sharded);
                ``False`` for video/audio<->text where text is replicated.
                No-op when SP is disabled.

        Returns:
            Attention output ``[B, L_local, D]`` (matches input layout).
        """
        B, L, _ = x.shape
        context = x if context is None else context

        # Sequence-Parallel state (sp_size==1 means SP disabled / no-op).
        sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
        use_async_ulysses = (
            _async_ulysses_enabled()
            and sp_size > 1
            and sp_context_sharded
            and not torch.is_grad_enabled()
        )

        if use_async_ulysses:
            sp_group = _get_sp_group()

            # V-first Ulysses: launch V seq->head all-to-all before Q/K
            # projection, normalization, and RoPE. The attention operands are
            # still waited before SDPA, so this is scheduling-only.
            v = self.to_v(context).view(B, -1, self.heads, self.dim_head)
            v_async = _seq_all_to_all_head_async(v, sp_group, sp_size)

            q = self.q_norm(self.to_q(x))
            if pe is not None:
                q = self._apply_rope(q, pe)
            q = q.view(B, -1, self.heads, self.dim_head)

            pack_qk_candidate = _async_ulysses_pack_qk_enabled()
            q_async = None
            if not pack_qk_candidate:
                q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)

            k = self.k_norm(self.to_k(context))
            if pe is not None:
                k = self._apply_rope(k, pe if k_pe is None else k_pe)
            k = k.view(B, -1, self.heads, self.dim_head)

            pack_qk = pack_qk_candidate and q.shape == k.shape
            if pack_qk:
                qk_async = _seq_all_to_all_head_many_async((q, k), sp_group, sp_size)
            else:
                if q_async is None:
                    q_async = _seq_all_to_all_head_async(q, sp_group, sp_size)
                k_async = _seq_all_to_all_head_async(k, sp_group, sp_size)

            v = v_async.wait()
            if pack_qk:
                q, k = qk_async.wait()
            else:
                q = q_async.wait()
                k = k_async.wait()

            if logit_log_scale is not None:
                if logit_log_scale.dim() == 3:
                    scale_4d = logit_log_scale.unsqueeze(-1)
                elif logit_log_scale.dim() == 4:
                    scale_4d = logit_log_scale
                else:
                    scale_4d = logit_log_scale.view(1, -1, 1, 1)
                q = q * scale_4d
        else:
            # Projections
            q = self.to_q(x)
            k = self.to_k(context)
            v = self.to_v(context)

            # Q/K Normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Apply RoPE if provided
            if pe is not None:
                q = self._apply_rope(q, pe)
                k = self._apply_rope(k, pe if k_pe is None else k_pe)

            # Apply log-ratio scaling to Q (PaLM-style Log-N Scaling).
            # When SP is OFF we keep the original ordering (BEFORE reshape) so
            # that bf16 numerics stay bit-identical to the pre-SP code path.
            # When SP is ON we defer the multiply to AFTER the all-to-all so
            # the scale broadcasts over the FULL sequence length.
            if sp_size == 1 and logit_log_scale is not None:
                q = q * logit_log_scale

            # Reshape for attention: [B, L, H, D_h]
            q = q.view(B, -1, self.heads, self.dim_head)
            k = k.view(B, -1, self.heads, self.dim_head)
            v = v.view(B, -1, self.heads, self.dim_head)

            # === Ulysses SP all-to-all: Seq-sharded -> Head-sharded ===
            if sp_size > 1:
                sp_group = _get_sp_group()
                # Query is always seq-sharded (its sequence dim is the model
                # output dim that is being parallelized).
                q = _seq_all_to_all_head(q, sp_group, sp_size)
                if sp_context_sharded:
                    # Self-attention or A2V/V2A: K/V also seq-sharded -> all-to-all.
                    k = _seq_all_to_all_head(k, sp_group, sp_size)
                    v = _seq_all_to_all_head(v, sp_group, sp_size)
                else:
                    # Text cross-attention: K/V are full on every rank, so we
                    # only need to keep our local 1/sp slice of heads to match
                    # Q's head shape after the all-to-all above.
                    H_local = self.heads // sp_size
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

        # Apply attention
        if block_mask is not None:
            if isinstance(block_mask, torch.Tensor):
                # Dense boolean mask -> SDPA path (OmniForcing-aligned)
                out = standard_attention_forward(q, k, v, block_mask)
            elif not FLEX_ATTENTION_AVAILABLE:
                raise RuntimeError(
                    "block_mask provided but flex_attention is not available. "
                    "PyTorch 2.2+ with CUDA is required for causal self-attention."
                )
            else:
                # === Flexattention Path (Self-Attention with BlockMask) ===
                out = flex_attention_forward(q, k, v, block_mask)

        elif cross_causal_mask is not None:
            # === Standard Attention with Dense Causal Mask (Cross-Attention) ===
            out = standard_attention_forward(q, k, v, cross_causal_mask)

        elif mask is not None:
            # === Standard Attention with Provided Mask (no temperature) ===
            out = standard_attention_forward(q, k, v, mask)

        else:
            # === Standard Attention (No Mask, no temperature) ===
            out = standard_attention_forward(q, k, v)

        # === Ulysses SP all-to-all: Head-sharded -> Seq-sharded ===
        if sp_size > 1:
            out = _head_all_to_all_seq(out, _get_sp_group(), sp_size)

        # Reshape and project output
        out = out.reshape(B, -1, self.inner_dim)

        # Apply per-head gating if enabled.
        # ``x`` is the local (SP-sharded) input, so gate_logits naturally
        # matches the local sequence length of ``out`` after head_all_to_all_seq.
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T_local, H)
            b, t, _ = out.shape
            out = out.view(b, t, self.heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)  # zero-init -> 1.0
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.inner_dim)

        return self.to_out(out)

    def _apply_rope(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply RoPE to input tensor. Supports both INTERLEAVED and SPLIT modes."""
        cos_freqs, sin_freqs = freqs_cis
        if self.rope_type == CausalRopeType.SPLIT:
            return apply_split_rotary_emb(x, cos_freqs, sin_freqs)
        elif self.rope_type == CausalRopeType.INTERLEAVED:
            return apply_interleaved_rotary_emb(x, cos_freqs, sin_freqs)
        else:
            raise ValueError(f"Unsupported rope_type: {self.rope_type}")


# ============================================================================
# Factory Functions
# ============================================================================

def create_causal_attention(
    query_dim: int,
    context_dim: Optional[int] = None,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """
    Factory function to create CausalLTXAttention with LTX-2 defaults.

    Args:
        query_dim: Query dimension
        context_dim: Context dimension (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
    Returns:
        Configured CausalLTXAttention instance
    """
    return CausalLTXAttention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_video_self_attention(
    dim: int = 4096,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """Create video self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_audio_self_attention(
    dim: int = 2048,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create audio self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_cross_attention(
    query_dim: int,
    context_dim: int,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create cross-attention module (A2V or V2A)."""
    return create_causal_attention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )
