import os
from enum import Enum
from typing import Protocol

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None

# === Optional Sequence Parallel support ===
# Lazy/optional import: SP infra lives in taomate.runtime_support.parallel which
# pulls in heavy training-only deps (lmdb, etc.). Falling back to no-op stubs
# keeps this module importable in inference-only environments and bit-equal
# when ``sp_size == 1``.
try:
    from taomate.runtime_support.parallel import (  # type: ignore[import-not-found]
        is_sp_enabled as _is_sp_enabled,
        get_sp_world_size as _get_sp_world_size,
        get_sp_rank as _get_sp_rank,
        get_sp_group as _get_sp_group,
        seq_all_to_all_head as _seq_all_to_all_head,
        head_all_to_all_seq as _head_all_to_all_seq,
        seq_all_to_all_head_many_async as _seq_all_to_all_head_many_async,
        seq_all_to_all_head_async as _seq_all_to_all_head_async,
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

    def _head_all_to_all_seq(x, group=None, sp_size=None):  # type: ignore[misc]
        return x

    def _seq_all_to_all_head_many_async(tensors, group=None, sp_size=None):  # type: ignore[misc]
        class _IdentityManyAsync:
            def __init__(self, values):
                self.values = values

            def wait(self):
                return self.values

        return _IdentityManyAsync(tuple(tensors))

    def _seq_all_to_all_head_async(x, group=None, sp_size=None):  # type: ignore[misc]
        class _IdentityAsync:
            def __init__(self, value):
                self.value = value

            def wait(self):
                return self.value

        return _IdentityAsync(x)


class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks."""
        backend_override = os.environ.get("LTX_ATTENTION_BACKEND", "").strip().lower()
        if backend_override:
            if backend_override in {"pytorch", "torch", "sdpa"}:
                return PytorchAttention()
            if backend_override in {"xformers", "xformer"}:
                return XFormersAttention()
            if backend_override in {"flash_attention_3", "flash3", "fa3"}:
                return FlashAttention3()
            raise ValueError(
                "Unsupported LTX_ATTENTION_BACKEND="
                f"{backend_override!r}; expected pytorch, xformers, or flash_attention_3"
            )

        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        # === Ulysses Sequence Parallel ===
        sp_context_sharded: bool = True,
        enable_sp: bool = True,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA). Must be built
                against the **full** sequence length when SP is enabled.
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
            sp_context_sharded: Whether ``context`` is SP-sharded along the
                sequence dimension. ``True`` for self-attention and AV cross-
                attention (both modalities are SP-sharded); ``False`` for text
                cross-attention where text is replicated on every SP rank.
                No-op when SP is disabled (``sp_size == 1``).
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed
        B = x.shape[0]

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            # ``enable_sp=False`` lets sub-modules (e.g. text encoder connector)
            # that share this Attention class opt out of SP entirely. Their
            # sequences are short and replicated on every SP rank, so a2a would
            # produce wrong shapes (seq * sp_size) vs the replicated mask.
            sp_size = _get_sp_world_size() if (enable_sp and _is_sp_enabled()) else 1
            sp_group = _get_sp_group() if sp_size > 1 else None
            use_async_ulysses = (
                sp_size > 1
                and sp_context_sharded
                and not torch.is_grad_enabled()
                and _async_ulysses_enabled()
            )

            v4_async = None
            if use_async_ulysses:
                # V-first Ulysses: start the V scatter before Q/K projection and
                # RoPE work. This only changes scheduling; all tensors are waited
                # before SDPA, so attention math stays unchanged.
                v4_async = _seq_all_to_all_head_async(
                    v.view(B, -1, self.heads, self.dim_head),
                    sp_group,
                    sp_size,
                )

            q = self.to_q(x)
            q = self.q_norm(q)
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)

            q4_raw = None
            q4_async = None
            qk4_async = None
            if use_async_ulysses:
                q4_raw = q.view(B, -1, self.heads, self.dim_head)
                if not _async_ulysses_pack_qk_enabled():
                    q4_async = _seq_all_to_all_head_async(
                        q4_raw,
                        sp_group,
                        sp_size,
                    )

            k = self.to_k(context)
            k = self.k_norm(k)
            if pe is not None:
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            k4_async = None
            if use_async_ulysses:
                assert q4_raw is not None
                k4_raw = k.view(B, -1, self.heads, self.dim_head)
                pack_qk = _async_ulysses_pack_qk_enabled() and q4_raw.shape == k4_raw.shape
                if pack_qk:
                    qk4_async = _seq_all_to_all_head_many_async((q4_raw, k4_raw), sp_group, sp_size)
                else:
                    if q4_async is None:
                        q4_async = _seq_all_to_all_head_async(
                            q4_raw,
                            sp_group,
                            sp_size,
                        )
                    k4_async = _seq_all_to_all_head_async(
                        k4_raw,
                        sp_group,
                        sp_size,
                    )

            if sp_size == 1:
                # Original path: bit-equal with the pre-SP code path.
                out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)
            elif use_async_ulysses:
                assert v4_async is not None
                if qk4_async is not None:
                    q4, k4 = qk4_async.wait()
                else:
                    assert q4_async is not None and k4_async is not None
                    q4 = q4_async.wait()
                    k4 = k4_async.wait()
                v4 = v4_async.wait()

                # SDPA on head-sharded layout: (B, T_full, H_local, D)
                out_4 = _sp_sdpa(q4, k4, v4, mask)
                # Back to seq-sharded: (B, T_local, H, D)
                out_4 = _head_all_to_all_seq(out_4, sp_group, sp_size)
                out = out_4.reshape(B, -1, self.heads * self.dim_head)
            else:
                # === Ulysses SP path ===
                # Reshape to (B, T, H, D) for all-to-all primitives.
                q4 = q.view(B, -1, self.heads, self.dim_head)
                k4 = k.view(B, -1, self.heads, self.dim_head)
                v4 = v.view(B, -1, self.heads, self.dim_head)

                # Q is always seq-sharded (the parallelized dim).
                q4 = _seq_all_to_all_head(q4, sp_group, sp_size)
                if sp_context_sharded:
                    # Self-attn / AV cross-attn: K/V also seq-sharded.
                    k4 = _seq_all_to_all_head(k4, sp_group, sp_size)
                    v4 = _seq_all_to_all_head(v4, sp_group, sp_size)
                else:
                    # Text cross-attn: K/V are full on every rank, narrow
                    # along head dim to match Q's local head shape.
                    H_local = self.heads // sp_size
                    head_start = _get_sp_rank() * H_local
                    k4 = k4.narrow(2, head_start, H_local).contiguous()
                    v4 = v4.narrow(2, head_start, H_local).contiguous()

                # SDPA on head-sharded layout: (B, T_full, H_local, D)
                out_4 = _sp_sdpa(q4, k4, v4, mask)
                # Back to seq-sharded: (B, T_local, H, D)
                out_4 = _head_all_to_all_seq(out_4, sp_group, sp_size)
                out = out_4.reshape(B, -1, self.heads * self.dim_head)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            b, t, _ = out.shape
            # Reshape to (B, T, H, D) for per-head gating
            out = out.view(b, t, self.heads, self.dim_head)
            # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
            gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
            out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
            # Reshape back to (B, T, H*D)
            out = out.view(b, t, self.heads * self.dim_head)

        return self.to_out(out)


def _sp_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention on head-sharded layout.

    Args:
        q: ``(B, Lq, H_local, D)``
        k: ``(B, Lk, H_local, D)``
        v: ``(B, Lk, H_local, D)``
        mask: Additive bias broadcastable to ``(B, H_local, Lq, Lk)``
            (typically ``(B, 1, Lq, Lk)``); built against the full sequence
            length on every SP rank.

    Returns:
        ``(B, Lq, H_local, D)``
    """
    # SDPA expects (B, H, L, D)
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)

    if mask is not None:
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)

    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=mask, dropout_p=0.0, is_causal=False
    )
    # Back to (B, L, H, D)
    return out.transpose(1, 2).contiguous()


def _async_ulysses_enabled() -> bool:
    return os.environ.get("ASYNC_ULYSSES", "0").strip().lower() in {"1", "true", "yes", "on"} or os.environ.get(
        "LTX_ASYNC_ULYSSES", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}


def _async_ulysses_pack_qk_enabled() -> bool:
    return os.environ.get("LTX_ASYNC_ULYSSES_PACK_QK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    } or os.environ.get("ASYNC_ULYSSES_PACK_QK", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
