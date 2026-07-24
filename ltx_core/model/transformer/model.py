from enum import Enum

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised

# === Optional Sequence Parallel support ===
# Same lazy-import pattern as ltx_causal: SP infra lives in
# taomate.runtime_support.parallel which pulls in heavy training-only deps. We fall
# back to no-op stubs so this module remains usable in inference-only
# environments and bit-equal when sp_size == 1.
try:
    from taomate.runtime_support.parallel import (  # type: ignore[import-not-found]
        is_sp_enabled as _is_sp_enabled,
        get_sp_world_size as _get_sp_world_size,
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

    def _split_sequence(x, dim=1, sp_size=None, sp_rank=None):  # type: ignore[misc]
        return x

    def _gather_sequence(x, dim=1, group=None, sp_size=None, sp_rank=None):  # type: ignore[misc]
        return x

    def _pad_sequence_to_multiple(x, multiple, dim=1, pad_value=0.0):  # type: ignore[misc]
        return x, x.shape[dim], x.shape[dim]

    def _unpad_sequence(x, original_length, dim=1):  # type: ignore[misc]
        return x


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class LTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        # === [Ulysses SP] Pad to multiple-of-sp_size and split per-token tensors ===
        # ``video_args`` / ``audio_args`` come out of ``preprocessor.prepare``
        # with the FULL (un-sharded) sequence on every rank. We pad the
        # sequence dim so it is divisible by ``sp_size`` and then take the
        # local SP shard for every per-token tensor that flows through
        # transformer blocks. Tensors that stay full on every rank
        # (text ``context``/``context_mask``, ``embedded_timestep``,
        # ``self_attention_mask``) are NOT split: they are either consumed
        # at the output stage after gather, or used internally by attention
        # which all-to-all's back to the full sequence first.
        #
        # Numerics guarantee: when sp_size == 1 every helper is a no-op,
        # the dataclass is returned unchanged, and this branch is bit-equal
        # with the original model on a single rank.
        sp_size = _get_sp_world_size() if _is_sp_enabled() else 1
        video_orig_len = None
        audio_orig_len = None
        if sp_size > 1:
            if video_args is not None:
                video_args, video_orig_len = self._sp_split_args(video_args, sp_size)
            if audio_args is not None:
                audio_args, audio_orig_len = self._sp_split_args(audio_args, sp_size)

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # === [Ulysses SP] Gather full sequence then unpad to original length ===
        # The output stage (norm/scale/shift/proj_out) operates on the full
        # original sequence; ``embedded_timestep`` was kept full+un-padded.
        if sp_size > 1:
            from dataclasses import replace as _dc_replace
            if video_out is not None:
                vx_full = _gather_sequence(video_out.x, dim=1, sp_size=sp_size)
                if video_orig_len is not None:
                    vx_full = _unpad_sequence(vx_full, video_orig_len, dim=1)
                video_out = _dc_replace(video_out, x=vx_full)
            if audio_out is not None:
                ax_full = _gather_sequence(audio_out.x, dim=1, sp_size=sp_size)
                if audio_orig_len is not None:
                    ax_full = _unpad_sequence(ax_full, audio_orig_len, dim=1)
                audio_out = _dc_replace(audio_out, x=ax_full)

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax

    # ------------------------------------------------------------------
    # Ulysses Sequence Parallel helpers
    # ------------------------------------------------------------------
    def _sp_split_args(
        self,
        args: "TransformerArgs",
        sp_size: int,
    ) -> tuple["TransformerArgs", int]:
        """Pad per-token tensors to a multiple of ``sp_size`` and split to local shard.

        Returns the rewritten ``TransformerArgs`` and the original (pre-pad)
        sequence length so the caller can ``unpad`` the output back to
        the user-visible length.

        Padding strategy:
          - ``x`` and per-token AdaLN tensors are right-padded with zeros
            along seq dim and then split.
          - ``positional_embeddings`` / ``cross_positional_embeddings`` are
            ``(cos, sin)`` tuples whose seq dim depends on rope type:
            INTERLEAVED -> ``(B, T, D)`` (seq_dim=1), SPLIT ->
            ``(B, H, T, D//2)`` (seq_dim=2). We auto-detect via ``ndim``
            and pad cos with 1.0, sin with 0.0 (identity rotation) so any
            spurious read on padded slots stays a no-op.
          - ``self_attention_mask`` (full ``(B, 1, T, T)`` additive bias)
            is kept full but padded along both query and key dims with the
            dtype's most-negative additive bias so attention to/from
            padded positions is fully suppressed.
          - When ``self_attention_mask`` is ``None`` and padding is
            actually applied (``T_pad > T_full``), we MATERIALIZE a
            minimal mask that only blocks the pad slots; without this the
            zero-filled pad keys would still consume softmax weight and
            silently dilute the real-token attention scores.
          - ``context``, ``context_mask``, ``embedded_timestep`` are NOT
            modified: they are either text (replicated on every rank) or
            consumed only at the output stage (after gather+unpad).
        """
        from dataclasses import replace as _dc_replace

        x = args.x
        T_full = x.shape[1]
        if T_full % sp_size == 0:
            T_pad = T_full
        else:
            T_pad = T_full + (sp_size - T_full % sp_size)

        def _pad_split_dim(t: torch.Tensor | None, seq_dim: int, pad_value: float = 0.0) -> torch.Tensor | None:
            """Pad ``t`` along ``seq_dim`` to T_pad, then split to local shard.

            Only applied when ``t.shape[seq_dim] == T_full`` (i.e. ``t`` is
            actually per-token along that dim); otherwise returned as-is.
            """
            if t is None:
                return None
            if t.shape[seq_dim] != T_full:
                return t
            if T_pad > T_full:
                t, _, _ = _pad_sequence_to_multiple(t, sp_size, dim=seq_dim, pad_value=pad_value)
            return _split_sequence(t, dim=seq_dim, sp_size=sp_size)

        def _pad_split_per_token(t: torch.Tensor | None) -> torch.Tensor | None:
            """Per-token tensors live with seq dim at axis=1."""
            return _pad_split_dim(t, seq_dim=1, pad_value=0.0)

        def _pad_split_pe(pe):
            """Pad+split a RoPE ``(cos, sin)`` tuple, auto-detecting seq dim.

            INTERLEAVED rope: cos/sin shape ``(B, T, D)`` -> seq_dim=1.
            SPLIT rope: cos/sin shape ``(B, H, T, D//2)`` -> seq_dim=2.
            cos is padded with 1.0, sin with 0.0 (identity rotation) so a
            spurious read on a pad slot is a no-op even before the mask
            blocks it.
            """
            if pe is None:
                return None

            def _one(p, is_cos: bool):
                if p is None:
                    return None
                seq_dim = 2 if p.ndim == 4 else 1
                pad_value = 1.0 if is_cos else 0.0
                return _pad_split_dim(p, seq_dim=seq_dim, pad_value=pad_value)

            if isinstance(pe, tuple):
                # Convention: (cos, sin)
                if len(pe) == 2:
                    return (_one(pe[0], is_cos=True), _one(pe[1], is_cos=False))
                return tuple(_one(p, is_cos=False) for p in pe)
            return _one(pe, is_cos=False)

        # 1. Pad+split the main token stream and per-token tensors
        x_local = _pad_split_per_token(x)
        pe_local = _pad_split_pe(args.positional_embeddings)
        cross_pe_local = _pad_split_pe(args.cross_positional_embeddings)
        timesteps_local = _pad_split_per_token(args.timesteps)
        cross_ss_local = _pad_split_per_token(args.cross_scale_shift_timestep)
        cross_gate_local = _pad_split_per_token(args.cross_gate_timestep)
        prompt_ts_local = _pad_split_per_token(args.prompt_timestep)

        # 2. self_attention_mask: (B, 1, T, T) — KEEP FULL, pad both dims.
        #    If user passed None AND we actually padded, materialize a
        #    minimal mask so pad keys/queries don't pollute softmax.
        self_attn_mask_full = args.self_attention_mask
        if T_pad > T_full:
            x_dtype = x.dtype
            B = x.shape[0]
            extra = T_pad - T_full
            if self_attn_mask_full is None:
                # Build a fresh (B, 1, T_pad, T_pad) additive-bias mask:
                #   real-real block: 0.0 (no bias, full attention)
                #   anything touching pad: finfo.min
                finfo_min = torch.finfo(x_dtype).min
                self_attn_mask_full = torch.zeros(
                    (B, 1, T_pad, T_pad), dtype=x_dtype, device=x.device
                )
                self_attn_mask_full[:, :, T_full:, :] = finfo_min  # padded queries
                self_attn_mask_full[:, :, :, T_full:] = finfo_min  # padded keys
            else:
                finfo_min = torch.finfo(self_attn_mask_full.dtype).min
                # Pad along the key (last) dim with finfo.min
                key_pad = torch.full(
                    (self_attn_mask_full.shape[0], self_attn_mask_full.shape[1], T_full, extra),
                    finfo_min,
                    dtype=self_attn_mask_full.dtype,
                    device=self_attn_mask_full.device,
                )
                self_attn_mask_full = torch.cat([self_attn_mask_full, key_pad], dim=-1)
                # Pad along the query dim with finfo.min
                q_pad = torch.full(
                    (self_attn_mask_full.shape[0], self_attn_mask_full.shape[1], extra, T_pad),
                    finfo_min,
                    dtype=self_attn_mask_full.dtype,
                    device=self_attn_mask_full.device,
                )
                self_attn_mask_full = torch.cat([self_attn_mask_full, q_pad], dim=-2)

        # 3. embedded_timestep: kept full+un-padded (only consumed after
        #    gather+unpad at the output stage, so we leave it as-is).

        new_args = _dc_replace(
            args,
            x=x_local,
            positional_embeddings=pe_local,
            cross_positional_embeddings=cross_pe_local,
            timesteps=timesteps_local,
            cross_scale_shift_timestep=cross_ss_local,
            cross_gate_timestep=cross_gate_local,
            prompt_timestep=prompt_ts_local,
            self_attention_mask=self_attn_mask_full,
        )
        return new_args, T_full


class LegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class X0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: LTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
