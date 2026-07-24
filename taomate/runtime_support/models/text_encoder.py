"""
Gemma Text Encoder Wrapper for DMD distillation.

Provides a simple interface for text encoding without prompt enhancement.
Just pure text -> context embedding conversion.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn


class GemmaTextEncoderWrapper(nn.Module):
    """
    Wrapper for Gemma text encoder to provide DMD-compatible interface.

    This wrapper:
    - Takes raw text prompts (no enhancement needed)
    - Returns conditional_dict with video_context and audio_context
    - Handles batched encoding
    """

    def __init__(
        self,
        text_encoder,
        embeddings_processor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            text_encoder: GemmaTextEncoder instance
            embeddings_processor: EmbeddingsProcessor instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def forward(
        self,
        text_prompts: List[str],
        padding_side: str = "left",
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text prompts to conditioning embeddings.

        Args:
            text_prompts: List of text prompts (already processed, no enhancement)
            padding_side: Padding side for tokenizer

        Returns:
            Dictionary containing:
                - video_context: [B, seq_len, dim] video conditioning
                - audio_context: [B, seq_len, dim] audio conditioning
                - attention_mask: [B, seq_len] attention mask
        """
        video_contexts = []
        audio_contexts = []
        attention_masks = []

        for prompt in text_prompts:
            # Step 1: encode text -> raw hidden states
            hidden_states, attention_mask = self.text_encoder.encode(
                text=prompt, padding_side=padding_side
            )
            # Step 2: process hidden states -> final embeddings
            output = self.embeddings_processor.process_hidden_states(
                hidden_states, attention_mask, padding_side
            )

            video_contexts.append(output.video_encoding)
            if output.audio_encoding is not None:
                audio_contexts.append(output.audio_encoding)
            attention_masks.append(output.attention_mask)

        # Stack batch
        video_context = torch.cat(video_contexts, dim=0)
        audio_context = torch.cat(audio_contexts, dim=0) if audio_contexts else None
        attention_mask = torch.cat(attention_masks, dim=0)

        return {
            "video_context": video_context,
            "audio_context": audio_context,
            "attention_mask": attention_mask,
        }

    def encode_batch(
        self,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() with default padding."""
        return self.forward(text_prompts)


def _drop_unused_meta_vision_modules(text_encoder) -> None:
    """Drop Gemma3 image-only modules that stayed on meta.

    Distillation only calls ``GemmaTextEncoder.encode()``, which runs the text
    language model via ``self.model.model(input_ids=...)``. Some Gemma3
    checkpoints used for text-only training do not materialize the vision tower
    weights, leaving those parameters on the meta device.  A later ``.to()``
    over the whole module would then fail even though those modules are unused.
    """
    gemma = getattr(text_encoder, "model", None)
    inner = getattr(gemma, "model", None)
    if inner is None:
        return

    for name in ("vision_tower", "multi_modal_projector"):
        module = getattr(inner, name, None)
        if module is None:
            continue
        has_meta = any(p.is_meta for p in module.parameters(recurse=True))
        has_meta = has_meta or any(b.is_meta for b in module.buffers(recurse=True))
        if has_meta:
            setattr(inner, name, None)


def create_text_encoder_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry=None,
    place_on_device: bool = False,
) -> GemmaTextEncoderWrapper:
    """
    Factory function to create GemmaTextEncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype
        registry: Optional StateDictRegistry for caching loaded state dicts
        place_on_device: Move the text encoder and embedding processor to
            ``device`` before returning. Keep this disabled for training/FSDP
            paths that expect to wrap CPU modules first.

    Returns:
        Configured GemmaTextEncoderWrapper
    """
    from ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    ledger = ModelLedger(
        dtype=dtype,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_path,
        registry=registry,
    )

    if not hasattr(ledger, "text_encoder_builder"):
        raise ValueError(
            "Text encoder not initialized. Please provide checkpoint_path and gemma_path."
        )
    ledger.text_encoder_builder = (
        ledger.text_encoder_builder.with_ignored_uninitialized_prefixes(
            (
                "model.model.vision_tower.",
                "model.model.multi_modal_projector.",
            )
        )
    )
    text_encoder = ledger.text_encoder_builder.build(
        device=ledger._target_device(), dtype=dtype
    )
    _drop_unused_meta_vision_modules(text_encoder)
    text_encoder = text_encoder.to(torch.device("cpu")).eval()
    # Keep on CPU by default – FSDP wrapping will shard and move to GPU.
    text_encoder = text_encoder.to(dtype=dtype)
    embeddings_processor = ledger.gemma_embeddings_processor()
    embeddings_processor = embeddings_processor.to(dtype=dtype)
    if place_on_device:
        text_encoder = text_encoder.to(device)
        embeddings_processor = embeddings_processor.to(device)

    wrapper = GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )

    return wrapper
