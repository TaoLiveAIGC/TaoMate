import copy
from typing import NamedTuple

import torch
from transformers import Gemma3Config
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.gemma3 import Gemma3ForConditionalGeneration

from ltx_core.loader import KeyValueOperationResult
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.text_encoders.gemma.config import GEMMA3_CONFIG_FOR_LTX
from ltx_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnector,
    Embeddings1DConnectorConfigurator,
)
from ltx_core.text_encoders.gemma.encoders.base_encoder import (
    GemmaTextEncoderModelBase,
)
from ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer


def _gemma3_rope_freqs(config):
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    sliding_rope = rope_scaling.get("sliding_attention", {})
    full_rope = rope_scaling.get("full_attention", rope_scaling)

    dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    base = getattr(
        config,
        "rope_local_base_freq",
        sliding_rope.get("rope_theta", getattr(config, "rope_theta", 10000.0)),
    )
    local_rope_freqs = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(dtype=torch.float) / dim)
    )

    rope_config = config
    if full_rope and full_rope is not rope_scaling:
        rope_config = copy.copy(config)
        rope_config.rope_scaling = full_rope
        if "rope_theta" in full_rope:
            rope_config.rope_theta = full_rope["rope_theta"]
    rope_type = (getattr(rope_config, "rope_scaling", None) or {}).get("rope_type", "default")
    inv_freqs, _ = ROPE_INIT_FUNCTIONS[rope_type](rope_config)
    return local_rope_freqs, inv_freqs


def _set_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value)


class AVGemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor
    attention_mask: torch.Tensor


class AVGemmaTextEncoderModel(GemmaTextEncoderModelBase):
    """
    AVGemma Text Encoder Model.
    This class combines the tokenizer, Gemma model, feature extractor from base class and a
    video and audio embeddings connectors to provide a preprocessing for audio-visual pipeline.
    """

    def __init__(
        self,
        feature_extractor_linear: GemmaFeaturesExtractorProjLinear,
        embeddings_connector: Embeddings1DConnector,
        audio_embeddings_connector: Embeddings1DConnector,
        tokenizer: LTXVGemmaTokenizer | None = None,
        model: Gemma3ForConditionalGeneration | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(
            feature_extractor_linear=feature_extractor_linear,
            tokenizer=tokenizer,
            model=model,
            dtype=dtype,
        )
        self.embeddings_connector = embeddings_connector.to(dtype=dtype)
        self.audio_embeddings_connector = audio_embeddings_connector.to(dtype=dtype)

    def _run_connectors(
        self, encoded_input: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        connector_attention_mask = self._convert_to_additive_mask(attention_mask, encoded_input.dtype)

        encoded, encoded_connector_attention_mask = self.embeddings_connector(
            encoded_input,
            connector_attention_mask,
        )

        # restore the mask values to int64
        attention_mask = (encoded_connector_attention_mask < 0.000001).to(torch.int64)
        attention_mask = attention_mask.reshape([encoded.shape[0], encoded.shape[1], 1])
        encoded = encoded * attention_mask

        encoded_for_audio, _ = self.audio_embeddings_connector(encoded_input, connector_attention_mask)

        return encoded, encoded_for_audio, attention_mask.squeeze(-1)

    def forward(self, text: str, padding_side: str = "left") -> AVGemmaEncoderOutput:
        encoded_inputs, attention_mask = self._preprocess_text(text, padding_side)
        video_encoding, audio_encoding, attention_mask = self._run_connectors(encoded_inputs, attention_mask)
        return AVGemmaEncoderOutput(video_encoding, audio_encoding, attention_mask)


class AVGemmaTextEncoderModelConfigurator(ModelConfigurator[AVGemmaTextEncoderModel]):
    @classmethod
    def from_config(cls: type["AVGemmaTextEncoderModel"], config: dict) -> "AVGemmaTextEncoderModel":
        feature_extractor_linear = GemmaFeaturesExtractorProjLinear.from_config(config)
        embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        audio_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(config)
        gemma_config = Gemma3Config.from_dict(GEMMA3_CONFIG_FOR_LTX.to_dict())
        with torch.device("meta"):
            model = Gemma3ForConditionalGeneration(gemma_config)
        return AVGemmaTextEncoderModel(
            model=model,
            feature_extractor_linear=feature_extractor_linear,
            embeddings_connector=embeddings_connector,
            audio_embeddings_connector=audio_embeddings_connector,
        )


AV_GEMMA_TEXT_ENCODER_KEY_OPS = (
    SDOps("AV_GEMMA_TEXT_ENCODER_KEY_OPS")
    # 1. Map the feature extractor
    .with_matching(prefix="text_embedding_projection.")
    .with_replacement("text_embedding_projection.", "feature_extractor_linear.")
    # 2. Map the connectors (fixing the swapped prefixes from before)
    .with_matching(prefix="model.diffusion_model.video_embeddings_connector.")
    .with_replacement("model.diffusion_model.video_embeddings_connector.", "embeddings_connector.")
    .with_matching(prefix="model.diffusion_model.audio_embeddings_connector.")
    .with_replacement("model.diffusion_model.audio_embeddings_connector.", "audio_embeddings_connector.")
    # 3. Map language model layers (note the double .model prefix)
    .with_matching(prefix="language_model.model.")
    .with_replacement("language_model.model.", "model.model.language_model.")
    # 4. Map the Vision Tower
    .with_matching(prefix="vision_tower.")
    .with_replacement("vision_tower.vision_model.", "__ltx_gemma_vision_tower__.")
    .with_replacement("vision_tower.", "__ltx_gemma_vision_tower__.")
    .with_replacement("__ltx_gemma_vision_tower__.", "model.model.vision_tower.")
    # 5. Map the Multi-Modal Projector
    .with_matching(prefix="multi_modal_projector.")
    .with_replacement("multi_modal_projector.", "model.model.multi_modal_projector.")
    .with_kv_operation(
        operation=lambda key, value: [
            KeyValueOperationResult(key, value),
            KeyValueOperationResult("model.lm_head.weight", value),
        ],
        key_prefix="model.model.language_model.embed_tokens.weight",
    )
)


def _resolve_vision_model(model: Gemma3ForConditionalGeneration) -> torch.nn.Module:
    vision_tower = model.model.vision_tower
    return getattr(vision_tower, "vision_model", vision_tower)


def create_and_populate(module: AVGemmaTextEncoderModel) -> AVGemmaTextEncoderModel:
    model = module.model
    v_model = _resolve_vision_model(model)
    l_model = model.model.language_model

    config = model.config.text_config
    local_rope_freqs, inv_freqs = _gemma3_rope_freqs(config)

    positions_length = len(v_model.embeddings.position_ids[0])
    position_ids = torch.arange(positions_length, dtype=torch.long, device="cpu").unsqueeze(0)
    v_model.embeddings.register_buffer("position_ids", position_ids)
    embed_scale = torch.tensor(model.config.text_config.hidden_size**0.5, device="cpu")
    l_model.embed_tokens.register_buffer("embed_scale", embed_scale)
    if hasattr(l_model, "rotary_emb_local"):
        _set_buffer(l_model.rotary_emb_local, "inv_freq", local_rope_freqs)
    if hasattr(l_model, "rotary_emb"):
        _set_buffer(l_model.rotary_emb, "inv_freq", inv_freqs)
        _set_buffer(l_model.rotary_emb, "sliding_attention_inv_freq", local_rope_freqs)
        _set_buffer(l_model.rotary_emb, "sliding_attention_original_inv_freq", local_rope_freqs)
        _set_buffer(l_model.rotary_emb, "full_attention_inv_freq", inv_freqs)
        _set_buffer(l_model.rotary_emb, "full_attention_original_inv_freq", inv_freqs)

    return module


GEMMA_MODEL_OPS = ModuleOps(
    name="GemmaModel",
    matcher=lambda module: hasattr(module, "model") and isinstance(module.model, Gemma3ForConditionalGeneration),
    mutator=create_and_populate,
)
