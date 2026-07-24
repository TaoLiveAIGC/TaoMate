import torch

from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessor,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoder,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.utils import find_matching_file


class ModelLedger:
    """Build the text, video and audio modules used by inference."""

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str,
        gemma_root_path: str | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.dtype = dtype
        self.device = device
        self.registry = registry or DummyRegistry()
        common = {"model_path": checkpoint_path, "registry": self.registry}
        self.video_decoder_builder = Builder(
            **common,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
        )
        self.video_encoder_builder = Builder(
            **common,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        self.audio_encoder_builder = Builder(
            **common,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
        )
        self.audio_decoder_builder = Builder(
            **common,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
        )
        self.vocoder_builder = Builder(
            **common,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
        )
        self.embeddings_processor_builder = Builder(
            **common,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
        )
        if gemma_root_path is not None:
            model_folder = find_matching_file(
                gemma_root_path,
                "model*.safetensors",
            ).parent
            weight_paths = tuple(str(path) for path in model_folder.rglob("*.safetensors"))
            self.text_encoder_builder = Builder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                registry=self.registry,
                module_ops=(GEMMA_MODEL_OPS, *module_ops_from_gemma_root(gemma_root_path)),
            )

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry):
            return self.device
        return torch.device("cpu")

    def _build(self, builder):
        return builder.build(
            device=self._target_device(),
            dtype=self.dtype,
        ).to(self.device).eval()

    def video_decoder(self) -> VideoDecoder:
        return self._build(self.video_decoder_builder)

    def video_encoder(self) -> VideoEncoder:
        return self._build(self.video_encoder_builder)

    def audio_encoder(self) -> AudioEncoder:
        return self._build(self.audio_encoder_builder)

    def audio_decoder(self) -> AudioDecoder:
        return self._build(self.audio_decoder_builder)

    def vocoder(self) -> Vocoder:
        return self._build(self.vocoder_builder)

    def gemma_embeddings_processor(self) -> EmbeddingsProcessor:
        return self._build(self.embeddings_processor_builder)

    def text_encoder(self) -> GemmaTextEncoder:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError("gemma_root_path is required for the text encoder")
        return self._build(self.text_encoder_builder)
