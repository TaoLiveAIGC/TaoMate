from typing import Any, Dict, List, Optional, Tuple

import torch

from ltx_causal.attention.mask_builder import compute_av_blocks
from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
from ltx_core.components.schedulers import LTX2Scheduler
from taomate.inference.cache import attach_segments, cache_segments, retain_streaming_context
from taomate.inference.memory import StreamingMemory

DENOISING_STEPS = (1000, 757, 522, 0)

def compute_denoising_sigmas(device: str) -> torch.Tensor:
    schedule = LTX2Scheduler().execute(steps=40)
    sigmas = []
    for step in DENOISING_STEPS:
        index = (schedule - step / 1000.0).abs().argmin().item()
        sigmas.append(schedule[index])
    return torch.stack(sigmas).to(device)

def load_generator(
    checkpoint_path: str,
    base_model_ckpt: str,
    video_height: int,
    video_width: int,
    dtype: torch.dtype,
) -> Tuple[CausalLTX2DiffusionWrapper, int]:
    from taomate.inference.model_runtime import load_model_generator

    return load_model_generator(
        model_ckpt_path=checkpoint_path,
        original_ckpt_path=base_model_ckpt,
        video_height=video_height,
        video_width=video_width,
        num_frame_per_block=3,
        num_frame_per_block_first=4,
        num_audio_sink_tokens=0,
        use_flex_attention=True,
        device="cpu",
        dtype=dtype,
        causal_rope_type="split",
        learned_memory_enabled=True,
        learned_memory_mode="memory_kv_side_branch",
        learned_memory_layer_interval=4,
        learned_memory_video_dim=512,
        learned_memory_audio_dim=256,
        learned_memory_heads=8,
        learned_memory_color_film=True,
        learned_memory_color_film_hidden_dim=256,
        use_mmap=True,
        use_ema=True,
    )

class StreamingGenerator:
    def __init__(
        self,
        pipeline: Any,
    ) -> None:
        self.pipeline = pipeline
        self.generator = pipeline.generator
        self.denoising_sigmas = pipeline.denoising_sigmas
        self.last_cache = None

    def _denoise_block(
        self,
        block: Any,
        batch_size: int,
        video_tail_shape: Tuple[int, ...],
        audio_channels: int,
        conditioning: Dict[str, Any],
        cache: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pipeline._denoise_block_with_kv(
            block=block,
            B=batch_size,
            video_tail_shape=video_tail_shape,
            audio_channels=audio_channels,
            conditional_dict=conditioning,
            kv_cache=cache,
            device=device,
            dtype=dtype,
        )

    def _stabilize_video(self, block_index: int, video: torch.Tensor) -> torch.Tensor:
        video, _ = self.pipeline._maybe_renorm_block(block_index, video, None)
        return video

    def _commit_clean_block(
        self,
        block: Any,
        video: torch.Tensor,
        audio: torch.Tensor,
        conditioning: Dict[str, Any],
        cache: Any,
        previous_layers: Optional[List[Any]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        if previous_layers is None:
            cache = self.generator.init_kv_cache()
        else:
            cache.layers = list(previous_layers)
        video_timestep = torch.zeros(
            (batch_size, video.shape[1]), device=device, dtype=dtype
        )
        audio_timestep = torch.zeros(
            (batch_size, audio.shape[1]), device=device, dtype=dtype
        )
        _, _, cache = self.generator.model.forward_inference(
            video_latent=video,
            audio_latent=audio,
            timesteps=video_timestep,
            audio_timesteps=audio_timestep,
            video_context=conditioning["video_context"],
            audio_context=conditioning["audio_context"],
            video_context_mask=conditioning.get("video_context_mask"),
            audio_context_mask=conditioning.get("audio_context_mask"),
            learned_memory_video=conditioning.get("learned_memory_video"),
            learned_memory_audio=conditioning.get("learned_memory_audio"),
            learned_memory_color=conditioning.get("learned_memory_color"),
            kv_cache=cache,
            video_start_frame=block.video_start,
            audio_start_frame=block.audio_start,
            include_audio_sinks=False,
            pyramid_policy=self.pipeline.pyramid_policy,
            kv_cache_only=True,
        )
        return cache

    @staticmethod
    def _cache_segment(block: Any, video_hw: Tuple[int, int]) -> Dict[str, Any]:
        return {
            "block_idx": int(block.block_idx),
            "video_tokens": int(block.video_frames * video_hw[0] * video_hw[1]),
            "audio_tokens": int(block.audio_frames),
        }

    @torch.no_grad()
    def generate_first_window(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        conditioning: Dict[str, Any],
        seed: int,
        memory: StreamingMemory,
        block_callback: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video, audio = self.pipeline.generate(
            video_shape=video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditioning,
            seed=seed,
            block_callback=block_callback,
            learned_memory_state=memory,
        )
        cache = self.pipeline.last_kv_cache
        segments = getattr(cache, "_interactive_segments", None)
        if cache is None or not segments:
            raise RuntimeError("Initial generation did not produce a clean KV cache.")
        attach_segments(cache, segments)
        self.last_cache = cache
        return video, audio

    @torch.no_grad()
    def generate_next_window(
        self,
        video_shape: Tuple[int, ...],
        audio_shape: Tuple[int, ...],
        conditioning: Dict[str, Any],
        seed: int,
        memory: StreamingMemory,
        cache: Any,
        first_block_index: int,
        block_callback: Optional[Any] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        batch_size = int(video_shape[0])
        block_count = 5
        total_frames = 4 + (int(first_block_index) + block_count - 1) * 3
        blocks = compute_av_blocks(
            total_frames,
            num_frame_per_block_first=4,
        )[
            first_block_index : first_block_index + block_count
        ]
        if len(blocks) != block_count:
            raise RuntimeError("Unable to construct continuous block positions.")

        video = torch.zeros(video_shape, device=device, dtype=dtype)
        audio = torch.zeros(audio_shape, device=device, dtype=dtype)
        audio_channels = int(audio_shape[2])
        video_hw = int(video_shape[-2]), int(video_shape[-1])
        segments = cache_segments(cache)

        video_offset = 0
        audio_offset = 0
        for block in blocks:
            previous_layers = list(cache.layers)
            block_conditioning = memory.with_conditional_memory(
                conditioning, device=device, dtype=dtype
            )
            block_video, block_audio = self._denoise_block(
                block,
                batch_size,
                tuple(video_shape[2:]),
                audio_channels,
                block_conditioning,
                cache,
                device,
                dtype,
            )
            block_video = self._stabilize_video(block.block_idx, block_video)
            block_video = memory.apply_color_memory(block_video)

            video_end = video_offset + int(block.video_frames)
            audio_end = audio_offset + int(block.audio_frames)
            video[:, video_offset:video_end] = block_video
            audio[:, audio_offset:audio_end] = block_audio
            if block_callback is not None:
                block_callback(block_video, block_audio, block)

            cache = self._commit_clean_block(
                block,
                block_video,
                block_audio,
                block_conditioning,
                cache,
                previous_layers,
                batch_size,
                device,
                dtype,
            )
            memory.update(block_video, block_audio)
            segments.append(self._cache_segment(block, video_hw))
            attach_segments(cache, segments)
            cache = retain_streaming_context(cache)
            segments = cache_segments(cache)
            video_offset = video_end
            audio_offset = audio_end
            torch.cuda.empty_cache()

        self.last_cache = cache
        return video, audio
