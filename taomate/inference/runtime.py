import gc
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)
from taomate.inference.cache import (
    attach_segments,
    cache_segments,
    retain_streaming_context,
)
from taomate.inference.config import InferenceConfig
from taomate.inference.memory import StreamingMemory
from taomate.inference.media import (
    RealtimeBlockStreamer,
    VIDEO_FPS,
    center_crop_video,
    decode_audio_latents,
    decode_video_latents,
    ensure_module_device,
    is_cuda_oom,
    tensor_tree_to_cpu,
    tensor_tree_to_device,
    write_valid_mp4,
)
from taomate.inference.model import (
    StreamingGenerator,
    compute_denoising_sigmas,
    load_generator,
)
from taomate.inference.model_runtime import KVCacheCausalPipeline
from taomate.inference.weights import (
    load_decoder_bundles,
    load_text_conditioner,
)

FRAMES_PER_WINDOW = 121
FIRST_WINDOW_LATENTS = 16
NEXT_WINDOW_LATENTS = 15
BLOCKS_PER_WINDOW = 5
FRAMES_PER_BLOCK = 3
FIRST_BLOCK_FRAMES = 4
AUDIO_FRAMES_PER_BLOCK = 25
MAX_WINDOWS = 12

@dataclass
class ModelRuntime:
    train_step: int
    video_vae: Any
    audio_vae: Any
    text_conditioner: Optional[Any]
    generator: StreamingGenerator
    first_video_shape: Tuple[int, ...]
    first_audio_shape: Tuple[int, ...]
    next_video_shape: Tuple[int, ...]
    next_audio_shape: Tuple[int, ...]

_RUNTIME_CACHE: "OrderedDict[Tuple[Any, ...], ModelRuntime]" = OrderedDict()

def _runtime_key(config: InferenceConfig, device: str) -> Tuple[Any, ...]:
    return (
        os.path.abspath(config.model_ckpt),
        os.path.abspath(config.base_model_ckpt),
        os.path.abspath(config.gemma_path),
        str(device),
        str(config.conditioning_device or ""),
        str(config.media_device or ""),
        int(config.video_height),
        int(config.video_width),
    )

def _distributed_context() -> Tuple[int, int, bool, str, torch.dtype]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2):
        raise RuntimeError(
            f"TaoMate supports one or two worker processes, got {world_size}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("TaoMate requires CUDA GPUs.")
    torch.cuda.set_device(local_rank)
    if world_size == 2 and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    return rank, local_rank, rank == 0, "cuda", torch.bfloat16

def _latent_shapes(
    video_height: int,
    video_width: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    if video_height % 32 or video_width % 32:
        raise ValueError("Video height and width must be divisible by 32.")
    video_shape = (1, FIRST_WINDOW_LATENTS, 128, video_height // 32, video_width // 32)
    first_audio_frames = compute_aligned_audio_frames(
        FIRST_WINDOW_LATENTS,
        num_frame_per_block_first=FIRST_BLOCK_FRAMES,
    )
    first_audio_shape = (1, first_audio_frames, 128)
    next_video_shape = (1, NEXT_WINDOW_LATENTS, 128, video_height // 32, video_width // 32)
    next_audio_shape = (1, BLOCKS_PER_WINDOW * AUDIO_FRAMES_PER_BLOCK, 128)
    return video_shape, first_audio_shape, next_video_shape, next_audio_shape

def _load_runtime(
    config: InferenceConfig,
    rank: int,
    device: str,
    dtype: torch.dtype,
) -> ModelRuntime:
    wrapper, train_step = load_generator(
        config.model_ckpt,
        config.base_model_ckpt,
        config.video_height,
        config.video_width,
        dtype,
    )
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size == 2:
        from ltx_causal.tensor_parallel import shard_model

        group = dist.new_group([0, 1])
        shard_model(wrapper, tp_rank=rank, tp_size=2, tp_group=group)
    wrapper = wrapper.to(device)
    wrapper.eval()
    gc.collect()
    torch.cuda.empty_cache()

    video_vae, audio_vae = load_decoder_bundles(config.base_model_ckpt, dtype)
    text_conditioner = None
    if rank == 0 and config.conditioning_device:
        print(f"[Load] Text conditioner on {config.conditioning_device}")
        text_conditioner = load_text_conditioner(
            config.base_model_ckpt,
            config.gemma_path,
            torch.device(config.conditioning_device),
            dtype,
        ).eval()
    if rank == 0 and config.media_device:
        print(f"[Load] Media decoders on {config.media_device}")
        ensure_module_device(video_vae.decoder, config.media_device)
        ensure_module_device(audio_vae, config.media_device)
    shapes = _latent_shapes(config.video_height, config.video_width)

    pipeline = KVCacheCausalPipeline(
        generator=wrapper,
        denoising_sigmas=compute_denoising_sigmas(device),
        num_frame_per_block=FRAMES_PER_BLOCK,
        num_frame_per_block_first=FIRST_BLOCK_FRAMES,
        num_audio_sink_tokens=0,
        context_noise=0.0,
        context_noise_max=0.0,
        context_noise_schedule="constant",
        max_prefix_blocks=2,
        block0_sink_enabled=True,
        prefix_renorm=True,
        prefix_renorm_alpha=1.0,
        learned_memory=True,
        learned_memory_video_downsample=4,
        learned_memory_audio_tokens=64,
        learned_memory_video_beta=0.15,
        learned_memory_audio_beta=0.10,
        learned_memory_video_anchor_tether=0.20,
        learned_memory_audio_anchor_tether=0.10,
        learned_memory_identity_anchor=True,
        learned_memory_identity_anchor_scale=1.0,
        learned_memory_ref_video_anchor=True,
        learned_memory_drift_gate=True,
        learned_memory_drift_gate_threshold=0.05,
        learned_memory_drift_gate_temperature=0.10,
        learned_memory_drift_gate_min=0.10,
        learned_memory_drift_gate_apply_to_color=True,
        learned_memory_color_alpha=0.04,
        learned_memory_color_proto_alpha=0.015,
        learned_memory_color_update_beta=0.03,
        learned_memory_color_anchor_tether=0.60,
        learned_memory_color_proto_grid=4,
        learned_memory_color_drift_threshold=2.0,
        learned_memory_color_max_correction=0.35,
        learned_memory_color_film=True,
    )
    pipeline.interactive_commit_final_clean_kv = True

    return ModelRuntime(
        train_step=train_step,
        video_vae=video_vae,
        audio_vae=audio_vae,
        text_conditioner=text_conditioner,
        generator=StreamingGenerator(pipeline),
        first_video_shape=shapes[0],
        first_audio_shape=shapes[1],
        next_video_shape=shapes[2],
        next_audio_shape=shapes[3],
    )

def _get_runtime(
    config: InferenceConfig,
    rank: int,
    device: str,
    dtype: torch.dtype,
) -> ModelRuntime:
    key = _runtime_key(config, device)
    if config.stream and key in _RUNTIME_CACHE:
        print("[Load] Reusing resident TaoMate runtime")
        _RUNTIME_CACHE.move_to_end(key)
        return _RUNTIME_CACHE[key]

    if config.stream and _RUNTIME_CACHE:
        _RUNTIME_CACHE.clear()
        gc.collect()
        torch.cuda.empty_cache()
    runtime = _load_runtime(config, rank, device, dtype)
    if config.stream:
        _RUNTIME_CACHE[key] = runtime
    return runtime

def _normalize_segments(case: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments = case.get("segments")
    if not isinstance(segments, list) or not 1 <= len(segments) <= MAX_WINDOWS:
        raise ValueError(f"Each case must contain 1 to {MAX_WINDOWS} segments.")
    normalized = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not str(segment.get("prompt", "")).strip():
            raise ValueError(f"Segment {index} must contain a non-empty prompt.")
        if "seed" not in segment:
            raise ValueError(f"Segment {index} must contain `seed`.")
        normalized.append(
            {
                "prompt": str(segment["prompt"]),
                "seed": int(segment["seed"]),
            }
        )
    return normalized

def _load_cases(config: InferenceConfig) -> List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    with open(config.benchmark_json, "r", encoding="utf-8") as handle:
        cases = json.load(handle)
    if not isinstance(cases, list):
        raise ValueError("Benchmark JSON must contain a list of cases.")
    cases = cases[max(0, int(config.case_start)) :]
    cases = cases[: int(config.max_cases)]
    return [(case, _normalize_segments(case)) for case in cases]

def _encode_prompts(
    config: InferenceConfig,
    prompts: Sequence[str],
    device: str,
    runtime_cached: bool,
    dtype: torch.dtype,
    text_conditioner: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    if text_conditioner is not None:
        print("[Encode] Using resident text conditioner")
        with torch.no_grad():
            return {
                prompt: tensor_tree_to_cpu(text_conditioner(prompt))
                for prompt in prompts
            }

    if config.conditioning_device:
        devices = [config.conditioning_device, "cpu"]
    else:
        devices = ["cpu"] if runtime_cached else [device, "cpu"]
    last_error: Optional[RuntimeError] = None
    for encode_device in devices:
        encoder = None
        try:
            print(f"[Encode] Loading text encoder on {encode_device}")
            encoder = load_text_conditioner(
                config.base_model_ckpt,
                config.gemma_path,
                torch.device(encode_device),
                dtype,
            )
            encoder.to(encode_device)
            encoder.eval()
            cache = {}
            with torch.no_grad():
                for prompt in prompts:
                    cache[prompt] = tensor_tree_to_cpu(
                        encoder(prompt)
                    )
            encoder.to("cpu")
            del encoder
            gc.collect()
            torch.cuda.empty_cache()
            return cache
        except RuntimeError as exc:
            if encode_device.startswith("cuda") and is_cuda_oom(exc):
                last_error = exc
                if encoder is not None:
                    try:
                        encoder.to("cpu")
                    except Exception:
                        pass
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                print("[Encode] CUDA memory exhausted; retrying on CPU")
                continue
            raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unable to encode prompts.")

def _prompt_cache(
    config: InferenceConfig,
    prompts: Sequence[str],
    rank: int,
    device: str,
    runtime_cached: bool,
    dtype: torch.dtype,
    text_conditioner: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    path = os.path.abspath(config.prompt_cache_path) if config.prompt_cache_path else None
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    if path:
        exists = [os.path.exists(path) if rank == 0 else False]
        if distributed:
            dist.broadcast_object_list(exists, src=0)
        if exists[0]:
            print(f"[Encode] Loading prompt cache from {path}")
            cache = torch.load(path, map_location="cpu", weights_only=False)
        elif rank == 0:
            cache = _encode_prompts(
                config,
                prompts,
                device,
                runtime_cached,
                dtype,
                text_conditioner,
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            temporary = f"{path}.tmp.{os.getpid()}"
            torch.save(cache, temporary)
            os.replace(temporary, path)
            if distributed:
                dist.barrier()
        else:
            dist.barrier()
            cache = torch.load(path, map_location="cpu", weights_only=False)
    else:
        cache = _encode_prompts(
            config,
            prompts,
            device,
            runtime_cached,
            dtype,
            text_conditioner,
        )

    missing = [prompt for prompt in prompts if prompt not in cache]
    if missing:
        raise KeyError(f"Prompt cache is missing {len(missing)} required prompts.")
    return cache

def _global_block(block_index: int) -> Any:
    total_frames = FIRST_BLOCK_FRAMES + int(block_index) * FRAMES_PER_BLOCK
    blocks = compute_av_blocks(
        total_frames,
        num_frame_per_block_first=FIRST_BLOCK_FRAMES,
    )
    return blocks[int(block_index)]

def _validate_prefix(video: torch.Tensor, audio: Optional[torch.Tensor]) -> None:
    if audio is None:
        raise RuntimeError("The generated audio prefix is missing.")
    expected = compute_aligned_audio_frames(
        int(video.shape[1]),
        num_frame_per_block_first=FIRST_BLOCK_FRAMES,
    )
    if int(audio.shape[1]) != expected:
        raise RuntimeError(
            f"Prefix alignment mismatch: {video.shape[1]} video frames require "
            f"{expected} audio frames, got {audio.shape[1]}."
        )

def _refresh_conditioning_cache(
    generator: StreamingGenerator,
    cache: Any,
    conditioning: Dict[str, Any],
    prefix_video: torch.Tensor,
    prefix_audio: torch.Tensor,
    next_video_shape: Tuple[int, ...],
    memory: StreamingMemory,
) -> Any:
    segments = cache_segments(cache)
    if not segments:
        raise RuntimeError("The retained cache has no block metadata.")
    _validate_prefix(prefix_video, prefix_audio)

    total_frames = int(prefix_video.shape[1]) + int(next_video_shape[1])
    local_blocks = compute_av_blocks(
        total_frames,
        num_frame_per_block_first=FIRST_BLOCK_FRAMES,
    )
    prefix_blocks = [
        block for block in local_blocks if int(block.video_end) <= prefix_video.shape[1]
    ]
    has_sink = int(segments[0]["block_idx"]) == 0
    warm_count = len(segments) - int(has_sink)
    warm_candidates = prefix_blocks[1:] if has_sink else prefix_blocks
    selected = (
        [prefix_blocks[0], *warm_candidates[-warm_count:]]
        if has_sink and warm_count
        else [prefix_blocks[0]]
        if has_sink
        else warm_candidates[-warm_count:]
    )
    if len(selected) != len(segments):
        raise RuntimeError("The retained latent prefix does not match the cache layout.")

    model = generator.generator.model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    rebuilt = None
    with torch.no_grad():
        for local_block, segment in zip(selected, segments):
            block = _global_block(int(segment["block_idx"]))
            video = prefix_video[
                :, int(local_block.video_start) : int(local_block.video_end)
            ].to(device=device, dtype=dtype)
            audio = prefix_audio[
                :, int(local_block.audio_start) : int(local_block.audio_end)
            ].to(device=device, dtype=dtype)
            block_conditioning = memory.with_conditional_memory(
                conditioning, device=device, dtype=dtype
            )
            _, _, rebuilt = model.forward_inference(
                video_latent=video,
                audio_latent=audio,
                timesteps=torch.zeros(
                    (video.shape[0], video.shape[1]), device=device, dtype=dtype
                ),
                audio_timesteps=torch.zeros(
                    (audio.shape[0], audio.shape[1]), device=device, dtype=dtype
                ),
                video_context=block_conditioning["video_context"],
                audio_context=block_conditioning["audio_context"],
                video_context_mask=block_conditioning.get("video_context_mask"),
                audio_context_mask=block_conditioning.get("audio_context_mask"),
                learned_memory_video=block_conditioning.get("learned_memory_video"),
                learned_memory_audio=block_conditioning.get("learned_memory_audio"),
                learned_memory_color=block_conditioning.get("learned_memory_color"),
                kv_cache=rebuilt,
                video_start_frame=int(block.video_start),
                audio_start_frame=int(block.audio_start),
                include_audio_sinks=False,
                pyramid_policy=generator.pipeline.pyramid_policy,
                kv_cache_only=True,
            )

    if rebuilt is None or len(rebuilt.layers) != len(cache.layers):
        raise RuntimeError("Conditioning cache refresh failed.")
    for retained_layer, rebuilt_layer in zip(cache.layers, rebuilt.layers):
        retained_layer.audio_self_k = rebuilt_layer.audio_self_k
        retained_layer.audio_self_v = rebuilt_layer.audio_self_v
        retained_layer.a2v_k = rebuilt_layer.a2v_k
        retained_layer.a2v_v = rebuilt_layer.a2v_v
        retained_layer.v2a_k = rebuilt_layer.v2a_k
        retained_layer.v2a_v = rebuilt_layer.v2a_v
    return attach_segments(cache, segments)

def _next_prefix(
    prefix_video: Optional[torch.Tensor],
    prefix_audio: Optional[torch.Tensor],
    sink_video: Optional[torch.Tensor],
    sink_audio: Optional[torch.Tensor],
    video: torch.Tensor,
    audio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if prefix_video is None:
        sink_video = video[:, :FIRST_BLOCK_FRAMES].detach().clone()
        sink_audio_frames = compute_aligned_audio_frames(
            FIRST_BLOCK_FRAMES,
            num_frame_per_block_first=FIRST_BLOCK_FRAMES,
        )
        sink_audio = audio[:, :sink_audio_frames].detach().clone()
        next_video = video.detach()
        next_audio = audio.detach()
    else:
        if sink_video is None or sink_audio is None or prefix_audio is None:
            raise RuntimeError("The persistent prefix is incomplete.")
        recent_video = torch.cat((prefix_video[:, FIRST_BLOCK_FRAMES:], video), dim=1)
        next_video = torch.cat((sink_video, recent_video[:, -12:]), dim=1)
        expected_audio = compute_aligned_audio_frames(
            int(next_video.shape[1]),
            num_frame_per_block_first=FIRST_BLOCK_FRAMES,
        )
        recent_audio_frames = expected_audio - int(sink_audio.shape[1])
        recent_audio = torch.cat((prefix_audio[:, sink_audio.shape[1] :], audio), dim=1)
        next_audio = torch.cat((sink_audio, recent_audio[:, -recent_audio_frames:]), dim=1)
    _validate_prefix(next_video, next_audio)
    return next_video, next_audio, sink_video, sink_audio

def preload_inference(config: InferenceConfig) -> None:
    rank, _local_rank, _is_writer, device, dtype = _distributed_context()
    _get_runtime(config, rank, device, dtype)
    if dist.is_initialized():
        dist.barrier()

def run_inference(config: InferenceConfig) -> None:
    rank, _local_rank, is_writer, device, dtype = _distributed_context()
    prepared_cases = _load_cases(config)
    prompts = sorted(
        {
            segment["prompt"]
            for _case, segments in prepared_cases
            for segment in segments
        }
    )
    runtime_key = _runtime_key(config, device)
    runtime_cached = bool(
        config.stream and runtime_key in _RUNTIME_CACHE
    )
    runtime = _get_runtime(config, rank, device, dtype) if config.conditioning_device else None
    prompt_cache = _prompt_cache(
        config,
        prompts,
        rank,
        device,
        runtime_cached,
        dtype,
        runtime.text_conditioner if runtime is not None else None,
    )
    if runtime is None:
        runtime = _get_runtime(config, rank, device, dtype)
    os.makedirs(config.output_dir, exist_ok=True)
    from torchvision.io import write_video

    if is_writer:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(
            f"[Run] {len(prepared_cases)} cases, "
            f"{config.video_width}x{config.video_height}, {world_size} GPU worker(s)"
        )

    for case_index, (case, segments) in enumerate(prepared_cases):
        case_id = str(case["case_id"])
        description = case.get("description") or case_id
        if is_writer:
            print(
                f"[Case {case_index + 1}/{len(prepared_cases)}] "
                f"{case_id}: {description}"
            )

        engine = runtime.generator
        memory = StreamingMemory()
        cache = None
        cache_prompt: Optional[str] = None
        prefix_video: Optional[torch.Tensor] = None
        prefix_audio: Optional[torch.Tensor] = None
        sink_video: Optional[torch.Tensor] = None
        sink_audio: Optional[torch.Tensor] = None
        decode_prefix: Optional[torch.Tensor] = None
        all_video: List[torch.Tensor] = []
        all_audio: List[torch.Tensor] = []
        global_block_offset = 0
        started = time.perf_counter()

        for segment_index, segment in enumerate(segments):
            prompt = segment["prompt"]
            seed = int(segment["seed"]) + segment_index
            conditioning = tensor_tree_to_device(prompt_cache[prompt], device)
            if prefix_video is not None and cache_prompt != prompt:
                cache = _refresh_conditioning_cache(
                    engine,
                    cache,
                    conditioning,
                    prefix_video,
                    prefix_audio,
                    runtime.next_video_shape,
                    memory,
                )

            streamer = None
            if config.stream and is_writer:
                streamer = RealtimeBlockStreamer(
                    write_video=write_video,
                    stream_dir=os.path.join(config.output_dir, f"{case_id}_streams"),
                    case_id=case_id,
                    segment_index=segment_index,
                    video_vae=runtime.video_vae,
                    audio_vae=runtime.audio_vae,
                    initial_video_prefix=decode_prefix,
                    output_video_height=config.output_video_height,
                    output_video_width=config.output_video_width,
                    device=config.media_device or device,
                )
                callback_index = 0

                def publish_block(
                    block_video: torch.Tensor,
                    block_audio: Optional[torch.Tensor],
                    _block: Any,
                ) -> None:
                    nonlocal callback_index
                    streamer.submit(callback_index, block_video, block_audio)
                    callback_index += 1

            else:
                publish_block = None

            window_started = time.perf_counter()
            try:
                if prefix_video is None:
                    video, audio = engine.generate_first_window(
                        runtime.first_video_shape,
                        runtime.first_audio_shape,
                        conditioning,
                        seed,
                        memory,
                        publish_block,
                    )
                else:
                    video, audio = engine.generate_next_window(
                        runtime.next_video_shape,
                        runtime.next_audio_shape,
                        conditioning,
                        seed,
                        memory,
                        cache,
                        global_block_offset,
                        publish_block,
                    )
                cache = retain_streaming_context(engine.last_cache)
                cache_prompt = prompt
                global_block_offset += BLOCKS_PER_WINDOW

                if is_writer and not config.stream:
                    all_video.append(video.detach().cpu())
                    all_audio.append(audio.detach().cpu())
                if is_writer and config.stream:
                    context = video.detach() if decode_prefix is None else torch.cat(
                        (decode_prefix, video.detach()), dim=1
                    )
                    decode_prefix = context[:, -8:].contiguous()

                prefix_video, prefix_audio, sink_video, sink_audio = _next_prefix(
                    prefix_video,
                    prefix_audio,
                    sink_video,
                    sink_audio,
                    video,
                    audio,
                )
            finally:
                if streamer is not None:
                    streamer.close()

            if is_writer:
                block_ids = ",".join(
                    str(item["block_idx"]) for item in cache_segments(cache)
                )
                print(
                    f"  Window {segment_index + 1}/{len(segments)} "
                    f"seed={seed} cache=[{block_ids}] "
                    f"time={time.perf_counter() - window_started:.1f}s"
                )
            del conditioning, video, audio
            torch.cuda.empty_cache()

        if not is_writer:
            continue
        if config.stream:
            print(f"  Stream generation complete in {time.perf_counter() - started:.1f}s")
            continue

        media_device = config.media_device or device
        video_latent = torch.cat(all_video, dim=1).to(media_device)
        pixels = decode_video_latents(
            runtime.video_vae,
            video_latent,
            media_device,
        )
        pixels = center_crop_video(
            pixels,
            config.output_video_height,
            config.output_video_width,
        )
        audio_latent = torch.cat(all_audio, dim=1).to(media_device)
        audio_wave, audio_rate = decode_audio_latents(
            runtime.audio_vae,
            audio_latent,
            media_device,
        )

        output_path = os.path.join(
            config.output_dir,
            f"{case_id}_step{runtime.train_step}.mp4",
        )
        write_valid_mp4(
            write_video,
            output_path,
            pixels,
            audio_wave,
            audio_rate,
        )
        duration = pixels.shape[0] / VIDEO_FPS
        print(
            f"  [Save] {output_path} ({pixels.shape[0]} frames, "
            f"{duration:.2f}s, {time.perf_counter() - started:.1f}s total)"
        )
        del video_latent, pixels, audio_wave
        gc.collect()
        torch.cuda.empty_cache()

    if not config.stream and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
