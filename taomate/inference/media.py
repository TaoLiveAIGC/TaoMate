import json
import os
import queue
import subprocess
import threading
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from ltx_core.model.video_vae.tiling import TilingConfig

VIDEO_FPS = 24

def center_crop_video(
    video: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    target_height = int(target_height or 0)
    target_width = int(target_width or 0)
    if target_height <= 0 and target_width <= 0:
        return video
    if video.ndim != 4:
        raise ValueError(f"Expected video [F,H,W,C], got {tuple(video.shape)}")

    height, width = int(video.shape[1]), int(video.shape[2])
    target_height = target_height or height
    target_width = target_width or width
    if (target_height, target_width) == (height, width):
        return video

    if target_height <= height and target_width <= width:
        crop_height, crop_width = target_height, target_width
    else:
        source_aspect = width / height
        target_aspect = target_width / target_height
        if source_aspect > target_aspect:
            crop_height = height
            crop_width = max(1, min(width, round(height * target_aspect)))
        else:
            crop_width = width
            crop_height = max(1, min(height, round(width / target_aspect)))

    top = max(0, (height - crop_height) // 2)
    left = max(0, (width - crop_width) // 2)
    cropped = video[
        :, top : top + crop_height, left : left + crop_width, :
    ].contiguous()
    if (crop_height, crop_width) == (target_height, target_width):
        return cropped
    resized = F.interpolate(
        cropped.permute(0, 3, 1, 2),
        size=(target_height, target_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return resized.permute(0, 2, 3, 1).contiguous()

def tensor_tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: tensor_tree_to_cpu(item) for key, item in value.items()}
    return value

def tensor_tree_to_device(value: Any, device: str) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: tensor_tree_to_device(item, device) for key, item in value.items()
        }
    return value

def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "out of memory" in message and any(
        marker in message for marker in ("cuda", "cublas", "gpu")
    )

def audio_to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim != 2:
        raise ValueError(f"Expected audio [C,N], got {tuple(audio.shape)}")
    return audio.mean(dim=0, keepdim=True).contiguous()

def ensure_module_device(module: torch.nn.Module, device: str) -> None:
    value = next(module.parameters(recurse=True), None)
    if value is None:
        value = next(module.buffers(recurse=True), None)
    current = None if value is None else str(value.device)
    if current != str(device):
        module.to(device)

def validate_video(path: str, expected_frames: int) -> None:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames,nb_frames",
            "-of",
            "json",
            path,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-1000:]}")
    streams = (json.loads(result.stdout or "{}").get("streams") or [])
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {path}")
    frame_count = int(
        streams[0].get("nb_read_frames") or streams[0].get("nb_frames") or 0
    )
    if frame_count != int(expected_frames):
        raise RuntimeError(
            f"ffprobe found {frame_count} frames in {path}, expected {expected_frames}"
        )

def _publish_video(temp_path: str, output_path: str, expected_frames: int) -> None:
    validate_video(temp_path, expected_frames)
    os.replace(temp_path, output_path)

def write_valid_mp4(
    write_video: Any,
    output_path: str,
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_rate: int,
) -> None:
    temp_path = f"{output_path}.tmp.{os.getpid()}.mp4"
    if os.path.exists(temp_path):
        os.remove(temp_path)
    write_video(
        temp_path,
        video,
        fps=VIDEO_FPS,
        audio_array=audio_to_mono(audio),
        audio_fps=int(audio_rate),
        audio_codec="aac",
    )
    _publish_video(temp_path, output_path, int(video.shape[0]))

def decode_video_latents(
    video_vae: Any,
    video_latent: torch.Tensor,
    device: str,
) -> torch.Tensor:
    ensure_module_device(video_vae.decoder, device)
    latent = video_latent.to(device).permute(0, 2, 1, 3, 4)
    chunks = []
    with torch.no_grad():
        for chunk in video_vae.decoder.decode_video(latent, TilingConfig.default()):
            chunks.append(chunk.cpu())
    return torch.cat(chunks, dim=0)

def decode_audio_latents(
    audio_vae: Any,
    audio_latent: torch.Tensor,
    device: str,
) -> Tuple[torch.Tensor, int]:
    ensure_module_device(audio_vae, device)
    latent = audio_latent.to(device)
    with torch.no_grad():
        waveform = audio_vae.decode_to_waveform(latent)
    audio = audio_to_mono(waveform[0].float().cpu())
    return audio, int(audio_vae.vocoder.output_sampling_rate)

def _write_preview_frame(output_path: str, pixels: torch.Tensor) -> None:
    from torchvision.io import write_jpeg

    temp_path = f"{output_path}.tmp.{os.getpid()}.jpg"
    frame = pixels[0].detach().to(dtype=torch.uint8).permute(2, 0, 1).contiguous()
    write_jpeg(frame, temp_path, quality=90)
    os.replace(temp_path, output_path)

class RealtimeBlockStreamer:
    def __init__(
        self,
        *,
        write_video: Any,
        stream_dir: str,
        case_id: str,
        segment_index: int,
        video_vae: Any,
        audio_vae: Any,
        initial_video_prefix: Optional[torch.Tensor],
        output_video_height: int,
        output_video_width: int,
        device: str,
    ) -> None:
        self._write_video = write_video
        self._stream_dir = stream_dir
        self._case_id = case_id
        self._segment_index = int(segment_index)
        self._video_vae = video_vae
        self._audio_vae = audio_vae
        self._output_height = int(output_video_height or 0)
        self._output_width = int(output_video_width or 0)
        self._device = device
        self._context = (
            initial_video_prefix.detach().to("cpu").contiguous()
            if initial_video_prefix is not None
            else None
        )
        self._queue: "queue.Queue[Optional[Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]" = queue.Queue(8)
        self._errors: List[str] = []
        self._publish_condition = threading.Condition()
        self._ready: Dict[int, Tuple[Optional[str], Optional[str]]] = {}
        self._next_publish = 0
        os.makedirs(stream_dir, exist_ok=True)
        ensure_module_device(video_vae.decoder, device)
        ensure_module_device(audio_vae, device)
        self._threads = [
            threading.Thread(
                target=self._run,
                name=f"taolive-stream-{case_id}-{index}",
                daemon=True,
            )
            for index in range(2)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        block_index: int,
        block_video: torch.Tensor,
        block_audio: torch.Tensor,
    ) -> None:
        video = block_video.detach().to("cpu").contiguous()
        audio = block_audio.detach().to("cpu").contiguous()
        context = self._context
        next_context = video if context is None else torch.cat((context, video), dim=1)
        self._context = next_context[:, -8:].contiguous()
        self._queue.put((int(block_index), video, audio, context))

    def close(self) -> None:
        for _ in self._threads:
            self._queue.put(None)
        self._queue.join()
        for thread in self._threads:
            thread.join()
        if self._errors:
            raise RuntimeError(self._errors[0])

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._write_one(*item)
            except Exception as exc:
                self._errors.append(str(exc))
                if item is not None:
                    self._mark_failed(item[0])
                print(f" [stream error: {exc}]", end="", flush=True)
            finally:
                self._queue.task_done()

    def _write_one(
        self,
        block_index: int,
        block_video: torch.Tensor,
        block_audio: torch.Tensor,
        context: Optional[torch.Tensor],
    ) -> None:
        stream_index = self._segment_index * 100 + block_index
        output_path = os.path.join(
            self._stream_dir,
            f"{self._case_id}_stream{stream_index:04d}.mp4",
        )
        ready_path = f"{output_path}.ready.{os.getpid()}.{threading.get_ident()}"

        audio_result: List[Optional[torch.Tensor]] = [None]
        audio_errors: List[BaseException] = []

        def decode_audio() -> None:
            try:
                audio_result[0], _ = decode_audio_latents(
                    self._audio_vae, block_audio, self._device
                )
            except BaseException as exc:
                audio_errors.append(exc)

        audio_thread = threading.Thread(target=decode_audio, daemon=True)
        audio_thread.start()

        full_latent = (
            block_video if context is None else torch.cat((context, block_video), dim=1)
        )
        pixels = decode_video_latents(
            self._video_vae, full_latent, self._device
        )
        if context is not None:
            context_frames = (int(context.shape[1]) - 1) * 8 + 1
            pixels = pixels[context_frames:].contiguous()
        pixels = center_crop_video(pixels, self._output_height, self._output_width)

        if stream_index == 0:
            _write_preview_frame(
                os.path.join(self._stream_dir, f"{self._case_id}_preview0000.jpg"),
                pixels,
            )

        audio_thread.join()
        if audio_errors:
            raise RuntimeError(str(audio_errors[0]))
        audio = audio_result[0]
        if audio is None:
            raise RuntimeError("Audio decoder returned no waveform.")
        write_valid_mp4(
            self._write_video,
            ready_path,
            pixels,
            audio=audio,
            audio_rate=int(self._audio_vae.vocoder.output_sampling_rate),
        )
        self._publish(block_index, ready_path, output_path)
        print(f" stream={os.path.basename(output_path)}", end="", flush=True)

    def _publish(self, block_index: int, ready_path: str, output_path: str) -> None:
        with self._publish_condition:
            self._ready[block_index] = (ready_path, output_path)
            self._publish_ready_locked()
            while block_index >= self._next_publish:
                self._publish_condition.wait(timeout=0.1)

    def _mark_failed(self, block_index: int) -> None:
        with self._publish_condition:
            self._ready[int(block_index)] = (None, None)
            self._publish_ready_locked()

    def _publish_ready_locked(self) -> None:
        while self._next_publish in self._ready:
            ready_path, output_path = self._ready.pop(self._next_publish)
            if ready_path and output_path:
                os.replace(ready_path, output_path)
            self._next_publish += 1
            self._publish_condition.notify_all()
