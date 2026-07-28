#!/usr/bin/env python3
"""FastAPI service for the TaoMate interactive demo."""

from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from contextlib import suppress
import difflib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"


def _now() -> float:
    return time.time()


def _utcish_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _safe_filename(name: str) -> str:
    stem = Path(name or "upload").stem
    suffix = Path(name or "").suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stem).strip("._") or "upload"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    return f"{stem}{suffix}"


def _safe_token(raw: str, *, prefix: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(raw or "")).strip("._")
    if not token:
        token = f"{prefix}_{uuid.uuid4().hex[:12]}"
    return token[:96]


def _normalize_scene_for_signature(scene: str) -> str:
    return re.sub(r"\s+", " ", str(scene or "").strip()).lower()


def _scene_signature(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "",
    refine_scene: bool = False,
) -> str:
    signature = {
        "scene": _normalize_scene_for_signature(scene),
        "template_id": str(template_id or "").strip(),
        "aspect_ratio": str(aspect_ratio or "").strip(),
    }
    if refine_scene:
        signature["refine_scene"] = True
    payload = json.dumps(
        signature,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _read_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _coerce_frames(duration_seconds: int) -> int:
    """Map requested seconds to LTX-valid frame count 1 + 8k."""
    seconds = max(5, min(int(duration_seconds or 5), 60))
    frames = int(round(seconds * 24))
    if (frames - 1) % 8 != 0:
        frames = ((frames - 1) // 8) * 8 + 1
    return max(121, frames)


ASPECT_PRESETS: Dict[str, Dict[str, Any]] = {
    "landscape": {"label": "横屏 864x480", "video_width": 864, "video_height": 480},
    "portrait": {"label": "竖屏 480x864", "video_width": 480, "video_height": 864},
}


def _resolve_aspect_ratio(raw: str) -> tuple[str, int, int]:
    key = (raw or "landscape").strip().lower().replace(" ", "")
    aliases = {
        "16:9": "landscape",
        "9:16": "portrait",
        "864x480": "landscape",
        "480x864": "portrait",
        "horizontal": "landscape",
        "vertical": "portrait",
    }
    key = aliases.get(key, key)
    if key not in ASPECT_PRESETS:
        raise HTTPException(status_code=400, detail="aspect_ratio must be landscape or portrait")
    preset = ASPECT_PRESETS[key]
    return key, int(preset["video_width"]), int(preset["video_height"])


def _resolve_internal_video_size(
    video_width: int,
    video_height: int,
    *,
    configured_width: int,
    configured_height: int,
    allow_cover_resize: bool = False,
) -> tuple[int, int]:
    """Resolve the generation canvas used before output framing."""
    output_width = int(video_width)
    output_height = int(video_height)
    internal_width = int(configured_width or 0)
    internal_height = int(configured_height or 0)
    if internal_width <= 0 or internal_height <= 0:
        return output_width, output_height
    if allow_cover_resize:
        return internal_width, internal_height
    if internal_width < output_width or internal_height < output_height:
        return output_width, output_height
    return internal_width, internal_height


def _generation_prompt_aspect_ratio(
    output_aspect_ratio: str,
    internal_width: int,
    internal_height: int,
) -> str:
    """Describe the canvas seen by DiT, not the delivery crop."""
    width = int(internal_width or 0)
    height = int(internal_height or 0)
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    return str(output_aspect_ratio or "landscape").strip().lower() or "landscape"


def _job_prompt_aspect_ratio(job: "JobState") -> str:
    return _generation_prompt_aspect_ratio(
        job.aspect_ratio,
        job.internal_video_width or job.video_width,
        job.internal_video_height or job.video_height,
    )


def _job_scene_prompt_signature(job: "JobState") -> str:
    return f"{job.scene_signature}:{_job_prompt_aspect_ratio(job)}"


@dataclass
class ServiceSettings:
    runs_root: Path = Path(os.environ.get("AVATAR_RUNS_ROOT", PROJECT_ROOT / "outputs" / "interactive_avatar_runs"))
    model_ckpt: str = os.environ.get("MODEL_CKPT", "")
    original_ckpt: str = os.environ.get("BASE_MODEL_CKPT", "")
    gemma_path: str = os.environ.get("GEMMA_PATH", "")
    max_segments: int = 6
    segment_seconds: int = 5
    reply_max_chars: int = 50
    prompt_speech_max_chars: int = 512
    dynamic_utterance_streaming: bool = True
    dynamic_prompt_use_history_tail: bool = True
    interactive_prompt_tail_chars: int = 64
    history_tail_in_speaker_says: bool = False
    speech_segment_visible_chars: int = 35
    num_frame_per_block: int = 3
    num_frame_per_block_first: int = 4
    video_height: int = 480
    video_width: int = 864
    internal_video_height: int = int(os.environ.get("INTERNAL_VIDEO_HEIGHT", "0"))
    internal_video_width: int = int(os.environ.get("INTERNAL_VIDEO_WIDTH", "0"))
    portrait_internal_video_height: int = 480
    portrait_internal_video_width: int = 864
    add_waiting_transition: bool = False
    llm_base_url: str = os.environ.get("DIALOGUE_API_BASE", "http://127.0.0.1:7864/v1").rstrip("/")
    llm_model: str = os.environ.get("DIALOGUE_MODEL_NAME", "taolive-dialogue")
    llm_api_key: str = os.environ.get("DIALOGUE_API_KEY", "EMPTY")
    runner_timeout_seconds: int = int(os.environ.get("AVATAR_RUNNER_TIMEOUT", "7200"))
    preload_worker_on_start: bool = True
    warmup_generate_on_start: bool = True
    realtime_stream_chunk_seconds: float = 1.0
    realtime_stream_blocks_per_chunk: int = 1
    asr_enabled: bool = True
    asr_model: str = "tiny"
    asr_model_path: str = os.environ.get("ASR_MODEL_PATH", str(PROJECT_ROOT / "models" / "whisper" / "tiny.pt"))
    asr_device: str = os.environ.get("ASR_DEVICE", "cuda:3")
    asr_allow_download: bool = False
    asr_preload_on_start: bool = True
    asr_min_seconds: float = 3.0
    asr_check_interval_seconds: float = 0.1
    asr_completion_ratio: float = 0.76
    asr_tail_fuzzy_ratio: float = 0.72
    asr_budget_chars_per_second: float = 7.0
    asr_budget_extra_seconds: float = 0.5
    asr_budget_min_coverage: float = 0.58
    asr_budget_hard_stop: bool = False
    asr_budget_medium_char_threshold: int = 31
    asr_budget_medium_min_seconds: float = 6.0
    asr_budget_long_char_threshold: int = 38
    asr_budget_long_min_seconds: float = 9.0
    asr_budget_comma_pause_seconds: float = 0.25
    asr_budget_sentence_pause_seconds: float = 0.45
    asr_budget_prewrite_chunks: int = 1
    asr_tail_seconds_after_done: float = 0.0
    asr_tail_blocks_after_done: int = 1
    asr_tail_silence_seconds: float = 0.25
    asr_tail_silence_dbfs: float = -45.0
    asr_final_timeout_seconds: float = 30.0
    asr_dynamic_stop: bool = True
    prompt_plan_cache_dir: Path = Path(os.environ.get("PROMPT_CACHE_DIR", PROJECT_ROOT / "outputs" / "interactive_avatar_prompt_cache"))
    interactive_video_prefix_enabled: bool = True
    fast_live5_prompt_planner: bool = False
    use_llm_prompt_planner: bool = True
    scene_prompt_pe_enabled: bool = True
    scene_prompt_pe_max_tokens: int = 900
    use_ltx_prompt_skeleton_cache: bool = True
    ltx_prompt_skeleton_max_examples: int = 2
    ltx_prompt_skeleton_max_tokens: int = 2200
    prompt_planner_max_tokens: int = 8192
    prompt_planner_max_examples: int = 32
    compact_ltx_prompt_for_latency: bool = False
    worker_status_poll_interval: float = 0.03
    sse_poll_interval: float = 0.03
    worker_queue_dir: Path = Path(os.environ.get("AVATAR_WORKER_QUEUE_DIR", PROJECT_ROOT / "outputs" / "interactive_avatar_worker_queue"))


SETTINGS = ServiceSettings()


PROMPT_PLANNER_PROFILE = "realtime_embodied_dialogue"
PROMPT_PLANNER_PROFILE_DESCRIPTION = (
    "Realtime embodied-dialogue prompt compiler with stable scene anchors, smooth controlled motion, "
    "upper-body camera tracking, clear user-directed actions, responsive facial emotion, clean photographic framing, "
    "and the exact spoken line isolated inside Speaker_1 says."
)


LANDSCAPE_CLOSE_COMPOSITION = (
    "The speaker fills about two thirds of the frame height from the top of the head to mid-torso, "
    "and the shoulders span roughly half the frame width. Compact headroom and slim side margins keep "
    "the face and upper torso dominant, while both hands may enter the lower quarter for small gestures. "
    "Only a narrow band of the furniture edge remains visible along the bottom of the frame."
)

PORTRAIT_CLOSE_COMPOSITION = (
    "The speaker fills about three quarters of the upright frame height from the top of the head to mid-torso. "
    "Compact headroom and slim side margins keep the face, shoulders, and upper torso dominant, while both hands "
    "may enter the lower quarter for small gestures. Only a narrow band of the furniture edge remains visible at the bottom."
)


LANDSCAPE_DIRECT_CAMERA_TAKE_CLAUSE = (
    "This is a direct natural camera take of the speaker in the physical setting. "
    f"{LANDSCAPE_CLOSE_COMPOSITION} "
    "An eye-level 50mm-equivalent rectilinear lens keeps facial proportions natural and background verticals straight. "
    "The broad soft key and gentle neutral fill create one coherent light direction, with controlled highlights and shadows attached to the objects that cast them. "
    "The speaker's clothing, hands, furniture edge, and physical setting continue naturally through the lower part of the frame from left to right. "
    "Every prop rests on a visible supporting surface; garment folds, furniture seams, hand contours, natural shadows, and perspective remain continuous all the way to the bottom edge. "
    "Every visible surface remains a photographed physical material with uninterrupted texture. "
    "Crisp eyes, natural skin microtexture, clean garment weave, well-resolved edges, and moderate depth of field give the image a polished high-end finish. "
    "The response is embodied through facial reaction, gesture, natural recorded voice, and synchronized mouth movement."
)

PORTRAIT_DIRECT_CAMERA_TAKE_CLAUSE = (
    "This is a direct natural portrait camera take of the speaker in the physical setting. "
    f"{PORTRAIT_CLOSE_COMPOSITION} "
    "An eye-level 50mm-equivalent rectilinear lens keeps facial proportions natural and background verticals straight. "
    "The broad soft key and gentle neutral fill create one coherent light direction, with controlled highlights and shadows attached to the objects that cast them. "
    "The upright frame is filled edge to edge by the photographed face, shoulders, torso, clothing, and softly focused surroundings. "
    "Every prop rests on a visible supporting surface; the torso, garment folds, hand contours, furniture seams, natural shadows, and perspective continue naturally beyond the bottom edge. "
    "Every visible surface remains a photographed physical material with uninterrupted texture. "
    "Crisp eyes, natural skin microtexture, clean garment weave, well-resolved edges, and moderate depth of field give the image a polished high-end finish. "
    "The response is embodied through facial reaction, gesture, natural recorded voice, and synchronized mouth movement."
)


class Segment(BaseModel):
    segment_id: int
    speech: str
    emotion: str = "calm and helpful"
    action: str = "keeps eye contact and makes a tiny nod"
    explicit_action: bool = False
    prompt: str


@dataclass(frozen=True)
class TurnPlan:
    speech: str
    emotion: str
    action: str
    explicit_action: bool = False
    language: str = "en"


class PreviewRequest(BaseModel):
    scene_description: str
    user_text: str
    mode: str = "auto"
    has_first_frame: bool = False
    aspect_ratio: str = "landscape"
    template_id: str = ""
    refine_scene: bool = False
    conversation_id: str = ""


class DemoRegisterRequest(BaseModel):
    title: str = ""
    caption: str = ""
    fps_label: str = ""


class JobSnapshot(BaseModel):
    task_id: str
    status: str
    phase: str
    created_at: float
    updated_at: float
    scene_description: str
    user_text: str
    mode: str
    reply: Optional[str] = None
    segments: List[Segment] = []
    videos: List[str] = []
    error: Optional[str] = None
    events_url: str
    videos_url: str


@dataclass
class JobState:
    task_id: str
    task_dir: Path
    scene_description: str
    user_text: str
    mode: str
    conversation_id: str = ""
    previous_job_id: Optional[str] = None
    prefix_state_in: Optional[str] = None
    prefix_state_out: Optional[str] = None
    aspect_ratio: str = "landscape"
    video_width: int = field(default_factory=lambda: SETTINGS.video_width)
    video_height: int = field(default_factory=lambda: SETTINGS.video_height)
    internal_video_width: int = 0
    internal_video_height: int = 0
    template_id: str = ""
    refine_scene: bool = False
    scene_signature: str = ""
    scene_prompt_text: str = ""
    scene_prompt_source: str = ""
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    status: str = "accepted"
    phase: str = "accepted"
    reply: Optional[str] = None
    response_language: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    speech_completion: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cancel_requested: bool = False
    generation_done: bool = False
    _video_seen_urls: set[str] = field(default_factory=set, repr=False)

    @property
    def events_file(self) -> Path:
        return self.task_dir / "events.jsonl"

    @property
    def status_file(self) -> Path:
        return self.task_dir / "status.json"

    @property
    def prompts_file(self) -> Path:
        return self.task_dir / "prompts.json"

    @property
    def output_dir(self) -> Path:
        return self.task_dir / "output" / f"infer_{self.task_id}"


JOBS: Dict[str, JobState] = {}
QUEUE: asyncio.Queue[str] = asyncio.Queue()
WORKER_TASK: Optional[asyncio.Task] = None
BACKGROUND_TASKS: "set[asyncio.Task[Any]]" = set()
PLAN_SEGMENTS_CACHE: "OrderedDict[Tuple[Any, ...], List[Dict[str, Any]]]" = OrderedDict()
PLAN_SEGMENTS_CACHE_SIZE = int(os.environ.get("AVATAR_PLAN_SEGMENTS_CACHE_SIZE", "64"))
SCENE_PROMPT_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, str]]" = OrderedDict()
SCENE_PROMPT_CACHE_SIZE = int(os.environ.get("AVATAR_SCENE_PROMPT_CACHE_SIZE", "64"))
LTX_PROMPT_SKELETON_CACHE: "OrderedDict[Tuple[Any, ...], Dict[str, str]]" = OrderedDict()
LTX_PROMPT_SKELETON_CACHE_SIZE = int(
    os.environ.get("AVATAR_LTX_PROMPT_SKELETON_CACHE_SIZE", "64")
)


def _persistent_prompt_cache_path(namespace: str, key: Tuple[Any, ...]) -> Path:
    key_json = json.dumps(list(key), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    return SETTINGS.prompt_plan_cache_dir / namespace / f"{digest}.json"


def _persistent_prompt_cache_get(
    namespace: str,
    key: Tuple[Any, ...],
) -> Optional[Dict[str, str]]:
    path = _persistent_prompt_cache_path(namespace, key)
    payload = _read_optional_json(path)
    if not isinstance(payload, dict) or payload.get("key") != list(key):
        return None
    value = payload.get("value")
    if not isinstance(value, dict):
        return None
    return {str(field): str(text) for field, text in value.items()}


def _persistent_prompt_cache_put(
    namespace: str,
    key: Tuple[Any, ...],
    value: Dict[str, str],
) -> None:
    path = _persistent_prompt_cache_path(namespace, key)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "key": list(key),
            "value": dict(value),
            "updated_at": _now(),
        }
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except OSError:
        with suppress(OSError):
            tmp.unlink()


app = FastAPI(title="TaoMate Interactive Demo", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _job_snapshot(job: JobState) -> Dict[str, Any]:
    return {
        "task_id": job.task_id,
        "status": job.status,
        "phase": job.phase,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "scene_description": job.scene_description,
        "user_text": job.user_text,
        "mode": job.mode,
        "conversation_id": job.conversation_id,
        "previous_job_id": job.previous_job_id,
        "prefix_state_in": job.prefix_state_in,
        "prefix_state_out": job.prefix_state_out,
        "is_continuation": bool(job.prefix_state_in),
        "aspect_ratio": job.aspect_ratio,
        "prompt_aspect_ratio": _job_prompt_aspect_ratio(job),
        "video_width": job.video_width,
        "video_height": job.video_height,
        "internal_video_width": job.internal_video_width or job.video_width,
        "internal_video_height": job.internal_video_height or job.video_height,
        "template_id": job.template_id,
        "refine_scene": job.refine_scene,
        "scene_signature": job.scene_signature,
        "scene_prompt_source": job.scene_prompt_source,
        "reply": job.reply,
        "response_language": job.response_language,
        "segments": job.segments,
        "videos": job.videos,
        "speech_completion": job.speech_completion,
        "error": job.error,
        "events_url": f"/api/jobs/{job.task_id}/events",
        "videos_url": f"/api/jobs/{job.task_id}/videos",
    }


def _demo_index_path() -> Path:
    return SETTINGS.runs_root / "demos.json"


def _read_demo_entries() -> List[Dict[str, Any]]:
    path = _demo_index_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [
        _normalize_demo_entry(item)
        for item in data
        if isinstance(item, dict)
    ]


def _write_demo_entries(entries: List[Dict[str, Any]]) -> None:
    path = _demo_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _demo_chunk_index(video_url: str) -> Optional[int]:
    match = re.search(r"_chunk([0-9]+)\.mp4$", str(video_url or ""))
    if match:
        return int(match.group(1))
    match = re.search(r"_stream([0-9]+)\.mp4$", str(video_url or ""))
    if match:
        return int(match.group(1)) // 100
    return None


def _demo_task_id(entry: Dict[str, Any]) -> str:
    return str(entry.get("task_id") or entry.get("id") or "").strip()


def _demo_stream_urls(entry: Dict[str, Any]) -> List[str]:
    urls = [
        str(url)
        for url in entry.get("videos") or []
        if isinstance(url, str) and re.search(r"_(stream|chunk)\d+\.mp4$", url)
    ]
    urls.sort(key=lambda url: (_demo_chunk_index(url) is None, _demo_chunk_index(url) or 0, url))
    if urls:
        return urls
    video_url = str(entry.get("video_url") or "")
    return [video_url] if video_url else []


def _stream_index_parts(index: int) -> Tuple[int, int]:
    # Legacy realtime inference used sparse stream ids: segment 0 -> 0000..0004,
    # segment 1 -> 0100..0104. New live requests use contiguous ids, but this
    # sort key still keeps legacy assets in the intended order.
    index = int(index)
    return index // 100, index % 100


def _preview_stream_index(path: Path) -> Optional[int]:
    match = re.search(r"_(?:stream|chunk)([0-9]+)\.mp4$", path.name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _preview_stream_ordinal(path: Path) -> Optional[int]:
    """Map legacy ``seg*100+block`` filenames onto contiguous block order."""
    stream_index = _preview_stream_index(path)
    if stream_index is None:
        return None
    if stream_index < 100:
        return stream_index
    segment_idx, block_idx = _stream_index_parts(stream_index)
    blocks_per_generation_chunk = max(
        1,
        int(
            round(
                SETTINGS.segment_seconds
                / max(0.1, SETTINGS.realtime_stream_chunk_seconds)
            )
        ),
    )
    return segment_idx * blocks_per_generation_chunk + block_idx


def _media_path_sort_key(path: Path) -> Tuple[int, int, int, str]:
    stream_match = re.search(r"_stream([0-9]+)\.mp4$", path.name)
    if stream_match:
        segment_idx, block_idx = _stream_index_parts(int(stream_match.group(1)))
        return (0, segment_idx, block_idx, path.as_posix())
    chunk_match = re.search(r"_chunk([0-9]+)\.mp4$", path.name)
    if chunk_match:
        return (1, int(chunk_match.group(1)), 0, path.as_posix())
    return (2, 0, 0, path.as_posix())


def _stop_after_block_count(job: JobState) -> Optional[int]:
    generation_result = _read_optional_json(
        job.task_dir / "interactive_generation_result.json"
    )
    if isinstance(generation_result, dict):
        committed = generation_result.get("committed_generated_block_count")
        try:
            if committed is not None:
                return max(0, int(committed))
        except (TypeError, ValueError):
            pass

    payloads: List[Dict[str, Any]] = []
    if isinstance(job.speech_completion, dict):
        payloads.append(job.speech_completion)
    for path in (job.task_dir / "asr_stop_control.json", job.task_dir / "speech_completion.json"):
        payload = _read_optional_json(path)
        if isinstance(payload, dict):
            payloads.append(payload)
    for payload in payloads:
        stop_after = (
            payload.get("stop_after_block_count")
            or payload.get("stop_after_blocks")
            or payload.get("stop_after_chunk_count")
        )
        if stop_after is None:
            continue
        if payload.get("completed") is False and not payload.get("prewritten"):
            continue
        try:
            return max(0, int(stop_after))
        except (TypeError, ValueError):
            continue
    return None


def _should_publish_preview_media(job: JobState, path: Path, stop_after: Optional[int]) -> bool:
    if stop_after is None:
        return True
    stream_ordinal = _preview_stream_ordinal(path)
    if stream_ordinal is None:
        return True
    return int(stream_ordinal) < int(stop_after)


def _normalize_demo_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Keep demo subtitles aligned to the selected 5s chunk."""
    demo = dict(entry)
    task_id = _demo_task_id(demo)
    if task_id:
        demo["live_events_url"] = f"/api/demos/{task_id}/live_events"
    video_url = str(demo.get("video_url") or "")
    chunk_idx = _demo_chunk_index(video_url)
    segments = demo.get("segments")
    if chunk_idx is not None and isinstance(segments, list) and chunk_idx < len(segments):
        segment = segments[chunk_idx]
        if isinstance(segment, dict):
            speech = str(segment.get("speech") or "").strip()
            demo["segments"] = [segment]
            if speech:
                demo["caption"] = speech
                demo["reply"] = speech
    return demo


def _public_missing_items(missing: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    public_items = []
    for item in missing:
        if not isinstance(item, dict):
            continue
        for key in item.keys():
            public_items.append({str(key): "missing"})
    return public_items


def _public_worker_heartbeat(heartbeat: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(heartbeat, dict):
        return None
    public = {}
    for key in ("time", "status", "phase", "model_residency", "model_loaded", "ready"):
        if key in heartbeat:
            public[key] = heartbeat[key]
    return public or {"available": True}


def _public_gpu_snapshot(gpu: Dict[str, Any]) -> Dict[str, Any]:
    public = {
        "available": bool(gpu.get("available")),
        "busy": bool(gpu.get("busy")),
        "busy_count": int(gpu.get("busy_count") or 0),
    }
    if gpu.get("error"):
        public["error"] = str(gpu.get("error"))
    processes = []
    for process in gpu.get("processes") or []:
        if not isinstance(process, dict):
            continue
        processes.append(
            {
                "gpu": process.get("gpu"),
                "type": process.get("type"),
                "sm": process.get("sm"),
                "mem": process.get("mem"),
            }
        )
    public["processes"] = processes
    return public


def _demo_from_job(
    job: JobState,
    *,
    title: str = "",
    caption: str = "",
    fps_label: str = "",
) -> Optional[Dict[str, Any]]:
    if not job.videos:
        return None
    video_url = job.videos[0]
    chunk_idx = _demo_chunk_index(video_url)
    demo_segments = job.segments
    if chunk_idx is not None and chunk_idx < len(job.segments):
        demo_segments = [job.segments[chunk_idx]]
    speech = ""
    if demo_segments:
        speech = " ".join(
            str(segment.get("speech", "")).strip()
            for segment in demo_segments
            if isinstance(segment, dict)
        ).strip()
    return _normalize_demo_entry({
        "id": job.task_id,
        "task_id": job.task_id,
        "title": title.strip() or "实时数字人样例",
        "caption": caption.strip() or speech or job.user_text,
        "fps_label": fps_label.strip() or "24 fps target",
        "video_url": video_url,
        "videos": job.videos,
        "mode": job.mode,
        "aspect_ratio": job.aspect_ratio,
        "video_width": job.video_width,
        "video_height": job.video_height,
        "scene_description": job.scene_description,
        "user_text": job.user_text,
        "reply": speech or job.reply,
        "segments": demo_segments,
        "created_at": job.created_at,
    })


def _persist_status(job: JobState) -> None:
    job.updated_at = _now()
    job.task_dir.mkdir(parents=True, exist_ok=True)
    job.status_file.write_text(
        json.dumps(_job_snapshot(job), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def _emit(job: JobState, event: str, payload: Dict[str, Any]) -> None:
    record = {
        "time": _now(),
        "event": event,
        "task_id": job.task_id,
        **payload,
    }
    job.task_dir.mkdir(parents=True, exist_ok=True)
    with job.events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    _persist_status(job)


def _dominant_response_language(text: str) -> str:
    value = str(text or "")
    cjk_count = len(re.findall(r"[\u3400-\u9fff]", value))
    latin_words = re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", value)
    if cjk_count and not latin_words:
        return "zh"
    if latin_words and not cjk_count:
        return "en"
    if not cjk_count and not latin_words:
        return "en"

    cjk_score = cjk_count / 2.0
    english_score = float(len(latin_words))
    if abs(cjk_score - english_score) > 0.5:
        return "zh" if cjk_score > english_score else "en"

    first_cjk = re.search(r"[\u3400-\u9fff]", value)
    first_latin = re.search(r"[A-Za-z]", value)
    if first_cjk and first_latin:
        return "zh" if first_cjk.start() < first_latin.start() else "en"
    return "zh" if cjk_count else "en"


def _reply_max_chars(language: str) -> int:
    if language == "en":
        return max(120, SETTINGS.reply_max_chars * 2)
    return max(1, SETTINGS.reply_max_chars)


def _reply_matches_language(text: str, language: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    return _dominant_response_language(value) == language


def _forced_reply_from_user_text(user_text: str) -> Optional[str]:
    """Respect explicit demo-style requests such as '请只说：...'."""
    text = re.sub(r"\s+", " ", str(user_text or "").strip())
    patterns = [
        r"(?:请)?只说[：:「“\"]?\s*(.+)",
        r"(?:请)?说[：:「“\"]\s*(.+)",
        r"(?:please\s+)?(?:only\s+)?say\s*(?:[:：]|[\"“])\s*(.+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.I)
        if not match:
            continue
        forced = match.group(1).strip().strip("。.!！?？\"“”'’‘」")
        forced = re.split(r"(?:。|！|!|？|\?)\s*(?:不要|别|禁止|无需)", forced, maxsplit=1)[0]
        forced = forced.strip("。.!！?？\"“”'’‘」")
        if forced:
            language = _dominant_response_language(forced)
            return _sanitize_speech_text(
                forced,
                max_chars=_reply_max_chars(language),
                language=language,
            )
    return None


def _call_openai_compatible_llm(messages: List[Dict[str, str]], max_tokens: int) -> Optional[str]:
    if not SETTINGS.llm_base_url:
        return None
    payload = {
        "model": SETTINGS.llm_model,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urlrequest.Request(
        f"{SETTINGS.llm_base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SETTINGS.llm_api_key}",
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, TimeoutError, urlerror.URLError, urlerror.HTTPError, json.JSONDecodeError):
        return None


def _persona_for_scene(scene: str, template_id: str = "", language: str = "zh") -> str:
    key = f"{template_id} {scene}".lower()
    if "business_consultant" in key or "consult" in key or "business" in key:
        return (
            "Mature business consultant: restrained, professional, and direct; give the judgment first, then one actionable suggestion."
            if language == "en"
            else "成熟商务顾问：表达克制、专业、直接，先给判断，再给一个可执行建议。"
        )
    if "tech_anchor" in key or "technology" in key or "tech" in key:
        return (
            "Technology presenter: clear and structured, with a talent for turning complex ideas into simple steps."
            if language == "en"
            else "科技讲解员：语气清晰、有条理，善于把复杂问题拆成简单步骤。"
        )
    if "education_coach" in key or "classroom" in key or "education" in key:
        return (
            "Course instructor: warm and encouraging, explaining in accessible language without lecturing."
            if language == "en"
            else "课程讲师：亲和、鼓励式表达，用容易理解的话解释，不说教。"
        )
    if "wellness_host" in key or "wellness" in key or "plant" in key:
        return (
            "Wellness host: unhurried, gentle, and steady, avoiding exaggerated promises."
            if language == "en"
            else "健康讲解员：语速放慢，语气温和稳定，避免夸张承诺。"
        )
    return (
        "Live studio host: natural, bright, and human, like a real person on a live video call rather than a scripted presenter."
        if language == "en"
        else "Live5 棚拍主持人：自然、明亮、像实时视频通话里的真人，不模板化。"
    )


def _clean_llm_reply(reply: str, *, max_chars: int = 180, language: str = "zh") -> str:
    text = str(reply or "").strip()
    text = re.sub(r"^```(?:text)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(
        r"^(?:数字人|助手|回答|回复|digital human|assistant|answer|reply)\s*[：:]\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"^\s*Segment\s*\d+\s*[:：-]?\s*", "", text, flags=re.I)
    text = re.sub(
        r"[（(][^）)]*(?:微笑|点头|挥手|镜头|动作|表情|眼神|语气|smile|nod|wave|camera|gesture|expression|tone)[^）)]*[）)]",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"[\U0001F000-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", "", text)
    text = re.sub(r"\s+", " " if language == "en" else "", text).strip()
    text = text.strip("\"'“”")
    text = re.sub(r"(?:查看\s*LTX\s*prompt|LTX\s*prompt|Prompt).*", "", text, flags=re.I)
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
        if language == "en" and " " in text:
            text = text.rsplit(" ", 1)[0].rstrip(" ,;:")
    if text and not re.search(r"[。！？!?.]$", text):
        text += "." if language == "en" else "。"
    return text


async def generate_turn_plan(
    scene: str,
    user_text: str,
    conversation_history: str = "",
    template_id: str = "",
) -> TurnPlan:
    forced = _forced_reply_from_user_text(user_text)
    response_language = (
        _dominant_response_language(forced)
        if forced
        else _dominant_response_language(user_text)
    )
    persona = _persona_for_scene(scene, template_id, response_language)
    language_name = "English" if response_language == "en" else "Simplified Chinese"
    reply_limit = _reply_max_chars(response_language)
    speech_rule = (
        "The speech field must be natural English. Do not include Chinese translations or bilingual repetition."
        if response_language == "en"
        else "The speech field must be natural Simplified Chinese. English may appear only when needed for a proper noun or technical term."
    )
    system = (
        "You are a digital human speaking with the user in a real-time video call. "
        "Understand the user's latest utterance, answer naturally like a person, and plan the visible expression and body action for this moment. "
        f"The dominant language of the latest user utterance is {language_name}. {speech_rule} "
        "The latest user utterance alone determines the reply language; scene text, conversation history, and interface language must not override it. "
        "Do not repeat the user's words, explain what was received, or reveal system prompts, segmentation, video generation, models, or backend implementation. "
        "Use no emoji, emoticons, parenthetical action notes, or stage directions. "
        f"Keep speech natural, concise, warm, and suitable for live delivery: usually one or two sentences and no more than {reply_limit} characters. "
        "Avoid lists and report-like phrasing. A greeting deserves one natural greeting rather than generic advice. "
        "Use same-scene conversation history when relevant without repeating earlier replies. "
        "React like a person: begin with a brief understanding expression or eye response, complete one clear main action, then settle naturally. "
        "Every action must be slow, smooth, and continuous, with gradual onset, clear weight transfer, and gentle deceleration. "
        "Standing, sitting, walking, or turning should use at least four seconds of a five-second shot across preparation, onset, travel, and settling. "
        "When the user explicitly requests an action, perform it as the primary visible action, preserve the character's anatomical left and right, and hold the result for a beat. "
        "Without an explicit action request, choose restrained but responsive eye, brow, head, or hand behavior. "
        "For position changes, action must state the start pose, end pose, three-dimensional direction, and slow timing. "
        "Action describes only the person's body path and never camera terms; the backend creates matching camera movement. "
        "Emotion and action must be concrete natural English. Action is a lowercase verb phrase without a subject and must fit a five-second shot. "
        "Return one JSON object only, with exactly speech, emotion, action, and explicit_action. "
        "Speech contains only the spoken reply; emotion describes the visible expression; action describes temporal movement; explicit_action says whether the user requested a visible action. "
        f"Conversation persona: {persona}"
    )
    history_note = ""
    if conversation_history.strip():
        history_note = (
            "Earlier speech in this same scene, for context only and never for verbatim repetition:\n"
            f"{conversation_history.strip()[-320:]}\n"
        )
    prompt = (
        f"Current scene, used only for identity and tone rather than a visual description: {scene}\n"
        f"{history_note}"
        f"Latest user utterance: {user_text}\n"
        + (f"The speech field must use exactly this text: {forced}\n" if forced else "")
        + "Return the JSON for this turn."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    result = await asyncio.to_thread(
        _call_openai_compatible_llm,
        messages,
        420,
    )
    raw_result = str(result or "").strip()
    parsed = _extract_json_object(raw_result) or {}
    speech_source = forced or str(parsed.get("speech") or "")
    if not speech_source and not parsed:
        speech_match = re.search(r'"speech"\s*:\s*"((?:\\.|[^"\\])*)"', raw_result)
        if speech_match:
            with suppress(json.JSONDecodeError):
                speech_source = json.loads(f'"{speech_match.group(1)}"')
        elif not raw_result.lstrip().startswith(("{", "[", "```")):
            speech_source = raw_result
    if (
        not forced
        and speech_source
        and not _reply_matches_language(speech_source, response_language)
    ):
        correction = await asyncio.to_thread(
            _call_openai_compatible_llm,
            [
                *messages,
                {"role": "assistant", "content": raw_result},
                {
                    "role": "user",
                    "content": (
                        f"The speech field used the wrong language. Return the same JSON structure again, "
                        f"but write speech only in {language_name}. Keep emotion, action, and explicit_action semantically unchanged."
                    ),
                },
            ],
            420,
        )
        corrected_raw = str(correction or "").strip()
        corrected_parsed = _extract_json_object(corrected_raw) or {}
        corrected_speech = str(corrected_parsed.get("speech") or "")
        if corrected_speech and _reply_matches_language(
            corrected_speech,
            response_language,
        ):
            raw_result = corrected_raw
            parsed = corrected_parsed
            speech_source = corrected_speech
    speech = _clean_llm_reply(
        speech_source,
        max_chars=reply_limit,
        language=response_language,
    )
    if not speech:
        raise RuntimeError("The dialogue service did not return a usable response.")

    fallback_action, fallback_emotion = _fallback_turn_behavior(user_text)
    requested = _explicit_action_from_user_text(user_text)
    raw_action = str(parsed.get("action") or fallback_action)
    raw_emotion = str(parsed.get("emotion") or fallback_emotion)
    explicit_value = parsed.get("explicit_action", False)
    explicit_action = (
        explicit_value
        if isinstance(explicit_value, bool)
        else str(explicit_value).strip().lower() in {"1", "true", "yes"}
    )
    if requested:
        raw_action, raw_emotion = requested
        explicit_action = True
    action = _safe_action(raw_action, 0)
    emotion = _safe_emotion(raw_emotion)
    return TurnPlan(
        speech=speech,
        emotion=emotion,
        action=action,
        explicit_action=explicit_action,
        language=response_language,
    )


WAITING_TRANSITIONS = {
    "en": [
        "Let me gather the key point, and I will continue in a moment.",
        "I will stay right here and wait for your next thought.",
        "I have that direction in mind, and we can keep going.",
        "I will pause on this point until you add more.",
    ],
    "zh": [
        "我先整理一下重点，马上接着说。",
        "我继续保持在这里，等你下一句话。",
        "这个方向我记住了，我们可以接着聊。",
        "我先停在这个要点上，等你继续补充。",
    ],
}


def _sanitize_speech_text(
    speech: str,
    max_chars: int = 48,
    *,
    ensure_terminal: bool = True,
    language: Optional[str] = None,
) -> str:
    language = language or _dominant_response_language(speech)
    speech = re.sub(
        r"\s+",
        " " if language == "en" else "",
        str(speech or "").strip(),
    )
    speech = speech.strip("\"'“”")
    speech = re.sub(r"^[，,、。！？!?；;：:]+", "", speech)
    if ensure_terminal:
        terminal = "." if language == "en" else "。"
        speech = re.sub(r"[，,、；;：:]+$", terminal, speech)
    else:
        speech = re.sub(r"[，,、；;：:]+$", "", speech)
    if len(speech) > max_chars:
        speech = speech[:max_chars].rstrip()
        if language == "en" and " " in speech:
            speech = speech.rsplit(" ", 1)[0].rstrip(" ,;:")
    if ensure_terminal and speech and not re.search(r"[。！？!?.]$", speech):
        speech += "." if language == "en" else "。"
    return speech


_SPEECH_PUNCTUATION = set(
    "，。！？!?；;：:、,.…—-~～（）()《》<>【】[]“”\"'‘’ \n\r\t"
)
_SPEECH_BREAK_PUNCTUATION_RE = re.compile(r"[，,、。！？!?；;：:]")


def _speech_visible_len(text: str) -> int:
    if _dominant_response_language(text) == "en":
        word_count = len(re.findall(r"[A-Za-z]+(?:['’-][A-Za-z]+)*", str(text or "")))
        return max(1, int(round(word_count * 2.8)))
    return sum(1 for ch in str(text or "") if ch not in _SPEECH_PUNCTUATION)


def _speech_pause_budget(text: str) -> Dict[str, Any]:
    speech = str(text or "").strip()
    # Do not charge the final sentence punctuation as an extra trailing pause.
    speech_body = re.sub(r"[。！？!?]+$", "", speech)
    comma_pause_count = len(re.findall(r"[，,、；;：:]", speech_body))
    sentence_pause_count = len(re.findall(r"[。！？!?]", speech_body))
    pause_seconds = (
        comma_pause_count * max(0.0, SETTINGS.asr_budget_comma_pause_seconds)
        + sentence_pause_count * max(0.0, SETTINGS.asr_budget_sentence_pause_seconds)
    )
    return {
        "pause_seconds": pause_seconds,
        "comma_pause_count": comma_pause_count,
        "sentence_pause_count": sentence_pause_count,
    }


def _split_speech(reply: str, max_segments: int) -> List[str]:
    language = _dominant_response_language(reply)
    reply = re.sub(r"\s+", " " if language == "en" else "", reply.strip())
    if not reply:
        return [WAITING_TRANSITIONS[language][0]]
    chunks: List[str] = []
    target_visible_chars = max(
        1,
        int(SETTINGS.speech_segment_visible_chars)
        * (2 if language == "en" else 1),
    )
    i = 0
    while i < len(reply):
        chunk_chars: List[str] = []
        visible = 0
        while i < len(reply):
            ch = reply[i]
            chunk_chars.append(ch)
            i += 1
            if ch not in _SPEECH_PUNCTUATION:
                visible += 1
            if visible >= target_visible_chars:
                if language == "en":
                    while i < len(reply) and reply[i] not in _SPEECH_PUNCTUATION:
                        chunk_chars.append(reply[i])
                        i += 1
                while i < len(reply) and reply[i] in _SPEECH_PUNCTUATION:
                    chunk_chars.append(reply[i])
                    i += 1
                break
        chunk = "".join(chunk_chars)
        if chunk:
            chunks.append(chunk)
    if not chunks:
        chunks = [reply]
    cleaned = [
        _sanitize_speech_text(
            chunk,
            max_chars=128 if language == "en" else 64,
            ensure_terminal=(idx == len(chunks) - 1),
            language=language,
        )
        for idx, chunk in enumerate(chunks)
    ]
    cleaned = [chunk for chunk in cleaned if chunk]
    max_segments = max(1, int(max_segments))
    if len(cleaned) <= max_segments:
        return cleaned
    kept = cleaned[: max_segments]
    kept[-1] = (
        "I can continue with the rest in the next turn."
        if language == "en"
        else "后面的我可以继续展开。"
    )
    return kept


def _extract_json_array(text: str) -> Optional[List[Any]]:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _clean_segment_speech(speech: str) -> str:
    language = _dominant_response_language(speech)
    return _sanitize_speech_text(
        speech,
        max_chars=128 if language == "en" else 48,
        language=language,
    )


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", str(text or "")))


def _looks_like_collapsed_english(text: str) -> bool:
    value = str(text or "")
    alpha_count = sum(char.isalpha() and char.isascii() for char in value)
    if alpha_count < 40:
        return False
    if value.count(" ") <= 1:
        return True
    collapsed_alpha = sum(len(run) for run in re.findall(r"[A-Za-z]{40,}", value))
    return collapsed_alpha >= max(80, int(alpha_count * 0.25))


def _explicit_action_from_user_text(user_text: str) -> Optional[Tuple[str, str]]:
    text = re.sub(r"\s+", "", str(user_text or "")).lower()
    if not text:
        return None

    cooperative = "attentive recognition softening into a warm cooperative smile"
    raise_terms = r"(?:举起|抬起|举高|抬高)"
    if re.search(r"(?:站起来|站起身?|起身|从.{0,6}(?:椅子|座位|沙发).{0,4}站起)", text):
        return (
            "remains seated during the opening second, then rises from the chair at roughly one-quarter normal standing speed, using the rest of the five-second shot: leans forward and shifts weight during the second second, lifts the hips only around the midpoint, gradually extends the knees and torso through the fourth second, reaches a stable upright pose only in the final frames, and keeps both feet at their original floor position",
            cooperative,
        )
    if re.search(r"(?:坐下来|坐下|坐回.{0,4}(?:椅子|座位|沙发))", text):
        return (
            "briefly acknowledges the request, turns toward the seat with measured weight transfer, bends the knees gradually, and lowers into the seat through one slow continuous controlled motion, then settles the upper body into a stable seated posture",
            cooperative,
        )
    if re.search(r"(?:走过去|走过来|向前走|往前走|向后退|往后退|(?:向|往)(?:左|右)(?:边|侧)?(?:走|移动)|走到.{0,8}|移动到.{0,8}|向.{0,6}移动|往.{0,6}移动)", text):
        if re.search(r"(?:向|往)左(?:边|侧)?(?:走|移动)|(?:走|移动)(?:到|向|往).{0,4}(?:左边|左侧)", text):
            path = "laterally toward the requested left side of the scene"
        elif re.search(r"(?:向|往)右(?:边|侧)?(?:走|移动)|(?:走|移动)(?:到|向|往).{0,4}(?:右边|右侧)", text):
            path = "laterally toward the requested right side of the scene"
        elif re.search(r"(?:走过来|向前走|往前走|向前移动|往前移动)", text):
            path = "forward toward the camera"
        elif re.search(r"(?:向后退|往后退|向后移动|往后移动|后退)", text):
            path = "backward away from the camera"
        else:
            path = "toward the requested position"
        return (
            f"briefly registers the request, shifts weight smoothly, then moves {path} with slow measured steps and continuous natural balance, keeping each phase readable before settling into a stable stance",
            cooperative,
        )
    if re.search(rf"(?:双手|两只手|两手).{{0,6}}{raise_terms}|{raise_terms}.{{0,6}}(?:双手|两只手|两手)", text):
        return (
            "briefly registers the request with an attentive look, then clearly raises both open hands beside the head, holds the complete pose visibly for a beat, and keeps the shoulders relaxed",
            cooperative,
        )
    if re.search(r"(?:竖起?|比).{0,4}(?:大拇指|拇指)|点赞", text):
        return (
            "responds with a pleased look, raises the anatomical right hand on the viewer-left side near the shoulder, forms a clear thumbs-up, and holds it visibly for a beat",
            "pleased approval with a genuine encouraging smile",
        )
    if re.search(r"(?:挥手|挥挥手|招手)", text):
        side = "left" if "左手" in text else "right"
        viewer_side = "right" if side == "left" else "left"
        return (
            f"brightens in recognition, raises the anatomical {side} hand on the viewer-{viewer_side} side beside the face, gives two clear relaxed waves, and holds the open palm briefly before settling",
            "warm recognition with a bright natural greeting smile",
        )
    if re.search(rf"(?:右手).{{0,6}}{raise_terms}|{raise_terms}.{{0,6}}(?:你的)?右手", text):
        return (
            "briefly registers the request with an attentive look, then clearly raises the anatomical right hand on the viewer-left side beside the head with an open palm, holds it visibly for a beat, and keeps the left hand relaxed",
            cooperative,
        )
    if re.search(rf"(?:左手).{{0,6}}{raise_terms}|{raise_terms}.{{0,6}}(?:你的)?左手", text):
        return (
            "briefly registers the request with an attentive look, then clearly raises the anatomical left hand on the viewer-right side beside the head with an open palm, holds it visibly for a beat, and keeps the right hand relaxed",
            cooperative,
        )
    return None


def _fallback_turn_behavior(user_text: str) -> Tuple[str, str]:
    text = str(user_text or "")
    if re.search(r"难过|伤心|担心|焦虑|害怕|不开心|糟糕", text):
        return (
            "pauses with softened eyes, gives a small understanding nod, and opens one hand gently near chest level in a reassuring response",
            "quiet concern softening into warm empathetic reassurance",
        )
    if re.search(r"谢谢|感谢|真好|很棒|厉害|喜欢", text):
        return (
            "brightens naturally, gives a small appreciative nod, and brings one open hand lightly toward the chest before relaxing it",
            "genuinely pleased and appreciative with a warm smile",
        )
    if re.search(r"为什么|怎么|什么|哪里|是否|吗[？?]?|[？?]", text):
        return (
            "briefly considers the question with a subtle eye movement, gives a thoughtful nod, and makes one measured open-palmed explaining gesture near chest level",
            "briefly thoughtful, then focused and warmly engaged",
        )
    if re.search(r"你好|嗨|哈喽|早上好|下午好|晚上好", text):
        return (
            "recognizes the greeting with brightened eyes, gives a small welcoming nod, and lifts one open hand in a relaxed compact greeting",
            "pleasant recognition growing into a bright welcoming smile",
        )
    return (
        "briefly gathers the thought with a subtle attentive eye movement, gives a small responsive nod, and lets one relaxed open hand gesture follow the rhythm of the reply",
        "attentive recognition developing into a natural engaged expression",
    )


def _safe_action(action: str, segment_id: int) -> str:
    fallback = [
        "briefly gathers the thought with an attentive eye movement, gives a small responsive nod, and lets one relaxed open hand gesture follow the reply",
        "briefly looks thoughtful, softens the expression, and makes one measured open-palmed gesture near chest level",
        "brightens slightly in recognition, blinks naturally, and gives a small warm nod toward the camera",
        "keeps steady eye contact, reacts with a subtle eyebrow movement, and lets the hands settle naturally after one compact gesture",
    ][segment_id % 4]
    action = re.sub(r"\s+", " ", str(action or "").strip())
    action = re.sub(
        r"^(?:Speaker_1|the speaker|she|he)\s+",
        "",
        action,
        flags=re.I,
    )
    if not action or _contains_cjk(action):
        return fallback
    banned = [
        "wide shot",
        "camera cut",
        "scene change",
    ]
    if any(term in action.lower() for term in banned):
        return fallback
    return action[:520]


def _safe_emotion(emotion: str) -> str:
    emotion = re.sub(r"\s+", " ", str(emotion or "").strip())
    if not emotion or _contains_cjk(emotion):
        return "attentive recognition developing into a natural engaged expression"
    return emotion[:120]


def _normalize_prompt_speech(text: str) -> str:
    text = re.sub(r"\s+", "", str(text or "").strip())
    text = text.strip("\"'“”")
    text = re.sub(r"^[，,、。！？!?；;：:]+", "", text)
    text = re.sub(r"[，,、；;：:]+$", "。", text)
    return text


def _tail_speech_for_prompt(text: str, max_chars: int = 96) -> str:
    text = _normalize_prompt_speech(text)
    if not text:
        return ""
    parts = [p for p in re.split(r"(?<=[。！？!?])", text) if p]
    tail = ""
    for part in reversed(parts or [text]):
        if len(tail) + len(part) > max_chars and tail:
            break
        tail = f"{part}{tail}"
    if not tail:
        tail = text[-max_chars:]
    return _sanitize_speech_text(tail, max_chars=max_chars)


def _last_speech_segment_for_prompt(text: str, max_chars: int = 64) -> str:
    text = _normalize_prompt_speech(text)
    max_chars = max(0, int(max_chars))
    if not text or max_chars <= 0:
        return ""
    break_positions = [match.start() for match in _SPEECH_BREAK_PUNCTUATION_RE.finditer(text)]
    if len(break_positions) >= 2:
        segment = text[break_positions[-2] + 1 :]
    elif len(break_positions) == 1 and break_positions[0] < len(text) - 1:
        segment = text[break_positions[0] + 1 :]
    else:
        segment = text
    segment = segment.lstrip("".join(_SPEECH_PUNCTUATION))
    tail_chars: List[str] = []
    visible = 0
    for ch in reversed(segment):
        tail_chars.append(ch)
        if ch not in _SPEECH_PUNCTUATION:
            visible += 1
        if visible >= max_chars:
            break
    tail = "".join(reversed(tail_chars)).lstrip("".join(_SPEECH_PUNCTUATION))
    return _sanitize_speech_text(
        tail,
        max_chars=max_chars + 4,
        ensure_terminal=bool(re.search(r"[。！？!?]$", tail)),
    )


def _history_tail_segment_for_prompt(
    prior_spoken_context: str,
    prior_spoken_tail_segment: str = "",
    *,
    max_chars: int = 64,
) -> str:
    """Extract the prior speech tail for the next Speaker_1 says line."""
    context_tail = _last_speech_segment_for_prompt(
        prior_spoken_context,
        max_chars=max_chars,
    )
    if context_tail:
        return context_tail
    return _last_speech_segment_for_prompt(
        prior_spoken_tail_segment,
        max_chars=max_chars,
    )


def _speech_for_prompt(parts: List[str], max_chars: int = 180) -> str:
    cleaned: List[str] = []
    for part in parts:
        speech = _normalize_prompt_speech(part)
        if not speech:
            continue
        sentences = [item for item in re.split(r"(?<=[。！？!?])", speech) if item]
        for sentence in sentences or [speech]:
            sentence = _normalize_prompt_speech(sentence)
            if not sentence:
                continue
            combined_so_far = "".join(cleaned)
            if combined_so_far and (
                combined_so_far.endswith(sentence)
                or sentence in combined_so_far
                or combined_so_far.endswith(sentence.rstrip("。！？!?"))
            ):
                continue
            cleaned.append(sentence)
    combined = "".join(cleaned)
    combined = re.sub(r"([。！？!?])\1+", r"\1", combined)
    if len(combined) > max_chars:
        pieces = [p for p in re.split(r"(?<=[。！？!?])", combined) if p]
        tail = ""
        for piece in reversed(pieces or [combined]):
            if len(tail) + len(piece) > max_chars and tail:
                break
            tail = f"{piece}{tail}"
        combined = tail or combined[-max_chars:]
    return _sanitize_speech_text(combined, max_chars=max_chars)


def _speech_target_for_prompt(text: str, max_chars: int = 64) -> str:
    raw = str(text or "").strip()
    return _sanitize_speech_text(
        raw,
        max_chars=max_chars,
        ensure_terminal=bool(re.search(r"[。！？!?]$", raw)),
    )


LIVE5_CANVAS_SMOKE_SCENE = (
    "eye-level tight medium close-up shot. a contemporary home office with an off-white plaster "
    "wall, a narrow walnut desk edge visible only at the bottom below the speaker's hands, one oak shelf holding two "
    "closed neutral books, and one small broad-leaf plant on the shelf. A large diffused "
    "window at camera left is the dominant key light and a white wall at camera right "
    "provides gentle neutral fill, producing consistent soft shadows. The gray upholstered "
    "chair visibly supports the seated speaker. Natural 50mm-equivalent perspective, "
    "straight background verticals, crisp facial detail, controlled highlights, clean "
    "material texture, moderate depth of field, and polished natural color"
)


LIVE5_CANVAS_SMOKE_APPEARANCE = (
    "Young male digital-human presenter with warm fair skin, neat short black hair, "
    "tidy eyebrows, a stable friendly face, and a charcoal gray zip-up hoodie over "
    "a white T-shirt"
)


ENGLISH_TEMPLATE_SCENES = {
    "live5_canvas_smoke": LIVE5_CANVAS_SMOKE_SCENE,
    "tech_anchor": (
        "eye-level tight medium close-up shot. a restrained professional technology studio with "
        "matte graphite acoustic wall panels, a narrow brushed-aluminum desk edge visible only at the bottom below the "
        "speaker's forearms, and one recessed oak shelf holding a small closed silver equipment "
        "case. All equipment remains on the rear shelf, leaving the foreground and the speaker's "
        "torso unobstructed. Every panel is matte and unmarked. A large frosted studio window "
        "outside the frame at camera left is the dominant key light and a white wall at camera "
        "right provides gentle neutral fill, producing one consistent shadow direction. A "
        "low-backed charcoal chair stays fully behind and visibly supports the seated speaker. "
        "Natural 50mm-equivalent perspective, straight background "
        "verticals, crisp facial detail, controlled highlights, clean material texture, "
        "moderate depth of field, and refined neutral color"
    ),
    "business_consultant": (
        "eye-level tight medium close-up shot. a refined executive meeting room with only a narrow band of an uncluttered "
        "matte light-gray meeting table visible at the bottom below the speaker's hands, a continuous walnut wall "
        "panel, and one small ceramic vase resting on a recessed sideboard. The clear foreground "
        "contains only the table surface and the speaker's hands. A large diffused window at "
        "camera left is the dominant key light and a white wall at camera right provides "
        "gentle neutral fill, producing consistent soft shadows. The upholstered chair "
        "visibly supports the seated speaker. Natural 50mm-equivalent perspective, straight "
        "background verticals, crisp facial detail, controlled highlights, clean material "
        "texture, moderate depth of field, and understated premium color"
    ),
    "education_coach": (
        "eye-level tight medium close-up shot. a bright modern classroom corner with a blank matte "
        "whiteboard mounted flush to the wall, a low oak bookcase holding three plain closed books "
        "with blank spines, one small plant on the bookcase, and a narrow pale gray table edge visible only at the bottom below the speaker's "
        "forearms. Every object is aligned with and supported by the furniture. A large diffused "
        "window at camera left is the dominant key light and a white wall at camera right provides "
        "gentle neutral fill, producing consistent soft shadows. The chair visibly supports the "
        "seated instructor. Natural 50mm-equivalent perspective, straight background verticals, "
        "crisp facial detail, controlled highlights, clean material texture, moderate depth of "
        "field, and fresh natural color"
    ),
    "wellness_host": (
        "eye-level tight medium close-up shot. a calm contemporary wellness studio with a pale "
        "mineral-plaster wall, a floor-length linen curtain, an oak sideboard holding one "
        "ceramic bowl and one neatly folded towel, and a potted olive tree standing in the "
        "corner. Every object rests on a clear physical support. Diffused daylight through "
        "the curtain at camera left is the dominant key light and a white wall at camera "
        "right provides gentle neutral fill, producing consistent soft shadows. The linen "
        "chair visibly supports the seated host. Natural 50mm-equivalent perspective, "
        "straight background verticals, crisp facial detail, controlled highlights, clean "
        "material texture, moderate depth of field, and calm natural color"
    ),
}


TEMPLATE_PHYSICAL_ANCHORS = {
    "live5_canvas_smoke": (
        "The physical anchors remain unchanged: only a narrow walnut desk edge stays visible at the bottom below the hands, "
        "the two closed books and broad-leaf plant stay on the single oak shelf, and the gray chair "
        "continues to support the seated speaker."
    ),
    "tech_anchor": (
        "The physical anchors remain unchanged: only a narrow brushed-aluminum desk edge stays visible at the bottom below the hands, "
        "the closed silver equipment case stays inside the rear oak shelf, the foreground and torso remain "
        "unobstructed, the graphite wall panels remain straight, and the low-backed charcoal chair stays behind "
        "and continues to support the seated speaker."
    ),
    "business_consultant": (
        "The physical anchors remain unchanged: only a narrow band of the uncluttered matte table stays visible at the bottom below the hands, the "
        "ceramic vase stays on the recessed sideboard, the walnut wall panel remains continuous, and the "
        "upholstered chair continues to support the seated speaker."
    ),
    "education_coach": (
        "The physical anchors remain unchanged: the blank whiteboard stays flush to the wall, the closed "
        "books with blank spines and the plant stay on the oak bookcase, only a narrow table edge stays visible at the bottom, and the chair "
        "continues to support the seated instructor."
    ),
    "wellness_host": (
        "The physical anchors remain unchanged: the ceramic bowl and folded towel stay on the oak sideboard, "
        "the olive tree stays rooted in its floor pot, the linen curtain hangs vertically, and the linen "
        "chair continues to support the seated host."
    ),
}


ENGLISH_TEMPLATE_APPEARANCES = {
    "live5_canvas_smoke": LIVE5_CANVAS_SMOKE_APPEARANCE,
    "tech_anchor": (
        "Young male technology presenter with warm fair skin, neat short black hair, "
        "and a navy casual blazer over a white shirt"
    ),
    "business_consultant": (
        "Mature female business consultant with soft fair skin, shoulder-length chestnut "
        "hair, natural makeup, and a light gray blazer with a white inner layer"
    ),
    "education_coach": (
        "Young female course instructor with clear glasses, a low ponytail, a gentle "
        "expression, and a pale green cardigan"
    ),
    "wellness_host": (
        "Older male wellness host with warm tan skin, short silver hair, a trustworthy "
        "smile, and an off-white knit top"
    ),
}


def _appearance_for_scene(
    scene: str,
    *,
    template_id: str = "",
    has_first_frame: bool = False,
) -> str:
    if has_first_frame:
        return (
            "the same person from the uploaded first-frame reference, preserving the same "
            "face, hairstyle, clothing, and body proportions"
        )
    template_appearance = ENGLISH_TEMPLATE_APPEARANCES.get(str(template_id or "").strip())
    if template_appearance:
        return template_appearance

    text = re.sub(r"\s+", " ", str(scene or "").strip())
    identity_terms = (
        "woman", "female", "girl", "lady", "man", "male", "boy", "presenter",
        "speaker", "host", "instructor", "teacher", "coach", "consultant", "athlete",
        "player", "chef", "doctor", "nurse",
    )
    appearance_terms = (
        "young", "older", "mature", "hair", "skin", "wearing", "dressed", "glasses",
        "jacket", "shirt", "jersey", "uniform", "hoodie", "sweater", "blazer", "cardigan",
        "dress", "clothing", "outfit",
    )
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        candidate = sentence.strip().strip(". ")
        lower = candidate.lower()
        if not any(term in lower for term in identity_terms):
            continue
        if not any(term in lower for term in appearance_terms):
            continue
        candidate = re.sub(
            r"^static\s+(?:medium(?:\s+close-up)?|close-up|waist-up)\s+shot(?:\s+of)?\s*[:,-]?\s*",
            "",
            candidate,
            flags=re.I,
        ).strip()
        if candidate:
            return candidate[:520].rstrip(". ")
    return (
        "A natural digital-human presenter whose gender, age, hairstyle, hair color, clothing, "
        "and body proportions match the scene description exactly"
    )


def _force_speaker_appearance(prompt: str, appearance: str) -> str:
    text = str(prompt or "")
    value = str(appearance or "").strip().rstrip(". ")
    if not text or not value:
        return text
    return re.sub(
        r"(Speaker_1's Appearance:\s*)[^\n]*",
        lambda match: f"{match.group(1)}{value}.",
        text,
        count=1,
    )


PROMPT_PLANNER_EXAMPLES = [
    {
        "name": "live studio canvas segment 1",
        "prompt": (
            "Summary: A polished interactive digital-human response begins in a cinematic live-studio desk scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a vivid live-demo studio with a matte ivory desk edge, warm practical lamps, translucent glass shelves, soft blue accent lights, a few green plants, and subtle reflections on the back wall. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a soft friendly smile, and a cream knit jacket over a light blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, makes a tiny nod, and lets the right hand lift slowly near chest level while speaking. The torso stays centered, the shoulders stay relaxed, and the movement feels smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm, welcoming, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The room tone is quiet and stable."
        ),
    },
    {
        "name": "live studio canvas segment 2 cumulative speech",
        "prompt": (
            "Summary: The same polished digital-human response continues in the identical cinematic live-studio desk scene, with the same face, outfit, lighting, framing, and background objects preserved.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the vivid live-demo studio remains unchanged: matte ivory desk edge, warm practical lamps, translucent glass shelves, soft blue accent lights, a few green plants, and subtle reflections on the back wall. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a soft friendly smile, and a cream knit jacket over a light blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps the face oriented toward the lens, makes a slow open-palmed gesture, then lets the hand settle near the desk. The torso stays centered and the movement remains small, steady, and conversational. \n"
            "Speaker_1's Facial Expression: attentive and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以直接告诉我你的问题，我会接着回答。\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke reference segment 1",
        "prompt": (
            "Summary: A premium real-time digital-human conversation begins in a cinematic canvas-backdrop live studio with a soft atmospheric portrait look.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a premium live5-style studio with a matte charcoal canvas backdrop, faint studio haze diffusion, a warm amber practical lamp on the left, a cool blue rim light on the glass shelf, a cream desk edge in the lower foreground, a small green plant, and subtle reflections on the dark wall. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a tiny welcoming nod, and lets one hand rise slowly near chest level before settling naturally. The torso stays centered, the shoulders stay relaxed, and the movement feels smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm, bright, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The studio room tone is quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke reference segment 2 cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues in the identical canvas-backdrop live studio, preserving the same face, jacket, desk edge, lamp glow, blue rim light, haze diffusion, and camera framing.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the premium live5-style studio remains physically unchanged: matte charcoal canvas backdrop, faint studio haze diffusion, warm amber practical lamp on the left, cool blue rim light on the glass shelf, cream desk edge in the lower foreground, small green plant, and subtle reflections on the dark wall. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps the face oriented toward the lens, makes a slow open-palmed explaining gesture, then lets the hand relax near the desk. The torso stays centered, the shoulders stay relaxed, and the motion remains small, steady, and conversational. \n"
            "Speaker_1's Facial Expression: attentive, relaxed, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以直接告诉我你的问题，我会接着回答。\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke reference segment 3 longer cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues as one coherent continuous video in the identical canvas-backdrop live studio, preserving the same face, cream jacket, pale blue shirt, charcoal backdrop, amber lamp, blue rim light, desk edge, plant, haze diffusion, and portrait color palette.\n"
            "Narration 3:\n"
            "eye-level tight medium close-up shot. the premium live5-style studio remains physically unchanged: matte charcoal canvas backdrop with soft woven texture, faint studio haze diffusion, warm amber practical lamp on the left, cool blue rim light along the glass shelf, cream desk edge in the lower foreground, small green plant, dark wall reflections, and a stable clean lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps steady eye contact with the lens, continues the explanation with a slow open-palmed hand movement near chest level, then lets the hand settle close to the desk while the torso stays centered. The shoulders stay relaxed, the chin angle remains consistent, and the motion feels small, smooth, and conversational. \n"
            "Speaker_1's Facial Expression: focused, friendly, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以直接告诉我你的问题，我会接着回答。我们可以先从最重要的一点说起。\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke reference segment 4 full-answer cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues in a single continuous live5-style continuous video, preserving the same face identity, cream textured jacket, pale blue shirt, charcoal canvas backdrop, amber practical lamp, cool blue rim light, cream desk edge, green plant, haze diffusion, and natural portrait color palette.\n"
            "Narration 4:\n"
            "eye-level tight medium close-up shot. the premium live5-style studio remains physically identical: matte charcoal canvas backdrop with visible soft woven texture, faint atmospheric studio haze, warm amber lamp glow on the left side, cool blue rim light tracing the glass shelf, cream desk edge low in frame, small green plant, subtle dark-wall reflections, and a stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft highlights across the cheeks, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps the face oriented toward the camera, maintains steady eye contact, makes a slow open-palmed gesture near chest level, then lets the hand settle back near the desk with the same centered posture. The shoulders stay relaxed, the chin angle stays consistent, and the movement remains smooth, small, and conversational. \n"
            "Speaker_1's Facial Expression: calm, focused, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以直接告诉我你的问题，我会接着回答。我们可以先从最重要的一点说起。然后我会把步骤讲得清楚一点。\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke second user turn handcrafted continuation",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues into a new user turn as one coherent live5-style continuous video, preserving the identical young male presenter identity, cream textured jacket, pale blue shirt, charcoal canvas backdrop, amber practical lamp, blue rim light, cream desk edge, green plant, haze diffusion, and vivid natural portrait color palette.\n"
            "Narration 5:\n"
            "eye-level tight medium close-up shot. the premium live5-style studio remains physically identical: matte charcoal canvas backdrop with visible woven texture, faint atmospheric haze diffusing the background, warm amber practical lamp glow on the left, cool blue rim light tracing the glass shelf, cream desk edge low in frame, a small green plant near the shelf, subtle dark-wall reflections, soft highlights across the cheeks, delicate eye catchlights, and a stable clean natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, gentle rim separation on the hair edge, rich layered background details, shallow depth of field, cinematic but stable live-demo portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the lens directly, gives a tiny listening nod as if receiving the follow-up question, then raises one hand slowly near chest level to answer before letting the hand settle back near the desk. The torso stays centered, the shoulders stay relaxed, the chin angle remains consistent, and the motion is smooth, small, and conversational. \n"
            "Speaker_1's Facial Expression: focused, friendly, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以，我们继续刚才的问题。第一点是先确认你真正想优化的目标。\"\n"
            "Speaker_1's Emotion: calm, focused, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke cinematic short greeting",
        "prompt": (
            "Summary: A premium real-time digital-human conversation begins with a short natural greeting in a cinematic canvas-backdrop live studio, preserving stable identity, natural lens geometry, detailed light direction, and a polished live-demo portrait look.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a premium live5-style studio with a matte charcoal canvas backdrop showing a fine woven texture, faint studio haze diffusion, a warm amber practical lamp glowing softly on the left, a cool blue rim light along a glass shelf, a cream desk edge in the lower foreground, a small green plant placed near the shelf, subtle dark-wall reflections, clean silhouette edges, soft cheek highlights, delicate catchlights in the eyes, and a stable clean lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives one tiny welcoming nod, and lets the right hand rise slowly near chest level in a relaxed open-palmed greeting before settling naturally. The torso stays centered, the shoulders stay relaxed, and the movement remains smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm, bright, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The studio room tone is quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke analytical answer segment 1",
        "prompt": (
            "Summary: A premium real-time digital-human conversation begins a concise analytical answer in the cinematic canvas-backdrop live studio, preserving stable identity, natural lens geometry, controlled portrait lighting, and a polished live-demo texture.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a premium live5-style studio with a matte charcoal canvas backdrop showing visible woven texture, faint atmospheric haze diffusion, a warm amber practical lamp glowing softly on the left, a cool blue rim light tracing the glass shelf, a cream desk edge low in the foreground, a small green plant placed beside the shelf, subtle reflections on the dark wall, clean silhouette edges, soft cheek highlights, delicate eye catchlights, and a stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a tiny thoughtful nod, and raises one hand slowly near chest level to introduce the key point before settling naturally. The torso stays centered, the shoulders stay relaxed, the chin angle remains consistent, and the movement remains small, smooth, and conversational. \n"
            "Speaker_1's Facial Expression: calm, focused, and helpful.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"这个问题先看结论。核心是让后续片段继续依赖同一个人物、光线和画面锚点。\"\n"
            "Speaker_1's Emotion: calm, focused, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke analytical answer segment 2 cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues the analytical answer in one coherent continuous live5-style video, preserving the identical presenter identity, cream textured jacket, pale blue shirt, charcoal canvas backdrop, amber lamp, blue rim light, cream desk edge, green plant, haze diffusion, and vivid natural portrait color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the premium live5-style studio remains physically identical: matte charcoal canvas backdrop with visible woven texture, faint atmospheric haze diffusion, warm amber practical lamp glow on the left, cool blue rim light along the glass shelf, cream desk edge low in frame, small green plant beside the shelf, dark-wall reflections, soft highlights across the cheeks, delicate eye catchlights, clean silhouette edges, and the same stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable live-demo portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps the face oriented toward the lens, continues with a slow open-palmed explaining gesture near chest level, then lets the hand relax near the desk while maintaining the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the motion remains smooth, small, and conversational. \n"
            "Speaker_1's Facial Expression: attentive, steady, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"这个问题先看结论。核心是让后续片段继续依赖同一个人物、光线和画面锚点。第二步是让每一段口播都接住前面的语义，不要像重新开场。\"\n"
            "Speaker_1's Emotion: attentive, steady, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "cinematic consultant second user turn handcrafted continuation",
        "prompt": (
            "Summary: The same professional real-time digital-human conversation continues into a new user turn in a refined live5-style consulting studio, preserving the same mature female consultant identity, light gray blazer, white inner layer, chestnut hairstyle, ivory desk edge, ceramic lamp, plant, notebooks, glass reflections, daylight, and calm cinematic portrait palette.\n"
            "Narration 4:\n"
            "eye-level tight medium close-up shot. the refined consulting studio remains physically identical: matte ivory desk edge low in frame, pale oak shelves arranged with cream notebooks, softly glowing white ceramic desk lamp on the left, a small green plant, muted silver pen tray, clear glass partition reflections on the right side, soft frontal daylight wrapping evenly across the face, faint warm practical glow behind the speaker, gentle hair rim light, delicate eye catchlights, and subtle highlights across the cheeks. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered office background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, gives a small reassuring nod, moves one hand slowly near the desk in an organized explaining gesture, then returns to the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the movement is measured, small, and professional. \n"
            "Speaker_1's Facial Expression: calm, trustworthy, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。可以，我们接着刚才的方向。先把结论说清楚，再补充原因。\"\n"
            "Speaker_1's Emotion: calm, professional, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. The office room tone remains quiet and stable."
        ),
    },
    {
        "name": "cinematic finance desk segment",
        "prompt": (
            "Summary: A composed digital-human advisor explains a point in a cinematic evening finance-studio desk scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a quiet evening finance studio with a walnut desk edge, a softly glowing brass lamp, dark green acoustic panels, a blurred city-window reflection, a small glass water cup, and neat paper notes placed low in frame. Bright balanced face lighting, natural warm colors, gentle rim light, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Middle-aged male advisor with fair skin, neatly combed dark hair, thin metal glasses, subtle smile lines, and a dark green knit cardigan over a white shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, makes a tiny nod, and slowly lifts one hand in a measured explaining gesture before resting it near the desk. The torso stays centered, the shoulders stay relaxed, and the gesture remains precise and conversational. \n"
            "Speaker_1's Facial Expression: thoughtful, steady, and friendly.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"这个问题可以从原因和做法两边看，我们先抓最关键的一点。\"\n"
            "Speaker_1's Emotion: calm, thoughtful, and helpful.\n"
            "Speaker_1's Voice Description: steady Mandarin voice, warm tone, unhurried conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays natural and precise. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "consulting office segment 1",
        "prompt": (
            "Summary: A professional interactive digital-human conversation begins in a refined consulting-office scene with a steady cinematic close-up look.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a bright consulting office desk with light wood shelves, clear glass partitions, a white ceramic desk lamp, a small green plant, a neat stack of cream notebooks, and soft frontal daylight falling evenly across the speaker's face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, layered office background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a small reassuring nod, and lets one hand move slowly near the desk as if calmly organizing the answer. The torso stays centered, the shoulders stay relaxed, and the motion remains smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm, trustworthy, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。\"\n"
            "Speaker_1's Emotion: calm and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The office room tone remains quiet and stable."
        ),
    },
    {
        "name": "consulting office segment 2 cumulative speech",
        "prompt": (
            "Summary: The same professional digital-human conversation continues in the identical consulting-office scene, preserving the same face, blazer, desk objects, daylight, color palette, and camera framing.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the bright consulting office remains physically unchanged: light wood shelves, clear glass partitions, white ceramic desk lamp, small green plant, cream notebooks, and soft frontal daylight across the speaker's face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, layered office background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 keeps steady eye contact with the lens, makes a slow open-palmed explaining gesture, then lets the hand settle near the desk. The torso stays centered, the shoulders stay relaxed, and the movement remains small and conversational. \n"
            "Speaker_1's Facial Expression: attentive and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。我会先听清楚你的问题，再用简单的话帮你梳理。\"\n"
            "Speaker_1's Emotion: calm, professional, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "consulting office segment 3 full-answer cumulative speech",
        "prompt": (
            "Summary: The same professional digital-human conversation continues as one coherent continuous consulting-office video, preserving the same mature female consultant identity, light gray blazer, white inner layer, chestnut hairstyle, desk objects, daylight, glass reflections, and calm color palette.\n"
            "Narration 3:\n"
            "eye-level tight medium close-up shot. the refined consulting office remains physically unchanged: matte ivory desk edge, light wood shelves, clear glass partitions, a white ceramic desk lamp, a small green plant, neat cream notebooks, soft frontal daylight, faint warm practical glow behind the speaker, and subtle reflections in the glass. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft highlights across the face, layered office background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, continues the explanation with a slow open-palmed motion near the desk, then returns to the same centered posture. The shoulders stay relaxed, the torso stays steady, and the movement remains measured, small, and professional. \n"
            "Speaker_1's Facial Expression: attentive, trustworthy, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。我会先听清楚你的问题，再用简单的话帮你梳理。如果需要，我也可以把结论和下一步分开说。\"\n"
            "Speaker_1's Emotion: calm, professional, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "classroom coach cumulative speech",
        "prompt": (
            "Summary: A warm interactive digital-human explanation continues in the same bright classroom coaching scene, keeping the same instructor identity and set dressing stable.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the sunny classroom corner remains unchanged with colorful flash cards, a clean whiteboard, picture books, pastel storage boxes, a small desk plant, and soft daylight from the left side. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered classroom details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young female course instructor with soft fair skin, clear glasses, a low ponytail, a gentle expression, and a pale green cardigan over a white top.\n"
            "Speaker_1's Actions: Speaker_1 keeps her face oriented toward the lens, raises one hand slowly to emphasize a point, and then returns to a relaxed centered posture. The gesture is small, smooth, and teacher-like. \n"
            "Speaker_1's Facial Expression: warm, patient, and encouraging.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"我们先把重点讲清楚。然后我会给你一个容易执行的小建议。\"\n"
            "Speaker_1's Emotion: calm, bright, and encouraging.\n"
            "Speaker_1's Voice Description: clear young female Mandarin voice, gentle teacher-like cadence, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. The classroom ambience stays quiet and stable."
        ),
    },
    {
        "name": "wellness host cumulative speech",
        "prompt": (
            "Summary: A calm interactive digital-human response continues in the same plant-filled consultation room, preserving a stable warm portrait look.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the calm consultation room remains unchanged with green leaves, light wood shelves, a ceramic diffuser, a woven basket, pale linen curtains, and soft window light wrapping around the speaker's face. Bright balanced lighting, natural warm colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Older male wellness host with warm tan skin, short silver hair, a trustworthy smile, soft wrinkles around the eyes, and an off-white knit top.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera with warm eye contact, slowly raises one hand near chest level, and makes a gentle downward calming gesture while speaking. The torso stays centered and the movement remains quiet and natural. \n"
            "Speaker_1's Facial Expression: calm, caring, and steady.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"先别着急，我们一步一步来。把最影响你的地方说出来，我会陪你慢慢拆开。\"\n"
            "Speaker_1's Emotion: calm, reassuring, and helpful.\n"
            "Speaker_1's Voice Description: warm mature Mandarin voice, measured pacing, clear articulation, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "education coach segment 2 cumulative speech",
        "prompt": (
            "Summary: The same warm interactive digital-human teaching response continues in the identical sunny classroom coaching scene, preserving the same instructor identity, pale green cardigan, glasses, whiteboard, flash cards, books, desk plant, soft daylight, and friendly color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the sunny classroom corner remains physically unchanged with colorful flash cards, a clean whiteboard, picture books, pastel storage boxes, a small desk plant, soft side daylight from a tall window, warm practical highlights on the back shelf, and gentle reflections on a laminated teaching card near the desk edge. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft highlights across the face, rich layered classroom details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young female course instructor with soft fair skin, clear glasses, a low ponytail, a gentle expression, and a pale green cardigan over a white top.\n"
            "Speaker_1's Actions: Speaker_1 keeps warm eye contact with the lens, slowly raises one hand to emphasize a point, then brings the hand back to a relaxed centered posture. The gesture stays small, smooth, and teacher-like while the torso remains steady. \n"
            "Speaker_1's Facial Expression: warm, patient, and encouraging.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"我们先把重点讲清楚。然后我会给你一个容易执行的小建议。你只要跟着第一步做就可以。\"\n"
            "Speaker_1's Emotion: calm, bright, and encouraging.\n"
            "Speaker_1's Voice Description: clear young female Mandarin voice, gentle teacher-like cadence, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. The classroom ambience stays quiet and stable."
        ),
    },
    {
        "name": "bright home-office desk",
        "prompt": (
            "Summary: A polished product-style digital-human demo begins in a bright cinematic home-office desk scene with stable portrait lighting and clear visual anchors.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a bright home office with a white desk edge in the foreground, colorful sticky notes pinned neatly on a cork board, a matte white desk lamp, two green plants, graphic posters, a clean monitor setup with soft screen reflections, pale wood shelves, and late-morning daylight diffused through a sheer curtain. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young woman with fair skin, tidy high ponytail, thin black glasses, rose blush, natural lip color, and a cobalt-blue blouse with a clean collar.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a tiny welcoming nod, and moves one hand slowly near chest level as if introducing a useful desk item. The torso stays centered, the shoulders stay relaxed, and the movement remains smooth and conversational. \n"
            "Speaker_1's Facial Expression: bright and welcoming.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"大家好，今天我在清爽家庭办公桌，想和你分享这个桌面收纳架。\"\n"
            "Speaker_1's Emotion: calm, bright, and engaged.\n"
            "Speaker_1's Voice Description: focused young female Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "colorful cafe counter host",
        "prompt": (
            "Summary: A polished interactive digital-human demo continues at a colorful cafe counter with a stable cinematic look.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a colorful cafe counter with glossy turquoise tiles, glass dessert stands, a red espresso machine, hanging green plants, warm wood shelves, ceramic cups arranged in soft rows, tiny highlights on the counter edge, and clear morning sunlight spilling gently from the side window. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male presenter with warm tan skin, neat black hair, tidy eyebrows, a friendly smile, a sky-blue shirt, and a white canvas apron.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera, keeps steady eye contact, and moves one hand slowly above the counter while speaking. The torso stays centered, the shoulders stay relaxed, and the gesture remains smooth and conversational. \n"
            "Speaker_1's Facial Expression: bright and welcoming.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。\"\n"
            "Speaker_1's Emotion: calm, bright, and engaged.\n"
            "Speaker_1's Voice Description: bright young male Mandarin voice, natural pacing, clear articulation, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "warm beauty counter advisor",
        "prompt": (
            "Summary: A polished interactive digital-human conversation continues in a luminous beauty counter scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a warm beauty counter with peach-toned wall panels, glass skincare bottles arranged on a cream tray, a softly glowing mirror edge, ivory shelves, fresh pale flowers, gold trim reflections, and smooth studio highlights across the countertop. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft highlights on the face, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Elegant woman in her late twenties with fair skin, long softly curled dark-brown hair, refined natural makeup, pearl earrings, and a satin ivory blouse.\n"
            "Speaker_1's Actions: Speaker_1 keeps steady eye contact with the lens, tilts the chin slightly, and lets both hands make a slow open-palmed explaining gesture near chest level before settling naturally. The torso stays centered, the shoulders stay relaxed, and the movement remains soft and precise. \n"
            "Speaker_1's Facial Expression: attentive and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"可以，我们先把重点说清楚，再慢慢往下展开。\"\n"
            "Speaker_1's Emotion: calm, warm, and professional.\n"
            "Speaker_1's Voice Description: elegant Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. The voice feels close, clean, and stable."
        ),
    },
    {
        "name": "bookstore conversation host",
        "prompt": (
            "Summary: A polished interactive digital-human response continues in a cozy bookstore conversation scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a cozy modern bookstore corner with walnut bookshelves, a brass reading lamp, small framed prints, a ceramic cup placed low on the table, soft amber practical lights, rows of softly blurred book spines, and gentle daylight from the side window. Bright balanced lighting, natural warm colors, soft portrait contrast, delicate catchlights in the eyes, delicate face highlights, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Middle-aged male host with fair skin, short neatly combed dark hair, subtle smile lines, thin metal glasses, and a dark green knit cardigan over a white shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a tiny nod, and slowly brings one hand up in a measured explaining gesture before settling it back down. The torso stays centered, the shoulders stay relaxed, and the gesture remains precise and conversational. \n"
            "Speaker_1's Facial Expression: thoughtful and friendly.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"这个问题可以从原因和做法两边看，我们先抓最关键的一点。\"\n"
            "Speaker_1's Emotion: calm, thoughtful, and helpful.\n"
            "Speaker_1's Voice Description: steady Mandarin voice, warm tone, unhurried conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth movement stays natural and precise."
        ),
    },
    {
        "name": "perfume counter host",
        "prompt": (
            "Summary: A polished interactive digital-human demo continues in an elegant product-display scene.\n"
            "Narration 1:\n"
            "static tight medium close-up shot. a perfume counter with crystal bottles, pastel flowers, gold trays, mirror reflections, cream product cards, tiny specular highlights on the glass edges, and bright boutique lighting wrapping softly across the face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young woman with fair skin, long straight brown hair, refined makeup, and an ivory blouse.\n"
            "Speaker_1's Actions: Speaker_1 keeps steady eye contact, holds a small featured object near chest level, then tilts it gently toward the camera while keeping the posture steady. The torso stays centered, the shoulders stay relaxed, and the hand motion remains slow and controlled. \n"
            "Speaker_1's Facial Expression: attentive and informative.\n"
            "Speaker_1's Held Objects:\nObject: citrus perfume bottle: A small elegant glass bottle held near chest level, clearly visible, with clean edges and pleasing color.\n"
            "Speech Attribution:\nSpeaker_1 says: \"第一眼看过去，它的颜色和质感都很舒服，放在镜头里也很上镜。\"\n"
            "Speaker_1's Emotion: calm, bright, and engaged.\n"
            "Speaker_1's Voice Description: elegant young female Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "consulting office presenter",
        "prompt": (
            "Summary: A polished interactive digital-human conversation continues in a stable consulting-office scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a bright consulting office desk with a matte ivory desk edge, light wood shelves, clear glass partitions, a white ceramic desk lamp, a small green plant, neat cream notebooks, subtle glass reflections, and soft frontal daylight falling evenly across the speaker's face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair, natural makeup, and a light gray blazer with a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, nods gently, and lets one hand move slowly near the desk while speaking. The torso stays centered, the shoulders stay relaxed, and the movement remains smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm and trustworthy.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。\"\n"
            "Speaker_1's Emotion: calm and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "sunny classroom teacher",
        "prompt": (
            "Summary: A polished interactive digital-human demo continues in a bright classroom scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a sunny classroom corner with colorful flash cards, a clean whiteboard, picture books, pastel storage boxes, a small desk plant, soft side daylight from a tall window, warm practical highlights on the back shelf, and gentle reflections on a laminated teaching card near the desk edge. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered classroom details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young female course instructor with clear glasses, a low ponytail, soft fair skin, and a pale green cardigan.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps warm eye contact, raises one hand slowly to emphasize a point, then returns to a relaxed centered posture. The gesture is small, smooth, and teacher-like. \n"
            "Speaker_1's Facial Expression: warm and encouraging.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"我会先把重点讲清楚，再给你一个容易执行的小建议。\"\n"
            "Speaker_1's Emotion: calm, bright, and engaged.\n"
            "Speaker_1's Voice Description: clear young female Mandarin voice, gentle teacher-like cadence, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5-grade technology anchor cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues in a cinematic blue-white live-demo technology studio, preserving the same face identity, navy blazer, glass desk edge, display panels, cool rim light, plant anchors, reflections, and stable portrait color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the cinematic technology studio remains physically identical: luminous glass desk edge in the lower foreground, translucent interface panels behind the speaker, soft product display lights, brushed metal trim, a small green plant on the right shelf, subtle screen reflections, clean morning light across the cheeks, and a cool rim light separating the silhouette from the background. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male technology presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a navy casual blazer over a white shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the lens directly, keeps steady eye contact, raises one hand slowly near chest level to mark the key point, then lets the hand settle back toward the desk. The torso stays centered, the shoulders stay relaxed, and the gesture remains smooth, precise, and conversational. \n"
            "Speaker_1's Facial Expression: focused, bright, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"我们先把核心问题讲清楚。第一步是确认目标，然后再看哪里最影响体验。\"\n"
            "Speaker_1's Emotion: calm, focused, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5-grade consulting studio cumulative speech",
        "prompt": (
            "Summary: The same professional digital-human conversation continues in a refined live5-style consulting studio, preserving the same mature female consultant identity, light gray blazer, white inner layer, chestnut hairstyle, ivory desk edge, ceramic lamp, plant, notebooks, glass reflections, daylight, and calm portrait color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the refined consulting studio remains physically identical: matte ivory desk edge low in frame, pale oak shelves, clear glass partitions, softly glowing white ceramic desk lamp on the left, small green plant, neat cream notebooks, muted silver pen tray, soft frontal daylight wrapping evenly across the face, faint warm practical glow behind the speaker, and subtle glass reflections on the right side. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, gentle rim light, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, gives a small reassuring nod, moves one hand slowly near the desk in an organized explaining gesture, then returns to the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the movement is measured, small, and professional. \n"
            "Speaker_1's Facial Expression: calm, trustworthy, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。你可以直接问我想聊的内容。我会先听清楚你的问题，再把结论讲得简单一点。\"\n"
            "Speaker_1's Emotion: calm, professional, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. The office room tone remains quiet and stable."
        ),
    },
    {
        "name": "live5-grade wellness portrait cumulative speech",
        "prompt": (
            "Summary: The same calm digital-human conversation continues in a premium plant-filled consultation studio, preserving the same older male host identity, off-white knit top, linen curtains, green leaves, ceramic diffuser, woven basket, warm lamp glow, window light, and stable natural color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the plant-filled consultation studio remains physically identical: layered green leaves along the back shelf, pale linen curtains, a ceramic diffuser on light wood, woven basket texture, soft window light wrapping across the face, faint amber lamp glow behind the speaker, muted cream wall reflections, and a clean natural lens perspective. Bright balanced lighting, natural warm colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Older male wellness host with warm tan skin, short silver hair, a trustworthy smile, soft wrinkles around the eyes, and an off-white knit top.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps warm eye contact, slowly raises one hand near chest level, then makes a gentle calming gesture and returns to a centered posture. The shoulders stay relaxed, the motion is smooth and quiet, and the expression remains steady. \n"
            "Speaker_1's Facial Expression: calm, caring, and steady.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"先别着急，我们一步一步来。你把最困扰的地方说出来，我会慢慢帮你拆开。\"\n"
            "Speaker_1's Emotion: calm, reassuring, and helpful.\n"
            "Speaker_1's Voice Description: warm mature Mandarin voice, measured pacing, clear articulation, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke follow-up question cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues after a user follow-up, preserving the identical young male presenter identity, cream textured jacket, pale blue shirt, charcoal canvas backdrop, amber practical lamp, blue rim light, cream desk edge, green plant, haze diffusion, and vivid natural portrait color palette.\n"
            "Narration 3:\n"
            "eye-level tight medium close-up shot. the premium live5-style canvas studio remains physically identical: matte charcoal backdrop with visible woven texture, faint atmospheric haze diffusion, warm amber lamp glow on the left, cool blue rim light tracing the glass shelf, cream desk edge low in the frame, a small green plant near the shelf, subtle dark-wall reflections, soft face highlights across the cheeks, delicate catchlights in the eyes, and a stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable live-demo portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, gives a tiny listening nod, then raises one hand slowly near chest level to answer the follow-up before settling back into the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the motion is smooth, small, and conversational. \n"
            "Speaker_1's Facial Expression: focused, friendly, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以，我们继续刚才的问题。第一点是先确认你真正想优化的目标。\"\n"
            "Speaker_1's Emotion: calm, focused, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "live5 canvas smoke cross-turn cumulative prompt without context note",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues into a new answer turn as one uninterrupted continuous live5-style studio portrait, preserving the identical young male presenter identity, cream textured jacket, pale blue shirt, charcoal woven canvas backdrop, warm amber lamp, cool blue rim light, cream desk edge, green plant, faint haze, and stable vivid-natural portrait colors.\n"
            "Narration 4:\n"
            "eye-level tight medium close-up shot. the premium live5-style canvas studio remains physically identical: matte charcoal backdrop with readable woven texture, faint atmospheric studio haze, warm amber practical lamp glow on the left, cool blue rim light tracing the glass shelf, cream desk edge low in the foreground, a small green plant near the shelf, subtle dark-wall reflections, soft cheek highlights, delicate eye catchlights, gentle rim separation on the hair edge, and the same stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable live-demo portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, gives a tiny attentive nod, slowly lifts one hand near chest level to begin the next answer, then lets the hand settle back near the desk while maintaining the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the motion is smooth, small, and conversational. \n"
            "Speaker_1's Facial Expression: focused, friendly, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。今天想聊点什么？可以，我们继续刚才的问题。第一点是先确认目标。第二点是让后面的回答自然接住前面的画面和语气。\"\n"
            "Speaker_1's Emotion: calm, focused, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
    {
        "name": "cinematic business consultant dense handcrafted prompt",
        "prompt": (
            "Summary: A professional real-time digital-human consultant gives a concise answer in a refined cinematic consulting studio, with stable face identity, controlled daylight, soft practical lighting, and a calm premium report-style portrait look.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a refined live5-style consulting studio with a matte ivory desk edge in the lower foreground, pale oak shelves arranged with cream notebooks, a softly glowing white ceramic desk lamp on the left, a small green plant, a muted silver pen tray, clear glass partitions catching faint reflections, warm practical highlights behind the speaker, and soft frontal daylight wrapping evenly across the face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, gentle rim light on the hair edge, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a small reassuring nod, and lets one hand move slowly near the desk as if calmly organizing the answer. The torso stays centered, the shoulders stay relaxed, the chin angle remains consistent, and the movement stays measured, small, and professional. \n"
            "Speaker_1's Facial Expression: calm, trustworthy, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"这个问题可以先看结论，再看原因。我会用两句话把重点说清楚。\"\n"
            "Speaker_1's Emotion: calm, professional, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The office room tone remains quiet and stable."
        ),
    },
    {
        "name": "live5-grade warm library host cumulative speech",
        "prompt": (
            "Summary: The same premium real-time digital-human conversation continues in a warm cinematic library studio, preserving the same senior female host identity, silk scarf, walnut desk edge, brass lamp glow, bookcase texture, side-window daylight, and stable natural portrait color palette.\n"
            "Narration 2:\n"
            "eye-level tight medium close-up shot. the warm library studio remains physically identical: walnut desk edge low in frame, tall bookshelves with softly blurred book spines, a brass reading lamp glowing on the left, a small porcelain cup, a muted green plant, textured cream curtains, gentle side-window daylight wrapping across the cheeks, faint amber practical highlights behind the speaker, and subtle glass reflections on a framed print. Bright balanced lighting, natural warm colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, gentle rim light on the hair edge, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Senior female host with fair skin, short softly waved silver-brown hair, refined natural makeup, a warm attentive face, a beige blazer, and a muted teal silk scarf.\n"
            "Speaker_1's Actions: Speaker_1 keeps direct eye contact with the lens, gives a tiny thoughtful nod, slowly raises one hand near chest level to underline the answer, then lets the hand settle near the desk while maintaining the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the motion is gentle, small, and conversational. \n"
            "Speaker_1's Facial Expression: thoughtful, warm, and reassuring.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"我们先把这件事讲清楚。核心不是把内容说得更多，而是让每一段都稳稳接住前面的画面和语气。\"\n"
            "Speaker_1's Emotion: calm, thoughtful, and helpful.\n"
            "Speaker_1's Voice Description: warm mature Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable."
        ),
    },
]


def _curated_prompt_example(template_id: str) -> Dict[str, str]:
    scene = ENGLISH_TEMPLATE_SCENES[template_id]
    appearance = ENGLISH_TEMPLATE_APPEARANCES[template_id]
    return {
        "name": f"curated {template_id}",
        "template_id": template_id,
        "prompt": (
            "Summary: A polished, physically coherent digital-human conversation begins in a restrained real set, "
            "preserving identity, camera geometry, light direction, and the exact placement of every visible object.\n"
            "Narration 1:\n"
            f"{scene}. {LANDSCAPE_CLOSE_COMPOSITION}  "
            f"{TEMPLATE_PHYSICAL_ANCHORS[template_id]} "
            "The dominant soft key and neutral fill retain one consistent shadow direction. "
            "The eye-level 50mm-equivalent lens preserves natural facial proportions and straight verticals. "
            "Crisp eyes, natural skin microtexture, clean garment weave, controlled highlights, well-resolved edges, "
            "and moderate depth of field create a clear high-end photographic image.\n"
            f"Speaker_1's Appearance: {appearance}.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a tiny natural nod, "
            "and makes one small slow hand gesture near chest level before returning to a supported centered posture. "
            "The shoulders remain relaxed and the motion stays smooth and conversational.\n"
            "Speaker_1's Facial Expression: calm, attentive, and natural.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好，很高兴见到你。\"\n"
            "Speaker_1's Emotion: calm, attentive, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, measured conversational pacing, "
            "and close stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with lip movement and the small gesture follows the spoken cadence. "
            "Background audio remains quiet and stable."
        ),
    }


REALISTIC_PROMPT_PLANNER_EXAMPLES = [
    _curated_prompt_example(template_id)
    for template_id in ENGLISH_TEMPLATE_SCENES
]


PROMPT_PLANNER_INPUT_OUTPUT_DEMOS = (
    "Input-output teaching pattern for the small LLM prompt compiler: the input is a JSON list of "
    "target segments with display_speech, prompt_speech, emotion, and action_hint. The output is "
    "not a chat reply and not a short UI template; it is JSON whose prompt field is a complete "
    "handcrafted LTX video prompt. For segment 1, prompt_speech contains the previous same-scene "
    "tail segment if available plus the current spoken line. For segment 2 and later, prompt_speech "
    "contains the previous current-turn segment plus the current segment. The compiler must place "
    "that exact local spoken span only inside "
    "Speech Attribution / Speaker_1 says. The visual part of the prompt should be as rich as the "
    "GOOD CASES: Summary plus dense Narration, stable Appearance, positive face-forward Actions, "
    "Facial Expression, Held Objects, Speech Attribution, Emotion, Voice Description, and "
    "Sound-Visual Alignment. A correct answer looks like the live5 canvas smoke analytical-answer "
    "cases; an incorrect answer looks like 'Segment 1 ... calm and helpful ... view prompt' or "
    "a short generic office description. When uncertain, adapt the live5 canvas smoke physical "
    "anchors: charcoal woven canvas, faint haze, amber lamp, blue rim light, cream desk edge, "
    "small plant, wall reflections, cheek highlights, eye catchlights, shallow depth of field, "
    "natural camera geometry, and vivid but natural colors."
)


PROMPT_PLANNER_FEWSHOT_DEMOS = (
    "Explicit few-shot demos for the small LLM prompt compiler:\n"
    "DEMO INPUT A:\n"
    "[{\"segment_id\":0,\"display_speech\":\"你好啊，我在。今天想聊点什么？\","
    "\"prompt_speech\":\"你好啊，我在。今天想聊点什么？\","
    "\"emotion\":\"calm, bright, and helpful\","
    "\"action_hint\":\"faces the camera directly, keeps steady eye contact, gives one tiny welcoming nod, and lets the right hand rise slowly near chest level\"}]\n"
    "DEMO OUTPUT A:\n"
    "[{\"segment_id\":0,\"emotion\":\"calm, bright, and helpful\","
    "\"action\":\"faces the camera directly, keeps steady eye contact, gives one tiny welcoming nod, and lets the right hand rise slowly near chest level\","
    "\"prompt\":\"Summary: A premium real-time digital-human conversation begins with a short natural greeting in a cinematic canvas-backdrop live studio, preserving stable identity, natural lens geometry, detailed light direction, and a polished live-demo portrait look.\\n"
    "Narration 1:\\n"
    "eye-level tight medium close-up shot. a premium live5-style studio with a matte charcoal canvas backdrop showing fine woven texture, faint studio haze diffusion, a warm amber practical lamp glowing softly on the left, a cool blue rim light along a glass shelf, a cream desk edge in the lower foreground, a small green plant placed near the shelf, subtle dark-wall reflections, clean silhouette edges, soft cheek highlights, delicate catchlights in the eyes, and a stable clean lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \\n"
    "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\\n"
    "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives one tiny welcoming nod, and lets the right hand rise slowly near chest level in a relaxed open-palmed greeting before settling naturally. The torso stays centered, the shoulders stay relaxed, and the movement remains smooth and conversational. \\n"
    "Speaker_1's Facial Expression: calm, bright, and attentive.\\n"
    "Speaker_1's Held Objects:\\nNone\\n"
    "Speech Attribution:\\nSpeaker_1 says: \\\"你好啊，我在。今天想聊点什么？\\\"\\n"
    "Speaker_1's Emotion: calm, bright, and helpful.\\n"
    "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\\n"
    "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. Background audio remains quiet and stable.\"}]\n"
    "DEMO INPUT B:\n"
    "[{\"segment_id\":1,\"display_speech\":\"第二步是让口播接住前面的语义。\","
    "\"prompt_speech\":\"这个问题先看结论。核心是让后续片段继续依赖同一个人物、光线和画面锚点。第二步是让口播接住前面的语义。\","
    "\"emotion\":\"attentive, steady, and reassuring\","
    "\"action_hint\":\"keeps the face oriented toward the lens, continues with a slow open-palmed explaining gesture near chest level, then lets the hand relax near the desk\"}]\n"
    "DEMO OUTPUT B:\n"
    "[{\"segment_id\":1,\"emotion\":\"attentive, steady, and reassuring\","
    "\"action\":\"keeps the face oriented toward the lens, continues with a slow open-palmed explaining gesture near chest level, then lets the hand relax near the desk\","
    "\"prompt\":\"Summary: The same premium real-time digital-human conversation continues the analytical answer in one coherent continuous live5-style video, preserving the identical presenter identity, cream textured jacket, pale blue shirt, charcoal canvas backdrop, amber lamp, blue rim light, cream desk edge, green plant, haze diffusion, and vivid natural portrait color palette.\\n"
    "Narration 2:\\n"
    "eye-level tight medium close-up shot. the premium live5-style studio remains physically identical: matte charcoal canvas backdrop with visible woven texture, faint atmospheric haze diffusion, warm amber practical lamp glow on the left, cool blue rim light along the glass shelf, cream desk edge low in frame, small green plant beside the shelf, dark-wall reflections, soft highlights across the cheeks, delicate eye catchlights, clean silhouette edges, and the same stable natural lens perspective. Bright balanced lighting, vivid but natural colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable live-demo portrait composition, clean digital-human demo look. \\n"
    "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a stable friendly face, and a cream textured jacket over a pale blue shirt.\\n"
    "Speaker_1's Actions: Speaker_1 keeps the face oriented toward the lens, continues with a slow open-palmed explaining gesture near chest level, then lets the hand relax near the desk while maintaining the same centered posture. The shoulders stay relaxed, the chin angle remains consistent, and the motion remains smooth, small, and conversational. \\n"
    "Speaker_1's Facial Expression: attentive, steady, and reassuring.\\n"
    "Speaker_1's Held Objects:\\nNone\\n"
    "Speech Attribution:\\nSpeaker_1 says: \\\"这个问题先看结论。核心是让后续片段继续依赖同一个人物、光线和画面锚点。第二步是让口播接住前面的语义。\\\"\\n"
    "Speaker_1's Emotion: attentive, steady, and helpful.\\n"
    "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\\n"
    "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable.\"}]\n"
    "DEMO INPUT C:\n"
    "[{\"segment_id\":0,\"display_speech\":\"这个问题可以先看结论，再看原因。\","
    "\"prompt_speech\":\"这个问题可以先看结论，再看原因。\","
    "\"emotion\":\"calm, professional, and helpful\","
    "\"action_hint\":\"faces the camera directly, keeps steady eye contact, gives a small reassuring nod, and lets one hand move slowly near the desk\"}]\n"
    "DEMO OUTPUT C:\n"
    "[{\"segment_id\":0,\"emotion\":\"calm, professional, and helpful\","
    "\"action\":\"faces the camera directly, keeps steady eye contact, gives a small reassuring nod, and lets one hand move slowly near the desk\","
    "\"prompt\":\"Summary: A professional real-time digital-human consultant gives a concise answer in a refined cinematic consulting studio, preserving stable face identity, controlled daylight, soft practical lighting, glass reflections, and a calm premium report-style portrait look.\\n"
    "Narration 1:\\n"
    "eye-level tight medium close-up shot. a refined live5-style consulting studio with a matte ivory desk edge in the lower foreground, pale oak shelves arranged with cream notebooks, a softly glowing white ceramic desk lamp on the left, a small green plant, a muted silver pen tray, clear glass partitions catching faint reflections, warm practical highlights behind the speaker, and soft frontal daylight wrapping evenly across the face. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, soft face highlights, gentle rim light on the hair edge, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \\n"
    "Speaker_1's Appearance: Mature female business consultant with soft fair skin, shoulder-length chestnut hair tucked neatly behind one ear, natural makeup, a light gray blazer, and a white inner layer.\\n"
    "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, gives a small reassuring nod, and lets one hand move slowly near the desk as if calmly organizing the answer. The torso stays centered, the shoulders stay relaxed, the chin angle remains consistent, and the movement stays measured, small, and professional. \\n"
    "Speaker_1's Facial Expression: calm, trustworthy, and attentive.\\n"
    "Speaker_1's Held Objects:\\nNone\\n"
    "Speech Attribution:\\nSpeaker_1 says: \\\"这个问题可以先看结论，再看原因。\\\"\\n"
    "Speaker_1's Emotion: calm, professional, and helpful.\\n"
    "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\\n"
    "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The office room tone remains quiet and stable.\"}]\n"
    "DEMO INPUT D:\n"
    "[{\"segment_id\":2,\"display_speech\":\"你把最困扰的地方说出来，我会慢慢帮你拆开。\","
    "\"prompt_speech\":\"先别着急，我们一步一步来。你把最困扰的地方说出来，我会慢慢帮你拆开。\","
    "\"emotion\":\"calm, reassuring, and helpful\","
    "\"action_hint\":\"faces the camera directly, keeps warm eye contact, slowly raises one hand near chest level, then makes a gentle calming gesture\"}]\n"
    "DEMO OUTPUT D:\n"
    "[{\"segment_id\":2,\"emotion\":\"calm, reassuring, and helpful\","
    "\"action\":\"faces the camera directly, keeps warm eye contact, slowly raises one hand near chest level, then makes a gentle calming gesture\","
    "\"prompt\":\"Summary: The same calm digital-human conversation continues in a premium plant-filled consultation studio, preserving the same older male host identity, off-white knit top, linen curtains, green leaves, ceramic diffuser, woven basket, warm lamp glow, window light, and stable natural color palette.\\n"
    "Narration 3:\\n"
    "eye-level tight medium close-up shot. the plant-filled consultation studio remains physically identical: layered green leaves along the back shelf, pale linen curtains, a ceramic diffuser on light wood, woven basket texture, soft window light wrapping across the face, faint amber lamp glow behind the speaker, muted cream wall reflections, gentle rim separation on the hair edge, delicate eye catchlights, soft cheek highlights, and a clean natural lens perspective. Bright balanced lighting, natural warm colors, soft portrait contrast, rich layered background details, shallow depth of field, cinematic but stable portrait composition, clean digital-human demo look. \\n"
    "Speaker_1's Appearance: Older male wellness host with warm tan skin, short silver hair, a trustworthy smile, soft wrinkles around the eyes, and an off-white knit top.\\n"
    "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps warm eye contact, slowly raises one hand near chest level, then makes a gentle calming gesture and returns to a centered posture. The shoulders stay relaxed, the motion is smooth and quiet, and the expression remains steady. \\n"
    "Speaker_1's Facial Expression: calm, caring, and steady.\\n"
    "Speaker_1's Held Objects:\\nNone\\n"
    "Speech Attribution:\\nSpeaker_1 says: \\\"先别着急，我们一步一步来。你把最困扰的地方说出来，我会慢慢帮你拆开。\\\"\\n"
    "Speaker_1's Emotion: calm, reassuring, and helpful.\\n"
    "Speaker_1's Voice Description: warm mature Mandarin voice, measured pacing, clear articulation, close and stable recording quality.\\n"
    "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact cumulative spoken line. Background audio remains quiet and stable.\"}]\n"
    "Key lesson from these demos: the prompt is complete, cinematic, and visual; different people and scenes still keep the live5-grade density; the cumulative spoken line appears only inside Speaker_1 says."
)


PROMPT_PLANNER_CONTRACT = (
    "Treat the task as cinematic LTX prompt compilation, not chat summarization. The small LLM must convert "
    "reply segments into full handcrafted LTX video prompts that resemble live5 canvas smoke and "
    "benchmark 1min prompts. Use the reference scene as the physical set, then expand it with "
    "cinematic details: exact shot scale, face lighting direction, practical lamp color, rim light, "
    "background props, glass or wall reflections, texture, depth of field, portrait color palette, "
    "and stable camera framing. Do not merely rephrase the UI scene description; synthesize a full "
    "artist-facing prompt from the good cases, so the result feels hand written, inspectable, and "
    "close to the live5 canvas smoke benchmark. For every later segment, keep the same physical anchors and put "
    "the exact local spoken span only inside Speech Attribution / Speaker_1 says. Never place "
    "a separate previous-context sentence in Narration. The generated prompt should read like a "
    "single complete video prompt that an artist could inspect, not like an instruction log. "
    "For interactive follow-up turns, do not restart the visual world and do not describe prior "
    "speech outside the dialogue; only Speaker_1 says carries the local spoken continuity span."
)


PROMPT_PLANNER_CURRICULUM = (
    "Curriculum for the small prompt compiler: study the good cases as a writing dataset. "
    "Every output should feel like it came from a human benchmark prompt writer: a fixed medium-close portrait, "
    "named physical anchors, inspectable light sources, stable face identity, and smooth small presenter motion. "
    "The live5 canvas smoke reference is the default quality bar: charcoal woven backdrop, faint haze diffusion, "
    "warm amber practical lamp, cool blue rim light, cream desk edge, small plant, dark-wall reflections, soft cheek "
    "highlights, delicate eye catchlights, shallow depth of field, and vivid but natural colors. "
    "When adapting other templates, translate that same density into the selected setting instead of returning a "
    "short generic office prompt. Learn from multiple cases at once: live5 canvas smoke teaches the density bar, "
    "the consulting, technology, wellness, classroom, library, cafe, and product-counter cases teach how to transfer "
    "that density to different people, ages, genders, and environments without weakening continuity. Segment prompts should be visually rich enough that a person can picture the "
    "frame before generation: foreground edge, speaker face, outfit texture, main light, rim light, practical glow, "
    "background objects, reflections, palette, and continuity anchors. Any non-English text in the final LTX prompt "
    "is confined to the exact local spoken span inside Speaker_1 says."
)


PROMPT_PLANNER_VISUAL_DENSITY_TERMS = (
    "medium close-up",
    "foreground",
    "desk edge",
    "supporting surface",
    "soft key",
    "neutral fill",
    "catchlights",
    "controlled highlights",
    "attached shadows",
    "texture",
    "material",
    "depth of field",
    "50mm-equivalent",
    "rectilinear",
    "straight verticals",
    "portrait",
    "color palette",
    "background objects",
    "stable",
    "physically coherent",
    "crisp eyes",
)


PROMPT_PLANNER_LEARNING_NOTES = (
    "The examples are teaching cases and should be treated as an in-context writing curriculum. "
    "Learn their structure, visual density, lighting vocabulary, and continuity discipline, then "
    "adapt them to the current reference scene instead of copying a single template name. Strong prompts behave like "
    "the benchmark 1min handwritten prompts: a complete Summary, a dense Narration paragraph, a "
    "stable Appearance description, a precise small Action description, a clear Facial Expression, "
    "Held Objects, Speech Attribution, Voice Description, and Sound-Visual Alignment. Good prompts "
    "repeat the same physical anchors in later segments: exact face identity, outfit, desk edge, "
    "lamp position, rim light, plant or shelf objects, background reflections, color palette, and "
    "natural camera composition. For live5 canvas smoke, preserve the charcoal canvas texture, faint haze, "
    "warm amber practical lamp, cool blue rim light, cream desk edge, small plant, dark-wall "
    "reflections, delicate eye catchlights, soft face highlights, and vivid but natural colors. "
    "For other templates, use the selected scene with the same level of concrete lighting and "
    "background detail. Later segments should sound like one continuous video by putting the exact "
    "prompt_speech spoken span inside Speaker_1 says; never add a separate sentence like 'previously "
    "generated spoken context' outside Speech Attribution. Do not output a simplified chat prompt, "
    "do not output only an action summary, and do not use broad negative lists. Describe what the "
    "speaker should do: face the lens, keep steady eye contact, speak naturally, and use small "
    "smooth hand gestures near chest level. A weak prompt is short, generic, office-like, or "
    "mostly conversational; reject that style and write a full cinematic handcrafted prompt instead. "
    "For follow-up turns, do not reset to a generic new answer prompt; preserve the same handcrafted "
    "scene language and let Speaker_1 says carry the local spoken continuity span. The planner "
    "should learn from multiple good cases, not from the short UI template text; if the current "
    "scene is sparse, expand it toward the live5 canvas smoke level of detail."
)


PROMPT_PLANNER_STYLE_GUIDE = (
    "Write prompts like a manually curated benchmark prompt, not a compressed chat template. "
    "The target quality is the live5 canvas smoke demo style: cinematic studio portrait, "
    "stable face identity, detailed light sources, concrete background anchors, and a "
    "complete handcrafted LTX prompt for each segment. "
    "Each prompt should have these exact sections: Summary, Narration, Speaker_1's Appearance, "
    "Speaker_1's Actions, Speaker_1's Facial Expression, Speaker_1's Held Objects, "
    "Speech Attribution, Speaker_1's Emotion, Speaker_1's Voice Description, and "
    "Sound-Visual Alignment. The visual writing should be concrete and cinematic: describe "
    "shot scale, stable camera, face lighting, rim or practical lights, background objects, "
    "color palette, depth of field, reflections or studio texture, and one small positive action. "
    "Every segment must be a complete standalone video prompt with "
    "a detailed scene paragraph; never output only a summary plus a short action line. Keep the "
    "scene physically identical across segments, including face identity, outfit, hairstyle, camera "
    "position, background objects, lighting, and color palette. "
    "For segment 2 and later, and for follow-up turns whose prompt_speech is longer than "
    "display_speech, the Speaker_1 says line may intentionally contain a local spoken span; "
    "use prompt_speech exactly as the full line and "
    "write the visual prompt as a continuation of the established take, not as an opening shot. "
    "Do not use short generic phrases such as 'office background' or 'stable presenter'. "
    "Do not use negative prohibition lists; describe the desired stable face-forward behavior instead. "
    "If the user does not choose a scene, adapt the live5 canvas smoke reference instead of inventing a generic office. "
    "Do not mention previous context, prompt planning, generated video, segment splitting, or model internals. "
    "Any non-English text is allowed only inside Speaker_1 says, and Speaker_1 says must exactly equal prompt_speech. "
    "The final video prompt itself should never mention prompt writing, benchmark examples, few-shot learning, "
    "prompt_speech, display_speech, or previous context; those are only compiler-side controls. "
    "The best outputs are dense, visually inspectable, and cinematic; the worst outputs look like short chatbot prompts. "
    "A follow-up prompt should read like the next shot of the same handcrafted video, with no visible reset."
)


PROMPT_PLANNER_SMALL_LLM_TRAINING_PROMPT = (
    "New prompt-compiler target: the small LLM should learn to write prompts that look like "
    "the live5 canvas smoke handwritten benchmark prompts, not like simplified interaction prompts. "
    "Use the selected high-quality cases as a compact writing dataset, and generalize from them instead "
    "of copying the short UI template. The model should imitate their "
    "structure, visual density, cinematic lighting, stable physical anchors, and cumulative-speech "
    "discipline. For an unspecified or weak scene, default toward the live5 canvas smoke look: matte "
    "charcoal woven canvas, faint studio haze, warm amber practical lamp, cool blue rim light, cream "
    "desk edge, green plant, dark-wall reflections, soft cheek highlights, delicate eye catchlights, "
    "shallow depth of field, tight medium close-up framing, and vivid but natural colors. For a chosen "
    "different persona or scene, keep the same quality level while changing only the concrete person, "
    "age, gender, outfit, and environment. Every output segment should feel like a dense cinematic "
    "video prompt that could be used directly for 1min benchmark-style LTX inference. The non-speech "
    "prompt language is English only; the complete local spoken span appears only in "
    "Speaker_1 says."
)


PROMPT_PLANNER_CASE_STUDY_GUIDE = (
    "Case-study guide for the small LLM: treat these as the reusable writing moves behind the "
    "live5 canvas smoke result, not as optional decoration. A strong prompt first establishes a "
    "fixed medium-close portrait with a named foreground edge, then locks the face identity, "
    "hair, outfit, camera angle, and background anchors. It next describes physical lighting in "
    "inspectable terms: warm practical lamp position, cool rim light, soft cheek highlights, eye "
    "catchlights, wall or glass reflections, and shallow depth of field. It then gives exactly one "
    "small face-forward action, such as a tiny nod or a slow open-palmed hand gesture near chest "
    "level. Later prompts must read as the next uninterrupted moment of the same video: reuse the "
    "same set, same clothing, same lights, same prop placement, same color palette, and same fixed "
    "lens framing. Speech continuity is not a separate visual note; the full local spoken span "
    "line is placed only in Speaker_1 says. For short greetings, still write the complete cinematic "
    "prompt. For analytical answers, keep the same studio grammar and let the action become a small "
    "explaining gesture. For follow-up turns, do not restart the world; preserve the handcrafted "
    "scene and let the dialogue line carry the new turn."
)


PROMPT_PLANNER_BAD_OUTPUT_EXAMPLES = (
    "Bad outputs to reject before returning JSON: "
    "1. A UI-facing summary such as 'Segment 1 / calm and helpful / view LTX prompt'. "
    "2. A simplified interaction prompt with one generic office sentence and one short action line. "
    "3. Any visible sentence like 'Previously generated spoken context', 'Speaker_1 has already said', "
    "'continue from this full context', or 'the current segment must continue'. "
    "4. Any prompt that describes the compiler, good cases, examples, prompt_speech, display_speech, "
    "segment planning, benchmark writing, or model internals. "
    "5. Any prompt where non-English text appears outside Speaker_1 says. "
    "Rewrite these failures as a complete live5-grade handcrafted LTX prompt with dense cinematic "
    "scene detail, stable identity anchors, inspectable lighting, and cumulative speech only inside "
    "Speaker_1 says."
)


PROMPT_PLANNER_QUALITY_RUBRIC = (
    "Before returning JSON, privately check every prompt against this rubric: "
    "1. It can stand alone as a complete LTX video prompt with Summary, Narration, Appearance, Actions, Facial Expression, Held Objects, Speech Attribution, Emotion, Voice Description, and Sound-Visual Alignment. "
    "2. The Narration paragraph is visually dense and cinematic, naming concrete light sources, face highlights, rim or practical lights, background props, reflections or texture, depth of field, and natural camera composition. "
    "3. The selected scene is not simplified into a generic office or generic presenter; if no scene is specified, use the live5 canvas smoke studio as the quality target. "
    "4. Later segments preserve the identical set, face, outfit, camera, colors, and lighting, while Speaker_1 says contains prompt_speech exactly. "
    "5. The prompt contains no meta language about prior context, few-shot examples, prompt generation, segments, models, or compilation. "
    "6. All non-speech text is English, and the local spoken line appears only inside Speaker_1 says. "
    "If any item fails, rewrite the prompt before output."
)


PROMPT_PLANNER_LIVE5_TEACHING_BRIEF = (
    "Live5 prompt-writing curriculum for the small LLM: learn from the good cases below as if they are a compact dataset of hand-written LTX benchmark prompts. "
    "The final JSON prompt field must be a complete cinematic video prompt, not a UI description, not a chat summary, and not a simplified interaction prompt. "
    "Use this grammar every time: Summary -> Narration -> Speaker_1's Appearance -> Speaker_1's Actions -> Speaker_1's Facial Expression -> Speaker_1's Held Objects -> Speech Attribution -> Speaker_1's Emotion -> Speaker_1's Voice Description -> Sound-Visual Alignment. "
    "The live5 canvas smoke case is the quality target: tight medium close-up, matte charcoal canvas or equivalently concrete scene texture, foreground desk edge, practical lamp glow, rim light, face highlights, catchlights, reflections, material detail, shallow depth of field, stable natural colors, and one small face-forward gesture. "
    "For other templates, transfer the same writing moves to the selected person and set instead of falling back to a generic office. "
    "For segment 2 and follow-up turns, keep the same world and put the exact local spoken span only inside Speaker_1 says. "
    "A strong output is cinematic and inspectable but not repetitive: roughly 1500 to 2600 characters per segment is enough when the scene, light, identity, action, voice, and alignment are all explicit. "
    "Never leak compiler words such as prompt_speech, previous context, few-shot, benchmark, prompt compiler, or segment plan into the returned video prompt."
)


def _select_prompt_planner_examples(template_id: str, max_examples: int) -> List[Dict[str, str]]:
    max_examples = max(1, int(max_examples or 4))
    key = str(template_id or "").strip()
    selected = [
        item for item in REALISTIC_PROMPT_PLANNER_EXAMPLES
        if item.get("template_id") == key
    ]
    for item in REALISTIC_PROMPT_PLANNER_EXAMPLES:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max_examples:
            break
    return selected[:max_examples]


def _english_scene_description(scene: str, *, template_id: str = "", has_first_frame: bool = False) -> str:
    scene_text = re.sub(r"\s+", " ", str(scene or "").strip()).rstrip("。.!！?？；;，, ")
    if scene_text and not _contains_cjk(scene_text) and _looks_like_compiled_scene_prompt(scene_text):
        return scene_text[:1100]
    template_scene = ENGLISH_TEMPLATE_SCENES.get(str(template_id or "").strip())
    if template_scene:
        return template_scene
    if scene_text and not _contains_cjk(scene_text):
        return scene_text[:520]
    if has_first_frame:
        return (
            "The clip follows the uploaded realistic first-frame reference in a natural "
            "tight medium close-up composition. The same person, setting, clothing, and lighting "
            "remain visually coherent throughout the shot"
        )
    return LIVE5_CANVAS_SMOKE_SCENE


def _scene_anchor_sentence_for_prompt(scene: str, *, template_id: str = "") -> str:
    template_anchor = TEMPLATE_PHYSICAL_ANCHORS.get(str(template_id or "").strip())
    if template_anchor:
        return template_anchor
    return (
        "Every named object remains in its stated position on a visible supporting surface. Furniture, walls, "
        "floor lines, body support, material texture, light direction, attached shadows, and lens perspective "
        "remain physically coherent and unchanged throughout the take."
    )


def _remove_locked_camera_language(value: str) -> str:
    text = str(value or "")
    text = re.sub(
        r"\bcamera\s+remains\s+(?:absolutely\s+)?(?:fixed|stable)\b\.?\s*(?:framing\s+unchanged\b\.?)?",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bframing\s+unchanged\b\.?", "", text, flags=re.I)
    replacements = (
        (r"\bfixed[- ]camera\s+framing\b", "natural camera composition"),
        (r"\bfixed[- ]camera\b", "continuous"),
        (r"\bfixed\s+lens\s+framing\b", "natural lens composition"),
        (
            r"\b(?:(?:same|stable|clean)\s+)*fixed\s+lens\s+(?:perspective|angle)\b",
            "natural lens perspective",
        ),
        (r"\bfixed[- ]lens\b", "constant-focal-length"),
        (r"\b(?:fixed|unchanged)\s+framing\b", "natural composition"),
        (
            r"\bstatic\s+(?=(?:eye-level\s+)?(?:tight\s+)?medium(?:\s+close-up)?\s+shot\b)",
            "",
        ),
    )
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.I)
    text = re.sub(r"[ \t]+([.,])", r"\1", text)
    text = re.sub(r"\.\s*\.", ".", text)
    return re.sub(r"[ \t]{2,}", " ", text)


def _ensure_direct_camera_take_prompt(
    prompt: str,
    *,
    aspect_ratio: str = "landscape",
) -> str:
    """Apply the direct-camera visual contract once for every planner path."""
    text = str(prompt or "")
    if not text or "photographed physical material with uninterrupted texture" in text.lower():
        return text
    marker = "\nSpeaker_1's Appearance:"
    if marker not in text:
        return text
    narration, remainder = text.split(marker, 1)
    clause = (
        PORTRAIT_DIRECT_CAMERA_TAKE_CLAUSE
        if str(aspect_ratio or "").strip().lower() == "portrait"
        else LANDSCAPE_DIRECT_CAMERA_TAKE_CLAUSE
    )
    return f"{narration.rstrip()} {clause}\n{marker.lstrip()}{remainder}"


def _looks_like_compiled_scene_prompt(scene: str) -> bool:
    text = re.sub(r"\s+", " ", str(scene or "").strip())
    if not text or _contains_cjk(text) or len(text) < 180:
        return False
    lower = text.lower()
    terms = [
        "medium close-up",
        "lighting",
        "camera",
        "lens",
        "face",
        "eye contact",
        "background",
        "portrait",
        "depth of field",
        "catchlights",
    ]
    return sum(1 for term in terms if term in lower) >= 6


def _scene_prompt_cache_key(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "",
    has_first_frame: bool = False,
    refine_scene: bool = False,
) -> Tuple[Any, ...]:
    key = (
        PROMPT_PLANNER_PROFILE,
        "scene_pe_v5",
        _normalize_scene_for_signature(scene),
        str(template_id or "").strip(),
        str(aspect_ratio or "").strip(),
        bool(has_first_frame),
    )
    return (*key, "refine_scene") if refine_scene else key


def _scene_prompt_cache_get(key: Tuple[Any, ...]) -> Optional[Dict[str, str]]:
    cached = SCENE_PROMPT_CACHE.get(key)
    if cached is not None:
        SCENE_PROMPT_CACHE.move_to_end(key)
        return dict(cached)
    cached = _persistent_prompt_cache_get("scene_prompt", key)
    if cached is not None:
        SCENE_PROMPT_CACHE[key] = dict(cached)
        SCENE_PROMPT_CACHE.move_to_end(key)
        while len(SCENE_PROMPT_CACHE) > SCENE_PROMPT_CACHE_SIZE:
            SCENE_PROMPT_CACHE.popitem(last=False)
        return dict(cached)
    return None


def _scene_prompt_cache_put(key: Tuple[Any, ...], value: Dict[str, str]) -> None:
    if SCENE_PROMPT_CACHE_SIZE <= 0:
        return
    SCENE_PROMPT_CACHE[key] = dict(value)
    SCENE_PROMPT_CACHE.move_to_end(key)
    while len(SCENE_PROMPT_CACHE) > SCENE_PROMPT_CACHE_SIZE:
        SCENE_PROMPT_CACHE.popitem(last=False)
    _persistent_prompt_cache_put("scene_prompt", key, value)


def _fallback_scene_prompt_pe(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "",
    has_first_frame: bool = False,
) -> str:
    scene_text = re.sub(r"\s+", " ", str(scene or "").strip()).rstrip("。.!！?？；;，, ")
    if scene_text and not _contains_cjk(scene_text) and len(scene_text) > 120:
        base = scene_text[:760].rstrip(". ")
    elif has_first_frame:
        base = (
            "eye-level tight medium close-up shot. The clip follows the uploaded realistic first-frame "
            "reference in a natural portrait composition with the same person, setting, "
            "clothing, lighting, and visual continuity"
        )
    else:
        base = ENGLISH_TEMPLATE_SCENES.get(str(template_id or "").strip(), LIVE5_CANVAS_SMOKE_SCENE)
        base = base.rstrip(". ")
    aspect_sentence = (
        PORTRAIT_CLOSE_COMPOSITION
        if str(aspect_ratio or "").lower() == "portrait"
        else LANDSCAPE_CLOSE_COMPOSITION
    )
    return (
        f"{base}. {aspect_sentence}. The person faces the camera directly, the face stays oriented toward the lens, "
        "and steady eye contact is maintained while speaking. The posture remains grounded and centered, with both forearms and hands "
        "naturally available in the lower quarter for one clear responsive gesture. An eye-level 50mm-equivalent lens preserves natural facial proportions "
        "and straight background verticals. One broad soft key and gentle neutral fill create a coherent light direction with controlled "
        "highlights and attached shadows. Every named object rests on a visible support. Crisp eyes, natural skin detail, visible outfit "
        "texture, well-resolved edges, moderate depth of field, and restrained background detail give the image a polished photographic finish. "
        "Garment folds, hand contours, furniture seams, shadows, and perspective continue naturally to the bottom edge."
    )


def _clean_scene_prompt_pe(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    raw = re.sub(r"^```(?:json|text)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                raw = str(parsed.get("scene") or parsed.get("prompt") or parsed.get("text") or "")
        except json.JSONDecodeError:
            pass
    raw = raw.strip().strip("\"'“”")
    raw = re.sub(r"(?is)speaker_1\s+says:.*", "", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    raw = _remove_locked_camera_language(raw)
    raw = re.sub(
        r"\blower[- ]third\b",
        "bottom part of the photographed scene",
        raw,
        flags=re.I,
    )
    raw = raw.rstrip("。.!！?？；;，, ")
    if not raw or _contains_cjk(raw) or _looks_like_collapsed_english(raw):
        return ""
    if not raw.lower().startswith("eye-level"):
        raw = f"eye-level tight medium close-up shot. {raw}"
    lower = raw.lower()
    required_tail: List[str] = []
    if "eye contact" not in lower or "oriented toward the lens" not in lower:
        required_tail.append(
            "The person faces the camera directly, keeps the face oriented toward the lens, and maintains steady eye contact."
        )
    if "hand gesture" not in lower and "hands" not in lower:
        required_tail.append(
            "Both forearms and hands remain naturally available inside the composition for one clear responsive gesture."
        )
    if "square frame" not in lower and "boxed" not in lower and "border" not in lower:
        required_tail.append(
            LANDSCAPE_CLOSE_COMPOSITION
        )
    if required_tail:
        raw = f"{raw}. {' '.join(required_tail)}"
    raw = re.sub(r"\s+", " ", raw).strip()
    if len(raw) < 220 or not _looks_like_compiled_scene_prompt(raw):
        return ""

    def _truncate_scene_text(value: str, max_chars: int, min_boundary: int = 700) -> str:
        value = re.sub(r"\s+", " ", str(value or "").strip())
        if len(value) <= max_chars:
            return value.rstrip(". ") + "."
        cut = value[:max_chars].rstrip()
        boundary = max(cut.rfind(". "), cut.rfind("."))
        if boundary >= min_boundary:
            cut = cut[: boundary + 1]
        return cut.rstrip(". ") + "."

    raw = _truncate_scene_text(raw, 1100)
    lower = raw.lower()
    final_tail: List[str] = []
    if "oriented toward the lens" not in lower:
        final_tail.append("The face stays oriented toward the lens throughout.")
    if "hand gesture" not in lower and "hand movement" not in lower and "hands" not in lower:
        final_tail.append("Both forearms and hands remain naturally available inside the composition for one clear responsive gesture.")
    if "boxed avatar" not in lower and "square frame" not in lower and "graphic border" not in lower:
        final_tail.append(LANDSCAPE_CLOSE_COMPOSITION)
    if final_tail:
        raw = f"{raw}. {' '.join(final_tail)}"
    raw = raw.replace("..", ".")
    return _truncate_scene_text(raw, 1500, min_boundary=900)


async def _compile_scene_prompt_pe(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "",
    has_first_frame: bool = False,
    refine_scene: bool = False,
) -> Dict[str, str]:
    key = _scene_prompt_cache_key(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
        refine_scene=refine_scene,
    )
    cached = _scene_prompt_cache_get(key)
    if cached:
        return cached

    provided = _clean_scene_prompt_pe(scene)
    if provided and not refine_scene:
        result = {"scene_prompt_text": provided, "scene_prompt_source": "provided_english"}
        _scene_prompt_cache_put(key, result)
        return result

    fallback = _fallback_scene_prompt_pe(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
    )
    if not SETTINGS.scene_prompt_pe_enabled:
        result = {"scene_prompt_text": fallback, "scene_prompt_source": "fallback_disabled"}
        _scene_prompt_cache_put(key, result)
        return result

    template_scene = ENGLISH_TEMPLATE_SCENES.get(str(template_id or "").strip(), LIVE5_CANVAS_SMOKE_SCENE)
    aspect_note = "vertical 9:16 portrait" if str(aspect_ratio or "").lower() == "portrait" else "horizontal 16:9 portrait"
    system = (
        "You are a senior prompt engineer for LTX long-video digital-human benchmark inference. "
        "Rewrite a user scene description into one concrete English scene paragraph for a realistic digital-human video. "
        "Return English only, one paragraph only, no JSON, no markdown, no labels, and no speech. "
        "The result must describe a physically buildable real set with at most three restrained background anchors. "
        "Every object must rest on a visible supporting surface, furniture and architecture must follow one coherent perspective, "
        "and the seated or standing body must have a clear physical support. Use an eye-level 50mm-equivalent rectilinear lens, "
        "straight architectural verticals, and moderate depth of field. Use one broad soft key as the dominant source and a gentle "
        "neutral fill so highlights are controlled and every shadow follows the same light direction. Prioritize crisp eyes, natural "
        "skin microtexture, clean garment weave, well-resolved edges, and quiet high-end photographic detail. "
        "Preserve the user's requested person, room, wardrobe, key objects, and mood; the template is style guidance only and must not replace that content. "
        "Use a tight medium close-up: the speaker fills about two thirds of the landscape frame height from head to mid-torso, "
        "the shoulders span roughly half the frame width, headroom and side margins stay compact, and only a narrow furniture edge "
        "is visible at the bottom. Keep both hands available in the lower quarter for compact gestures. "
        "Continue the photographed torso, garment folds, hand contours, furniture seams, natural shadows, and perspective naturally through the bottom edge. "
        "Use positive declarative action descriptions: the person faces the camera directly, the face stays oriented toward the lens, "
        "eye contact stays steady, posture is grounded, and both forearms and hands remain naturally available inside the frame for one clear responsive gesture. "
        "Keep the set restrained: add no decorative haze, colored rim lights, reflective partitions, floating graphics, visible writing, "
        "or extra props beyond what is needed to make the requested room recognizable."
    )
    user = (
        f"Raw scene description:\n{scene or '(empty)'}\n\n"
        f"Template id: {template_id or '(none)'}\n"
        f"Aspect: {aspect_note}\n"
        f"Uploaded first frame reference: {bool(has_first_frame)}\n\n"
        f"Template visual reference, only as style guidance:\n{template_scene}\n\n"
        "Write the final scene paragraph now. It should start with 'eye-level tight medium close-up shot.' "
        "Make it specific, realistic, and suitable for repeated 5-second conversational chunks."
    )
    compiled = ""
    try:
        raw = await asyncio.to_thread(
            _call_openai_compatible_llm,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            SETTINGS.scene_prompt_pe_max_tokens,
        )
        compiled = _clean_scene_prompt_pe(raw or "")
    except Exception:
        compiled = ""
    if compiled:
        result = {"scene_prompt_text": compiled, "scene_prompt_source": "llm_pe"}
    else:
        result = {"scene_prompt_text": fallback, "scene_prompt_source": "fallback"}
    _scene_prompt_cache_put(key, result)
    return result


async def _resolve_scene_prompt_text(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "",
    has_first_frame: bool = False,
    refine_scene: bool = False,
    conversation_meta: Optional[Dict[str, Any]] = None,
    scene_signature: str = "",
    task_dir: Optional[Path] = None,
) -> Dict[str, str]:
    meta = conversation_meta if isinstance(conversation_meta, dict) else {}
    cached_text = str(meta.get("scene_prompt_text") or "").strip()
    cached_signature = str(meta.get("scene_prompt_signature") or meta.get("scene_signature") or "")
    if cached_text and cached_signature == str(scene_signature or "") and _looks_like_compiled_scene_prompt(cached_text):
        return {
            "scene_prompt_text": cached_text,
            "scene_prompt_source": str(meta.get("scene_prompt_source") or "conversation_cache"),
        }
    resolved = await _compile_scene_prompt_pe(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
        refine_scene=refine_scene,
    )
    if task_dir:
        with suppress(OSError):
            (task_dir / "scene_prompt_pe.json").write_text(
                json.dumps(
                    {
                        "scene_prompt_signature": scene_signature,
                        "scene_prompt_source": resolved.get("scene_prompt_source"),
                        "scene_prompt_text": resolved.get("scene_prompt_text"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return resolved


def _normalize_planned_segments(
    raw_segments: List[Any],
    *,
    fallback_speeches: List[str],
    max_segments: int,
    add_waiting_transition: bool,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in raw_segments:
        if len(normalized) >= max_segments:
            break
        if not isinstance(item, dict):
            continue
        speech = _clean_segment_speech(item.get("speech") or item.get("text"))
        if not speech:
            continue
        seg_id = len(normalized)
        normalized.append(
            {
                "segment_id": seg_id,
                "speech": speech,
                "emotion": _safe_emotion(str(item.get("emotion") or "calm and helpful")),
                "action": _safe_action(str(item.get("action") or ""), seg_id),
                "is_transition": bool(item.get("is_transition", False)),
            }
        )
    if not normalized:
        for speech in fallback_speeches[:max_segments]:
            seg_id = len(normalized)
            normalized.append(
                {
                    "segment_id": seg_id,
                    "speech": speech,
                    "emotion": "calm and helpful",
                    "action": _safe_action("", seg_id),
                    "is_transition": False,
                }
            )
    if add_waiting_transition and len(normalized) < max_segments:
        if not normalized or not normalized[-1].get("is_transition"):
            seg_id = len(normalized)
            response_language = _dominant_response_language(
                " ".join(fallback_speeches)
            )
            transitions = WAITING_TRANSITIONS[response_language]
            normalized.append(
                {
                    "segment_id": seg_id,
                    "speech": transitions[seg_id % len(transitions)],
                    "emotion": "calm and ready to continue",
                    "action": "keeps a relaxed posture, blinks naturally, and makes a very small nod",
                    "is_transition": True,
                }
            )
    for idx, segment in enumerate(normalized):
        segment["segment_id"] = idx
    return normalized[:max_segments]


_CAMERA_FOLLOW_ACTION_RE = re.compile(
    r"\b(?:stand(?:s|ing)?[- ]+up|get(?:s|ting)?\s+up|ris(?:e|es|ing)\s+from|"
    r"rais(?:e|es|ed|ing)\s+(?:the\s+)?(?:hips|body)|reach(?:es|ed|ing)?\s+(?:an?\s+)?upright|"
    r"sit(?:s|ting)?\s+down|lower(?:s|ing)?\s+into\s+(?:a|the)\s+seat|"
    r"walk(?:s|ing)?|step(?:s|ping)?\s+(?:forward|backward|sideways|laterally|left|right|toward|away|across|into|out)|"
    r"move(?:s|d|ing)?\s+(?:forward|backward|sideways|laterally|left|right|toward|away|across|into|out|around|to)|"
    r"approach(?:es|ing)?|back(?:s|ing)?\s+away|cross(?:es|ing)?\s+the\s+room|"
    r"jump(?:s|ing)?|crouch(?:es|ing)?|kneel(?:s|ing)?|dance(?:s|d|ing)?|"
    r"change(?:s|d|ing)?\s+position)\b",
    flags=re.I,
)


def _action_requires_camera_follow(action: str) -> bool:
    return bool(_CAMERA_FOLLOW_ACTION_RE.search(str(action or "")))


def _camera_motion_clause(action: str) -> str:
    action_text = str(action or "")
    if re.search(
        r"\b(?:stand(?:s|ing)?[- ]+up|get(?:s|ting)?\s+up|ris(?:e|es|ing)\s+from|"
        r"rais(?:e|es|ed|ing)\s+(?:the\s+)?(?:hips|body)|reach(?:es|ed|ing)?\s+(?:an?\s+)?upright)\b",
        action_text,
        flags=re.I,
    ):
        return (
            "This is one continuous physically filmed take. The camera begins a gentle upward tilt and crane just before Speaker_1 starts rising, then follows the body upward at the same slow speed while dollying back only enough to preserve a natural medium-full scale. "
            "The center of Speaker_1's upper torso stays inside a small central zone around the exact frame center in every intermediate frame; the complete head, comfortable headroom, shoulders, chest, and waist remain visible from the seated pose through the final standing pose. "
            "The camera eases to a stop only after Speaker_1 settles, and coherent room parallax makes the gradual camera movement physically clear."
        )
    if re.search(r"\b(?:sit(?:s|ting)?\s+down|lower(?:s|ing)?\s+into\s+(?:a|the)\s+seat)\b", action_text, flags=re.I):
        return (
            "This is one continuous physically filmed take. The camera begins a gentle downward tilt and crane just before Speaker_1 starts lowering, then follows the body downward at the same slow speed while dollying forward only enough to preserve a natural medium-full scale. "
            "The center of Speaker_1's upper torso stays inside a small central zone around the exact frame center in every intermediate frame; the complete head, comfortable headroom, shoulders, chest, and waist remain visible from the standing pose through the final seated pose. "
            "The camera eases to a stop only after Speaker_1 settles, and coherent room parallax makes the gradual camera movement physically clear."
        )
    if _action_requires_camera_follow(action_text):
        return (
            "This is one continuous physically filmed take. The camera starts moving just before Speaker_1 leaves the settled pose and continuously matches the complete three-dimensional movement path with a slow damped pan, tilt, truck, or dolly in the corresponding direction. "
            "The center of Speaker_1's upper torso stays inside a small central zone around the exact frame center in every intermediate frame; the complete head, comfortable headroom, shoulders, chest, and waist remain visible throughout. "
            "The camera matches the speaker's speed, eases to a stop only after the body settles, and coherent background parallax makes the gradual tracking motion physically clear."
        )
    return (
        "This is one continuous calmly filmed take. The center of Speaker_1's upper torso stays inside a small central zone around the exact frame center. "
        "Speaker_1 fills about two thirds "
        "of the frame height from head to mid-torso, with compact headroom, shoulders spanning roughly half the frame width, and hands entering "
        "the lower quarter for compact gestures."
    )


def _apply_motion_camera_prompt(prompt: str, action: str) -> str:
    text = _remove_locked_camera_language(prompt)
    follows = _action_requires_camera_follow(action)
    camera_clause = _camera_motion_clause(action)
    if follows:
        text = re.sub(
            r"The lens holds a steady realistic medium close-up[^.]*\.",
            "The moving camera maintains a natural medium-full composition with Speaker_1's upper torso centered through every intermediate frame.",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bmedium close-up framing\b", "adaptive medium-full framing", text, flags=re.I)
        text = re.sub(r"\bmedium close-up(?: shot)?\b", "medium-full shot", text, flags=re.I)
        text = re.sub(
            r"((?:Identity|The identity),[^.]*?color palette),?\s+and lens distance (?:stays?|remains?) stable\b",
            r"\1 remain coherent while camera position adjusts smoothly",
            text,
            flags=re.I,
        )
        text = re.sub(
            r",?\s+and lens distance (?:stays?|remains?) stable\b",
            ", while camera position adjusts smoothly",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:the\s+)?lens distance (?:stays?|remains?) stable\b",
            "camera position adjusts smoothly with the tracking motion",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bbackground placement\b", "background geometry", text, flags=re.I)
    if camera_clause not in text:
        marker = "\nSpeaker_1's Appearance:"
        text = text.replace(marker, f" {camera_clause}{marker}", 1)
    return text


def _speaker_action_section(action: str, *, explicit_action: bool) -> str:
    action = _safe_action(action, 0)
    if explicit_action:
        pose_clause = (
            "The movement reaches a clear readable pose on the correct anatomical side"
            if re.search(r"\banatomical\s+(?:left|right)\b|\bboth\s+(?:open\s+)?hands\b", action, flags=re.I)
            else "The movement reaches a clear readable complete pose"
        )
        motion = (
            f"This is the primary visible action of the shot. {pose_clause}, "
            "stays visible for a beat, and receives the full visual emphasis while the supporting body motion remains natural. "
            "Any large body movement uses at least four seconds of this five-second shot: a settled preparation, a gradual start, slow continuous travel, and gentle deceleration into a stable pose. "
            "Adjacent frames advance the body only a small amount, and every intermediate pose has crisp high-shutter temporal detail, one coherent body silhouette, clean limb contours, and stable clothing texture."
        )
    else:
        motion = (
            "A subtle facial reaction leads the movement, and the compact gesture follows the natural rhythm of the reply with smooth continuous motion, "
            "gentle acceleration and deceleration, and a comfortable settled finish."
        )
    return (
        f"Speaker_1's Actions: Speaker_1 {action}. {motion} "
        "The shoulders stay relaxed and the face remains oriented toward the lens."
    )


def _force_speaker_action(prompt: str, action: str, *, explicit_action: bool) -> str:
    text = str(prompt or "")
    replacement = _speaker_action_section(action, explicit_action=explicit_action)
    pattern = r"Speaker_1's Actions:\s*.+?\nSpeaker_1's Facial Expression:"
    if re.search(pattern, text, flags=re.S):
        text = re.sub(
            pattern,
            f"{replacement}\nSpeaker_1's Facial Expression:",
            text,
            count=1,
            flags=re.S,
        )
    if explicit_action:
        text = re.sub(
            r"Speech is synchronized with actions and lip movements\.?",
            "Speech is synchronized with lip movements, while the slower primary body action continues independently across the full shot.",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"Facial reaction and gesture follow the spoken cadence[^.]*\.",
            "Facial reaction accompanies the spoken cadence, while the slower primary body action continues independently across the full shot.",
            text,
            flags=re.I,
        )
    return text


def _action_alignment_clause(explicit_action: bool) -> str:
    if explicit_action:
        return (
            "The facial recognition beat begins just before the first spoken word. The shorter spoken line and the body motion proceed independently: the primary action continues through four slow readable phases across the entire five-second shot, even after speech ends, and reaches its final pose only in the last frames."
        )
    return (
        "A subtle facial response begins just before speech, and the compact gesture follows the cadence of the spoken reply."
    )


def _voice_description_for_speech(speech: str) -> str:
    language = _dominant_response_language(speech)
    spoken_language = "English" if language == "en" else "Mandarin"
    return (
        f"natural {spoken_language} voice, clear articulation, measured conversational pacing, "
        "close and stable recording quality"
    )


def _sound_alignment_for_speech(speech: str) -> str:
    spoken_language = (
        "English"
        if _dominant_response_language(speech) == "en"
        else "Mandarin"
    )
    return (
        f"The natural {spoken_language} voice is carried by the audio track and synchronized lip movement. "
        "Facial reaction and gesture follow the spoken cadence while the clean photographic image remains visually uninterrupted. "
        "Background audio remains quiet and stable"
    )


def _force_prompt_speech_language(prompt: str, speech: str) -> str:
    text = str(prompt or "")
    voice = _voice_description_for_speech(speech)
    text = re.sub(
        r"Speaker_1's Voice Description:\s*[^\n]*",
        f"Speaker_1's Voice Description: {voice}.",
        text,
        count=1,
        flags=re.I,
    )
    spoken_language = (
        "English"
        if _dominant_response_language(speech) == "en"
        else "Mandarin"
    )
    marker = "Sound-Visual Alignment:"
    if marker in text:
        before, after = text.split(marker, 1)
        after = re.sub(
            r"\b(?:English|Mandarin)\b(?=\s+voice\b)",
            spoken_language,
            after,
            flags=re.I,
        )
        text = f"{before}{marker}{after}"
    return text


def _build_ltx_prompt(
    scene: str,
    speech: str,
    segment_id: int,
    total_segments: int,
    has_first_frame: bool,
    action: Optional[str] = None,
    emotion: str = "calm and helpful",
    is_transition: bool = False,
    template_id: str = "",
    aspect_ratio: str = "landscape",
    continues_from_prior_context: bool = False,
    prior_context_speech: str = "",
    explicit_action: bool = False,
) -> str:
    appearance = _appearance_for_scene(
        scene,
        template_id=template_id,
        has_first_frame=has_first_frame,
    )
    action_pool = [
        "briefly gathers the thought with an attentive eye movement, gives a small responsive nod, and lets one relaxed open hand gesture follow the reply",
        "briefly looks thoughtful, softens the expression, and makes one measured open-palmed gesture near chest level",
        "brightens slightly in recognition, blinks naturally, and gives a small warm nod toward the camera",
        "keeps steady eye contact, reacts with a subtle eyebrow movement, and lets the hands settle naturally after one compact gesture",
    ]
    action = _safe_action(action or action_pool[segment_id % len(action_pool)], segment_id)
    emotion = _safe_emotion(emotion)
    if continues_from_prior_context:
        phase = "continues the established live conversation"
    elif total_segments <= 1:
        phase = "begins and holds a complete short response"
    elif segment_id == 0:
        phase = "begins"
    elif segment_id == total_segments - 1:
        phase = "holds a quiet attentive beat" if is_transition else "continues and finishes"
    else:
        phase = "continues"
    transition_note = ""
    if is_transition:
        transition_note = (
            " The speaker holds a calm attentive pause, keeps the same posture and eye contact, "
            "and lets the voice remain close and natural."
        )
    scene_text = _english_scene_description(scene, template_id=template_id, has_first_frame=has_first_frame)
    anchor_sentence = _scene_anchor_sentence_for_prompt(scene_text, template_id=template_id)
    is_portrait = str(aspect_ratio or "").strip().lower() == "portrait"
    summary_subject = "portrait camera take" if is_portrait else "digital-human conversation"
    composition_sentence = (
        PORTRAIT_CLOSE_COMPOSITION
        if is_portrait
        else LANDSCAPE_CLOSE_COMPOSITION
    )
    return (
        f"Summary: A realistic {summary_subject} {phase}, preserving stable identity, lens geometry, light direction, natural portrait colors, and readable background objects.\n"
        f"Narration {segment_id + 1}:\n"
        f"{scene_text}. "
        f"{composition_sentence} {anchor_sentence} "
        f"Identity, clothing, hairstyle, body proportions, lighting, background objects, and color palette stay visually consistent. "
        f"The image keeps soft cheek highlights, delicate eye catchlights, visible outfit texture, shallow depth of field, stable natural colors, and clear light direction.{transition_note}\n"
        f"Speaker_1's Appearance: {appearance}.\n\n"
        f"{_speaker_action_section(action, explicit_action=explicit_action)}\n"
        f"Speaker_1's Facial Expression: {emotion}.\n"
        f"Speaker_1's Held Objects:\nNone\n"
        f"Speech Attribution:\n"
        f"Speaker_1 says: \"{speech}\"\n"
        f"Speaker_1's Emotion: {emotion}.\n"
        f"Speaker_1's Voice Description: {_voice_description_for_speech(speech)}.\n"
        f"Sound-Visual Alignment: {_sound_alignment_for_speech(speech)}. "
        f"{_action_alignment_clause(explicit_action)} Background audio remains quiet and stable."
    )


def _compact_ltx_prompt_for_latency(prompt: str) -> str:
    """Remove repeated visual-anchor prose while keeping the LTX prompt contract."""
    if not SETTINGS.compact_ltx_prompt_for_latency:
        return prompt
    text = str(prompt or "")
    if not text:
        return text

    split_marker = "\nSpeech Attribution:\n"
    if split_marker in text:
        before, after = text.split(split_marker, 1)
        suffix = split_marker + after
    else:
        before, suffix = text, ""

    removals = [
        r"\s*The opening frame establishes the exact physical anchors:[^.]*\.\s*",
        r"\s*The speaker is already settled into a natural live conversation posture\.\s*",
        r"\s*Identity, clothing, hairstyle, body proportions, (?:light direction|lighting), background (?:placement|objects), and color palette (?:remain|stay) (?:visually )?consistent\.\s*",
        r"\s*The image (?:keeps|reads as)[^.]*?(?:inspectable light direction|stable natural colors|vivid but natural colors)\.\s*",
        r"\s*The gesture reaches a small natural emphasis and settles back into the same stable centered posture\.\s*",
    ]
    for pattern in removals:
        before = re.sub(pattern, " ", before, flags=re.I | re.S)

    before = re.sub(r"[ \t]{2,}", " ", before)
    before = re.sub(r"\n{3,}", "\n\n", before)
    return before.strip() + suffix


def _force_speaker_says(prompt: str, speech: str) -> str:
    prompt = str(prompt or "").strip()
    raw_speech = str(speech or "").strip()
    speech = _sanitize_speech_text(
        raw_speech,
        max_chars=180,
        ensure_terminal=bool(re.search(r"[。！？!?]$", raw_speech)),
    )
    replacement = f'Speaker_1 says: "{speech}"'
    pattern = r"Speaker_1 says:\s*\"[^\"]*\""
    if re.search(pattern, prompt):
        return re.sub(pattern, replacement, prompt, count=1)
    if "Speech Attribution:" in prompt:
        return prompt.replace("Speech Attribution:", f"Speech Attribution:\n{replacement}", 1)
    return f"{prompt.rstrip()}\nSpeech Attribution:\n{replacement}"


def _prompt_has_cjk_outside_speech(prompt: str) -> bool:
    without_speech = re.sub(
        r"Speaker_1 says:\s*\"[^\"]*\"",
        'Speaker_1 says: ""',
        str(prompt or ""),
    )
    return _contains_cjk(without_speech)


def _prompt_outside_speech(prompt: str) -> str:
    return re.sub(
        r"Speaker_1 says:\s*\"[^\"]*\"",
        'Speaker_1 says: ""',
        str(prompt or ""),
    )


def _valid_llm_ltx_prompt(prompt: str, speech: str) -> bool:
    prompt = str(prompt or "")
    if len(prompt) < 1200:
        return False
    required = [
        "Summary:",
        "Narration",
        "medium close-up",
        "Speaker_1's Appearance",
        "Speaker_1's Actions",
        "Speaker_1's Facial Expression",
        "Speaker_1's Held Objects",
        "Speech Attribution",
        "Speaker_1 says:",
        "Speaker_1's Emotion",
        "Speaker_1's Voice Description",
        "Sound-Visual Alignment",
    ]
    if any(item not in prompt for item in required):
        return False
    visual_terms = [
        "lighting",
        "light",
        "colors",
        "natural colors",
        "background",
        "background objects",
        "readable background",
        "depth of field",
        "catchlights",
        "rim light",
        "reflections",
        "studio",
        "room",
        "face",
        "portrait",
        "palette",
        "highlights",
        "cheek highlights",
        "window light",
        "daylight",
        "lamp",
        "texture",
        "objects",
    ]
    if sum(1 for item in visual_terms if item in prompt) < 5:
        return False
    narration_match = re.search(
        r"Narration(?:\s+[0-9]+)?\s*:\s*\n?(.+?)\nSpeaker_1's Appearance",
        prompt,
        flags=re.S,
    )
    if not narration_match or len(narration_match.group(1).strip()) < 520:
        return False
    narration_lower = narration_match.group(1).lower()
    density_score = sum(1 for item in PROMPT_PLANNER_VISUAL_DENSITY_TERMS if item in narration_lower)
    if density_score < 3:
        return False
    light_source_terms = (
        "dominant key light",
        "soft key",
        "softbox",
        "diffused window",
        "diffused daylight",
        "window light",
        "daylight",
        "desk lamp",
    )
    physical_anchor_terms = (
        "supporting surface",
        "visibly supports",
        "rests on",
        "rests flat",
        "mounted flush",
        "rooted in",
        "texture",
        "straight background verticals",
        "rectilinear",
    )
    if not any(item in narration_lower for item in light_source_terms):
        return False
    if not any(item in narration_lower for item in physical_anchor_terms):
        return False
    if prompt.count("Speaker_1 says:") != 1:
        return False
    outside_speech = _prompt_outside_speech(prompt)
    outside_lower = outside_speech.lower()
    if re.search(r"(^|\n)\s*segment\s+[0-9]+", outside_speech, flags=re.I):
        return False
    forbidden = [
        "previously generated",
        "previously said",
        "previously spoken",
        "previous context",
        "previous-context",
        "has already said",
        "already said",
        "full context",
        "continue from this",
        "prior segment",
        "earlier segment",
        "the segment feels",
        "this segment",
        "per-segment",
        "continuity note",
        "generated context",
        "prompt_speech",
        "display_speech",
        "segment plan",
        "prompt compiler",
        "prompt behaves",
        "video prompt",
        "prompt writing",
        "small llm",
        "task as prompt compilation",
        "style guide",
        "few-shot",
        "few shot",
        "good case",
        "good cases",
        "input-output",
        "teaching pattern",
        "benchmark example",
        "benchmark prompt",
        "hand-crafted benchmark",
        "simplified interactive",
        "short template",
        "chat template",
        "handwritten prompt",
        "model internals",
        "查看 LTX",
        "do not ",
        "don't ",
        "cannot ",
        "must not ",
        "no walking",
        "no large",
        "no body rotation",
        "no camera movement",
        "no scene change",
        "without moving",
    ]
    if any(item in outside_lower for item in forbidden):
        return False
    actions_match = re.search(
        r"Speaker_1's Actions:\s*(.+?)\nSpeaker_1's Facial Expression",
        prompt,
        flags=re.S,
    )
    if not actions_match:
        return False
    actions_lower = actions_match.group(1).lower()
    positive_action_terms = [
        "faces",
        "eye contact",
        "toward the lens",
        "toward the camera",
        "tiny nod",
        "small nod",
        "slow",
        "smooth",
        "hand",
        "gesture",
        "centered",
    ]
    if sum(1 for item in positive_action_terms if item in actions_lower) < 2:
        return False
    if _prompt_has_cjk_outside_speech(prompt):
        return False
    raw_speech = str(speech or "").strip()
    speech_target = _sanitize_speech_text(
        raw_speech,
        max_chars=180,
        ensure_terminal=bool(re.search(r"[。！？!?]$", raw_speech)),
    )
    if speech_target not in prompt:
        return False
    return True


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = str(text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _select_prompt_skeleton_examples(template_id: str, max_examples: int) -> List[Dict[str, str]]:
    max_examples = max(1, int(max_examples or 2))
    key = str(template_id or "").strip()
    selected = [
        item for item in REALISTIC_PROMPT_PLANNER_EXAMPLES
        if item.get("template_id") == key
    ]
    for item in REALISTIC_PROMPT_PLANNER_EXAMPLES:
        if item in selected:
            continue
        selected.append(item)
        if len(selected) >= max_examples:
            break
    return selected[:max_examples]


def _compact_skeleton_example_prompt(prompt: str, max_chars: int = 1300) -> str:
    text = _prompt_outside_speech(prompt)
    text = re.sub(r"\nSpeech Attribution:.*", "", text, flags=re.S)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars].rstrip()
    boundary = max(cut.rfind("\n"), cut.rfind(". "))
    if boundary > max_chars * 0.65:
        cut = cut[: boundary + 1]
    return cut.rstrip()


def _ltx_prompt_skeleton_cache_key(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "landscape",
    has_first_frame: bool = False,
) -> Tuple[Any, ...]:
    return (
        PROMPT_PLANNER_PROFILE,
        "ltx_prompt_skeleton_v8",
        _normalize_scene_for_signature(scene),
        str(template_id or "").strip(),
        str(aspect_ratio or "").strip().lower(),
        bool(has_first_frame),
        int(SETTINGS.ltx_prompt_skeleton_max_examples),
    )


def _ltx_prompt_skeleton_cache_get(key: Tuple[Any, ...]) -> Optional[Dict[str, str]]:
    cached = LTX_PROMPT_SKELETON_CACHE.get(key)
    if cached is not None:
        LTX_PROMPT_SKELETON_CACHE.move_to_end(key)
        return dict(cached)
    cached = _persistent_prompt_cache_get("ltx_skeleton", key)
    if cached is not None:
        LTX_PROMPT_SKELETON_CACHE[key] = dict(cached)
        LTX_PROMPT_SKELETON_CACHE.move_to_end(key)
        while len(LTX_PROMPT_SKELETON_CACHE) > LTX_PROMPT_SKELETON_CACHE_SIZE:
            LTX_PROMPT_SKELETON_CACHE.popitem(last=False)
        return dict(cached)
    return None


def _ltx_prompt_skeleton_cache_put(key: Tuple[Any, ...], value: Dict[str, str]) -> None:
    if LTX_PROMPT_SKELETON_CACHE_SIZE <= 0:
        return
    LTX_PROMPT_SKELETON_CACHE[key] = dict(value)
    LTX_PROMPT_SKELETON_CACHE.move_to_end(key)
    while len(LTX_PROMPT_SKELETON_CACHE) > LTX_PROMPT_SKELETON_CACHE_SIZE:
        LTX_PROMPT_SKELETON_CACHE.popitem(last=False)
    _persistent_prompt_cache_put("ltx_skeleton", key, value)


def _fallback_ltx_prompt_skeleton(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "landscape",
    has_first_frame: bool = False,
) -> Dict[str, str]:
    scene_text = _english_scene_description(
        scene,
        template_id=template_id,
        has_first_frame=has_first_frame,
    )
    scene_clause = (
        scene_text.rstrip(". ")
        if scene_text.lower().startswith("eye-level")
        else f"eye-level tight medium close-up shot. {scene_text.rstrip('. ')}"
    )
    anchor_sentence = _scene_anchor_sentence_for_prompt(scene_text, template_id=template_id)
    is_portrait = str(aspect_ratio or "").strip().lower() == "portrait"
    composition_sentence = (
        PORTRAIT_CLOSE_COMPOSITION
        if is_portrait
        else LANDSCAPE_CLOSE_COMPOSITION
    )
    appearance = _appearance_for_scene(
        scene,
        template_id=template_id,
        has_first_frame=has_first_frame,
    )
    narration = (
        f"{scene_clause}. {composition_sentence} {anchor_sentence} "
        "An eye-level 50mm-equivalent rectilinear lens preserves natural facial proportions and straight architectural verticals. "
        "The scene's broad soft key and gentle neutral fill keep one consistent shadow direction and controlled highlights. "
        "Every named object remains on a visible support, the chair supports the seated body, and the hands remain anatomically connected "
        "while moving freely above the furniture edge. Crisp eyes, natural skin microtexture, clean garment weave, well-resolved edges, "
        "moderate depth of field, and restrained natural color create a clear high-end photographic image. "
        "Identity, outfit, object placement, light direction, color, and lens distance stay stable. Every visible surface remains a "
        "photographed physical material with uninterrupted texture. The person faces the camera directly, keeps the face oriented toward "
        "the lens, and maintains steady eye contact while speaking."
    )
    summary_subject = "portrait camera take" if is_portrait else "digital-human conversation"
    return {
        "summary_begin": (
            f"A polished {summary_subject} begins in a restrained physically coherent setting, "
            "preserving identity, lens geometry, light direction, natural colors, and exact object placement"
        ),
        "summary_continue": (
            f"The established {summary_subject} continues in the same physically coherent setting, "
            "preserving identity, lens geometry, light direction, natural colors, and exact object placement"
        ),
        "narration": narration,
        "appearance": appearance,
        "base_action": (
            "faces the camera directly, keeps the face oriented toward the lens, maintains steady eye contact, "
            "and lets one hand make a small slow conversational movement near chest level"
        ),
        "emotion": "calm and helpful",
        "voice_description": (
            "natural spoken voice, clear articulation, measured conversational pacing, close and stable recording quality"
        ),
        "sound_alignment": (
            "The spoken voice is carried by the audio track and synchronized lip movement. "
            "Facial reaction and gesture follow the spoken cadence while the clean photographic image remains visually uninterrupted. "
            "Background audio remains quiet and stable."
        ),
        "source": "deterministic_scene_skeleton",
    }


def _clean_ltx_skeleton_field(value: Any, fallback: str, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip()).strip("\"'“”")
    text = re.sub(
        r"\blower[- ]third\b",
        "bottom part of the photographed scene",
        text,
        flags=re.I,
    )
    lower = text.lower()
    banned = [
        "speaker_1 says",
        "prompt_speech",
        "display_speech",
        "previously",
        "previous context",
        "has already said",
        "few-shot",
        "few shot",
        "benchmark",
        "prompt compiler",
        "video prompt",
        "do not ",
        "don't ",
        "cannot ",
        "must not ",
        "no walking",
        "no large",
    ]
    if (
        not text
        or _contains_cjk(text)
        or _looks_like_collapsed_english(text)
        or any(item in lower for item in banned)
    ):
        return fallback
    return text[:max_chars].rstrip(". ")


def _normalize_ltx_prompt_skeleton(
    raw: Optional[Dict[str, Any]],
    fallback: Dict[str, str],
) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return dict(fallback)
    skeleton = {
        "summary_begin": _clean_ltx_skeleton_field(
            raw.get("summary_begin"),
            fallback["summary_begin"],
            max_chars=420,
        ),
        "summary_continue": _clean_ltx_skeleton_field(
            raw.get("summary_continue"),
            fallback["summary_continue"],
            max_chars=420,
        ),
        "narration": _clean_ltx_skeleton_field(
            raw.get("narration"),
            fallback["narration"],
            max_chars=1700,
        ),
        "appearance": _clean_ltx_skeleton_field(
            raw.get("appearance"),
            fallback["appearance"],
            max_chars=520,
        ),
        "base_action": _clean_ltx_skeleton_field(
            raw.get("base_action"),
            fallback["base_action"],
            max_chars=260,
        ),
        "emotion": _clean_ltx_skeleton_field(
            raw.get("emotion"),
            fallback["emotion"],
            max_chars=80,
        ),
        "voice_description": _clean_ltx_skeleton_field(
            raw.get("voice_description"),
            fallback["voice_description"],
            max_chars=220,
        ),
        "sound_alignment": _clean_ltx_skeleton_field(
            raw.get("sound_alignment"),
            fallback["sound_alignment"],
            max_chars=320,
        ),
        "source": "llm_scene_skeleton",
    }
    narration_lower = skeleton["narration"].lower()
    density_score = sum(1 for item in PROMPT_PLANNER_VISUAL_DENSITY_TERMS if item in narration_lower)
    if len(skeleton["narration"]) < 520 or density_score < 3:
        skeleton["narration"] = fallback["narration"]
    for key in ("summary_begin", "summary_continue", "narration"):
        skeleton[key] = _remove_locked_camera_language(skeleton[key]).strip()
    if "medium close-up" not in skeleton["narration"].lower():
        skeleton["narration"] = f"eye-level tight medium close-up shot. {skeleton['narration']}"
    return skeleton


def _render_ltx_prompt_from_skeleton(
    skeleton: Dict[str, str],
    *,
    speech: str,
    segment_id: int,
    total_segments: int,
    has_first_frame: bool,
    action: Optional[str] = None,
    emotion: str = "calm and helpful",
    continues_from_prior_context: bool = False,
    is_transition: bool = False,
    explicit_action: bool = False,
) -> str:
    summary_key = "summary_continue" if (continues_from_prior_context or segment_id > 0) else "summary_begin"
    summary = _remove_locked_camera_language(
        str(skeleton.get(summary_key) or skeleton.get("summary_begin") or "")
    ).rstrip(". ")
    narration = _remove_locked_camera_language(str(skeleton.get("narration") or "")).rstrip(". ")
    appearance = str(skeleton.get("appearance") or "").rstrip(". ")
    action_text = _safe_action(action or skeleton.get("base_action") or "", segment_id)
    emotion_text = _safe_emotion(emotion or skeleton.get("emotion") or "calm and helpful")
    voice = _voice_description_for_speech(speech)
    alignment = _sound_alignment_for_speech(speech)
    if is_transition:
        action_text = (
            "faces the camera directly, keeps a calm centered posture, maintains steady eye contact, "
            "and lets the hands settle into a quiet attentive pause"
        )
        explicit_action = False
    if total_segments > 1 and segment_id == total_segments - 1 and not is_transition:
        summary = summary.replace("conversation continues", "conversation continues and closes naturally")
    prompt = (
        f"Summary: {summary}.\n"
        f"Narration {segment_id + 1}:\n"
        f"{narration}.\n"
        f"Speaker_1's Appearance: {appearance}.\n\n"
        f"{_speaker_action_section(action_text, explicit_action=explicit_action)}\n"
        f"Speaker_1's Facial Expression: {emotion_text}.\n"
        f"Speaker_1's Held Objects:\nNone\n"
        f"Speech Attribution:\n"
        f"Speaker_1 says: \"{speech}\"\n"
        f"Speaker_1's Emotion: {emotion_text}.\n"
        f"Speaker_1's Voice Description: {voice}.\n"
        f"Sound-Visual Alignment: {alignment}. {_action_alignment_clause(explicit_action)}"
    )
    return _force_prompt_speech_language(_force_speaker_says(prompt, speech), speech)


async def _compile_ltx_prompt_skeleton(
    scene: str,
    *,
    template_id: str = "",
    aspect_ratio: str = "landscape",
    has_first_frame: bool = False,
) -> Optional[Dict[str, str]]:
    if not SETTINGS.use_ltx_prompt_skeleton_cache:
        return None
    fallback = _fallback_ltx_prompt_skeleton(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
    )
    if str(template_id or "").strip() in ENGLISH_TEMPLATE_SCENES:
        fallback["source"] = "curated_template_skeleton"
        return fallback
    if not SETTINGS.use_llm_prompt_planner:
        return fallback

    key = _ltx_prompt_skeleton_cache_key(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
    )
    cached = _ltx_prompt_skeleton_cache_get(key)
    if cached:
        cached["appearance"] = fallback["appearance"]
        cached_source = str(cached.get("source") or "")
        cached["source"] = (
            "llm_scene_skeleton_cache"
            if cached_source == "llm_scene_skeleton"
            else (cached_source or "scene_skeleton_cache")
        )
        return cached

    scene_text = _english_scene_description(scene, template_id=template_id, has_first_frame=has_first_frame)
    appearance = fallback["appearance"]
    selected_examples = _select_prompt_skeleton_examples(
        template_id,
        SETTINGS.ltx_prompt_skeleton_max_examples,
    )
    aspect_note = (
        PORTRAIT_CLOSE_COMPOSITION
        if str(aspect_ratio or "").strip().lower() == "portrait"
        else LANDSCAPE_CLOSE_COMPOSITION
    )
    examples = "\n\n".join(
        f"GOOD SKELETON CASE {idx + 1} - {item['name']}:\n"
        f"{_compact_skeleton_example_prompt(item['prompt'])}"
        for idx, item in enumerate(selected_examples)
    )
    system = (
        "You are a prompt engineer for a realistic LTX digital-human video model. "
        "Create one reusable English scene skeleton for a continuing live conversation. "
        "Return JSON only with these string keys: summary_begin, summary_continue, narration, appearance, "
        "base_action, emotion, voice_description, sound_alignment. "
        "Do not include Speaker_1 says, dialogue text, previous-context notes, segment labels, model notes, or negative prohibition lists. "
        "All text must be English. The narration should describe a restrained, physically buildable set with at most three "
        "background anchors, exact object supports, stable identity, and one coherent lighting setup. Use an eye-level "
        "tight medium close-up in which the speaker fills about two thirds of frame height from head to mid-torso, with compact "
        "headroom, slim side margins, hands available in the lower quarter, and only a narrow furniture edge at the bottom. Use a "
        "50mm-equivalent rectilinear lens, natural facial proportions, straight architectural verticals, moderate depth of field, "
        "one broad soft key, gentle neutral fill, controlled highlights, and shadows that remain attached to the objects casting them. "
        "Prioritize crisp eyes, natural skin microtexture, clean garment weave, well-resolved edges, restrained natural color, and quiet "
        "high-end photographic detail. Treat the result as raw unedited camera footage, not a broadcast or social-media program. "
        "Carry garment folds, anatomically connected hands, furniture seams, natural shadows, and perspective continuously through the "
        "bottom edge. Keep every prop on a visible supporting surface and preserve its exact placement. "
        "The appearance field must preserve the supplied person, gender, hairstyle, hair color, and clothing exactly."
    )
    user = (
        f"Scene to turn into a reusable skeleton:\n{scene_text}\n\n"
        f"Appearance to preserve:\n{appearance}\n\n"
        f"Frame composition to preserve:\n{aspect_note}\n\n"
        "Use only a few high-quality examples below as writing guidance. Keep the final JSON concise and physically precise; "
        "the narration should be roughly 650 to 1100 English characters.\n\n"
        f"{examples}\n\n"
        "Now write the reusable skeleton JSON. It will be filled locally with different Speaker_1 says lines later."
    )
    try:
        raw = await asyncio.to_thread(
            _call_openai_compatible_llm,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            min(SETTINGS.ltx_prompt_skeleton_max_tokens, SETTINGS.prompt_planner_max_tokens),
        )
        parsed = _extract_json_object(raw or "")
        skeleton = _normalize_ltx_prompt_skeleton(parsed, fallback)
    except Exception:
        skeleton = dict(fallback)
    skeleton["appearance"] = appearance

    probe = _render_ltx_prompt_from_skeleton(
        skeleton,
        speech="你好。",
        segment_id=0,
        total_segments=1,
        has_first_frame=has_first_frame,
        action=skeleton.get("base_action"),
        emotion=skeleton.get("emotion", "calm and helpful"),
    )
    if not _valid_llm_ltx_prompt(probe, "你好。"):
        skeleton = dict(fallback)
    _ltx_prompt_skeleton_cache_put(key, skeleton)
    return dict(skeleton)


async def _plan_ltx_prompts_with_scene_skeleton(
    *,
    scene: str,
    template_id: str,
    aspect_ratio: str,
    has_first_frame: bool,
    segment_plans: List[Dict[str, Any]],
    prompt_speeches: List[str],
    prior_context_speeches: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if not SETTINGS.use_ltx_prompt_skeleton_cache:
        return None
    if not segment_plans or len(segment_plans) != len(prompt_speeches):
        return None
    skeleton = await _compile_ltx_prompt_skeleton(
        scene,
        template_id=template_id,
        aspect_ratio=aspect_ratio,
        has_first_frame=has_first_frame,
    )
    if not skeleton:
        return None

    planned: List[Dict[str, Any]] = []
    planner_source = str(skeleton.get("source") or "scene_skeleton")
    for idx, plan in enumerate(segment_plans):
        emotion = _safe_emotion(str(plan.get("emotion") or skeleton.get("emotion") or "calm and helpful"))
        action = _safe_action(str(plan.get("action") or skeleton.get("base_action") or ""), idx)
        explicit_action = bool(plan.get("explicit_action", False))
        continues = bool(prior_context_speeches[idx] or idx > 0)
        prompt = _render_ltx_prompt_from_skeleton(
            skeleton,
            speech=prompt_speeches[idx],
            segment_id=idx,
            total_segments=len(segment_plans),
            has_first_frame=has_first_frame,
            action=action,
            emotion=emotion,
            continues_from_prior_context=continues,
            is_transition=bool(plan.get("is_transition", False)),
            explicit_action=explicit_action,
        )
        source = planner_source
        if not _valid_llm_ltx_prompt(prompt, prompt_speeches[idx]):
            prompt = _build_ltx_prompt(
                scene,
                prompt_speeches[idx],
                idx,
                len(segment_plans),
                has_first_frame,
                action=action,
                emotion=emotion,
                is_transition=bool(plan.get("is_transition", False)),
                template_id=template_id,
                aspect_ratio=aspect_ratio,
                continues_from_prior_context=continues,
                prior_context_speech=prior_context_speeches[idx],
                explicit_action=explicit_action,
            )
            source = "deterministic_fallback"
        planned.append(
            {
                "prompt": prompt,
                "emotion": emotion,
                "action": action,
                "prompt_planner": source,
            }
        )
    return planned


async def _plan_ltx_prompts_with_llm(
    *,
    scene: str,
    template_id: str,
    aspect_ratio: str,
    has_first_frame: bool,
    segment_plans: List[Dict[str, Any]],
    prompt_speeches: List[str],
) -> Optional[List[Dict[str, Any]]]:
    if not SETTINGS.use_llm_prompt_planner:
        return None
    if not segment_plans or len(segment_plans) != len(prompt_speeches):
        return None

    scene_text = _english_scene_description(scene, template_id=template_id, has_first_frame=has_first_frame)
    appearance = _appearance_for_scene(
        scene,
        template_id=template_id,
        has_first_frame=has_first_frame,
    )
    selected_examples = _select_prompt_planner_examples(
        template_id,
        SETTINGS.prompt_planner_max_examples,
    )
    examples = "\n\n".join(
        f"GOOD CASE {idx + 1} - {item['name']}:\n{item['prompt']}"
        for idx, item in enumerate(selected_examples)
    )
    target_segments = [
        {
            "segment_id": int(plan["segment_id"]),
            "display_speech": str(plan.get("speech") or ""),
            "prompt_speech": prompt_speeches[idx],
            "emotion": str(plan.get("emotion") or "calm and helpful"),
            "action_hint": str(plan.get("action") or ""),
            "explicit_action": bool(plan.get("explicit_action", False)),
        }
        for idx, plan in enumerate(segment_plans)
    ]
    system = (
        "You are a prompt compiler for a realistic LTX digital-human video model. "
        "Transform dialogue segments into complete benchmark-style video prompts, not chat summaries. "
        "The output must be valid JSON only: an array of objects with segment_id, prompt, emotion, and action. "
        "Every prompt must use the sections Summary, Narration, Speaker_1's Appearance, "
        "Speaker_1's Actions, Speaker_1's Facial Expression, Speaker_1's Held Objects, "
        "Speech Attribution, Speaker_1's Emotion, Speaker_1's Voice Description, and Sound-Visual Alignment. "
        "All non-speech prompt text must be English; the exact local spoken line may appear only inside Speaker_1 says. "
        "Speaker_1 says must exactly equal prompt_speech. For follow-up turns, prompt_speech already contains "
        "the previous same-scene speech tail when needed, so do not add any separate previous-context note. "
        "Keep the same person, camera geometry, clothing, single coherent lighting setup, exact object placement, and natural color palette across segments. "
        "Use a restrained physically buildable set and a tight eye-level medium close-up in which the speaker fills about two thirds "
        "of frame height from head to mid-torso, with compact headroom, hands in the lower quarter, and a narrow furniture edge at the bottom. "
        "Use a 50mm-equivalent rectilinear lens, straight verticals, moderate depth of field, "
        "one broad soft key, gentle neutral fill, controlled highlights, attached shadows, clear object supports, crisp eyes, natural skin detail, "
        "clean garment weave, and well-resolved edges. "
        "Use positive direct motion descriptions: face the camera, steady eye contact, small slow hand movement, relaxed shoulders. "
        "When explicit_action is true, action_hint is the primary visible action and must be completed clearly on the stated anatomical side and held for a beat. "
        "Reject segment labels, UI text, compiler notes, previous-context sentences, and negative prohibition lists."
    )
    user = (
        f"Reference scene to adapt:\n{scene_text}\n\n"
        f"Reference appearance to preserve:\n{appearance}\n\n"
        f"Frame orientation: {'upright portrait' if str(aspect_ratio or '').strip().lower() == 'portrait' else 'horizontal landscape'}.\n\n"
        "Prompt compiler contract:\n"
        "Use the selected scene as the exact real physical set. Write a complete LTX prompt with a tight medium close-up, "
        "the speaker filling about two thirds of frame height from head to mid-torso, compact headroom, shoulders spanning roughly "
        "half the frame width, hands available in the lower quarter, and only a narrow furniture edge visible at the bottom. "
        "Use one light direction, no more than three restrained background anchors, visible support for every prop and the body, "
        "stable face identity, stable outfit, natural colors, and one small face-forward action. Later segments and follow-up "
        "turns are the next moment of the same take. "
        "Do not restart the world. Do not mention prior context outside the dialogue line.\n\n"
        f"Selected benchmark-style prompt cases:\n{examples}\n\n"
        "Target segments. For each segment, Speaker_1 says must exactly equal prompt_speech. "
        "The prompt must be a complete standalone video prompt, not a short template. "
        "Describe the exact scene, coherent light direction, controlled face highlights, object supports, restrained background anchors, "
        "natural color palette, moderate depth of field, and one small stable presenter action. "
        "For segment 2 and later, or any follow-up target whose prompt_speech is longer than display_speech, "
        "prompt_speech already contains the needed local spoken context window; "
        "do not shorten it, paraphrase it, or add any separate 'previously said' sentence outside Speech Attribution. "
        "Output every segment as a realistic benchmark-style LTX prompt with the same structure as the selected cases, "
        "including concrete props, identity preservation, stable natural colors, and speech-audio alignment. "
        "If prompt_speech is longer than display_speech, that is intentional for long-video continuity; "
        "the Summary and Narration should say the established conversation continues, not that a new opening frame begins.\n"
        f"{json.dumps(target_segments, ensure_ascii=False, indent=2)}"
    )
    result = await asyncio.to_thread(
        _call_openai_compatible_llm,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        SETTINGS.prompt_planner_max_tokens,
    )
    parsed = _extract_json_array(result or "")
    if not parsed or len(parsed) < len(segment_plans):
        return None

    planned: List[Dict[str, Any]] = []
    for idx, plan in enumerate(segment_plans):
        item = parsed[idx]
        if not isinstance(item, dict):
            return None
        prompt = _force_speaker_says(str(item.get("prompt") or ""), prompt_speeches[idx])
        prompt = _force_speaker_appearance(prompt, appearance)
        if not _valid_llm_ltx_prompt(prompt, prompt_speeches[idx]):
            return None
        planned.append(
            {
                "prompt": prompt,
                "emotion": _safe_emotion(str(item.get("emotion") or plan.get("emotion") or "calm and helpful")),
                "action": _safe_action(str(item.get("action") or plan.get("action") or ""), idx),
            }
        )
    return planned


async def plan_segments(
    scene: str,
    reply: str,
    has_first_frame: bool,
    max_segments: int,
    add_waiting_transition: bool,
    template_id: str = "",
    aspect_ratio: str = "landscape",
    prior_spoken_context: str = "",
    prior_spoken_tail_segment: str = "",
    turn_action: str = "",
    turn_emotion: str = "",
    explicit_action: bool = False,
) -> List[Dict[str, Any]]:
    history_tail_chars = max(0, int(SETTINGS.interactive_prompt_tail_chars))
    history_tail_segment = _history_tail_segment_for_prompt(
        prior_spoken_context,
        prior_spoken_tail_segment,
        max_chars=history_tail_chars,
    )
    cache_key: Tuple[Any, ...] = (
        PROMPT_PLANNER_PROFILE,
        SETTINGS.prompt_planner_max_examples,
        SETTINGS.use_llm_prompt_planner,
        SETTINGS.use_ltx_prompt_skeleton_cache,
        SETTINGS.ltx_prompt_skeleton_max_examples,
        SETTINGS.ltx_prompt_skeleton_max_tokens,
        str(scene or ""),
        str(reply or ""),
        bool(has_first_frame),
        int(max_segments),
        bool(add_waiting_transition),
        str(template_id or ""),
        str(aspect_ratio or "").strip().lower(),
        history_tail_segment,
        SETTINGS.dynamic_utterance_streaming,
        SETTINGS.dynamic_prompt_use_history_tail,
        history_tail_chars,
        SETTINGS.history_tail_in_speaker_says,
        int(SETTINGS.reply_max_chars),
        str(turn_action or ""),
        str(turn_emotion or ""),
        bool(explicit_action),
    )
    cached_segments = PLAN_SEGMENTS_CACHE.get(cache_key)
    if cached_segments is not None:
        PLAN_SEGMENTS_CACHE.move_to_end(cache_key)
        return copy.deepcopy(cached_segments)

    speech_budget = max(1, max_segments - (1 if add_waiting_transition and max_segments > 1 else 0))
    if SETTINGS.dynamic_utterance_streaming:
        fallback_speeches = [
            _speech_target_for_prompt(
                reply,
                max_chars=max(1, int(SETTINGS.prompt_speech_max_chars)),
            )
        ]
        effective_add_waiting_transition = False
    else:
        fallback_speeches = _split_speech(reply, speech_budget)
        compact_reply = re.sub(r"\s+", "", reply or "")
        effective_add_waiting_transition = (
            add_waiting_transition and len(compact_reply) > 45
        )
    segment_plans = _normalize_planned_segments(
        [],
        fallback_speeches=fallback_speeches,
        max_segments=max_segments,
        add_waiting_transition=effective_add_waiting_transition,
    )
    for idx, plan in enumerate(segment_plans):
        if idx == 0:
            plan["action"] = _safe_action(turn_action or str(plan.get("action") or ""), idx)
            plan["emotion"] = _safe_emotion(turn_emotion or str(plan.get("emotion") or ""))
            plan["explicit_action"] = bool(explicit_action)
        else:
            plan["explicit_action"] = False
    segments = []
    prompt_speeches: List[str] = []
    continuity_speeches: List[str] = []
    prior_context_speeches: List[str] = []
    for idx, plan in enumerate(segment_plans):
        current_speech = str(plan.get("speech") or "")
        prior_context_speech = (
            history_tail_segment
            if idx == 0
            else str(segment_plans[idx - 1].get("speech") or "")
        )
        previous_speech = (
            history_tail_segment
            if idx == 0
            else str(segment_plans[idx - 1].get("speech") or "")
        )
        if SETTINGS.dynamic_utterance_streaming and not SETTINGS.dynamic_prompt_use_history_tail:
            previous_speech = "" if idx == 0 else previous_speech
            prior_context_speech = "" if idx == 0 else prior_context_speech
        if idx == 0 and history_tail_segment and not SETTINGS.history_tail_in_speaker_says:
            previous_speech = ""
        if previous_speech:
            prompt_speech = _speech_for_prompt(
                [previous_speech, current_speech],
                max_chars=max(1, int(SETTINGS.prompt_speech_max_chars)),
            )
            if current_speech and not re.search(r"[。！？!?]$", current_speech.strip()):
                prompt_speech = re.sub(r"[。！？!?]+$", "", prompt_speech)
        else:
            prompt_speech = _speech_target_for_prompt(
                current_speech,
                max_chars=max(1, int(SETTINGS.prompt_speech_max_chars)),
            )
        if not prompt_speech:
            prompt_speech = _speech_target_for_prompt(
                current_speech,
                max_chars=max(1, int(SETTINGS.prompt_speech_max_chars)),
            )
        continuity_speeches.append(prompt_speech)
        prompt_speeches.append(prompt_speech)
        prior_context_speeches.append(prior_context_speech)
    llm_prompts = None
    if not SETTINGS.fast_live5_prompt_planner:
        llm_prompts = await _plan_ltx_prompts_with_scene_skeleton(
            scene=scene,
            template_id=template_id,
            aspect_ratio=aspect_ratio,
            has_first_frame=has_first_frame,
            segment_plans=segment_plans,
            prompt_speeches=prompt_speeches,
            prior_context_speeches=prior_context_speeches,
        )
        if llm_prompts is None and not SETTINGS.use_ltx_prompt_skeleton_cache:
            llm_prompts = await _plan_ltx_prompts_with_llm(
                scene=scene,
                template_id=template_id,
                aspect_ratio=aspect_ratio,
                has_first_frame=has_first_frame,
                segment_plans=segment_plans,
                prompt_speeches=prompt_speeches,
            )
    locked_appearance = _appearance_for_scene(
        scene,
        template_id=template_id,
        has_first_frame=has_first_frame,
    )
    for i, plan in enumerate(segment_plans):
        prompt_speech = prompt_speeches[i]
        segment_explicit_action = bool(plan.get("explicit_action", False))
        if llm_prompts:
            prompt = llm_prompts[i]["prompt"]
            emotion = (
                str(plan.get("emotion") or "")
                if segment_explicit_action
                else llm_prompts[i]["emotion"]
            )
            action = (
                str(plan.get("action") or "")
                if segment_explicit_action
                else llm_prompts[i]["action"]
            )
        else:
            emotion = plan.get("emotion", "calm and helpful")
            action = plan.get("action")
            prompt = _build_ltx_prompt(
                scene,
                prompt_speech,
                i,
                len(segment_plans),
                has_first_frame,
                action=action,
                emotion=emotion,
                is_transition=bool(plan.get("is_transition", False)),
                template_id=template_id,
                aspect_ratio=aspect_ratio,
                continues_from_prior_context=bool(
                    history_tail_segment or i > 0
                ),
                prior_context_speech=prior_context_speeches[i],
                explicit_action=segment_explicit_action,
            )
        prompt = _force_speaker_appearance(prompt, locked_appearance)
        prompt = _force_speaker_action(
            prompt,
            action or "",
            explicit_action=segment_explicit_action,
        )
        prompt = _ensure_direct_camera_take_prompt(
            prompt,
            aspect_ratio=aspect_ratio,
        )
        prompt = _apply_motion_camera_prompt(prompt, action or "")
        prompt = _force_prompt_speech_language(prompt, prompt_speech)
        prompt = _compact_ltx_prompt_for_latency(prompt)
        segments.append(
            {
                "segment_id": i,
                "speech": plan["speech"],
                "prompt_speech": prompt_speech,
                "continuity_speech": continuity_speeches[i],
                "prior_context_speech": prior_context_speeches[i],
                "emotion": emotion,
                "action": action or "minimal presenter gesture",
                "explicit_action": segment_explicit_action,
                "is_transition": bool(plan.get("is_transition", False)),
                "prompt": prompt,
                "prompt_planner": (
                    "deterministic_live5_fast"
                    if SETTINGS.fast_live5_prompt_planner
                    else (
                        str(llm_prompts[i].get("prompt_planner") or "llm_fewshot")
                        if llm_prompts
                        else "deterministic_fallback"
                    )
                ),
                "prompt_profile": PROMPT_PLANNER_PROFILE,
            }
        )
    if PLAN_SEGMENTS_CACHE_SIZE > 0:
        PLAN_SEGMENTS_CACHE[cache_key] = copy.deepcopy(segments)
        PLAN_SEGMENTS_CACHE.move_to_end(cache_key)
        while len(PLAN_SEGMENTS_CACHE) > PLAN_SEGMENTS_CACHE_SIZE:
            PLAN_SEGMENTS_CACHE.popitem(last=False)
    return segments


def _write_prompt_file(job: JobState, first_frame_path: Optional[Path]) -> None:
    case: Dict[str, Any] = {
        "case_id": job.task_id,
        "description": "interactive digital-human conversation",
        "seed": 20260702,
    }
    if SETTINGS.dynamic_utterance_streaming and len(job.segments) == 1:
        segment = job.segments[0]
        case.update(
            {
                "prompt": segment["prompt"],
                "prompt_repeat_count": 1,
                "prompt_window_id": 0,
                "prompt_speech": segment.get("prompt_speech", ""),
                "display_speech": segment.get("speech", ""),
            }
        )
    else:
        segments = []
        for segment in job.segments:
            segments.append(
                {
                    "segment_id": int(segment["segment_id"]),
                    "prompt": segment["prompt"],
                    "seed": 20260702 + int(segment["segment_id"]),
                }
            )
        case["segments"] = segments
    if first_frame_path:
        case["conditioning_mode"] = "i2v"
        case["first_frame_path"] = str(first_frame_path)
        case["image"] = [[str(first_frame_path), 0, 0.9]]
    job.prompts_file.write_text(
        json.dumps([case], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


_SPEAKER_SAYS_RE = re.compile(r'((Speaker[_ ]?\d+)\s+says:\s*")([^"]*)(")')


def _speaker_says_texts(prompt: str) -> List[str]:
    return [match.group(3) for match in _SPEAKER_SAYS_RE.finditer(str(prompt or ""))]


def _log_model_prompts(job: JobState) -> None:
    debug_path = job.task_dir / "model_prompts_debug.jsonl"
    rows: List[Dict[str, Any]] = []
    print(
        f"[ModelPrompt:server] task={job.task_id} segments={len(job.segments)} "
        f"prompt_file={job.prompts_file}",
        flush=True,
    )
    for segment in job.segments:
        prompt = str(segment.get("prompt") or "")
        says = _speaker_says_texts(prompt)
        row = {
            "task_id": job.task_id,
            "segment_id": int(segment.get("segment_id") or 0),
            "display_speech": str(segment.get("speech") or ""),
            "prompt_speech": str(segment.get("prompt_speech") or ""),
            "continuity_speech": str(segment.get("continuity_speech") or ""),
            "speaker_says": says,
            "prompt_chars": len(prompt),
            "prompt": prompt,
        }
        rows.append(row)
        print(
            "[ModelPrompt:server] "
            f"segment={row['segment_id']} display_speech={row['display_speech']} "
            f"prompt_speech={row['prompt_speech']} "
            f"continuity_speech={row['continuity_speech']} speaker_says={says}",
            flush=True,
        )
    try:
        with debug_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[ModelPrompt:server][WARN] failed to write {debug_path}: {exc}", flush=True)


async def _discover_videos(job: JobState, emit_new: bool = True) -> List[str]:
    videos: List[str] = []
    media_urls: List[str] = []
    stop_after = _stop_after_block_count(job)
    if job.output_dir.exists():
        paths: List[Path] = []
        for pattern in (
            "*.mp4",
            "*_streams/*.mp4",
            "*_chunks/*.mp4",
            "*_streams/*.jpg",
            "*_streams/*.jpeg",
            "*_streams/*.png",
        ):
            paths.extend(job.output_dir.glob(pattern))
        unique_paths = sorted(set(paths), key=_media_path_sort_key)
        for path in unique_paths:
            if ".tmp." in path.name:
                continue
            if not _should_publish_preview_media(job, path, stop_after):
                continue
            try:
                rel = path.relative_to(job.task_dir)
            except ValueError:
                continue
            url = f"/media/{job.task_id}/{rel.as_posix()}"
            media_urls.append(url)
            if path.suffix.lower() == ".mp4":
                videos.append(url)
    # `job._video_seen_urls` tracks SSE asset emission, not mere filesystem
    # discovery. Status polling uses emit_new=False; marking files as seen there
    # would let `/api/jobs/{id}` swallow early stream assets before the live SSE
    # watcher has a chance to publish them.
    job.videos = list(videos)
    new_urls = [url for url in media_urls if url not in job._video_seen_urls]
    if emit_new and new_urls:
        for url in new_urls:
            kind = "video" if url.lower().endswith(".mp4") else "image"
            await _emit(job, "asset", {"kind": kind, "url": url})
        job._video_seen_urls.update(new_urls)
    return videos


_ASR_MODEL: Any = None
_ASR_MODEL_ERROR: Optional[str] = None
_ASR_MODEL_WARMUP_ATTEMPTED = False
_ASR_MODEL_WARMUP_LOCK = threading.Lock()


def _stream_chunk_paths(job: JobState) -> List[Path]:
    if not job.output_dir.exists():
        return []
    return sorted(
        {
            path
            for path in job.output_dir.glob("*_streams/*.mp4")
            if ".tmp." not in path.name
        },
        key=_media_path_sort_key,
    )


def _stream_asr_audio_paths(job: JobState) -> List[Path]:
    """Return only the contiguous, atomically published ASR WAV prefix."""
    if not job.output_dir.exists():
        return []
    indexed: Dict[int, Path] = {}
    for path in job.output_dir.glob("*_streams/*.asr.wav"):
        if ".tmp." in path.name or ".ready." in path.name:
            continue
        match = re.search(r"_stream([0-9]+)\.asr\.wav$", path.name)
        if match:
            indexed[int(match.group(1))] = path
    contiguous: List[Path] = []
    expected = 0
    while expected in indexed:
        contiguous.append(indexed[expected])
        expected += 1
    return contiguous


def _asr_observation_paths(job: JobState) -> List[Path]:
    audio_paths = _stream_asr_audio_paths(job)
    video_paths = _stream_chunk_paths(job)
    return audio_paths if audio_paths and len(audio_paths) >= len(video_paths) else video_paths


_ASR_TRADITIONAL_TO_SIMPLIFIED = str.maketrans(
    {
        "們": "们", "這": "这", "個": "个", "會": "会", "來": "来", "說": "说", "話": "话",
        "講": "讲", "員": "员", "輕": "轻", "鬆": "松", "體": "体", "學": "学", "習": "习",
        "時": "时", "間": "间", "裡": "里", "裏": "里", "為": "为", "與": "与", "對": "对",
        "關": "关", "於": "于", "點": "点", "幫": "帮", "問": "问", "題": "题", "專": "专",
        "業": "业", "顧": "顾", "劃": "划", "決": "决", "策": "策", "職": "职", "場": "场",
        "規": "规", "諮": "咨", "詢": "询", "讓": "让", "現": "现", "實": "实", "穩": "稳",
        "定": "定", "顏": "颜", "顯": "显", "視": "视", "頻": "频", "聲": "声", "音": "音",
        "醫": "医", "藥": "药", "營": "营", "養": "养", "師": "师", "課": "课", "程": "程",
        "愛": "爱", "樂": "乐", "歡": "欢", "識": "识", "讀": "读", "處": "处", "長": "长",
        "應": "应", "該": "该", "後": "后", "邊": "边", "還": "还", "續": "续", "產": "产",
        "開": "开", "啟": "启", "閉": "闭", "轉": "转", "復": "复", "雜": "杂", "簡": "简",
        "單": "单", "確": "确", "認": "认", "記": "记", "錄": "录", "數": "数", "據": "据",
        "儀": "仪", "態": "态", "動": "动", "靜": "静", "聽": "听", "遲": "迟", "緩": "缓",
        "優": "优", "質": "质", "線": "线", "內": "内", "氣": "气", "親": "亲", "請": "请",
        "謝": "谢", "曉": "晓", "鄉": "乡", "響": "响", "緊": "紧", "寬": "宽", "軟": "软",
        "體": "体", "薑": "姜", "蔣": "蒋", "覺": "觉", "僅": "仅", "種": "种",
        "達": "达", "銷": "销", "兩": "两", "壓": "压", "慣": "惯", "從": "从",
    }
)


_ASR_BUDGET_CONTENT_MIN_LENGTH_RATIO = 0.82
_ASR_BUDGET_CONTENT_MIN_TARGET_PROGRESS = 0.88
_ASR_BUDGET_CONTENT_MAX_PROGRESS_GAIN = 0.02
_ASR_BUDGET_CONTENT_MAX_COVERAGE_GAIN = 0.02


_ASR_MATCH_REPLACEMENTS = (
    ("您", "你"),
    ("讲解人", "讲解员"),
    ("讲解元", "讲解员"),
    ("解说员", "讲解员"),
    ("身体见", "身体健康"),
    ("身體見", "身体健康"),
    ("健康见", "健康"),
    ("青松", "轻松"),
    ("清松", "轻松"),
    ("清淡", "轻松"),
    ("三步", "散步"),
    ("前青松", "先轻松"),
    ("前清松", "先轻松"),
    ("每天前", "每天先"),
    ("开水", "开始"),
)


def _normalize_speech_for_match(text: str) -> str:
    text = str(text or "").translate(_ASR_TRADITIONAL_TO_SIMPLIFIED)
    for src, dst in _ASR_MATCH_REPLACEMENTS:
        text = text.replace(src, dst)
    normalized = "".join(
        ch
        for ch in text
        if ch not in _SPEECH_PUNCTUATION and not ch.isspace()
    ).lower()
    return normalized


def _lcs_len(a: str, b: str) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if ca == cb else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _speech_match_metrics(transcript: str, target: str) -> Dict[str, Any]:
    hyp = _normalize_speech_for_match(transcript)
    ref = _normalize_speech_for_match(target)
    lcs = _lcs_len(hyp, ref)
    coverage = float(lcs / max(1, len(ref)))
    matcher = difflib.SequenceMatcher(None, hyp, ref, autojunk=False)
    seq_ratio = float(matcher.ratio()) if hyp and ref else 0.0
    matching_blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    last_target_match_end = max(
        (block.b + block.size for block in matching_blocks),
        default=0,
    )
    target_progress = float(last_target_match_end / max(1, len(ref)))
    transcript_target_length_ratio = float(len(hyp) / max(1, len(ref)))
    tail_len = min(8, max(1, len(ref)))
    tail = ref[-tail_len:] if ref else ""
    tail_covered = bool(tail and tail in hyp)
    tail_window = hyp[-max(16, tail_len * 2):] if hyp else ""
    tail_lcs = _lcs_len(tail_window, tail) if tail else 0
    tail_coverage = float(tail_lcs / max(1, len(tail))) if tail else 0.0
    tail_suffix = hyp[-tail_len:] if tail and hyp else ""
    tail_suffix_ratio = (
        float(difflib.SequenceMatcher(None, tail_suffix, tail).ratio())
        if tail_suffix and tail else 0.0
    )
    tail_end_len = min(4, len(tail))
    tail_end = tail[-tail_end_len:] if tail_end_len else ""
    hyp_end = hyp[-tail_end_len:] if tail_end_len and hyp else ""
    tail_end_ratio = (
        float(difflib.SequenceMatcher(None, hyp_end, tail_end).ratio())
        if hyp_end and tail_end else 0.0
    )
    tail_suffix_aligned = bool(
        tail_suffix_ratio >= 0.68
        and tail_end_ratio >= 0.65
    )
    tail_fuzzy_covered = bool(
        tail_covered
        or (tail and tail_coverage >= SETTINGS.asr_tail_fuzzy_ratio)
        or tail_suffix_aligned
    )
    return {
        "transcript_norm": hyp,
        "target_norm": ref,
        "lcs_chars": lcs,
        "target_chars": len(ref),
        "coverage": coverage,
        "sequence_ratio": seq_ratio,
        "transcript_chars": len(hyp),
        "transcript_target_length_ratio": transcript_target_length_ratio,
        "target_progress": target_progress,
        "target_remaining_chars": max(0, len(ref) - last_target_match_end),
        "tail_covered": tail_covered,
        "tail_fuzzy_covered": tail_fuzzy_covered,
        "tail_lcs_chars": tail_lcs,
        "tail_chars": len(tail),
        "tail_coverage": tail_coverage,
        "tail_suffix_ratio": tail_suffix_ratio,
        "tail_end_ratio": tail_end_ratio,
        "tail_suffix_aligned": tail_suffix_aligned,
    }


def _resolve_asr_model_spec() -> Tuple[str, bool]:
    explicit_path = Path(SETTINGS.asr_model_path).expanduser() if SETTINGS.asr_model_path else None
    if explicit_path and explicit_path.is_file():
        return str(explicit_path), True
    project_path = PROJECT_ROOT / "models" / "whisper" / f"{SETTINGS.asr_model}.pt"
    if project_path.is_file():
        return str(project_path), True
    return SETTINGS.asr_model or "tiny", False


def _warm_up_asr_model(model: Any) -> None:
    """Exercise Whisper's encoder and decoder once before the first request."""
    global _ASR_MODEL_WARMUP_ATTEMPTED
    with _ASR_MODEL_WARMUP_LOCK:
        if _ASR_MODEL_WARMUP_ATTEMPTED:
            return
        _ASR_MODEL_WARMUP_ATTEMPTED = True
        started = time.perf_counter()
        try:
            import numpy as np

            model.transcribe(
                np.zeros(3 * 16000, dtype=np.float32),
                language="zh",
                task="transcribe",
                fp16=str(SETTINGS.asr_device).startswith("cuda"),
                condition_on_previous_text=False,
                verbose=None,
            )
            print(
                f"[ASR] inference warmup completed in "
                f"{(time.perf_counter() - started) * 1000.0:.1f}ms",
                flush=True,
            )
        except Exception as exc:
            # Warmup is an optimization. Keep the loaded model available if a
            # backend does not accept synthetic in-memory audio.
            print(f"[ASR] inference warmup skipped: {exc}", flush=True)


def _load_asr_model() -> Any:
    global _ASR_MODEL, _ASR_MODEL_ERROR
    if _ASR_MODEL is not None:
        _warm_up_asr_model(_ASR_MODEL)
        return _ASR_MODEL
    if _ASR_MODEL_ERROR:
        raise RuntimeError(_ASR_MODEL_ERROR)
    try:
        import whisper  # type: ignore

        model_spec, is_local_file = _resolve_asr_model_spec()
        if not is_local_file and not SETTINGS.asr_allow_download:
            raise RuntimeError(
                "ASR model is not available locally and AVATAR_ASR_ALLOW_DOWNLOAD=0 "
                f"(model={model_spec}, checked={SETTINGS.asr_model_path})"
            )
        _ASR_MODEL = whisper.load_model(model_spec, device=SETTINGS.asr_device or "cpu")
        _warm_up_asr_model(_ASR_MODEL)
        return _ASR_MODEL
    except Exception as exc:
        _ASR_MODEL_ERROR = str(exc)
        raise


def _extract_audio_for_asr(paths: List[Path], wav_path: Path) -> None:
    if not paths:
        raise RuntimeError("no stream chunks for ASR")
    if all(path.name.endswith(".asr.wav") for path in paths):
        tmp_path = wav_path.with_name(
            f"{wav_path.stem}.tmp.{os.getpid()}{wav_path.suffix}"
        )
        try:
            with wave.open(str(tmp_path), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(16000)
                for path in paths:
                    with wave.open(str(path), "rb") as reader:
                        params = (
                            reader.getnchannels(),
                            reader.getsampwidth(),
                            reader.getframerate(),
                        )
                        if params != (1, 2, 16000):
                            raise RuntimeError(
                                f"unexpected ASR sidecar format {params} for {path}"
                            )
                        writer.writeframes(reader.readframes(reader.getnframes()))
            os.replace(tmp_path, wav_path)
        finally:
            with suppress(OSError):
                tmp_path.unlink()
        return
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract stream audio for ASR")
    list_path = wav_path.with_suffix(".txt")
    def _concat_file_line(path: Path) -> str:
        escaped = str(path.resolve()).replace("'", "'\\''")
        return f"file '{escaped}'"

    list_path.write_text(
        "\n".join(_concat_file_line(path) for path in paths) + "\n",
        encoding="utf-8",
    )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "ffmpeg audio extraction failed")[-1000:])


def _audio_endpoint_metrics(paths: List[Path]) -> Dict[str, Any]:
    """Measure whether the latest direct-audio block ends in real silence."""
    base: Dict[str, Any] = {
        "tail_audio_available": False,
        "tail_audio_dbfs": None,
        "block_audio_dbfs": None,
        "tail_silence": False,
        "tail_silence_seconds": SETTINGS.asr_tail_silence_seconds,
        "tail_silence_threshold_dbfs": SETTINGS.asr_tail_silence_dbfs,
    }
    if not paths or not paths[-1].name.endswith(".asr.wav"):
        return base
    try:
        import numpy as np

        with wave.open(str(paths[-1]), "rb") as reader:
            if (
                reader.getnchannels(),
                reader.getsampwidth(),
                reader.getframerate(),
            ) != (1, 2, 16000):
                return base
            samples = np.frombuffer(
                reader.readframes(reader.getnframes()),
                dtype=np.int16,
            ).astype(np.float32) / 32768.0
        if samples.size == 0:
            return base
        tail_samples = max(
            1,
            int(round(16000 * max(0.05, SETTINGS.asr_tail_silence_seconds))),
        )

        def _dbfs(values: Any) -> float:
            rms = float(np.sqrt(np.mean(values * values)))
            return float(20.0 * math.log10(max(rms, 1e-8)))

        block_dbfs = _dbfs(samples)
        tail_dbfs = _dbfs(samples[-tail_samples:])
        base.update(
            {
                "tail_audio_available": True,
                "tail_audio_dbfs": round(tail_dbfs, 3),
                "block_audio_dbfs": round(block_dbfs, 3),
                "tail_silence": tail_dbfs <= SETTINGS.asr_tail_silence_dbfs,
            }
        )
    except (OSError, ValueError, wave.Error):
        return base
    return base


def _speech_budget_timing(target: str, chunk_count: int) -> Dict[str, Any]:
    chunk_seconds = max(0.1, SETTINGS.realtime_stream_chunk_seconds)
    seconds_observed = max(0, int(chunk_count)) * chunk_seconds
    target_visible_chars = max(1, _speech_visible_len(target))
    budget_chars_per_second = max(0.1, SETTINGS.asr_budget_chars_per_second)
    pause_budget = _speech_pause_budget(target)
    expected_seconds = target_visible_chars / budget_chars_per_second + max(
        0.0,
        SETTINGS.asr_budget_extra_seconds,
    ) + float(pause_budget["pause_seconds"])
    min_budget_seconds = SETTINGS.asr_min_seconds
    if target_visible_chars >= max(1, SETTINGS.asr_budget_medium_char_threshold):
        min_budget_seconds = max(
            min_budget_seconds,
            SETTINGS.asr_budget_medium_min_seconds,
        )
    if target_visible_chars >= max(1, SETTINGS.asr_budget_long_char_threshold):
        min_budget_seconds = max(
            min_budget_seconds,
            SETTINGS.asr_budget_long_min_seconds,
        )
    budget_seconds = max(min_budget_seconds, expected_seconds)
    stop_after_chunk_count = max(1, int(math.ceil(budget_seconds / chunk_seconds)))
    budget_ready = max(0, int(chunk_count)) >= stop_after_chunk_count
    near_budget_ready = seconds_observed >= max(
        SETTINGS.asr_min_seconds,
        expected_seconds * 0.85,
    )
    return {
        "target_visible_chars": target_visible_chars,
        "expected_seconds": expected_seconds,
        **pause_budget,
        "budget_seconds": budget_seconds,
        "budget_stop_after_chunk_count": stop_after_chunk_count,
        "budget_ready": budget_ready,
        "near_budget_ready": near_budget_ready,
        "seconds_observed": seconds_observed,
    }


def _speech_completion_target_details(job: JobState) -> Dict[str, Any]:
    """Return only the speech expected in this job's newly generated audio.

    A continuation prompt intentionally prepends the previous turn's tail to
    ``Speaker_1 says`` so the latent/audio prefix and text condition describe
    one continuous clip.  That historical tail is already present in the
    prefix and is not repeated in the new audio sidecars watched by ASR.
    """
    targets: List[str] = []
    for segment in job.segments or []:
        if not isinstance(segment, dict):
            continue
        target = str(segment.get("speech") or segment.get("display_speech") or "").strip()
        if target:
            targets.append(target)
    if targets:
        max_chars = max(
            2048,
            int(SETTINGS.prompt_speech_max_chars) * max(1, len(targets)),
        )
        return {
            "target": _speech_for_prompt(targets, max_chars=max_chars),
            "target_source": "speech",
            "target_part_count": len(targets),
        }
    reply = str(job.reply or "").strip()
    if reply:
        return {
            "target": reply,
            "target_source": "reply",
            "target_part_count": 1,
        }

    # Compatibility for old persisted jobs that predate structured speech.
    for segment in job.segments or []:
        if not isinstance(segment, dict):
            continue
        speaker_says = _speaker_says_texts(str(segment.get("prompt") or ""))
        if speaker_says:
            return {
                "target": _speech_for_prompt(
                    speaker_says,
                    max_chars=max(1, int(SETTINGS.prompt_speech_max_chars)),
                ),
                "target_source": "speaker_says_legacy",
                "target_part_count": len(speaker_says),
            }
    return {"target": "", "target_source": "none", "target_part_count": 0}


def _speech_completion_target(job: JobState) -> str:
    return str(_speech_completion_target_details(job)["target"])


def _asr_stop_target(base_chunk_count: int) -> Dict[str, Any]:
    chunk_seconds = max(0.1, SETTINGS.realtime_stream_chunk_seconds)
    blocks_per_chunk = max(1, int(SETTINGS.realtime_stream_blocks_per_chunk))
    tail_chunks = int(
        math.ceil(max(0.0, SETTINGS.asr_tail_seconds_after_done) / chunk_seconds)
    )
    tail_blocks = max(0, int(SETTINGS.asr_tail_blocks_after_done))
    stop_after_block_count = (
        max(0, int(base_chunk_count)) + tail_chunks
    ) * blocks_per_chunk + tail_blocks
    stop_after_chunk_count = int(
        math.ceil(stop_after_block_count / blocks_per_chunk)
    )
    return {
        "tail_seconds_after_done": SETTINGS.asr_tail_seconds_after_done,
        "tail_chunks_after_done": tail_chunks,
        "tail_blocks_after_done": tail_blocks,
        "stop_after_chunk_count": stop_after_chunk_count,
        "stop_after_block_count": stop_after_block_count,
        "recommended_stop_after_seconds": (
            stop_after_block_count / blocks_per_chunk
        ) * chunk_seconds,
    }


def _budget_stop_payload(job: JobState, chunk_count: int) -> Dict[str, Any]:
    target_details = _speech_completion_target_details(job)
    target = str(target_details["target"])
    timing = _speech_budget_timing(target, chunk_count)
    budget_stop_after_chunk_count = int(timing["budget_stop_after_chunk_count"])
    stop_target = _asr_stop_target(budget_stop_after_chunk_count)
    return {
        "enabled": True,
        "available": True,
        "completed": True,
        "completion_source": (
            "duration_budget"
            if int(chunk_count) >= budget_stop_after_chunk_count
            else "duration_budget_scheduled"
        ),
        "target": target,
        "target_source": target_details["target_source"],
        "target_part_count": target_details["target_part_count"],
        "transcript": "",
        "chunk_count": chunk_count,
        **timing,
        "budget_min_coverage": SETTINGS.asr_budget_min_coverage,
        **stop_target,
    }


def _run_asr_completion_check(job: JobState, paths: List[Path]) -> Dict[str, Any]:
    profile_started = time.perf_counter()
    target_details = _speech_completion_target_details(job)
    target = str(target_details["target"])
    asr_dir = job.task_dir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)
    wav_path = asr_dir / f"asr_{len(paths):04d}.wav"
    stage_started = time.perf_counter()
    _extract_audio_for_asr(paths, wav_path)
    extract_ms = (time.perf_counter() - stage_started) * 1000.0
    stage_started = time.perf_counter()
    model = _load_asr_model()
    model_ready_ms = (time.perf_counter() - stage_started) * 1000.0
    stage_started = time.perf_counter()
    result = model.transcribe(
        str(wav_path),
        language=_dominant_response_language(target),
        task="transcribe",
        fp16=str(SETTINGS.asr_device).startswith("cuda"),
        condition_on_previous_text=False,
        verbose=False,
    )
    transcribe_ms = (time.perf_counter() - stage_started) * 1000.0
    stage_started = time.perf_counter()
    transcript = str(result.get("text") or "").strip()
    metrics = _speech_match_metrics(transcript, target)
    match_ms = (time.perf_counter() - stage_started) * 1000.0
    stage_started = time.perf_counter()
    endpoint_metrics = _audio_endpoint_metrics(paths)
    endpoint_ms = (time.perf_counter() - stage_started) * 1000.0
    timing = _speech_budget_timing(target, len(paths))
    budget_ready = bool(timing["budget_ready"])
    near_budget_ready = bool(timing["near_budget_ready"])
    strong_coverage_ready = bool(
        metrics["coverage"] >= SETTINGS.asr_completion_ratio
        and metrics["tail_fuzzy_covered"]
    )
    suffix_completion_ready = bool(
        metrics["coverage"] >= max(0.50, SETTINGS.asr_budget_min_coverage - 0.08)
        and metrics["tail_suffix_aligned"]
    )
    endpoint_completion_ready = bool(
        budget_ready
        and metrics["coverage"] >= max(0.50, SETTINGS.asr_budget_min_coverage - 0.08)
        and endpoint_metrics["tail_silence"]
    )
    # Tiny Whisper often transcribes a fully spoken sentence with a corrupted
    # final phrase. Requiring an exact fuzzy suffix then lets generation run
    # into repetition. This fallback only activates at the duration budget and
    # requires both near-complete transcript length and late target progress,
    # so a genuinely unfinished sentence is not cut at its midpoint.
    budget_content_candidate = bool(
        budget_ready
        and metrics["coverage"] >= SETTINGS.asr_budget_min_coverage
        and metrics["transcript_target_length_ratio"]
        >= _ASR_BUDGET_CONTENT_MIN_LENGTH_RATIO
        and metrics["target_progress"] >= _ASR_BUDGET_CONTENT_MIN_TARGET_PROGRESS
    )
    budget_content_ready = False
    completed = bool(
        strong_coverage_ready
        or suffix_completion_ready
        or endpoint_completion_ready
        or (
            metrics["coverage"] >= max(0.58, SETTINGS.asr_completion_ratio - 0.18)
            and metrics["tail_fuzzy_covered"]
        )
    )
    completion_source = None
    if strong_coverage_ready:
        completion_source = "asr_strong_coverage"
    elif suffix_completion_ready:
        completion_source = "asr_suffix"
    elif endpoint_completion_ready:
        completion_source = "asr_endpoint"
    elif completed:
        completion_source = "asr_fuzzy_tail"
    stop_target = _asr_stop_target(len(paths))
    payload: Dict[str, Any] = {
        "enabled": True,
        "available": True,
        "completed": completed,
        "completion_source": completion_source,
        "target": target,
        "target_source": target_details["target_source"],
        "target_part_count": target_details["target_part_count"],
        "transcript": transcript,
        "chunk_count": len(paths),
        "observation_source": (
            "audio_sidecar"
            if paths and all(path.name.endswith(".asr.wav") for path in paths)
            else "mp4"
        ),
        **timing,
        "strong_coverage_ready": strong_coverage_ready,
        "suffix_completion_ready": suffix_completion_ready,
        "endpoint_completion_ready": endpoint_completion_ready,
        "budget_content_candidate": budget_content_candidate,
        "budget_content_ready": budget_content_ready,
        "budget_min_coverage": SETTINGS.asr_budget_min_coverage,
        "tail_seconds_after_done": stop_target["tail_seconds_after_done"],
        "tail_chunks_after_done": stop_target["tail_chunks_after_done"],
        "tail_blocks_after_done": stop_target["tail_blocks_after_done"],
        "stop_after_chunk_count": (
            stop_target["stop_after_chunk_count"] if completed else None
        ),
        "stop_after_block_count": (
            stop_target["stop_after_block_count"] if completed else None
        ),
        "recommended_stop_after_seconds": (
            stop_target["recommended_stop_after_seconds"] if completed else None
        ),
        "profile_ms": {
            "extract_audio": round(extract_ms, 3),
            "model_ready": round(model_ready_ms, 3),
            "transcribe": round(transcribe_ms, 3),
            "match": round(match_ms, 3),
            "endpoint": round(endpoint_ms, 3),
            "total_before_result_write": round(
                (time.perf_counter() - profile_started) * 1000.0,
                3,
            ),
        },
        **endpoint_metrics,
        **metrics,
    }
    (asr_dir / f"asr_{len(paths):04d}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "[Profile:ASR] "
        f"chunks={len(paths)} extract={extract_ms:.1f}ms "
        f"model={model_ready_ms:.1f}ms transcribe={transcribe_ms:.1f}ms "
        f"match={match_ms:.1f}ms endpoint={endpoint_ms:.1f}ms "
        f"silence={int(bool(endpoint_metrics['tail_silence']))} "
        f"completed={int(completed)}",
        flush=True,
    )
    return payload


def _confirm_budget_content_plateau(
    payload: Dict[str, Any],
    previous_payload: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Confirm an uncertain sentence tail only after ASR progress plateaus.

    A single near-complete snapshot cannot distinguish a corrupted final phrase
    from a final phrase that is still being spoken. Requiring two consecutive
    budget candidates with no meaningful target progress preserves the latter
    while stopping the former before it runs into repetition.
    """
    if payload.get("completed") or not payload.get("budget_content_candidate"):
        return payload
    if not isinstance(previous_payload, dict) or not previous_payload.get(
        "budget_content_candidate"
    ):
        return payload
    current_chunk = int(payload.get("chunk_count") or 0)
    previous_chunk = int(previous_payload.get("chunk_count") or 0)
    if current_chunk <= previous_chunk:
        return payload

    progress_gain = float(payload.get("target_progress") or 0.0) - float(
        previous_payload.get("target_progress") or 0.0
    )
    coverage_gain = float(payload.get("coverage") or 0.0) - float(
        previous_payload.get("coverage") or 0.0
    )
    if (
        progress_gain > _ASR_BUDGET_CONTENT_MAX_PROGRESS_GAIN
        or coverage_gain > _ASR_BUDGET_CONTENT_MAX_COVERAGE_GAIN
    ):
        return payload

    stop_target = _asr_stop_target(current_chunk)
    payload = dict(payload)
    payload.update(
        {
            "completed": True,
            "completion_source": "asr_budget_content_plateau",
            "budget_content_ready": True,
            "budget_content_progress_gain": progress_gain,
            "budget_content_coverage_gain": coverage_gain,
            **stop_target,
        }
    )
    return payload


def _align_asr_stop_to_decision_time(job: JobState, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Align the stop target to when the ASR decision reaches the live generator.

    ASR runs asynchronously.  By the time a completed result for N observed
    chunks returns, the generator may already have emitted more chunks.  The stop
    target is based on the stream count at decision time plus the configured
    optional tail.
    """
    if not payload.get("completed"):
        return payload
    emitted_chunks_now = len(_asr_observation_paths(job))
    observed_chunks = int(payload.get("chunk_count") or 0)
    decision_base_chunks = max(observed_chunks, emitted_chunks_now)
    stop_target = _asr_stop_target(decision_base_chunks)
    payload = dict(payload)
    payload.update(
        {
            "chunks_available_at_decision": emitted_chunks_now,
            **stop_target,
        }
    )
    return payload


def _write_asr_stop_control(job: JobState, payload: Dict[str, Any]) -> None:
    control_path = job.task_dir / "asr_stop_control.json"
    predicted_stop_after_chunks = (
        payload.get("predicted_stop_after_chunk_count")
        or payload.get("budget_stop_after_chunk_count")
    )
    predicted_stop_after_blocks = payload.get("predicted_stop_after_block_count")
    if predicted_stop_after_blocks is None and predicted_stop_after_chunks is not None:
        predicted_stop_after_blocks = int(predicted_stop_after_chunks) * max(
            1,
            int(SETTINGS.realtime_stream_blocks_per_chunk),
        )
    blocks_per_chunk = max(1, int(SETTINGS.realtime_stream_blocks_per_chunk))
    observed_chunks = max(0, int(payload.get("chunk_count") or 0))
    control: Dict[str, Any] = {
        "enabled": bool(payload.get("enabled", True)),
        "available": payload.get("available"),
        "completed": payload.get("completed"),
        "completion_source": payload.get("completion_source"),
        "chunk_count": payload.get("chunk_count"),
        "observation_source": payload.get("observation_source"),
        "seconds_observed": payload.get("seconds_observed"),
        "target_visible_chars": payload.get("target_visible_chars"),
        "target_source": payload.get("target_source"),
        "target_part_count": payload.get("target_part_count"),
        "budget_seconds": payload.get("budget_seconds"),
        "budget_stop_after_chunk_count": payload.get("budget_stop_after_chunk_count"),
        "budget_min_coverage": payload.get("budget_min_coverage"),
        "budget_ready": payload.get("budget_ready"),
        "near_budget_ready": payload.get("near_budget_ready"),
        "strong_coverage_ready": payload.get("strong_coverage_ready"),
        "suffix_completion_ready": payload.get("suffix_completion_ready"),
        "endpoint_completion_ready": payload.get("endpoint_completion_ready"),
        "budget_content_candidate": payload.get("budget_content_candidate"),
        "budget_content_ready": payload.get("budget_content_ready"),
        "budget_content_progress_gain": payload.get("budget_content_progress_gain"),
        "budget_content_coverage_gain": payload.get("budget_content_coverage_gain"),
        "coverage": payload.get("coverage"),
        "transcript_target_length_ratio": payload.get("transcript_target_length_ratio"),
        "target_progress": payload.get("target_progress"),
        "tail_fuzzy_covered": payload.get("tail_fuzzy_covered"),
        "profile_ms": payload.get("profile_ms"),
        "predicted_stop_after_chunk_count": predicted_stop_after_chunks,
        "predicted_stop_after_block_count": predicted_stop_after_blocks,
        "observed_block_count": observed_chunks * blocks_per_chunk,
        "tail_seconds_after_done": SETTINGS.asr_tail_seconds_after_done,
        "tail_chunks_after_done": payload.get("tail_chunks_after_done"),
        "tail_blocks_after_done": payload.get(
            "tail_blocks_after_done",
            SETTINGS.asr_tail_blocks_after_done,
        ),
        "stop_after_chunk_count": payload.get("stop_after_chunk_count"),
        "stop_after_block_count": payload.get("stop_after_block_count"),
        "prewritten": payload.get("prewritten"),
        "error": payload.get("error"),
        "updated_at": _now(),
    }
    if payload.get("available") is False:
        control["disabled"] = True
    tmp = control_path.with_suffix(control_path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(control, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, control_path)


def _prewrite_budget_stop_control(job: JobState) -> None:
    """Expose the duration-budget stop before the first generated chunk exists.

    The resident inference worker decides whether to append another causal chunk at the
    chunk boundary.  If the service only writes the budget stop after seeing
    stream chunks, a short utterance can race with that boundary and get one
    unnecessary continuation chunk.  This control file is only a maximum
    duration guard; the ASR watcher can still overwrite it with an earlier
    completed decision once audio is available.
    """
    if not (SETTINGS.asr_enabled and SETTINGS.asr_dynamic_stop):
        return
    payload = _budget_stop_payload(job, 0)
    if SETTINGS.asr_budget_hard_stop:
        payload["completion_source"] = "duration_budget_prewrite"
    else:
        predicted_chunks = int(payload.get("stop_after_chunk_count") or 1)
        predicted_blocks = int(payload.get("stop_after_block_count") or predicted_chunks)
        payload.update(
            {
                "completed": False,
                "completion_source": "duration_budget_prediction",
                "predicted_stop_after_chunk_count": predicted_chunks,
                "predicted_stop_after_block_count": predicted_blocks,
                "stop_after_chunk_count": None,
                "stop_after_block_count": None,
                "recommended_stop_after_seconds": None,
            }
        )
    payload["prewritten"] = True
    _write_asr_stop_control(job, payload)


def _asr_public_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    keep = {
        "enabled",
        "available",
        "completed",
        "completion_source",
        "target",
        "target_source",
        "target_part_count",
        "transcript",
        "chunk_count",
        "seconds_observed",
        "target_visible_chars",
        "expected_seconds",
        "pause_seconds",
        "comma_pause_count",
        "sentence_pause_count",
        "budget_seconds",
        "budget_stop_after_chunk_count",
        "budget_ready",
        "near_budget_ready",
        "strong_coverage_ready",
        "suffix_completion_ready",
        "endpoint_completion_ready",
        "budget_content_candidate",
        "budget_content_ready",
        "budget_content_progress_gain",
        "budget_content_coverage_gain",
        "budget_min_coverage",
        "chunks_available_at_decision",
        "tail_seconds_after_done",
        "tail_chunks_after_done",
        "tail_blocks_after_done",
        "stop_after_chunk_count",
        "stop_after_block_count",
        "recommended_stop_after_seconds",
        "observation_source",
        "coverage",
        "sequence_ratio",
        "transcript_chars",
        "transcript_target_length_ratio",
        "target_progress",
        "target_remaining_chars",
        "tail_covered",
        "tail_fuzzy_covered",
        "tail_coverage",
        "tail_suffix_ratio",
        "tail_end_ratio",
        "tail_suffix_aligned",
        "tail_audio_available",
        "tail_audio_dbfs",
        "block_audio_dbfs",
        "tail_silence",
        "tail_silence_seconds",
        "tail_silence_threshold_dbfs",
        "profile_ms",
        "error",
    }
    return {key: payload[key] for key in keep if key in payload}


async def _watch_speech_completion(job: JobState) -> None:
    if not SETTINGS.asr_enabled:
        job.speech_completion = {"enabled": False, "completed": None}
        return
    min_chunks = max(
        1,
        int(math.ceil(SETTINGS.asr_min_seconds / max(0.1, SETTINGS.realtime_stream_chunk_seconds))),
    )
    last_checked = 0
    last_check_time = 0.0
    emitted_unavailable = False
    pending_asr_task: Optional[asyncio.Task[Dict[str, Any]]] = None

    async def _publish_payload(payload: Dict[str, Any]) -> None:
        job.speech_completion = payload
        _write_asr_stop_control(job, payload)
        (job.task_dir / "speech_completion.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        await _emit(
            job,
            "speech_completion" if payload.get("completed") else "asr_partial",
            _asr_public_payload(payload),
        )

    while True:
        paths = _asr_observation_paths(job)
        chunk_count = len(paths)
        budget_timing = _speech_budget_timing(_speech_completion_target(job), chunk_count)
        budget_stop_after_chunk_count = int(budget_timing["budget_stop_after_chunk_count"])
        budget_prewrite_at = max(
            min_chunks,
            budget_stop_after_chunk_count - max(0, int(SETTINGS.asr_budget_prewrite_chunks)),
        )
        if (
            SETTINGS.asr_budget_hard_stop
            and chunk_count >= min_chunks
            and chunk_count >= budget_prewrite_at
        ):
            payload = _budget_stop_payload(job, chunk_count)
            await _publish_payload(payload)
            if pending_asr_task is not None and not pending_asr_task.done():
                pending_asr_task.cancel()
            return

        if pending_asr_task is not None and pending_asr_task.done():
            try:
                payload = pending_asr_task.result()
            except Exception as exc:
                payload = {
                    "enabled": True,
                    "available": False,
                    "completed": None,
                    "chunk_count": last_checked,
                    "seconds_observed": last_checked * max(0.1, SETTINGS.realtime_stream_chunk_seconds),
                    "error": str(exc),
                }
                if emitted_unavailable:
                    pending_asr_task = None
                    if job.generation_done:
                        return
                    await asyncio.sleep(max(0.05, SETTINGS.asr_check_interval_seconds))
                    continue
                emitted_unavailable = True
            pending_asr_task = None
            payload = _confirm_budget_content_plateau(payload, job.speech_completion)
            payload = _align_asr_stop_to_decision_time(job, payload)
            await _publish_payload(payload)
            if payload.get("completed"):
                return

        should_check = (
            chunk_count >= min_chunks
            and chunk_count != last_checked
            and pending_asr_task is None
            and _now() - last_check_time >= max(0.1, SETTINGS.asr_check_interval_seconds)
        )
        if should_check:
            last_checked = chunk_count
            last_check_time = _now()
            pending_asr_task = asyncio.create_task(
                asyncio.to_thread(_run_asr_completion_check, job, list(paths))
            )
        if job.generation_done:
            if chunk_count == last_checked or chunk_count < min_chunks:
                if not job.speech_completion:
                    job.speech_completion = {
                        "enabled": True,
                        "available": True,
                        "completed": False,
                        "chunk_count": chunk_count,
                        "seconds_observed": chunk_count * max(0.1, SETTINGS.realtime_stream_chunk_seconds),
                    }
                return
        if job.status in {"failed", "canceled"} or job.cancel_requested:
            if pending_asr_task is not None and not pending_asr_task.done():
                pending_asr_task.cancel()
            return
        await asyncio.sleep(
            max(
                0.02,
                min(
                    SETTINGS.asr_check_interval_seconds,
                    SETTINGS.worker_status_poll_interval,
                ),
            )
        )


def _gpu_process_snapshot() -> Dict[str, Any]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": str(exc), "processes": []}
    if proc.returncode != 0:
        return {
            "available": False,
            "error": proc.stderr.strip()[-1000:],
            "processes": [],
        }
    processes = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            processes.append(
                {
                    "gpu": int(parts[0]),
                    "type": parts[2],
                    "sm": int(parts[3]),
                    "mem": int(parts[4]),
                }
            )
        except ValueError:
            continue
    busy = [p for p in processes if p["sm"] > 10 or p["mem"] > 10]
    return {
        "available": True,
        "busy": bool(busy),
        "busy_count": len(busy),
        "processes": processes,
    }


_GPU_SNAPSHOT_CACHE: Dict[str, Any] = {
    "time": 0.0,
    "value": {"available": False, "error": "not sampled yet", "processes": []},
}
_GPU_SNAPSHOT_TTL_SECONDS = 3.0


async def _gpu_process_snapshot_cached() -> Dict[str, Any]:
    """Return a UI GPU snapshot without blocking the FastAPI event loop."""
    now = _now()
    cached_time = float(_GPU_SNAPSHOT_CACHE.get("time") or 0.0)
    cached_value = _GPU_SNAPSHOT_CACHE.get("value")
    if isinstance(cached_value, dict) and now - cached_time < _GPU_SNAPSHOT_TTL_SECONDS:
        return cached_value
    snapshot = await asyncio.to_thread(_gpu_process_snapshot)
    _GPU_SNAPSHOT_CACHE["time"] = _now()
    _GPU_SNAPSHOT_CACHE["value"] = snapshot
    return snapshot


def _conversation_dir(conversation_id: str) -> Path:
    safe_id = _safe_token(conversation_id, prefix="conv")
    return SETTINGS.runs_root / "_sessions" / safe_id


def _conversation_state_path(conversation_id: str) -> Path:
    return _conversation_dir(conversation_id) / "prefix_state.pt"


def _conversation_meta_path(conversation_id: str) -> Path:
    return _conversation_dir(conversation_id) / "meta.json"


def _spoken_parts_from_segments(segments: List[Dict[str, Any]]) -> List[str]:
    return [
        str(segment.get("speech") or "").strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("speech") or "").strip()
    ]


def _last_spoken_part_from_segments(segments: List[Dict[str, Any]]) -> str:
    parts = _spoken_parts_from_segments(segments)
    if not parts:
        return ""
    spoken = _speech_for_prompt(parts, max_chars=800)
    return _last_speech_segment_for_prompt(
        spoken,
        max_chars=max(1, int(SETTINGS.interactive_prompt_tail_chars)),
    )


def _merge_spoken_history(*parts: str, max_chars: int = 800) -> str:
    return _speech_for_prompt([part for part in parts if str(part or "").strip()], max_chars=max_chars)


def _write_conversation_meta(conversation_id: str, meta: Dict[str, Any]) -> None:
    meta_path = _conversation_meta_path(conversation_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_meta = meta_path.with_suffix(".json.tmp")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_meta, meta_path)


_PENDING_CONVERSATION_META_KEYS = (
    "pending_job_id",
    "pending_spoken_history",
    "pending_last_spoken_segment",
)


def _clear_pending_conversation_meta(
    meta: Dict[str, Any],
    *,
    job_id: Optional[str] = None,
) -> Dict[str, Any]:
    cleaned = dict(meta)
    pending_job_id = str(cleaned.get("pending_job_id") or "")
    if job_id and pending_job_id and pending_job_id != str(job_id):
        return cleaned
    for key in _PENDING_CONVERSATION_META_KEYS:
        cleaned.pop(key, None)
    return cleaned


def _inference_argv(job: JobState) -> List[str]:
    argv = [
        "--model_ckpt",
        SETTINGS.model_ckpt,
        "--original_ckpt",
        SETTINGS.original_ckpt,
        "--gemma_path",
        SETTINGS.gemma_path,
        "--benchmark_json",
        str(job.prompts_file),
        "--prompt_cache_path",
        str(job.task_dir / "prompt_cache.pt"),
        "--output_dir",
        str(job.output_dir),
        "--video_height",
        str(job.internal_video_height or job.video_height),
        "--video_width",
        str(job.internal_video_width or job.video_width),
        "--interactive_stop_control_path",
        str(job.task_dir / "asr_stop_control.json"),
    ]
    if (job.internal_video_height or job.video_height) != job.video_height:
        argv.extend(["--output_video_height", str(job.video_height)])
    if (job.internal_video_width or job.video_width) != job.video_width:
        argv.extend(["--output_video_width", str(job.video_width)])
    if job.prefix_state_in:
        argv.extend(["--interactive_prefix_state_in", job.prefix_state_in])
    if job.prefix_state_out:
        argv.extend(["--interactive_prefix_state_out", job.prefix_state_out])
    return argv



def _worker_request_path(job: JobState) -> Path:
    return SETTINGS.worker_queue_dir / "pending" / f"{job.task_id}.json"


def _worker_model_loaded(worker_heartbeat: Optional[Dict[str, Any]]) -> bool:
    if not worker_heartbeat:
        return False
    residency = str(worker_heartbeat.get("model_residency") or "")
    return residency in {"resident_runtime_cache_loaded", "loaded", "warm"}


async def _enqueue_worker_warmup() -> Optional[Path]:
    """Ask the persistent worker to load the exact runtime used by web jobs."""
    queue_dir = SETTINGS.worker_queue_dir
    pending_dir = queue_dir / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("running", "done", "failed"):
        (queue_dir / subdir).mkdir(parents=True, exist_ok=True)

    task_id = f"warmup_{_utcish_stamp()}"
    task_dir = SETTINGS.runs_root / "_warmup" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    warmup_job = JobState(
        task_id=task_id,
        task_dir=task_dir,
        scene_description="",
        user_text="",
        mode="t2av",
        aspect_ratio="landscape",
        video_width=ASPECT_PRESETS["landscape"]["video_width"],
        video_height=ASPECT_PRESETS["landscape"]["video_height"],
        internal_video_width=SETTINGS.internal_video_width or ASPECT_PRESETS["landscape"]["video_width"],
        internal_video_height=SETTINGS.internal_video_height or ASPECT_PRESETS["landscape"]["video_height"],
        template_id="",
    )
    if SETTINGS.warmup_generate_on_start:
        warmup_prompt = (
            "Summary: A polished interactive digital-human warmup clip begins in a vivid cinematic live-studio desk scene.\n"
            "Narration 1:\n"
            "eye-level tight medium close-up shot. a vivid live-demo studio with a matte ivory desk edge, warm practical lamps, translucent glass shelves, soft blue accent lights, a few green plants, and subtle reflections on the back wall. Bright balanced lighting, vivid but natural colors, soft portrait contrast, delicate catchlights in the eyes, rich layered background details, shallow depth of field, clean digital-human demo look. \n"
            "Speaker_1's Appearance: Young male digital-human presenter with warm fair skin, neat short black hair, tidy eyebrows, a soft friendly smile, and a cream knit jacket over a light blue shirt.\n"
            "Speaker_1's Actions: Speaker_1 faces the camera directly, keeps steady eye contact, makes a tiny nod, and lets the right hand lift slowly near chest level while speaking. The torso stays centered, the shoulders stay relaxed, and the movement feels smooth and conversational. \n"
            "Speaker_1's Facial Expression: calm, welcoming, and attentive.\n"
            "Speaker_1's Held Objects:\nNone\n"
            "Speech Attribution:\nSpeaker_1 says: \"你好啊，我在。\"\n"
            "Speaker_1's Emotion: calm, bright, and helpful.\n"
            "Speaker_1's Voice Description: natural Mandarin voice, clear articulation, slow conversational pacing, close and stable recording quality.\n"
            "Sound-Visual Alignment: Speech is synchronized with actions and lip movements. Mouth motion stays gentle and matches the exact words. The room tone is quiet and stable."
        )
        warmup_job.prompts_file.write_text(
            json.dumps(
                [
                    {
                        "case_id": task_id,
                        "description": "interactive avatar resident runtime warmup",
                        "seed": 20260703,
                        "segments": [
                            {
                                "segment_id": 0,
                                "prompt": warmup_prompt,
                                "seed": 20260703,
                            }
                        ],
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        warmup_stop_control = {
            "enabled": True,
            "available": True,
            "completed": True,
            "completion_source": "warmup_prewrite",
            "target": "你好啊，我在。",
            "transcript": "",
            "chunk_count": 0,
            "seconds_observed": 0.0,
            "tail_seconds_after_done": 0.0,
            "tail_chunks_after_done": 0,
            "tail_blocks_after_done": 0,
            "stop_after_chunk_count": 2,
            "stop_after_block_count": 2,
            "prewritten": True,
            "updated_at": _now(),
        }
        (task_dir / "asr_stop_control.json").write_text(
            json.dumps(warmup_stop_control, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (task_dir / "speech_completion.json").write_text(
            json.dumps(warmup_stop_control, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    request = {
        "task_id": task_id,
        "warmup": True,
        "warmup_generate": SETTINGS.warmup_generate_on_start,
        "prompt_file": str(warmup_job.prompts_file),
        "output_dir": str(warmup_job.output_dir),
        "status_file": str(task_dir / "worker_status.json"),
        "argv": _inference_argv(warmup_job),
        "aspect_ratio": warmup_job.aspect_ratio,
        "video_width": warmup_job.video_width,
        "video_height": warmup_job.video_height,
        "internal_video_width": warmup_job.internal_video_width or warmup_job.video_width,
        "internal_video_height": warmup_job.internal_video_height or warmup_job.video_height,
        "created_at": _now(),
    }
    request_path = pending_dir / f"{task_id}.json"
    tmp_path = request_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, request_path)
    return request_path


async def _run_model_persistent_worker(job: JobState) -> None:
    queue_dir = SETTINGS.worker_queue_dir
    pending_dir = queue_dir / "pending"
    status_file = job.task_dir / "worker_status.json"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("running", "done", "failed"):
        (queue_dir / subdir).mkdir(parents=True, exist_ok=True)

    request = {
        "task_id": job.task_id,
        "prompt_file": str(job.prompts_file),
        "output_dir": str(job.output_dir),
        "status_file": str(status_file),
        "argv": _inference_argv(job),
        "aspect_ratio": job.aspect_ratio,
        "prompt_aspect_ratio": _job_prompt_aspect_ratio(job),
        "video_width": job.video_width,
        "video_height": job.video_height,
        "internal_video_width": job.internal_video_width or job.video_width,
        "internal_video_height": job.internal_video_height or job.video_height,
        "template_id": job.template_id,
        "conversation_id": job.conversation_id,
        "previous_job_id": job.previous_job_id,
        "prefix_state_in": job.prefix_state_in,
        "prefix_state_out": job.prefix_state_out,
        "created_at": _now(),
    }
    request_path = _worker_request_path(job)
    tmp_path = request_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, request_path)

    (job.task_dir / "runner.env").write_text(
        "\n".join(
            [
                f"RUNNER_MODE=persistent_worker",
                f"WORKER_QUEUE_DIR={queue_dir}",
                f"REQUEST_PATH={request_path}",
                f"MODEL_CKPT={SETTINGS.model_ckpt}",
                f"PROMPT_FILE={job.prompts_file}",
                f"OUTPUT_DIR={job.output_dir}",
                f"ASPECT_RATIO={job.aspect_ratio}",
                f"PROMPT_ASPECT_RATIO={_job_prompt_aspect_ratio(job)}",
                f"VIDEO_WIDTH={job.video_width}",
                f"VIDEO_HEIGHT={job.video_height}",
                f"INTERNAL_VIDEO_WIDTH={job.internal_video_width or job.video_width}",
                f"INTERNAL_VIDEO_HEIGHT={job.internal_video_height or job.video_height}",
                f"TEMPLATE_ID={job.template_id}",
                f"CONVERSATION_ID={job.conversation_id}",
                f"PREFIX_STATE_IN={job.prefix_state_in or ''}",
                f"PREFIX_STATE_OUT={job.prefix_state_out or ''}",
                "ARGV=" + " ".join(request["argv"]),
            ]
        ) + "\n",
        encoding="utf-8",
    )
    await _emit(
        job,
        "stage",
        {
            "phase": "queued_for_persistent_worker",
            "queue_dir": str(queue_dir),
            "request": str(request_path),
        },
    )

    last_status = None
    started = _now()
    while True:
        if job.cancel_requested:
            cancel_file = job.task_dir / "cancel.requested"
            cancel_file.write_text(str(_now()), encoding="utf-8")
            raise RuntimeError("job canceled while waiting for persistent worker")
        await _discover_videos(job)
        if status_file.exists():
            try:
                status = json.loads(status_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                status = {}
            status_key = json.dumps(status, sort_keys=True, ensure_ascii=False)
            if status_key != last_status:
                last_status = status_key
                phase = status.get("phase") or status.get("status") or "worker"
                job.phase = phase
                await _emit(job, "runner_log", {"phase": phase, "line": status_key})
            if status.get("status") == "succeeded":
                await _discover_videos(job)
                return
            if status.get("status") == "failed":
                raise RuntimeError(status.get("error") or f"persistent worker failed: {status}")
        if _now() - started > SETTINGS.runner_timeout_seconds:
            raise RuntimeError(
                f"persistent worker timeout after {SETTINGS.runner_timeout_seconds}s"
            )
        await asyncio.sleep(max(0.02, SETTINGS.worker_status_poll_interval))


async def _run_model(job: JobState) -> None:
    return await _run_model_persistent_worker(job)


async def _process_job(job: JobState) -> None:
    first_frame_path = None
    conversation_meta: Dict[str, Any] = {}
    request_file = job.task_dir / "request.json"
    if request_file.exists():
        try:
            request_data = json.loads(request_file.read_text(encoding="utf-8"))
            first_frame_raw = request_data.get("first_frame")
            if first_frame_raw and Path(first_frame_raw).is_file():
                first_frame_path = Path(first_frame_raw)
        except (OSError, json.JSONDecodeError):
            first_frame_path = None
    try:
        if job.cancel_requested:
            raise RuntimeError("job canceled before processing")
        job.status = "running"
        job.phase = "llm_replying"
        await _emit(job, "stage", {"phase": job.phase})
        conversation_meta = (
            _read_optional_json(_conversation_meta_path(job.conversation_id))
            if job.conversation_id else None
        ) or {}
        if str(conversation_meta.get("scene_signature") or "") != str(job.scene_signature or ""):
            conversation_meta = {}
        conversation_history = str(conversation_meta.get("spoken_history") or "")
        conversation_tail_segment = str(conversation_meta.get("last_spoken_segment") or "")
        prompt_aspect_ratio = _job_prompt_aspect_ratio(job)
        scene_prompt_signature = _job_scene_prompt_signature(job)
        planning_template_id = "" if job.refine_scene else job.template_id
        turn_plan = await generate_turn_plan(
            job.scene_description,
            job.user_text,
            conversation_history=conversation_history,
            template_id=planning_template_id,
        )
        job.reply = turn_plan.speech
        job.response_language = turn_plan.language
        (job.task_dir / "reply.json").write_text(
            json.dumps(
                {
                    "reply": turn_plan.speech,
                    "language": turn_plan.language,
                    "emotion": turn_plan.emotion,
                    "action": turn_plan.action,
                    "explicit_action": turn_plan.explicit_action,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        await _emit(
            job,
            "llm_reply",
            {"reply": job.reply, "language": job.response_language},
        )

        job.phase = "prompt_expanding"
        scene_prompt = await _resolve_scene_prompt_text(
            job.scene_description,
            template_id=job.template_id,
            aspect_ratio=prompt_aspect_ratio,
            has_first_frame=first_frame_path is not None,
            refine_scene=job.refine_scene,
            conversation_meta=conversation_meta,
            scene_signature=scene_prompt_signature,
            task_dir=job.task_dir,
        )
        job.scene_prompt_text = str(scene_prompt.get("scene_prompt_text") or job.scene_description)
        job.scene_prompt_source = str(scene_prompt.get("scene_prompt_source") or "unknown")
        await _emit(
            job,
            "scene_prompt_pe",
            {
                "scene_prompt_source": job.scene_prompt_source,
                "scene_prompt_signature": scene_prompt_signature,
                "prompt_aspect_ratio": prompt_aspect_ratio,
            },
        )
        job.segments = await plan_segments(
            job.scene_prompt_text,
            job.reply,
            has_first_frame=first_frame_path is not None,
            max_segments=min(SETTINGS.max_segments, 8),
            add_waiting_transition=SETTINGS.add_waiting_transition,
            template_id=planning_template_id,
            aspect_ratio=prompt_aspect_ratio,
            prior_spoken_context=conversation_history,
            prior_spoken_tail_segment=conversation_tail_segment,
            turn_action=turn_plan.action,
            turn_emotion=turn_plan.emotion,
            explicit_action=turn_plan.explicit_action,
        )
        (job.task_dir / "segments.json").write_text(
            json.dumps(job.segments, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_prompt_file(job, first_frame_path)
        _log_model_prompts(job)
        _prewrite_budget_stop_control(job)
        await _emit(job, "prompt_ready", {"segments": job.segments})
        if job.conversation_id:
            pending_spoken_history = _merge_spoken_history(
                conversation_history,
                *_spoken_parts_from_segments(job.segments),
                max_chars=800,
            )
            pending_meta = {
                **conversation_meta,
                "conversation_id": job.conversation_id,
                "pending_job_id": job.task_id,
                "pending_spoken_history": pending_spoken_history,
                "pending_last_spoken_segment": _last_spoken_part_from_segments(job.segments),
                "video_width": job.video_width,
                "video_height": job.video_height,
                "internal_video_width": job.internal_video_width or job.video_width,
                "internal_video_height": job.internal_video_height or job.video_height,
                "aspect_ratio": job.aspect_ratio,
                "prompt_aspect_ratio": prompt_aspect_ratio,
                "template_id": job.template_id,
                "scene_signature": job.scene_signature,
                "scene_description": job.scene_description,
                "scene_prompt_signature": scene_prompt_signature,
                "scene_prompt_source": job.scene_prompt_source,
                "scene_prompt_text": job.scene_prompt_text,
                "model_ckpt": SETTINGS.model_ckpt,
                "updated_at": _now(),
            }
            if not SETTINGS.interactive_video_prefix_enabled:
                pending_meta.pop("prefix_state_path", None)
            _write_conversation_meta(job.conversation_id, pending_meta)

        job.phase = "queued_on_gpu"
        await _emit(job, "stage", {"phase": job.phase})
        job.generation_done = False
        asr_task: Optional[asyncio.Task[None]] = None
        if SETTINGS.asr_enabled:
            asr_task = asyncio.create_task(_watch_speech_completion(job))
        try:
            await _run_model(job)
        finally:
            job.generation_done = True
            if asr_task is not None:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        asr_task,
                        timeout=max(0.1, SETTINGS.asr_final_timeout_seconds),
                    )
                if not asr_task.done():
                    asr_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await asr_task
        await _discover_videos(job)
        if not job.videos:
            raise RuntimeError(
                "generation completed without playable video assets; "
                "inspect realtime stream decode and mux logs"
            )
        prefix_state_ready = bool(
            SETTINGS.interactive_video_prefix_enabled
            and job.conversation_id
            and job.prefix_state_out
            and Path(job.prefix_state_out).exists()
        )
        if job.conversation_id:
            latest_meta = _read_optional_json(_conversation_meta_path(job.conversation_id)) or conversation_meta
            spoken_history = _merge_spoken_history(
                str(latest_meta.get("spoken_history") or conversation_history),
                *_spoken_parts_from_segments(job.segments),
                max_chars=800,
            )
            meta = {
                **latest_meta,
                "conversation_id": job.conversation_id,
                "last_job_id": job.task_id,
                "spoken_history": spoken_history,
                "last_spoken_segment": _last_spoken_part_from_segments(job.segments),
                "turn_count": int(latest_meta.get("turn_count") or conversation_meta.get("turn_count") or 0) + 1,
                "video_width": job.video_width,
                "video_height": job.video_height,
                "internal_video_width": job.internal_video_width or job.video_width,
                "internal_video_height": job.internal_video_height or job.video_height,
                "aspect_ratio": job.aspect_ratio,
                "prompt_aspect_ratio": prompt_aspect_ratio,
                "template_id": job.template_id,
                "scene_signature": job.scene_signature,
                "scene_description": job.scene_description,
                "scene_prompt_signature": scene_prompt_signature,
                "scene_prompt_source": job.scene_prompt_source,
                "scene_prompt_text": job.scene_prompt_text,
                "model_ckpt": SETTINGS.model_ckpt,
                "updated_at": _now(),
            }
            if prefix_state_ready:
                meta["prefix_state_path"] = job.prefix_state_out
            else:
                meta.pop("prefix_state_path", None)
            meta = _clear_pending_conversation_meta(meta, job_id=job.task_id)
            _write_conversation_meta(job.conversation_id, meta)
            if prefix_state_ready:
                await _emit(
                    job,
                    "continuation_state",
                    {
                        "conversation_id": job.conversation_id,
                        "prefix_state": job.prefix_state_out,
                        "previous_job_id": job.previous_job_id,
                    },
                )

        job.status = "succeeded"
        job.phase = "succeeded"
        await _emit(job, "done", {"status": job.status, "videos": job.videos})
    except Exception as exc:
        job.status = "canceled" if job.cancel_requested else "failed"
        job.phase = job.status
        job.error = str(exc)
        if job.conversation_id:
            try:
                latest_meta = _read_optional_json(
                    _conversation_meta_path(job.conversation_id)
                ) or {}
                if str(latest_meta.get("pending_job_id") or "") == job.task_id:
                    latest_meta = _clear_pending_conversation_meta(
                        latest_meta,
                        job_id=job.task_id,
                    )
                    latest_meta["updated_at"] = _now()
                    _write_conversation_meta(job.conversation_id, latest_meta)
            except Exception as cleanup_exc:
                print(
                    "[ConversationState][WARN] failed to roll back pending "
                    f"state for {job.task_id}: {cleanup_exc}",
                    flush=True,
                )
        await _emit(job, "error", {"message": job.error})
    finally:
        _persist_status(job)


async def _worker_loop() -> None:
    while True:
        task_id = await QUEUE.get()
        job = JOBS.get(task_id)
        if job is None:
            QUEUE.task_done()
            continue
        try:
            await _process_job(job)
        finally:
            QUEUE.task_done()


def _load_persisted_jobs(limit: int = 50) -> None:
    """Recover recent job metadata so media/status survive API restarts."""
    if not SETTINGS.runs_root.exists():
        return
    status_files = sorted(
        SETTINGS.runs_root.glob("*/tsk_*/status.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:limit]
    for status_file in status_files:
        task_dir = status_file.parent
        request_file = task_dir / "request.json"
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            request = (
                json.loads(request_file.read_text(encoding="utf-8"))
                if request_file.exists()
                else {}
            )
        except (OSError, json.JSONDecodeError):
            continue
        task_id = status.get("task_id") or request.get("task_id") or task_dir.name
        if task_id in JOBS:
            continue
        scene_description = request.get("scene_description") or status.get("scene_description") or ""
        aspect_ratio = request.get("aspect_ratio") or status.get("aspect_ratio") or "landscape"
        template_id = request.get("template_id") or status.get("template_id") or ""
        refine_scene = bool(request.get("refine_scene") or status.get("refine_scene"))
        job = JobState(
            task_id=task_id,
            task_dir=task_dir,
            scene_description=scene_description,
            user_text=request.get("user_text") or status.get("user_text") or "",
            mode=request.get("mode") or status.get("mode") or "t2av",
            conversation_id=request.get("conversation_id") or status.get("conversation_id") or "",
            previous_job_id=request.get("previous_job_id") or status.get("previous_job_id"),
            prefix_state_in=request.get("prefix_state_in") or status.get("prefix_state_in"),
            prefix_state_out=request.get("prefix_state_out") or status.get("prefix_state_out"),
            aspect_ratio=aspect_ratio,
            video_width=int(request.get("video_width") or status.get("video_width") or SETTINGS.video_width),
            video_height=int(request.get("video_height") or status.get("video_height") or SETTINGS.video_height),
            internal_video_width=int(
                request.get("internal_video_width")
                or status.get("internal_video_width")
                or request.get("video_width")
                or status.get("video_width")
                or SETTINGS.internal_video_width
                or SETTINGS.video_width
            ),
            internal_video_height=int(
                request.get("internal_video_height")
                or status.get("internal_video_height")
                or request.get("video_height")
                or status.get("video_height")
                or SETTINGS.internal_video_height
                or SETTINGS.video_height
            ),
            template_id=template_id,
            refine_scene=refine_scene,
            scene_signature=(
                request.get("scene_signature")
                or status.get("scene_signature")
                or _scene_signature(
                    scene_description,
                    template_id=template_id,
                    aspect_ratio=aspect_ratio,
                    refine_scene=refine_scene,
                )
            ),
            created_at=float(status.get("created_at") or task_dir.stat().st_mtime),
            updated_at=float(status.get("updated_at") or status_file.stat().st_mtime),
            status=status.get("status") or "failed",
            phase=status.get("phase") or "restored",
            reply=status.get("reply"),
            response_language=status.get("response_language") or "",
            segments=status.get("segments") or [],
            videos=status.get("videos") or [],
            error=status.get("error"),
        )
        if job.status in {"accepted", "running"}:
            job.status = "failed"
            job.phase = "interrupted"
            job.error = job.error or "API process restarted before this job finished."
            _persist_status(job)
        JOBS[task_id] = job


@app.on_event("startup")
async def _startup() -> None:
    SETTINGS.runs_root.mkdir(parents=True, exist_ok=True)
    _load_persisted_jobs()
    if SETTINGS.asr_enabled and SETTINGS.asr_preload_on_start:
        try:
            await asyncio.to_thread(_load_asr_model)
            print(
                f"[ASR] preloaded model={SETTINGS.asr_model} "
                f"path={SETTINGS.asr_model_path or '<auto>'} device={SETTINGS.asr_device}",
                flush=True,
            )
        except Exception as exc:
            print(f"[ASR] preload failed: {exc}", flush=True)
            raise RuntimeError("ASR preload failed") from exc
    global WORKER_TASK
    if WORKER_TASK is None or WORKER_TASK.done():
        WORKER_TASK = asyncio.create_task(_worker_loop())
    if SETTINGS.preload_worker_on_start:
        await _enqueue_worker_warmup()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.head("/")
async def index_head() -> Response:
    return Response(status_code=200)


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    return {"ok": True, "time": _now()}


@app.get("/api/health")
async def api_health() -> Dict[str, Any]:
    return await healthz()


def _dialogue_service_ready() -> bool:
    request = urlrequest.Request(
        f"{SETTINGS.llm_base_url}/models",
        headers={"Authorization": f"Bearer {SETTINGS.llm_api_key}"},
    )
    try:
        with urlrequest.urlopen(request, timeout=2) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urlerror.URLError):
        return False


async def _require_dialogue_service_ready() -> None:
    if not await asyncio.to_thread(_dialogue_service_ready):
        raise HTTPException(
            status_code=503,
            detail="The dialogue model is still starting. Retry after the service is ready.",
        )


@app.get("/readyz")
async def readyz() -> Dict[str, Any]:
    missing: List[str] = []
    if not SETTINGS.model_ckpt or not Path(SETTINGS.model_ckpt).is_file():
        missing.append("model_checkpoint")
    if not SETTINGS.original_ckpt or not Path(SETTINGS.original_ckpt).is_file():
        missing.append("base_model_checkpoint")
    if not SETTINGS.gemma_path or not Path(SETTINGS.gemma_path).is_dir():
        missing.append("text_encoder")

    dialogue_ready = await asyncio.to_thread(_dialogue_service_ready)
    if not dialogue_ready:
        missing.append("dialogue_service")

    asr_ready = not SETTINGS.asr_enabled or _ASR_MODEL is not None
    if not asr_ready:
        missing.append("asr_model")

    heartbeat = _read_optional_json(
        SETTINGS.worker_queue_dir / "worker_heartbeat.json"
    )
    worker_model_loaded = _worker_model_loaded(heartbeat)
    worker_lag_seconds = None
    worker_ready = False
    if heartbeat and isinstance(heartbeat.get("time"), (int, float)):
        worker_lag_seconds = max(0.0, _now() - float(heartbeat["time"]))
        worker_ready = worker_lag_seconds <= 45.0 and worker_model_loaded
    if not worker_ready:
        missing.append("resident_worker")

    return {
        "ready": not missing,
        "missing": missing,
        "dialogue_ready": dialogue_ready,
        "asr_ready": asr_ready,
        "worker_ready": worker_ready,
        "worker_model_loaded": worker_model_loaded,
        "worker_lag_seconds": worker_lag_seconds,
    }


@app.get("/api/system")
async def system() -> Dict[str, Any]:
    gpu = await _gpu_process_snapshot_cached()
    heartbeat = _read_optional_json(
        SETTINGS.worker_queue_dir / "worker_heartbeat.json"
    )
    active_jobs = [
        _job_snapshot(job)
        for job in JOBS.values()
        if job.status in {"accepted", "running"}
    ]
    return {
        "time": _now(),
        "gpu": _public_gpu_snapshot(gpu),
        "queue_depth": QUEUE.qsize(),
        "worker_heartbeat": _public_worker_heartbeat(heartbeat),
        "active_jobs": [
            {
                "task_id": job.get("task_id"),
                "status": job.get("status"),
                "phase": job.get("phase"),
                "created_at": job.get("created_at"),
                "updated_at": job.get("updated_at"),
            }
            for job in active_jobs
        ],
    }


@app.post("/api/preview")
async def preview_plan(request: PreviewRequest) -> Dict[str, Any]:
    scene = request.scene_description.strip()
    text = request.user_text.strip()
    requested_mode = request.mode.strip().lower()
    aspect_key, video_width, video_height = _resolve_aspect_ratio(request.aspect_ratio)
    configured_internal_width = (
        SETTINGS.portrait_internal_video_width
        if aspect_key == "portrait"
        else SETTINGS.internal_video_width
    )
    configured_internal_height = (
        SETTINGS.portrait_internal_video_height
        if aspect_key == "portrait"
        else SETTINGS.internal_video_height
    )
    internal_video_width, internal_video_height = _resolve_internal_video_size(
        video_width,
        video_height,
        configured_width=configured_internal_width,
        configured_height=configured_internal_height,
        allow_cover_resize=aspect_key == "portrait",
    )
    prompt_aspect_ratio = _generation_prompt_aspect_ratio(
        aspect_key,
        internal_video_width,
        internal_video_height,
    )
    if requested_mode not in {"auto", "t2av", "i2av"}:
        raise HTTPException(status_code=400, detail="mode must be auto, t2av, or i2av")
    if not scene:
        raise HTTPException(status_code=400, detail="scene_description is required")
    if not text:
        raise HTTPException(status_code=400, detail="user_text is required")
    await _require_dialogue_service_ready()
    effective_mode = (
        "i2av"
        if requested_mode == "i2av" or (requested_mode == "auto" and request.has_first_frame)
        else "t2av"
    )
    safe_preview_conversation_id = (
        _safe_token(request.conversation_id, prefix="conv")
        if str(request.conversation_id or "").strip()
        else ""
    )
    conversation_meta = (
        _read_optional_json(_conversation_meta_path(safe_preview_conversation_id))
        if safe_preview_conversation_id
        else None
    ) or {}
    preview_scene_signature = _scene_signature(
        scene,
        template_id=request.template_id,
        aspect_ratio=aspect_key,
        refine_scene=request.refine_scene,
    )
    preview_scene_prompt_signature = (
        f"{preview_scene_signature}:{prompt_aspect_ratio}"
    )
    if str(conversation_meta.get("scene_signature") or "") != preview_scene_signature:
        conversation_meta = {}
    conversation_history = str(conversation_meta.get("spoken_history") or "")
    conversation_tail_segment = str(conversation_meta.get("last_spoken_segment") or "")
    planning_template_id = "" if request.refine_scene else request.template_id
    turn_plan = await generate_turn_plan(
        scene,
        text,
        conversation_history=conversation_history,
        template_id=planning_template_id,
    )
    reply = turn_plan.speech
    scene_prompt = await _resolve_scene_prompt_text(
        scene,
        template_id=request.template_id,
        aspect_ratio=prompt_aspect_ratio,
        has_first_frame=bool(request.has_first_frame or effective_mode == "i2av"),
        refine_scene=request.refine_scene,
        conversation_meta=conversation_meta,
        scene_signature=preview_scene_prompt_signature,
        task_dir=None,
    )
    scene_prompt_text = str(scene_prompt.get("scene_prompt_text") or scene)
    segments = await plan_segments(
        scene_prompt_text,
        reply,
        has_first_frame=bool(request.has_first_frame or effective_mode == "i2av"),
        max_segments=min(SETTINGS.max_segments, 8),
        add_waiting_transition=SETTINGS.add_waiting_transition,
        template_id=planning_template_id,
        aspect_ratio=prompt_aspect_ratio,
        prior_spoken_context=conversation_history,
        prior_spoken_tail_segment=conversation_tail_segment,
        turn_action=turn_plan.action,
        turn_emotion=turn_plan.emotion,
        explicit_action=turn_plan.explicit_action,
    )
    return {
        "mode": effective_mode,
        "reply": reply,
        "response_language": turn_plan.language,
        "segments": segments,
        "aspect_ratio": aspect_key,
        "prompt_aspect_ratio": prompt_aspect_ratio,
        "video_width": video_width,
        "video_height": video_height,
        "internal_video_width": internal_video_width,
        "internal_video_height": internal_video_height,
        "template_id": request.template_id,
        "conversation_id": safe_preview_conversation_id,
        "previous_job_id": str(conversation_meta.get("last_job_id") or "") or None,
        "scene_prompt_source": scene_prompt.get("scene_prompt_source"),
        "scene_prompt_text": scene_prompt_text,
        "segment_seconds": SETTINGS.segment_seconds,
        "speech_segment_visible_chars": SETTINGS.speech_segment_visible_chars,
        "num_frame_per_block": SETTINGS.num_frame_per_block,
        "num_frame_per_block_first": SETTINGS.num_frame_per_block_first,
        "add_waiting_transition": SETTINGS.add_waiting_transition,
        "prompt_planner_profile": PROMPT_PLANNER_PROFILE,
        "prompt_planner_good_case_count": len(PROMPT_PLANNER_EXAMPLES),
        "uses_gpu": False,
    }


@app.post("/api/jobs")
async def create_job(
    scene_description: str = Form(...),
    user_text: str = Form(...),
    mode: str = Form("auto"),
    aspect_ratio: str = Form("landscape"),
    template_id: str = Form(""),
    refine_scene: bool = Form(False),
    conversation_id: str = Form(""),
    first_frame: Optional[UploadFile] = File(None),
) -> JSONResponse:
    scene = scene_description.strip()
    text = user_text.strip()
    requested_mode = mode.strip().lower()
    aspect_key, video_width, video_height = _resolve_aspect_ratio(aspect_ratio)
    configured_internal_width = (
        SETTINGS.portrait_internal_video_width
        if aspect_key == "portrait"
        else SETTINGS.internal_video_width
    )
    configured_internal_height = (
        SETTINGS.portrait_internal_video_height
        if aspect_key == "portrait"
        else SETTINGS.internal_video_height
    )
    internal_video_width, internal_video_height = _resolve_internal_video_size(
        video_width,
        video_height,
        configured_width=configured_internal_width,
        configured_height=configured_internal_height,
        allow_cover_resize=aspect_key == "portrait",
    )
    if requested_mode not in {"auto", "t2av", "i2av"}:
        raise HTTPException(status_code=400, detail="mode must be auto, t2av, or i2av")
    if not scene:
        raise HTTPException(status_code=400, detail="scene_description is required")
    if not text:
        raise HTTPException(status_code=400, detail="user_text is required")
    await _require_dialogue_service_ready()
    task_id = f"tsk_{_utcish_stamp()}_{uuid.uuid4().hex[:8]}"
    task_dir = SETTINGS.runs_root / time.strftime("%Y-%m-%d") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    uploaded_first_frame = None
    if first_frame and first_frame.filename:
        uploaded_first_frame = task_dir / f"first_frame{Path(_safe_filename(first_frame.filename)).suffix}"
        with uploaded_first_frame.open("wb") as f:
            shutil.copyfileobj(first_frame.file, f)

    if requested_mode == "i2av" and uploaded_first_frame is None:
        raise HTTPException(status_code=400, detail="i2av mode requires first_frame")
    effective_mode = (
        "i2av"
        if requested_mode == "i2av" or (requested_mode == "auto" and uploaded_first_frame is not None)
        else "t2av"
    )
    first_frame_for_inference = uploaded_first_frame if effective_mode == "i2av" else None
    safe_conversation_id = _safe_token(conversation_id, prefix="conv")
    safe_template_id = template_id.strip()[:80]
    current_scene_signature = _scene_signature(
        scene,
        template_id=safe_template_id,
        aspect_ratio=aspect_key,
        refine_scene=refine_scene,
    )
    conversation_dir = _conversation_dir(safe_conversation_id)
    conversation_dir.mkdir(parents=True, exist_ok=True)
    conversation_meta = _read_optional_json(_conversation_meta_path(safe_conversation_id)) or {}
    state_path = _conversation_state_path(safe_conversation_id)
    state_available = SETTINGS.interactive_video_prefix_enabled and state_path.exists()
    state_compatible = (
        SETTINGS.interactive_video_prefix_enabled
        and state_available
        and int(conversation_meta.get("video_width") or 0) == video_width
        and int(conversation_meta.get("video_height") or 0) == video_height
        and int(conversation_meta.get("internal_video_width") or video_width) == internal_video_width
        and int(conversation_meta.get("internal_video_height") or video_height) == internal_video_height
        and str(conversation_meta.get("template_id") or "") == safe_template_id
        and str(conversation_meta.get("scene_signature") or "") == current_scene_signature
        and str(conversation_meta.get("model_ckpt") or "") == SETTINGS.model_ckpt
    )
    prefix_state_in = (
        str(state_path)
        if SETTINGS.interactive_video_prefix_enabled
        and state_compatible
        and uploaded_first_frame is None
        else None
    )
    previous_job_id = (
        str(conversation_meta.get("last_job_id") or "")
        if prefix_state_in else None
    ) or None

    job = JobState(
        task_id=task_id,
        task_dir=task_dir,
        scene_description=scene,
        user_text=text,
        mode=effective_mode,
        conversation_id=safe_conversation_id,
        previous_job_id=previous_job_id,
        prefix_state_in=prefix_state_in,
        prefix_state_out=(
            str(state_path) if SETTINGS.interactive_video_prefix_enabled else None
        ),
        aspect_ratio=aspect_key,
        video_width=video_width,
        video_height=video_height,
        internal_video_width=internal_video_width,
        internal_video_height=internal_video_height,
        template_id=safe_template_id,
        refine_scene=refine_scene,
        scene_signature=current_scene_signature,
    )
    JOBS[task_id] = job
    (task_dir / "request.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "scene_description": scene,
                "user_text": text,
                "requested_mode": requested_mode,
                "mode": job.mode,
                "conversation_id": job.conversation_id,
                "previous_job_id": job.previous_job_id,
                "prefix_state_in": job.prefix_state_in,
                "prefix_state_out": job.prefix_state_out,
                "aspect_ratio": job.aspect_ratio,
                "prompt_aspect_ratio": _job_prompt_aspect_ratio(job),
                "video_width": job.video_width,
                "video_height": job.video_height,
                "internal_video_width": job.internal_video_width or job.video_width,
                "internal_video_height": job.internal_video_height or job.video_height,
                "template_id": job.template_id,
                "refine_scene": job.refine_scene,
                "scene_signature": job.scene_signature,
                "first_frame": str(first_frame_for_inference) if first_frame_for_inference else None,
                "uploaded_first_frame": str(uploaded_first_frame) if uploaded_first_frame else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _persist_status(job)
    await _emit(job, "accepted", {"status": job.status, "mode": job.mode})
    await QUEUE.put(task_id)
    return JSONResponse(_job_snapshot(job), status_code=202)


@app.get("/api/jobs")
async def list_jobs() -> Dict[str, Any]:
    return {"jobs": [_job_snapshot(job) for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]}


@app.get("/api/jobs/{task_id}")
async def get_job(task_id: str) -> Dict[str, Any]:
    job = JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await _discover_videos(job, emit_new=False)
    return _job_snapshot(job)


@app.delete("/api/jobs/{task_id}")
async def cancel_job(task_id: str) -> Dict[str, Any]:
    job = JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    job.cancel_requested = True
    job.status = "canceled"
    job.phase = "canceled"
    await _emit(job, "canceled", {"status": job.status})
    return _job_snapshot(job)


@app.get("/api/jobs/{task_id}/videos")
async def get_videos(task_id: str) -> Dict[str, Any]:
    job = JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await _discover_videos(job, emit_new=False)
    return {"task_id": task_id, "videos": job.videos}


@app.get("/api/demos")
async def list_demos() -> Dict[str, Any]:
    entries = _read_demo_entries()
    known_ids = {str(item.get("task_id") or item.get("id")) for item in entries}
    fallback = []
    for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True):
        if job.status != "succeeded" or not job.videos or job.task_id in known_ids:
            continue
        demo = _demo_from_job(job, title="最近生成样例")
        if demo:
            fallback.append(demo)
        if len(fallback) >= 2:
            break
    return {"demos": entries + fallback}


@app.post("/api/demos/from_job/{task_id}")
async def register_demo(task_id: str, request: DemoRegisterRequest) -> Dict[str, Any]:
    job = JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    await _discover_videos(job, emit_new=False)
    demo = _demo_from_job(
        job,
        title=request.title,
        caption=request.caption,
        fps_label=request.fps_label,
    )
    if demo is None:
        raise HTTPException(status_code=400, detail="job has no videos yet")
    entries = [
        item for item in _read_demo_entries()
        if str(item.get("task_id") or item.get("id")) != task_id
    ]
    entries.insert(0, demo)
    _write_demo_entries(entries[:8])
    return {"demo": demo, "count": len(entries[:8])}


@app.get("/api/demos/{task_id}/live_events")
async def demo_live_events(
    task_id: str,
    initial_delay: float = 2.0,
    interval: float = 1.0,
) -> StreamingResponse:
    entries = _read_demo_entries()
    demo = next((item for item in entries if _demo_task_id(item) == task_id), None)
    if demo is None:
        fallback_job = JOBS.get(task_id)
        if fallback_job and fallback_job.status == "succeeded":
            await _discover_videos(fallback_job, emit_new=False)
            demo = _demo_from_job(fallback_job, title="最近生成样例")
    if demo is None:
        raise HTTPException(status_code=404, detail="demo not found")
    urls = _demo_stream_urls(demo)
    if not urls:
        raise HTTPException(status_code=400, detail="demo has no playable videos")
    initial_delay = max(0.0, min(float(initial_delay), 5.0))
    interval = max(0.2, min(float(interval), 3.0))

    async def stream() -> AsyncIterator[str]:
        started = _now()
        yield f"data: {json.dumps({'event': 'stage', 'phase': 'demo_live_buffering', 'task_id': task_id, 'delay': initial_delay}, ensure_ascii=False)}\n\n"
        if initial_delay > 0:
            await asyncio.sleep(initial_delay)
        for idx, url in enumerate(urls):
            payload = {
                "event": "asset",
                "kind": "video",
                "url": url,
                "task_id": task_id,
                "index": idx,
                "elapsed": round(_now() - started, 3),
                "source": "demo_reference",
            }
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if idx < len(urls) - 1:
                await asyncio.sleep(interval)
        yield f"data: {json.dumps({'event': 'done', 'task_id': task_id, 'status': 'succeeded', 'source': 'demo_reference'}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'event': 'closed', 'task_id': task_id, 'status': 'succeeded'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/jobs/{task_id}/events")
async def job_events(task_id: str) -> StreamingResponse:
    job = JOBS.get(task_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")

    async def stream() -> AsyncIterator[str]:
        offset = 0
        while True:
            if job.events_file.exists():
                with job.events_file.open("rb") as f:
                    f.seek(offset)
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        offset = f.tell()
                        yield f"data: {line.decode('utf-8', errors='replace').rstrip()}\n\n"
            if job.status in {"succeeded", "failed", "canceled"}:
                yield f"data: {json.dumps({'event': 'closed', 'task_id': task_id, 'status': job.status}, ensure_ascii=False)}\n\n"
                return
            await asyncio.sleep(max(0.02, SETTINGS.sse_poll_interval))

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/media/{task_id}/{relative_path:path}")
async def media(task_id: str, relative_path: str) -> FileResponse:
    job = JOBS.get(task_id)
    if not job:
        # Allow serving files after an API restart if the caller gives a valid
        # date-scoped path through the task id is not supported yet.
        raise HTTPException(status_code=404, detail="job not found")
    target = (job.task_dir / relative_path).resolve()
    root = job.task_dir.resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=403, detail="invalid media path")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="media not found")
    return FileResponse(str(target))
