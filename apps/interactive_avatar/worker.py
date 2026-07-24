#!/usr/bin/env python3
"""Persistent model worker for the interactive avatar service."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taomate.inference.streaming_runtime import parse_args as parse_runtime_args
from taomate.inference.streaming_runtime import run_inference
from taomate.inference.streaming_runtime import warmup_windowed_model_runtime


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _is_rank0() -> bool:
    return _rank() == 0


def _init_control_group(world_size: int):
    """Use a CPU process group for lightweight queue-control messages.

    The main model path keeps using the default NCCL group.  Reusing NCCL for
    idle queue polling is surprisingly expensive on some machines because every
    empty poll still launches GPU communication work.
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return None
    backend = os.environ.get("AVATAR_WORKER_CONTROL_BACKEND", "gloo").strip().lower()
    if not backend or backend in {"none", "default", "nccl"}:
        return None
    try:
        return dist.new_group(backend=backend)
    except Exception as exc:
        if _is_rank0():
            print(
                f"[Worker] Failed to create {backend!r} control group; "
                f"falling back to default process group: {exc}",
                flush=True,
            )
        return None


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _init_queue(root: Path) -> Dict[str, Path]:
    dirs = {
        "pending": root / "pending",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _claim_next(dirs: Dict[str, Path]) -> Optional[Path]:
    for pending in sorted(dirs["pending"].glob("*.json")):
        running = dirs["running"] / pending.name
        try:
            pending.rename(running)
            return running
        except FileNotFoundError:
            continue
        except OSError:
            continue
    return None


def _build_argv(request: Dict[str, Any]) -> List[str]:
    argv = list(request["argv"])
    prompt_file = str(request["prompt_file"])
    output_dir = str(request["output_dir"])
    if "--benchmark_json" not in argv:
        argv.extend(["--benchmark_json", prompt_file])
    if "--output_dir" not in argv:
        argv.extend(["--output_dir", output_dir])
    return argv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive avatar persistent model worker")
    parser.add_argument("--queue_dir", required=True)
    parser.add_argument("--poll_interval", type=float, default=1.0)
    parser.add_argument("--idle_heartbeat_interval", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    control_group = _init_control_group(world_size)

    queue_dir = Path(args.queue_dir).resolve()
    dirs = _init_queue(queue_dir)
    heartbeat_path = queue_dir / "worker_heartbeat.json"
    rank = _rank()

    if _is_rank0():
        _write_json(
            heartbeat_path,
            {
                "status": "started",
                "time": time.time(),
                "rank": rank,
                "world_size": world_size,
                "backend": "streaming",
            },
        )

    last_idle_heartbeat = 0.0
    model_loaded = False
    while True:
        running_path_s = ""
        if _is_rank0():
            claimed = _claim_next(dirs)
            running_path_s = str(claimed) if claimed else ""
            now = time.time()
            if not claimed and now - last_idle_heartbeat >= args.idle_heartbeat_interval:
                _write_json(
                    heartbeat_path,
                    {
                        "status": "idle",
                        "time": now,
                        "queue_dir": str(queue_dir),
                        "model_residency": (
                            "resident_runtime_cache_loaded"
                            if model_loaded
                            else "resident_runtime_cache_not_loaded"
                        ),
                        "backend": "streaming",
                    },
                )
                last_idle_heartbeat = now

        if dist.is_available() and dist.is_initialized():
            obj = [running_path_s]
            if control_group is not None:
                dist.broadcast_object_list(obj, src=0, group=control_group)
            else:
                dist.broadcast_object_list(obj, src=0)
            running_path_s = obj[0]

        if not running_path_s:
            time.sleep(max(0.02, float(args.poll_interval)))
            continue

        running_path = Path(running_path_s)
        request = _read_json(running_path)
        status_file = Path(request["status_file"])
        task_id = request.get("task_id") or running_path.stem
        is_warmup = bool(request.get("warmup", False))

        if _is_rank0():
            _write_json(
                status_file,
                {
                    "task_id": task_id,
                    "status": "running",
                    "phase": "worker_warming_model" if is_warmup else "worker_running",
                    "time": time.time(),
                    "running_path": str(running_path),
                },
            )
            _write_json(
                heartbeat_path,
                {
                    "status": "warming_model" if is_warmup else "running",
                    "task_id": task_id,
                    "time": time.time(),
                    "running_path": str(running_path),
                    "model_residency": (
                        "resident_runtime_cache_loaded"
                        if model_loaded
                        else "resident_runtime_cache_not_loaded"
                    ),
                    "backend": "streaming",
                },
            )

        previous_status_file_env = os.environ.get("AVATAR_WORKER_STATUS_FILE")
        os.environ["AVATAR_WORKER_STATUS_FILE"] = str(status_file)
        try:
            if is_warmup:
                inference_args = parse_runtime_args(_build_argv(request))
                warmup_windowed_model_runtime(inference_args, destroy_process_group=False)
                model_loaded = True
                if bool(request.get("warmup_generate", False)):
                    if _is_rank0():
                        _write_json(
                            heartbeat_path,
                            {
                                "status": "warming_generation",
                                "task_id": task_id,
                                "time": time.time(),
                                "running_path": str(running_path),
                                "model_residency": "resident_runtime_cache_loaded",
                                "backend": "streaming",
                            },
                        )
                    run_inference(inference_args, destroy_process_group=False)
            else:
                inference_args = parse_runtime_args(_build_argv(request))
                run_inference(inference_args, destroy_process_group=False)
                model_loaded = True
        except Exception as exc:
            if _is_rank0():
                tb = traceback.format_exc()
                _write_json(
                    status_file,
                    {
                        "task_id": task_id,
                        "status": "failed",
                        "phase": "worker_failed",
                        "time": time.time(),
                        "error": str(exc),
                        "traceback": tb,
                    },
                )
                failed_path = dirs["failed"] / running_path.name
                try:
                    running_path.rename(failed_path)
                except OSError:
                    pass
            # A rank-local CUDA error can leave peers blocked in recv/send. Let
            # torchrun terminate the full worker group instead of deadlocking
            # on a barrier that the other ranks can never reach.
            raise
        finally:
            if previous_status_file_env is None:
                os.environ.pop("AVATAR_WORKER_STATUS_FILE", None)
            else:
                os.environ["AVATAR_WORKER_STATUS_FILE"] = previous_status_file_env

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        if _is_rank0():
            _write_json(
                status_file,
                {
                    "task_id": task_id,
                    "status": "succeeded",
                    "phase": "worker_warm" if is_warmup else "worker_succeeded",
                    "time": time.time(),
                    "output_dir": request["output_dir"],
                    "model_residency": (
                        "resident_runtime_cache_loaded"
                        if model_loaded
                        else "resident_runtime_cache_not_loaded"
                    ),
                    "backend": "streaming",
                },
            )
            _write_json(
                heartbeat_path,
                {
                    "status": "idle",
                    "time": time.time(),
                    "queue_dir": str(queue_dir),
                    "model_residency": (
                        "resident_runtime_cache_loaded"
                        if model_loaded
                        else "resident_runtime_cache_not_loaded"
                    ),
                    "backend": "streaming",
                },
            )
            done_path = dirs["done"] / running_path.name
            try:
                running_path.rename(done_path)
            except OSError:
                pass


if __name__ == "__main__":
    main()
