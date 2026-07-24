#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from taomate.inference import InferenceConfig, run_inference

def parse_args(argv: Optional[List[str]] = None) -> InferenceConfig:
    parser = argparse.ArgumentParser(description="TaoMate 1-minute inference")
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--base_model_ckpt", required=True)
    parser.add_argument("--gemma_path", required=True)
    parser.add_argument(
        "--benchmark_json",
        default="configs/inference/benchmark_1min_windowed.json",
    )
    parser.add_argument("--prompt_cache_path")
    parser.add_argument("--output_dir", default="outputs/taomate_1min")
    parser.add_argument("--video_height", type=int, default=512)
    parser.add_argument("--video_width", type=int, default=768)
    parser.add_argument("--output_video_height", type=int, default=0)
    parser.add_argument("--output_video_width", type=int, default=0)
    parser.add_argument("--case_start", type=int, default=0)
    parser.add_argument("--max_cases", type=int, default=1)
    return InferenceConfig(**vars(parser.parse_args(argv)))

def main() -> None:
    run_inference(parse_args())

if __name__ == "__main__":
    main()
