from dataclasses import dataclass
from typing import Optional

@dataclass
class InferenceConfig:
    model_ckpt: str
    base_model_ckpt: str
    gemma_path: str
    benchmark_json: str = "configs/inference/benchmark_1min_windowed.json"
    output_dir: str = "outputs/taomate_1min"
    prompt_cache_path: Optional[str] = None
    video_height: int = 512
    video_width: int = 768
    output_video_height: int = 0
    output_video_width: int = 0
    case_start: int = 0
    max_cases: int = 1
    stream: bool = False
    conditioning_device: Optional[str] = None
    media_device: Optional[str] = None
