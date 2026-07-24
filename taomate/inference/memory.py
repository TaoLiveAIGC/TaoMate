from taomate.runtime_support.learned_memory import LearnedMemoryState


class StreamingMemory(LearnedMemoryState):
    def __init__(self) -> None:
        super().__init__(
            enabled=True,
            video_downsample=4,
            audio_tokens=64,
            video_beta=0.15,
            audio_beta=0.10,
            video_anchor_tether=0.20,
            audio_anchor_tether=0.10,
            identity_anchor_enabled=True,
            identity_anchor_scale=1.0,
            reference_anchor_enabled=True,
            drift_gate_enabled=True,
            drift_gate_threshold=0.05,
            drift_gate_temperature=0.10,
            drift_gate_min=0.10,
            drift_gate_apply_to_color=True,
            color_enabled=True,
            color_alpha=0.04,
            color_proto_alpha=0.015,
            color_update_beta=0.03,
            color_anchor_tether=0.60,
            color_proto_grid=4,
            color_drift_threshold=2.0,
            color_max_correction=0.35,
            color_film_enabled=True,
        )
