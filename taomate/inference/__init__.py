from .config import InferenceConfig

__all__ = ["InferenceConfig", "preload_inference", "run_inference"]


def preload_inference(*args, **kwargs):
    from .runtime import preload_inference as implementation

    return implementation(*args, **kwargs)


def run_inference(*args, **kwargs):
    from .runtime import run_inference as implementation

    return implementation(*args, **kwargs)
