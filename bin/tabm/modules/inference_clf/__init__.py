from .inference import inference_v0
from .temperature import inference_v1
from .tta_noise import inference_v2

INFERENCE_MAP = {0: inference_v0, 1: inference_v1, 2: inference_v2}

__all__ = ["inference_v0", "inference_v1", "inference_v2", "INFERENCE_MAP"]
