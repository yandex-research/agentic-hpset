from .inference import inference_v0
from .tta import inference_v1
from .temp_scaling import inference_v2
from .mc_dropout import inference_v3

INFERENCE_MAP = {0: inference_v0, 1: inference_v1, 2: inference_v2, 3: inference_v3}

__all__ = [
    "inference_v0",
    "inference_v1",
    "inference_v2",
    "inference_v3",
    "INFERENCE_MAP",
]
