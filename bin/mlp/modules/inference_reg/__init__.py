from .inference import inference_v0
from .inference_clipped import inference_v2
from .inference_mcdropout import inference_v1

INFERENCE_MAP = {0: inference_v0, 1: inference_v1, 2: inference_v2}

__all__ = ["inference_v0", "inference_v1", "inference_v2", "INFERENCE_MAP"]
