from .clipped import inference_v7
from .geometric import inference_v3
from .inference import inference_v0
from .median import inference_v1
from .quantile_avg import inference_v5
from .trimmed_mean import inference_v2
from .tta_noise import inference_v4
from .weighted_variance import inference_v6

INFERENCE_MAP = {
    0: inference_v0,
    1: inference_v1,
    2: inference_v2,
    3: inference_v3,
    4: inference_v4,
    5: inference_v5,
    6: inference_v6,
    7: inference_v7,
}

__all__ = [
    "inference_v0",
    "inference_v1",
    "inference_v2",
    "inference_v3",
    "inference_v4",
    "inference_v5",
    "inference_v6",
    "inference_v7",
    "INFERENCE_MAP",
]
