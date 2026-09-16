"""Static registry: one local file per distinct implementation."""

from .baseline import inference_v0 as _v0
from .v1_input_noise_tta import inference_input_noise_tta_v1 as _v1
from .v2_temperature import inference_temperature_v2 as _v2
from .v3_mc_dropout import inference_mc_dropout_v3 as _v3

INFERENCE_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
