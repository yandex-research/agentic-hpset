"""Static registry: one local file per distinct implementation."""

from .baseline import inference_v0 as _v0
from .v2_temperature import inference_temperature_v2 as _v1

INFERENCE_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
