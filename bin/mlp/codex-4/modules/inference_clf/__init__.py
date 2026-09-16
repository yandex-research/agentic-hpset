"""Static registry: one local file per distinct implementation."""

from .baseline import inference_v0 as _v0
from .v1_prior_calibration import inference_prior_calibration_v1 as _v1

INFERENCE_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
