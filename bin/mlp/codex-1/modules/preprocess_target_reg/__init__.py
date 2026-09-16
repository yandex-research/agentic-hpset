"""Static registry: one local file per distinct implementation."""

from .baseline import target_preprocess_v0 as _v0
from .reg_target_clipped import target_preprocess_v2 as _v1

TARGET_PREPROCESS_REG_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0, 1},
}
ALIASES: dict[int, int] = {}
