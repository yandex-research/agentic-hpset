"""Static registry: one local file per distinct implementation."""

from .baseline import target_preprocess_v0 as _v0

TARGET_PREPROCESS_REG_MAP = {
    0: _v0,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0},
}
ALIASES: dict[int, int] = {}
