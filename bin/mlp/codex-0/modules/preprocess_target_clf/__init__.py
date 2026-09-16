"""Static registry: one local file per distinct implementation."""

from .baseline import target_preprocess_v0 as _v0

TARGET_PREPROCESS_CLF_MAP = {
    0: _v0,
}
APPLICABLE = {
    'classification': {0},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
