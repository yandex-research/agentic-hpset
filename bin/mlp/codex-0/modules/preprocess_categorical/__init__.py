"""Static registry: one local file per distinct implementation."""

from .baseline import categorical_preprocess_v0 as _v0

CAT_PREPROCESS_MAP = {
    0: _v0,
}
APPLICABLE = {
    'classification': {0},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
