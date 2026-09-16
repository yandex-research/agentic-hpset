"""Static registry: one local file per distinct implementation."""

from .baseline import target_preprocess_v0 as _v0
from .reg_target_robust import target_preprocess_v1 as _v1
from .reg_target_winsor_standard import target_preprocess_v4 as _v2

TARGET_PREPROCESS_REG_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0, 1, 2},
}
ALIASES: dict[int, int] = {}
