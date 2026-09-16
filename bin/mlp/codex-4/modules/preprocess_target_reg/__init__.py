"""Static registry: one local file per distinct implementation."""

from .baseline import target_preprocess_v0 as _v0
from .reg_v1 import target_preprocess_v1 as _v1
from .reg_v2 import target_preprocess_v2 as _v2
from .reg_v6 import target_preprocess_v6 as _v3

TARGET_PREPROCESS_REG_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0, 1, 2, 3},
}
ALIASES: dict[int, int] = {}
