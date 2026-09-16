"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_v4 import build_mlp_v4 as _v1
from .clf_mlp_v5 import build_mlp_v5 as _v2
from .clf_mlp_v7 import build_mlp_v7 as _v3
from .reg_mlp_gelu import build_mlp_v3_gelu as _v4

MODEL_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': {0, 4},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
