"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_v1_layernorm_silu import build_mlp_v1 as _v1
from .clf_mlp_v4_feature_gate import build_mlp_v4 as _v2
from .clf_mlp_v5_maxout import build_mlp_v5 as _v3
from .reg_wide_deep_v1 import build_mlp_v1 as _v4
from .reg_dense_skip_v3 import build_mlp_v3 as _v5

MODEL_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3},
    'regression': {0, 4, 5},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
