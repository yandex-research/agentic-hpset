"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_v1_layernorm_silu import build_mlp_v1 as _v1
from .clf_mlp_v5_feature_gate import build_mlp_v5 as _v2
from .reg_normalized_mlp import build_normalized_mlp as _v3
from .reg_glu_mlp import build_glu_mlp as _v4
from .reg_input_gated_mlp import build_input_gated_mlp as _v5
from .reg_multihead_mlp import build_multihead_mlp as _v6
from .reg_packed_mini_ensemble import build_packed_mini_ensemble as _v7
from .reg_wide_deep_mlp import build_wide_deep_mlp as _v8

MODEL_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
    7: _v7,
    8: _v8,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 3, 4, 5, 6, 7, 8},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
