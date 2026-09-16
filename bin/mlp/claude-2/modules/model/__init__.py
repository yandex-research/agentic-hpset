"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_feature_gate import build_mlp_v4 as _v1
from .clf_mlp_fm import build_mlp_v5 as _v2
from .clf_mlp_zeroinit import build_mlp_v6 as _v3
from .clf_mlp_dropconnect import build_mlp_v7 as _v4
from .clf_mlp_l0 import build_mlp_v8 as _v5
from .reg_dcn_v2 import build_mlp_v1 as _v6
from .reg_wide_deep import build_mlp_v3 as _v7
from .reg_moe_mlp import build_mlp_v6 as _v8
from .reg_low_rank_hypernet import build_mlp_v8 as _v9

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
    9: _v9,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4, 5},
    'regression': {0, 6, 7, 8, 9},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
