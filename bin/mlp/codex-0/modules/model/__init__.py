"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_v1_layernorm import build_mlp_v1 as _v1
from .clf_mlp_v4_residual_gate import build_mlp_v4 as _v2
from .clf_mlp_v5_dense import build_mlp_v5 as _v3
from .clf_mlp_v6_maxout import build_mlp_v6 as _v4
from .clf_mlp_v10_input_norm_leaky import build_mlp_v10 as _v5
from .clf_mlp_v11_dropconnect import build_mlp_v11 as _v6
from .clf_mlp_v12_highway import build_mlp_v12 as _v7
from .reg_model_v1 import build_model_v1 as _v8
from .reg_model_v2 import build_model_v2 as _v9
from .reg_model_v3 import build_model_v3 as _v10
from .reg_model_v4 import build_model_v4 as _v11
from .reg_model_v5 import build_model_v5 as _v12
from .reg_model_v6 import build_model_v6 as _v13
from .reg_model_v7 import build_model_v7 as _v14
from .reg_model_v8 import build_model_v8 as _v15
from .reg_model_v9 import build_model_v9 as _v16

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
    10: _v10,
    11: _v11,
    12: _v12,
    13: _v13,
    14: _v14,
    15: _v15,
    16: _v16,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4, 5, 6, 7},
    'regression': {0, 8, 9, 10, 11, 12, 13, 14, 15, 16},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
