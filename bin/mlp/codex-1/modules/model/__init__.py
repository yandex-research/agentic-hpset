"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_batch_norm_v1 import build_mlp_v1 as _v1
from .clf_layer_norm_silu_v2 import build_mlp_v2 as _v2
from .clf_gated_v3 import build_mlp_v3 as _v3
from .clf_feature_dropout_v4 import build_mlp_v4 as _v4
from .reg_gated_residual import build_mlp_v2 as _v5
from .reg_batchnorm import build_mlp_v4 as _v6
from .reg_input_gated import build_mlp_v5 as _v7

MODEL_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
    7: _v7,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 5, 6, 7},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
