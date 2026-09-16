"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_v1 import build_mlp_v1 as _v1
from .clf_mlp_v3 import build_mlp_v3 as _v2
from .clf_mlp_v5 import build_mlp_v5 as _v3
from .clf_mlp_v6 import build_mlp_v6 as _v4
from .clf_mlp_v7 import build_mlp_v7 as _v5
from .reg_resnet import build_resnet_v0 as _v6
from .reg_mlp_gelu_ln import build_mlp_gelu_ln as _v7
from .reg_glu_mlp import build_glu_mlp as _v8
from .reg_densenet_mlp import build_densenet_mlp as _v9
from .reg_bottleneck_mlp import build_bottleneck_mlp as _v10

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
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4, 5},
    'regression': {0, 6, 7, 8, 9, 10},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
