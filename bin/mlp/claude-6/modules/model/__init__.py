"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_se import build_mlp_v2 as _v1
from .clf_mlp_stochastic_depth import build_mlp_v3 as _v2
from .clf_mlp_dropconnect import build_mlp_v4 as _v3
from .clf_mlp_orthogonal import build_mlp_v5 as _v4
from .clf_mlp_film import build_mlp_v6 as _v5
from .clf_mlp_topk import build_mlp_v7 as _v6
from .clf_mlp_spectral import build_mlp_v8 as _v7
from .reg_resnet import build_mlp_v1 as _v8
from .reg_glu import build_mlp_v3 as _v9
from .reg_bn import build_mlp_v4 as _v10
from .reg_gelu import build_mlp_v5 as _v11
from .reg_swiglu import build_mlp_v6 as _v12

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
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4, 5, 6, 7},
    'regression': {0, 8, 9, 10, 11, 12},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
