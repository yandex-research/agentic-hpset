"""Static registry: one local file per distinct implementation."""

from .baseline import MLPModel, build_mlp_v0 as _v0
from .clf_mlp_input_affine import build_mlp_v1 as _v1
from .clf_mlp_sparse_gate import build_mlp_v6 as _v2
from .reg_mlp_v1 import build_mlp_v1 as _v3
from .reg_mlp_v2 import build_mlp_v2 as _v4
from .reg_mlp_v4 import build_mlp_v4 as _v5

MODEL_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 3, 4, 5},
}
ALIASES: dict[int, int] = {}

__all__ = ['MODEL_MAP', 'MLPModel']
