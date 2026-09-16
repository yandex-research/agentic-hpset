"""Static registry: one local file per distinct implementation."""

from .baseline import build_adamw as _v0
from .v1_lookahead import optimizer_lookahead_v1 as _v1

OPTIMIZER_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
