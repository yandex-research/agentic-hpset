"""Static registry: one local file per distinct implementation."""

from .baseline import loss_v0 as _v0
from .v1_huber import loss_huber_v1 as _v1

LOSS_REG_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0, 1},
}
ALIASES: dict[int, int] = {}
