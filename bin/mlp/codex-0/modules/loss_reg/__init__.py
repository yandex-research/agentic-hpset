"""Static registry: one local file per distinct implementation."""

from .baseline import loss_v0 as _v0
from .v1_huber import loss_huber_v1 as _v1
from .v2_heteroscedastic import loss_heteroscedastic_v2 as _v2

LOSS_REG_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': set(),
    'regression': {0, 1, 2},
}
ALIASES: dict[int, int] = {}
