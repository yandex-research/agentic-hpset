"""Static registry: one local file per distinct implementation."""

from .baseline import loss_v0 as _v0

LOSS_CLF_MAP = {
    0: _v0,
}
APPLICABLE = {
    'classification': {0},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
