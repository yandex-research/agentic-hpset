"""Static registry: one local file per distinct implementation."""

from .baseline import loss_v0 as _v0
from .v1_label_smoothing import loss_label_smoothing_v1 as _v1

LOSS_CLF_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': set(),
}
ALIASES: dict[int, int] = {}
