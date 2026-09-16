"""Static registry: one local file per distinct implementation."""

from .baseline import train_v0 as _v0
from .v1_cosine import train_cosine_v1 as _v1
from .v2_ema import train_ema_v2 as _v2

TRAIN_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': {0, 1, 2},
    'regression': {0, 2},
}
ALIASES: dict[int, int] = {}
