"""Static registry: one local file per distinct implementation."""

from .baseline import train_v0 as _v0
from .v1_cosine_warmup import train_cosine_warmup_v1 as _v1
from .v4_ema import train_ema_v4 as _v2

TRAIN_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0, 2},
}
ALIASES: dict[int, int] = {}
