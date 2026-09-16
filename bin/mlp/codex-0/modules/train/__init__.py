"""Static registry: one local file per distinct implementation."""

from .baseline import train_v0 as _v0
from .v1_cosine import train_cosine_v1 as _v1
from .v2_ema import train_ema_v2 as _v2
from .v4_sam import train_sam_v4 as _v3
from .v5_mixup import train_mixup_v5 as _v4
from .v6_ema_cosine import train_ema_cosine_v6 as _v5

TRAIN_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
}
APPLICABLE = {
    'classification': {0, 1},
    'regression': {0, 1, 2, 3, 4, 5},
}
ALIASES: dict[int, int] = {}
