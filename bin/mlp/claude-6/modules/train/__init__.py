"""Static registry: one local file per distinct implementation."""

from .baseline import train_v0 as _v0
from .v3_swa import train_swa_v3 as _v1
from .v4_ema import train_ema_v4 as _v2
from .v5_cosine_warmup import train_cosine_warmup_v5 as _v3
from .v6_adversarial import train_adversarial_v6 as _v4

TRAIN_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4},
    'regression': {0, 2, 3},
}
ALIASES: dict[int, int] = {}
