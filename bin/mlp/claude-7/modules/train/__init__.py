"""Static registry: one local file per distinct implementation."""

from .baseline import train_v0 as _v0
from .v1_cosine_warmup import train_cosine_warmup_v1 as _v1
from .v2_mixup import train_mixup_v2 as _v2
from .v3_sam import train_sam_v3 as _v3
from .v4_swa import train_swa_v4 as _v4
from .v5_ema import train_ema_v5 as _v5
from .v6_cosine import train_cosine_v6 as _v6
from .v7_warmup import train_warmup_v7 as _v7
from .v9_target_noise import train_target_noise_v9 as _v8

TRAIN_MAP = {
    0: _v0,
    1: _v1,
    2: _v2,
    3: _v3,
    4: _v4,
    5: _v5,
    6: _v6,
    7: _v7,
    8: _v8,
}
APPLICABLE = {
    'classification': {0, 1, 2, 3, 4, 5},
    'regression': {0, 5, 6, 7, 8},
}
ALIASES: dict[int, int] = {}
