from .cosine_lr import train_v1
from .cutmix import train_v8
from .ema import train_v4
from .mixup import train_v3
from .negative_correlation import train_v6
from .sam import train_v5
from .swa import train_v2
from .train import train_v0
from .weight_decorrelation import train_v7

TRAIN_MAP = {
    0: train_v0,
    1: train_v1,
    2: train_v2,
    3: train_v3,
    4: train_v4,
    5: train_v5,
    6: train_v6,
    7: train_v7,
    8: train_v8,
}

__all__ = [
    "train_v0",
    "train_v1",
    "train_v2",
    "train_v3",
    "train_v4",
    "train_v5",
    "train_v6",
    "train_v7",
    "train_v8",
    "TRAIN_MAP",
]
