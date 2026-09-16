from .ema import train_member_ema
from .mixup import train_member_mixup
from .realmlp import TrainMemberResult, train_member_v0


TRAIN_MAP = {
    0: train_member_v0,
    1: train_member_ema,
    2: train_member_mixup,
}

__all__ = [
    "TrainMemberResult",
    "train_member_v0",
    "train_member_ema",
    "train_member_mixup",
    "TRAIN_MAP",
]
