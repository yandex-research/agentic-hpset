from .cosine import train_cosine_v0
from .cosine_warmup import train_cosine_warmup_v0
from .ema import train_ema_v0
from .ensemble import train_ensemble_v0
from .gradnoise import train_gradnoise_v0
from .mixup import train_mixup_v0
from .soup import train_soup_v0
from .swa import train_swa_v0
from .swap_noise import train_swap_noise_v0
from .swp import train_swp_v0
from .train import train_v0
from .warmrestart import train_warmrestart_v0

TRAIN_MAP = {
    0: train_v0,
    1: train_cosine_v0,
    2: train_warmrestart_v0,
    3: train_ensemble_v0,
    4: train_swa_v0,
    5: train_mixup_v0,
    6: train_gradnoise_v0,
    7: train_cosine_warmup_v0,
    8: train_ema_v0,
    9: train_swp_v0,
    10: train_swap_noise_v0,
    12: train_soup_v0,
}

__all__ = [
    "train_v0",
    "train_cosine_v0",
    "train_warmrestart_v0",
    "train_ensemble_v0",
    "train_swa_v0",
    "train_mixup_v0",
    "train_gradnoise_v0",
    "train_cosine_warmup_v0",
    "train_ema_v0",
    "train_swp_v0",
    "train_swap_noise_v0",
    "train_soup_v0",
    "TRAIN_MAP",
]
