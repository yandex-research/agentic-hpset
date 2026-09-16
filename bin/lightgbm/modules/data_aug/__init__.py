from .bootstrap import bootstrap_aug
from .cat_dropout import cat_dropout_aug
from .column_swap_noise import column_swap_noise_aug
from .concat_perturbed_copy import concat_perturbed_copy_aug
from .feature_nan_dropout import feature_nan_dropout_aug
from .gaussian_noise import gaussian_noise_aug
from .identity import identity_aug
from .intra_target_swap import intra_target_swap_aug
from .mean_imputation_dropout import mean_imputation_dropout_aug
from .mixup_same_target import mixup_same_target_aug
from .oversample_minority import oversample_minority_aug
from .random_duplicate import random_duplicate_aug
from .rank_jitter import rank_jitter_aug
from .stratified_bootstrap import stratified_bootstrap_aug
from .swap_low_variance import swap_low_variance_aug
from .target_quantile_bootstrap import target_quantile_bootstrap_aug
from .uniform_jitter import uniform_jitter_aug

DATA_AUG_MAP = {
    0: identity_aug,
    1: bootstrap_aug,
    2: gaussian_noise_aug,
    3: feature_nan_dropout_aug,
    4: column_swap_noise_aug,
    5: stratified_bootstrap_aug,
    6: oversample_minority_aug,
    7: mixup_same_target_aug,
    8: cat_dropout_aug,
    9: concat_perturbed_copy_aug,
    10: intra_target_swap_aug,
    11: mean_imputation_dropout_aug,
    12: random_duplicate_aug,
    13: uniform_jitter_aug,
    14: target_quantile_bootstrap_aug,
    15: swap_low_variance_aug,
    16: rank_jitter_aug,
}

__all__ = [
    "DATA_AUG_MAP",
    "bootstrap_aug",
    "cat_dropout_aug",
    "column_swap_noise_aug",
    "concat_perturbed_copy_aug",
    "feature_nan_dropout_aug",
    "gaussian_noise_aug",
    "identity_aug",
    "intra_target_swap_aug",
    "mean_imputation_dropout_aug",
    "mixup_same_target_aug",
    "oversample_minority_aug",
    "random_duplicate_aug",
    "rank_jitter_aug",
    "stratified_bootstrap_aug",
    "swap_low_variance_aug",
    "target_quantile_bootstrap_aug",
    "uniform_jitter_aug",
]
