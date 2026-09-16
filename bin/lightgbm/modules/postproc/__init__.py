from .classification.clip_renormalize import clip_renormalize_postproc
from .classification.identity_clf import identity_clf
from .classification.label_smoothing import label_smoothing_postproc
from .classification.min_proba_floor import min_proba_floor_postproc
from .classification.power_sharpen import power_sharpen_postproc
from .classification.power_smooth import power_smooth_postproc
from .classification.prior_correction import prior_correction_postproc
from .classification.temperature_scaling import temperature_scaling_postproc
from .classification.train_prior_blend import train_prior_blend_postproc
from .regression.clip_train_quantile import clip_train_quantile_postproc
from .regression.clip_train_range import clip_train_range_postproc
from .regression.identity import identity_postproc
from .regression.integer_round import integer_round_postproc
from .regression.iqr_winsorize import iqr_winsorize_postproc
from .regression.nonneg_clip import nonneg_clip_postproc
from .regression.quantile_match import quantile_match_postproc
from .regression.shrink_to_train_mean import shrink_to_train_mean_postproc
from .regression.softclip_tanh import softclip_tanh_postproc


POSTPROC_REGRESSION_MAP = {
    0: identity_postproc,
    1: clip_train_range_postproc,
    2: clip_train_quantile_postproc,
    3: quantile_match_postproc,
    4: integer_round_postproc,
    5: softclip_tanh_postproc,
    6: shrink_to_train_mean_postproc,
    7: nonneg_clip_postproc,
    8: iqr_winsorize_postproc,
}

POSTPROC_CLASSIFICATION_MAP = {
    0: identity_clf,
    1: clip_renormalize_postproc,
    2: label_smoothing_postproc,
    3: temperature_scaling_postproc,
    4: prior_correction_postproc,
    5: train_prior_blend_postproc,
    6: power_sharpen_postproc,
    7: min_proba_floor_postproc,
    8: power_smooth_postproc,
}


def get_postproc_map(task_type: str) -> dict:
    if task_type == "regression":
        return POSTPROC_REGRESSION_MAP
    if task_type in ("binclass", "multiclass"):
        return POSTPROC_CLASSIFICATION_MAP
    raise ValueError(f"Unsupported task_type: {task_type!r}")


__all__ = [
    "POSTPROC_REGRESSION_MAP",
    "POSTPROC_CLASSIFICATION_MAP",
    "clip_renormalize_postproc",
    "clip_train_quantile_postproc",
    "clip_train_range_postproc",
    "get_postproc_map",
    "identity_clf",
    "identity_postproc",
    "integer_round_postproc",
    "iqr_winsorize_postproc",
    "label_smoothing_postproc",
    "min_proba_floor_postproc",
    "nonneg_clip_postproc",
    "power_sharpen_postproc",
    "power_smooth_postproc",
    "prior_correction_postproc",
    "quantile_match_postproc",
    "shrink_to_train_mean_postproc",
    "softclip_tanh_postproc",
    "temperature_scaling_postproc",
    "train_prior_blend_postproc",
]
