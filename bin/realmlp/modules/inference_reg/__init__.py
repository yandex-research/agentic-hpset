from .realmlp import (
    predict_ensemble_reg,
    predict_members_reg_v0,
    raw_member_predict,
    score_regressor,
)


INFERENCE_REG_MAP = {
    0: predict_members_reg_v0,
}

__all__ = [
    "predict_ensemble_reg",
    "predict_members_reg_v0",
    "raw_member_predict",
    "score_regressor",
    "INFERENCE_REG_MAP",
]
