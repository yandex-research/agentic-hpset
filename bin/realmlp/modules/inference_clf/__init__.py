from .realmlp import (
    average_classification_logits,
    predict_logits_ensemble,
    predict_proba_members_v0,
    raw_member_predict,
    score_classifier,
)


INFERENCE_CLF_MAP = {
    0: predict_proba_members_v0,
}

__all__ = [
    "average_classification_logits",
    "predict_logits_ensemble",
    "predict_proba_members_v0",
    "raw_member_predict",
    "score_classifier",
    "INFERENCE_CLF_MAP",
]
