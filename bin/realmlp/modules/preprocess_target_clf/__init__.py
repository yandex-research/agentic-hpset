# Classification target preprocessing is applied ONCE at the wrapper level
# (`realmlp_agentic.main`), which uses only the integer-encoded labels and the
# recovered `classes` array for `out_dim` — it deliberately discards the inverse
# closure. The predicted probability columns are therefore compared against the
# canonical (lib.data ordinal) label order by `task.calculate_metrics`. Any
# variant that permutes the label order (e.g. a frequency-sorted encoding)
# misaligns the probability columns and silently ruins the predictions. Only the
# canonical order-preserving v0 encoder is safe here, so it is the sole variant.
from .realmlp import target_preprocess_clf_v0


TARGET_PREPROCESS_CLF_MAP = {
    0: target_preprocess_clf_v0,
}

__all__ = [
    "target_preprocess_clf_v0",
    "TARGET_PREPROCESS_CLF_MAP",
]
