from .realmlp import target_preprocess_reg_v0


TARGET_PREPROCESS_REG_MAP = {
    0: target_preprocess_reg_v0,
}

__all__ = [
    "target_preprocess_reg_v0",
    "TARGET_PREPROCESS_REG_MAP",
]
