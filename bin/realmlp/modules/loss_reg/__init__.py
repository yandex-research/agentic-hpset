from .huber import regression_loss_step_huber
from .realmlp import regression_loss_step_v0


LOSS_REG_MAP = {
    0: regression_loss_step_v0,
    1: regression_loss_step_huber,
}

__all__ = [
    "regression_loss_step_v0",
    "regression_loss_step_huber",
    "LOSS_REG_MAP",
]
