from .focal import classification_loss_step_focal
from .realmlp import classification_loss_step_v0


LOSS_CLF_MAP = {
    0: classification_loss_step_v0,
    1: classification_loss_step_focal,
}

__all__ = [
    "classification_loss_step_v0",
    "classification_loss_step_focal",
    "LOSS_CLF_MAP",
]
