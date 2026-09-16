from ._variant import TabMVariantModel
from .bilinear_tabm import build_tabm_v12
from .column_dropout_tabm import build_tabm_v11
from .feature_dropout_tabm import build_tabm_v3
from .film_tabm import build_tabm_v5
from .geglu_tabm import build_tabm_v1
from .input_gates_tabm import InputGatedTabMModel, build_tabm_v8
from .linear_residual_tabm import LinearResidualTabMModel, build_tabm_v7
from .lora_tabm import build_tabm_v13
from .prenorm_tabm import build_tabm_v14
from .se_tabm import build_tabm_v10
from .soft_moe_tabm import build_tabm_v4
from .sparse_hidden_tabm import SparseHiddenTabMModel, build_tabm_v9
from .swiglu_rms_tabm import build_tabm_v2
from .tabm import TabMModel, build_tabm_v0
from .wide_shallow_tabm import build_tabm_v6

MODEL_MAP = {
    0: build_tabm_v0,
    1: build_tabm_v1,
    2: build_tabm_v2,
    3: build_tabm_v3,
    4: build_tabm_v4,
    5: build_tabm_v5,
    6: build_tabm_v6,
    7: build_tabm_v7,
    8: build_tabm_v8,
    9: build_tabm_v9,
    10: build_tabm_v10,
    11: build_tabm_v11,
    12: build_tabm_v12,
    13: build_tabm_v13,
    14: build_tabm_v14,
}

__all__ = [
    "TabMModel",
    "build_tabm_v0",
    "build_tabm_v1",
    "build_tabm_v2",
    "build_tabm_v3",
    "build_tabm_v4",
    "build_tabm_v5",
    "build_tabm_v6",
    "LinearResidualTabMModel",
    "build_tabm_v7",
    "InputGatedTabMModel",
    "build_tabm_v8",
    "SparseHiddenTabMModel",
    "build_tabm_v9",
    "TabMVariantModel",
    "build_tabm_v10",
    "build_tabm_v11",
    "build_tabm_v12",
    "build_tabm_v13",
    "build_tabm_v14",
    "MODEL_MAP",
]
