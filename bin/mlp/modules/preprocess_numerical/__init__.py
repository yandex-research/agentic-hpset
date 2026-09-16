from .numerical import numerical_preprocess_v0
from .numerical_standard import numerical_preprocess_v1
from .numerical_power import numerical_preprocess_v2
from .numerical_winsorized import numerical_preprocess_v3
from .numerical_rankgauss import numerical_preprocess_v4
from .numerical_engineered import numerical_preprocess_v5
from .numerical_uniform import numerical_preprocess_v6
from .numerical_rowstats import numerical_preprocess_v7

NUM_PREPROCESS_MAP = {
    0: numerical_preprocess_v0,
    1: numerical_preprocess_v1,
    2: numerical_preprocess_v2,
    3: numerical_preprocess_v3,
    4: numerical_preprocess_v4,
    5: numerical_preprocess_v5,
    6: numerical_preprocess_v6,
    7: numerical_preprocess_v7,
}

__all__ = [
    "numerical_preprocess_v0",
    "numerical_preprocess_v1",
    "numerical_preprocess_v2",
    "numerical_preprocess_v3",
    "numerical_preprocess_v4",
    "numerical_preprocess_v5",
    "numerical_preprocess_v6",
    "numerical_preprocess_v7",
    "NUM_PREPROCESS_MAP",
]
