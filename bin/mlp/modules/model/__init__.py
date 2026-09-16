from .mlp import MLPModel, build_mlp_v0
from .resmlp import ResMLPModel, build_mlp_v1
from .gelu_mlp import GELUMLPModel, build_mlp_v2
from .glu_mlp import GLUMLPModel, build_mlp_v3
from .featdrop_mlp import FeatureDropoutMLPModel, build_mlp_v4
from .bottleneck_mlp import BottleneckMLPModel, build_mlp_v5
from .se_mlp import SEMLPModel, build_mlp_v6
from .prenorm_mlp import PreNormMLPModel, build_mlp_v7
from .wsgelu_mlp import WSGELUMLP, build_mlp_v8
from .scaledinit_mlp import ScaledInitMLP, build_mlp_v9
from .widenarrow_mlp import WideNarrowMLP, build_mlp_v10
from .spectral_mlp import SpectralMLP, build_mlp_v11
from .wide_mlp import WideMLPModel, build_mlp_v12

MODEL_MAP = {
    0: build_mlp_v0,
    1: build_mlp_v1,
    2: build_mlp_v2,
    3: build_mlp_v3,
    4: build_mlp_v4,
    5: build_mlp_v5,
    6: build_mlp_v6,
    7: build_mlp_v7,
    8: build_mlp_v8,
    9: build_mlp_v9,
    10: build_mlp_v10,
    11: build_mlp_v11,
    12: build_mlp_v12,
}

__all__ = [
    "MLPModel", "build_mlp_v0",
    "ResMLPModel", "build_mlp_v1",
    "GELUMLPModel", "build_mlp_v2",
    "GLUMLPModel", "build_mlp_v3",
    "FeatureDropoutMLPModel", "build_mlp_v4",
    "BottleneckMLPModel", "build_mlp_v5",
    "SEMLPModel", "build_mlp_v6",
    "PreNormMLPModel", "build_mlp_v7",
    "WSGELUMLP", "build_mlp_v8",
    "ScaledInitMLP", "build_mlp_v9",
    "WideNarrowMLP", "build_mlp_v10",
    "SpectralMLP", "build_mlp_v11",
    "WideMLPModel", "build_mlp_v12",
    "MODEL_MAP",
]
