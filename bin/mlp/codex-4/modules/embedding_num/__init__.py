"""Static registry: one local file per distinct implementation."""

from .baseline import build_num_embedding_v0 as _v0
from .reg_num_embedding_v2 import build_num_embedding_v2 as _v1

NUM_EMBEDDING_MAP = {
    0: _v0,
    1: _v1,
}
APPLICABLE = {
    'classification': {0},
    'regression': {0, 1},
}
ALIASES: dict[int, int] = {}
