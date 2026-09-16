"""Static registry: one local file per distinct implementation."""

from .baseline import build_adamw as _v0

OPTIMIZER_MAP = {
    0: _v0,
}
APPLICABLE = {
    'classification': {0},
    'regression': {0},
}
ALIASES: dict[int, int] = {}
