from __future__ import annotations

from ._base import _latin, exact_or_sample, random_samples
from .latin import permute_latin
from .none import permute_none
from .random import permute_random
from .shift import permute_shift

# index -> permutation generator (n_items, base, n_needed, rng) -> list[list[int]]
FEATURE_PERMUTATION_MAP = {
    0: permute_latin,
    1: permute_none,
    2: permute_random,
    3: permute_shift,
}
FEATURE_PERMUTATION_NAMES = {0: "latin", 1: "none", 2: "random", 3: "shift"}

__all__ = [
    "FEATURE_PERMUTATION_MAP",
    "FEATURE_PERMUTATION_NAMES",
    "permute_latin",
    "permute_none",
    "permute_random",
    "permute_shift",
    "_latin",
    "exact_or_sample",
    "random_samples",
]
