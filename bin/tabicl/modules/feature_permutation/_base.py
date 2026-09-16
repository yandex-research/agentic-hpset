from __future__ import annotations

import itertools
import sys
from copy import deepcopy


def _latin(n: int, rng) -> list[list[int]]:
    def shuffle_t(matrix):
        square = deepcopy(matrix)
        rng.shuffle(square)
        transposed = list(zip(*square))
        rng.shuffle(transposed)
        return transposed

    def build(symbols):
        if len(symbols) == 1:
            return [symbols]
        sym = rng.choice(symbols)
        symbols.remove(sym)
        square = build(symbols)
        square.append(square[0].copy())
        for index in range(len(square)):
            square[index].insert(index, sym)
        return square

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 100000))
    try:
        return [list(item) for item in shuffle_t(build(list(range(n))))]
    finally:
        sys.setrecursionlimit(old_limit)


def exact_or_sample(base: list[int], n_needed: int, rng) -> list[list[int]]:
    perms = [list(perm) for perm in itertools.permutations(base)]
    return rng.sample(perms, min(max(n_needed, 1), len(perms)))


def random_samples(base: list[int], n_needed: int, rng) -> list[list[int]]:
    return [rng.sample(base, len(base)) for _ in range(max(n_needed, 1))]


__all__ = ["_latin", "exact_or_sample", "random_samples"]
