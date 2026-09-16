from __future__ import annotations

from ._base import _latin, random_samples


def permute_latin(
    n_items: int, base: list[int], n_needed: int, rng
) -> list[list[int]]:
    if n_items <= 4000:
        return _latin(n_items, rng)
    return random_samples(base, n_needed, rng)


__all__ = ["permute_latin"]
