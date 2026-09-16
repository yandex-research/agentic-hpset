from __future__ import annotations

from ._base import exact_or_sample, random_samples


def permute_random(
    n_items: int, base: list[int], n_needed: int, rng
) -> list[list[int]]:
    if n_items <= 5:
        return exact_or_sample(base, n_needed, rng)
    return random_samples(base, n_needed, rng)


__all__ = ["permute_random"]
