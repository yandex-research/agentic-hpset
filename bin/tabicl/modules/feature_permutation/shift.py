from __future__ import annotations


def permute_shift(
    n_items: int, base: list[int], n_needed: int, rng
) -> list[list[int]]:
    return [base[-offset:] + base[:-offset] for offset in range(n_items)]


__all__ = ["permute_shift"]
