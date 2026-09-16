"""Search space, sampler, and budget for the modular LightGBM family.

Unlike the deep families, LightGBM's search has two parts that live with the code
they drive:

  * :func:`suggest_lightgbm_params` (in ``bin.lightgbm.model``) — the gradient
    boosting hyperparameters (learning rate, num_leaves, fractions, L2 …),
  * :func:`sample_implementation_indices` (in ``bin.lightgbm.pipeline``) — the
    per-stage module choices ``feature_idx`` / ``data_aug_idx`` / ``postproc_idx``
    plus the ``feature_mode`` (append vs replace).

This module re-exports both and provides a single :func:`sample_hps` that draws a
full trial in the exact order used during tuning, so a run is reproducible from
the sampler + trial budget below. Index 0 of every stage is the identity/base
transform, so the non-agentic ``lightgbm`` runs are this pipeline with every
``*_idx`` = 0.
"""

from __future__ import annotations

from typing import Any

import optuna

from bin.lightgbm.model import (
    LIGHTGBM_FIXED_PARAMS,
    resolve_lightgbm_search_params,
    suggest_lightgbm_params,
)
from bin.lightgbm.pipeline import (
    FEATURE_MODE_KEY,
    FEATURE_MODES,
    REGISTRY_KEY_MAP,
    get_implementation_indices,
    get_registries,
    resolve_impl,
    sample_implementation_indices,
)

__all__ = [
    "LIGHTGBM_FIXED_PARAMS",
    "resolve_lightgbm_search_params",
    "suggest_lightgbm_params",
    "sample_implementation_indices",
    "REGISTRY_KEY_MAP",
    "FEATURE_MODES",
    "FEATURE_MODE_KEY",
    "get_registries",
    "get_implementation_indices",
    "resolve",
    "sample_hps",
    "n_trials_for",
    "make_sampler",
]


def resolve(arg_name: str, idx: int, task_type: str) -> Any:
    """Return the builder for stage ``arg_name`` (``*_idx``) at ``idx``."""
    registry_name = REGISTRY_KEY_MAP[arg_name]
    registries = get_registries(task_type)
    return resolve_impl(registries[registry_name], idx, registry_name)


def sample_hps(trial: optuna.Trial, task_type: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sample one trial as ``(lightgbm_params, implementation_config)``.

    Sampling order matches the tuning objective: LightGBM params first, then the
    per-stage implementation indices + feature mode.
    """
    lightgbm_params = suggest_lightgbm_params(trial)
    impl_config = sample_implementation_indices(trial, task_type)
    return lightgbm_params, impl_config


# ---------------------------------------------------------------------------
# Optimizer + budget
# ---------------------------------------------------------------------------

# Optuna study: minimize the validation metric with a TPE sampler.
N_TRIALS_DEFAULT = 200
N_TRIALS_LARGE = 100  # tabred/* and microsoft (large datasets)


def is_large_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("tabred/") or dataset_name == "microsoft"


def n_trials_for(dataset_name: str) -> int:
    return N_TRIALS_LARGE if is_large_dataset(dataset_name) else N_TRIALS_DEFAULT


def make_sampler(seed: int, *, n_startup_trials: int = 20):
    """The TPE sampler used for the search (``direction="minimize"``)."""
    return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
