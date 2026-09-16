"""Search space, budget, and stage resolution for the modular RealMLP.

RealMLP differs from the other families: its hyperparameters are not sampled in
Python. Every stage reads what it needs from a shared ``cfg`` dict (with in-code
defaults in ``bin/realmlp/modules/``), and the search space — including the
per-stage ``implementation_indices`` — is declared in :data:`SEARCH_SPACE_PATH`
(``search_space.toml``) and drawn by an Optuna TPE loop.

This module exposes that space, the stage-resolution helper, and the trial
budget, so a run is fully specified by the TOML + sampler + budget below. Feed a
report's ``config.model`` (which carries ``implementation_indices``) back through
``bin.realmlp.pipeline.run`` (passing the full experiment config) to reconstruct
the model. Index 0 of every stage is the
meta-tuned RealMLP-TD default, so the non-agentic ``realmlp`` method corresponds
to all stage indices = 0 (equivalently, plain ``bin.realmlp.base``).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from bin.realmlp.modules import (
    IMPLEMENTATION_REGISTRIES,
    STAGE_KEYS,
    registry_name,
    resolve,
)

__all__ = [
    "SEARCH_SPACE_PATH",
    "load_search_space",
    "SEARCH_SPACE",
    "STAGE_KEYS",
    "IMPLEMENTATION_REGISTRIES",
    "registry_name",
    "resolve",
    "N_STARTUP_TRIALS",
    "n_trials_for",
]

SEARCH_SPACE_PATH = Path(__file__).with_name("search_space.toml")


def load_search_space() -> dict:
    """Parse ``search_space.toml`` into a dict (the DSL is documented there)."""
    with SEARCH_SPACE_PATH.open("rb") as f:
        return tomllib.load(f)


SEARCH_SPACE = load_search_space()

# Optuna TPESampler warmup used during tuning.
N_STARTUP_TRIALS = 20
N_TRIALS_DEFAULT = 200
N_TRIALS_LARGE = 100  # tabred/* and microsoft (large datasets)


def is_large_dataset(dataset_name: str) -> bool:
    return dataset_name.startswith("tabred/") or dataset_name == "microsoft"


def n_trials_for(dataset_name: str) -> int:
    return N_TRIALS_LARGE if is_large_dataset(dataset_name) else N_TRIALS_DEFAULT
