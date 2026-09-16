"""Search space and sampler for the modular NanoTabICL (TabICL) family.

TabICL does in-context learning — there is no gradient training. A "run" is an
*evaluation recipe*: which feature preprocessing basket, feature permutation,
augmentation basket, ensemble reducer, and post-processing basket to apply around
the pretrained NanoTabICL model (``model_idx`` is always 0 — the single model).

The recipe is chosen from :func:`implementation_choices` (task-dispatched), and
:func:`sample_hps` draws one recipe exactly as the tuning objective does. Feeding
the sampled ``implementation_indices`` back through
``bin.tabicl.pipeline.build_evaluator_from_indices`` (or ``resolve_evaluation_spec``)
reconstructs the evaluator. Each ``exp/tuned/{tabicl,agentic-tabicl}`` report records
both the basket ``implementation_indices`` and the resolved concrete
``evaluation_spec``.

The non-agentic ``tabicl`` method is the fixed default NanoTabICL ensemble recipe;
``agentic-tabicl`` searches the space below.
"""

from __future__ import annotations

from typing import Any

import optuna

from bin.tabicl.pipeline import (
    build_evaluator_from_indices,
    implementation_choices,
    resolve_evaluation_spec,
    total_grid_size,
    validate_implementation_indices,
)

__all__ = [
    "implementation_choices",
    "resolve_evaluation_spec",
    "build_evaluator_from_indices",
    "validate_implementation_indices",
    "total_grid_size",
    "sample_hps",
    "n_trials_for",
    "make_sampler",
]


def sample_hps(trial: optuna.Trial, task_type: str | None) -> dict[str, int]:
    """Sample one evaluation recipe as ``implementation_indices``.

    Matches the tuning objective: one categorical draw per axis over the
    task-appropriate ``implementation_choices``.
    """
    choices = implementation_choices(task_type)
    return {
        key: int(trial.suggest_categorical(key, values))
        for key, values in choices.items()
    }


def n_trials_for(task_type: str | None) -> int:
    """Size of the full implementation grid for this task type.

    The search covers the grid (TPE-guided); this is its upper bound.
    """
    return total_grid_size(task_type)


def make_sampler(seed: int):
    """The multivariate TPE sampler used for the search."""
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)
