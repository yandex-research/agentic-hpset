"""Per-stage registries for one self-contained agent-ablation MLP."""

from __future__ import annotations

from typing import Any

from .embedding_cat import (
    ALIASES as CAT_EMBEDDING_ALIASES,
    APPLICABLE as CAT_EMBEDDING_APPLICABLE,
    CAT_EMBEDDING_MAP,
)
from .embedding_num import (
    ALIASES as NUM_EMBEDDING_ALIASES,
    APPLICABLE as NUM_EMBEDDING_APPLICABLE,
    NUM_EMBEDDING_MAP,
)
from .inference_clf import (
    ALIASES as INFERENCE_CLF_ALIASES,
    APPLICABLE as INFERENCE_CLF_APPLICABLE,
    INFERENCE_MAP as INFERENCE_CLF_MAP,
)
from .inference_reg import (
    ALIASES as INFERENCE_REG_ALIASES,
    APPLICABLE as INFERENCE_REG_APPLICABLE,
    INFERENCE_MAP as INFERENCE_REG_MAP,
)
from .loss_clf import (
    ALIASES as LOSS_CLF_ALIASES,
    APPLICABLE as LOSS_CLF_APPLICABLE,
    LOSS_CLF_MAP,
)
from .loss_reg import (
    ALIASES as LOSS_REG_ALIASES,
    APPLICABLE as LOSS_REG_APPLICABLE,
    LOSS_REG_MAP,
)
from .model import ALIASES as MODEL_ALIASES, APPLICABLE as MODEL_APPLICABLE, MODEL_MAP
from .optimizer import (
    ALIASES as OPTIMIZER_ALIASES,
    APPLICABLE as OPTIMIZER_APPLICABLE,
    OPTIMIZER_MAP,
)
from .preprocess_categorical import (
    ALIASES as CAT_PREPROCESS_ALIASES,
    APPLICABLE as CAT_PREPROCESS_APPLICABLE,
    CAT_PREPROCESS_MAP,
)
from .preprocess_numerical import (
    ALIASES as NUM_PREPROCESS_ALIASES,
    APPLICABLE as NUM_PREPROCESS_APPLICABLE,
    NUM_PREPROCESS_MAP,
)
from .preprocess_target_clf import (
    ALIASES as TARGET_PREPROCESS_CLF_ALIASES,
    APPLICABLE as TARGET_PREPROCESS_CLF_APPLICABLE,
    TARGET_PREPROCESS_CLF_MAP,
)
from .preprocess_target_reg import (
    ALIASES as TARGET_PREPROCESS_REG_ALIASES,
    APPLICABLE as TARGET_PREPROCESS_REG_APPLICABLE,
    TARGET_PREPROCESS_REG_MAP,
)
from .train import ALIASES as TRAIN_ALIASES, APPLICABLE as TRAIN_APPLICABLE, TRAIN_MAP


STAGE_KEYS: tuple[str, ...] = (
    'numerical_preprocess',
    'categorical_preprocess',
    'target_preprocess',
    'num_embedding',
    'cat_embedding',
    'model',
    'train',
    'optimizer',
    'inference',
    'loss',
)

IMPLEMENTATION_REGISTRIES: dict[str, dict[int, Any]] = {
    'numerical_preprocess': NUM_PREPROCESS_MAP,
    'categorical_preprocess': CAT_PREPROCESS_MAP,
    'target_preprocess_clf': TARGET_PREPROCESS_CLF_MAP,
    'target_preprocess_reg': TARGET_PREPROCESS_REG_MAP,
    'num_embedding': NUM_EMBEDDING_MAP,
    'cat_embedding': CAT_EMBEDDING_MAP,
    'model': MODEL_MAP,
    'train': TRAIN_MAP,
    'optimizer': OPTIMIZER_MAP,
    'inference_clf': INFERENCE_CLF_MAP,
    'inference_reg': INFERENCE_REG_MAP,
    'loss_clf': LOSS_CLF_MAP,
    'loss_reg': LOSS_REG_MAP,
}

APPLICABLE_INDICES = {
    'numerical_preprocess': NUM_PREPROCESS_APPLICABLE,
    'categorical_preprocess': CAT_PREPROCESS_APPLICABLE,
    'target_preprocess_clf': TARGET_PREPROCESS_CLF_APPLICABLE,
    'target_preprocess_reg': TARGET_PREPROCESS_REG_APPLICABLE,
    'num_embedding': NUM_EMBEDDING_APPLICABLE,
    'cat_embedding': CAT_EMBEDDING_APPLICABLE,
    'model': MODEL_APPLICABLE,
    'train': TRAIN_APPLICABLE,
    'optimizer': OPTIMIZER_APPLICABLE,
    'inference_clf': INFERENCE_CLF_APPLICABLE,
    'inference_reg': INFERENCE_REG_APPLICABLE,
    'loss_clf': LOSS_CLF_APPLICABLE,
    'loss_reg': LOSS_REG_APPLICABLE,
}

IMPLEMENTATION_ALIASES = {
    'numerical_preprocess': NUM_PREPROCESS_ALIASES,
    'categorical_preprocess': CAT_PREPROCESS_ALIASES,
    'target_preprocess_clf': TARGET_PREPROCESS_CLF_ALIASES,
    'target_preprocess_reg': TARGET_PREPROCESS_REG_ALIASES,
    'num_embedding': NUM_EMBEDDING_ALIASES,
    'cat_embedding': CAT_EMBEDDING_ALIASES,
    'model': MODEL_ALIASES,
    'train': TRAIN_ALIASES,
    'optimizer': OPTIMIZER_ALIASES,
    'inference_clf': INFERENCE_CLF_ALIASES,
    'inference_reg': INFERENCE_REG_ALIASES,
    'loss_clf': LOSS_CLF_ALIASES,
    'loss_reg': LOSS_REG_ALIASES,
}

_TASK_DISPATCHED = {'target_preprocess', 'inference', 'loss'}


def _task_family(task_type: str) -> str:
    return 'regression' if task_type == 'regression' else 'classification'


def registry_name(stage_key: str, task_type: str) -> str:
    if stage_key not in STAGE_KEYS:
        raise ValueError(f'Unknown stage key: {stage_key}')
    if stage_key in _TASK_DISPATCHED:
        suffix = 'reg' if task_type == 'regression' else 'clf'
        return f'{stage_key}_{suffix}'
    return stage_key


def applicable_indices(stage_key: str, task_type: str) -> tuple[int, ...]:
    name = registry_name(stage_key, task_type)
    return tuple(sorted(APPLICABLE_INDICES[name][_task_family(task_type)]))


def resolve(stage_key: str, idx: int, task_type: str) -> Any:
    name = registry_name(stage_key, task_type)
    choices = applicable_indices(stage_key, task_type)
    if idx not in IMPLEMENTATION_REGISTRIES[name] or idx not in choices:
        raise ValueError(
            f'Unknown or inapplicable {name} index {idx} for {task_type}. '
            f'Available: {list(choices)}'
        )
    return IMPLEMENTATION_REGISTRIES[name][idx]


for _name, _registry in IMPLEMENTATION_REGISTRIES.items():
    if tuple(sorted(_registry)) != tuple(range(len(_registry))):
        raise RuntimeError(f'Non-contiguous registry {_name}: {sorted(_registry)}')


__all__ = [
    'APPLICABLE_INDICES',
    'IMPLEMENTATION_ALIASES',
    'IMPLEMENTATION_REGISTRIES',
    'STAGE_KEYS',
    'applicable_indices',
    'registry_name',
    'resolve',
]
