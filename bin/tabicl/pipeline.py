"""Single orchestration entry point for the unified NanoTabICL evaluation.

This module wires the pluggable per-variant registries under ``bin/modules`` into
the recipe grid (``EvaluationSpec`` / ``implementation_choices`` /
``total_grid_size``), the per-seed ensemble-member construction
(``build_members``), and the in-context inference loop (``TabICLEvaluator``).

The registries themselves are dumb maps; everything that decides *how to sample*
a recipe and *how to run* it lives here. ``bin/tune.py`` and ``bin/eval.py`` import
``build_evaluator_from_indices``, ``resolve_evaluation_spec``,
``implementation_choices`` and ``total_grid_size`` from here.
"""

from __future__ import annotations

import itertools
import os
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bin._pipeline import RunResult, dataset_path, split_name

from bin.tabicl.core import (
    classification_metrics,
    is_classification_task,
    normalize_task_type,
    regression_metrics,
)
from bin.tabicl.modules.augmentation import (
    AUGMENTATION_BASKETS,
    AUGMENTATION_NAMES,
    apply_augmentation,
)
from bin.tabicl.modules.ensemble import ENSEMBLE_NAMES, aggregate_prediction_pool
from bin.tabicl.modules.feature_permutation import (
    FEATURE_PERMUTATION_MAP,
    FEATURE_PERMUTATION_NAMES,
)
from bin.tabicl.modules.feature_preproc import (
    FEATURE_PREPROC_BASKETS,
    FEATURE_PREPROC_MAP,
    FEATURE_PREPROC_NAMES,
    FeatureEncoder,
)
from bin.tabicl.modules.postproc_clf import (
    CLASSIFICATION_POSTPROC_BASKETS,
    CLASSIFICATION_POSTPROC_MAP,
    CLASSIFICATION_POSTPROC_NAMES,
    ClassificationTargetEncoder,
    undo_class_permutation,
)
from bin.tabicl.modules.postproc_reg import (
    REGRESSION_POSTPROC_BASKETS,
    REGRESSION_POSTPROC_MAP,
    REGRESSION_POSTPROC_NAMES,
)

PARTS = ("train", "val", "test")

# Fixed ensemble size (no longer a tuned axis); class-permutation axis (classification only)
N_MEMBERS = 8
CLASS_PERMUTATION_REGISTRY = {0: "shift"}
CLASS_PERMUTATION_NAMES = {0: "shift"}


# ---------------------------------------------------------------------------
# Recipe spec + members
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationSpec:
    name: str
    feature_preprocs: tuple[int, ...]
    postprocs: tuple[int, ...]
    feature_permutation: int
    ensemble: int
    n_members: int
    class_permutation: int | None = None
    augmentations: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["feature_preproc_names"] = [
            FEATURE_PREPROC_NAMES[index] for index in self.feature_preprocs
        ]
        value["feature_permutation_name"] = FEATURE_PERMUTATION_NAMES[
            self.feature_permutation
        ]
        value["ensemble_name"] = ENSEMBLE_NAMES[self.ensemble]
        value["augmentation_names"] = [
            AUGMENTATION_NAMES[index] for index in self.augmentations
        ]
        if self.class_permutation is not None:
            value["class_permutation_name"] = CLASS_PERMUTATION_NAMES[
                self.class_permutation
            ]
        return value


@dataclass(frozen=True)
class EvaluationMember:
    feature_preproc: int
    postproc: int
    feature_permutation: tuple[int, ...]
    augmentation: int = 0
    class_permutation: tuple[int, ...] | None = None

    def to_dict(self, task_type: str) -> dict[str, Any]:
        postproc_names = postproc_registry_for_task(task_type)
        value: dict[str, Any] = {
            "feature_preproc": self.feature_preproc,
            "feature_preproc_name": FEATURE_PREPROC_NAMES[self.feature_preproc],
            "postproc": self.postproc,
            "postproc_name": postproc_names[self.postproc],
            "feature_permutation": list(self.feature_permutation),
            "augmentation": self.augmentation,
            "augmentation_name": AUGMENTATION_NAMES[self.augmentation],
        }
        if self.class_permutation is not None:
            value["class_permutation"] = list(self.class_permutation)
        return value


def postproc_registry_for_task(task_type: str | None) -> dict[int, str]:
    return (
        CLASSIFICATION_POSTPROC_NAMES
        if is_classification_task(task_type)
        else REGRESSION_POSTPROC_NAMES
    )


def postproc_baskets_for_task(task_type: str | None) -> dict[int, tuple[int, ...]]:
    return (
        CLASSIFICATION_POSTPROC_BASKETS
        if is_classification_task(task_type)
        else REGRESSION_POSTPROC_BASKETS
    )


def ensemble_choices_for_task(task_type: str | None) -> list[int]:
    if is_classification_task(task_type):
        return sorted(ENSEMBLE_NAMES)
    return [0, 1, 5]


def implementation_choices(task_type: str | None) -> dict[str, list[int]]:
    return {
        "model_idx": [0],
        "feature_preproc_basket_idx": sorted(FEATURE_PREPROC_BASKETS),
        "augmentation_basket_idx": sorted(AUGMENTATION_BASKETS),
        "postproc_basket_idx": sorted(postproc_baskets_for_task(task_type)),
        "feature_permutation_idx": sorted(FEATURE_PERMUTATION_NAMES),
        "ensemble_idx": ensemble_choices_for_task(task_type),
    }


def validate_implementation_indices(
    task_type: str | None, implementation_indices: dict[str, int]
) -> None:
    choices = implementation_choices(task_type)
    missing = [key for key in choices if key not in implementation_indices]
    if missing:
        raise ValueError(f"Missing implementation index keys: {missing}")
    bad = {
        key: value
        for key, value in implementation_indices.items()
        if key in choices and value not in choices[key]
    }
    if bad:
        raise ValueError(f"Invalid implementation indices for {task_type}: {bad}")


def resolve_evaluation_spec(
    task_type: str | None, implementation_indices: dict[str, int]
) -> EvaluationSpec:
    validate_implementation_indices(task_type, implementation_indices)
    task_name = normalize_task_type(task_type)
    feature_preproc_basket = implementation_indices["feature_preproc_basket_idx"]
    augmentation_basket = implementation_indices["augmentation_basket_idx"]
    postproc_basket = implementation_indices["postproc_basket_idx"]
    ensemble = implementation_indices["ensemble_idx"]
    spec = EvaluationSpec(
        name=(
            f"{task_name}:fp{feature_preproc_basket}:aug{augmentation_basket}:"
            f"pp{postproc_basket}:"
            f"perm{implementation_indices['feature_permutation_idx']}:"
            f"ens{ensemble}"
        ),
        feature_preprocs=FEATURE_PREPROC_BASKETS[feature_preproc_basket],
        postprocs=postproc_baskets_for_task(task_name)[postproc_basket],
        feature_permutation=implementation_indices["feature_permutation_idx"],
        ensemble=ensemble,
        n_members=N_MEMBERS,
        class_permutation=0 if is_classification_task(task_name) else None,
        augmentations=AUGMENTATION_BASKETS[augmentation_basket],
    )
    if not is_valid_spec(task_name, spec):
        raise ValueError(f"Invalid evaluation spec for {task_type}: {spec}")
    return spec


def is_valid_spec(task_type: str | None, spec: EvaluationSpec) -> bool:
    if spec.ensemble == 2 and not is_classification_task(task_type):
        return False
    return True


def iter_valid_implementation_indices(task_type: str | None):
    choices = implementation_choices(task_type)
    keys = tuple(choices)
    for values in itertools.product(*(choices[key] for key in keys)):
        indices = dict(zip(keys, values, strict=True))
        try:
            resolve_evaluation_spec(task_type, indices)
        except ValueError:
            continue
        yield indices


def total_grid_size(task_type: str | None) -> int:
    return sum(1 for _ in iter_valid_implementation_indices(task_type))


def build_permutations(
    n_items: int,
    method_idx: int,
    n_needed: int,
    seed: int,
) -> list[tuple[int, ...]]:
    if n_items <= 0:
        raise ValueError("No items available for permutation.")
    base = list(range(n_items))
    rng = random.Random(seed)
    if n_needed <= 1:
        values = [base]
    else:
        values = FEATURE_PERMUTATION_MAP[method_idx](n_items, base, n_needed, rng)
    if not values:
        values = [base]
    while len(values) < n_needed:
        values.extend(deepcopy(values))
    return [tuple(value) for value in values[:n_needed]]


def build_class_permutations(
    n_classes: int,
    class_permutation: int | None,
    n_needed: int,
    seed: int,
) -> list[tuple[int, ...] | None]:
    if class_permutation is None:
        return [None] * n_needed
    if n_classes <= 0:
        raise ValueError("No classes available for class permutation.")
    method = CLASS_PERMUTATION_NAMES[class_permutation]
    if method != "shift":
        raise ValueError(f"Unknown class permutation method: {method}")
    base = list(range(n_classes))
    values = [tuple(base[-offset:] + base[:-offset]) for offset in range(n_classes)]
    while len(values) < n_needed:
        values.extend(values)
    rng = random.Random(seed + 1)
    rng.shuffle(values)
    return values[:n_needed]


def build_members(
    spec: EvaluationSpec,
    *,
    n_features: int,
    seed: int,
    n_classes: int | None = None,
) -> list[EvaluationMember]:
    """Expand the recipe baskets into the pool of ensemble members.

    Each member is one ``(feature_preproc, postproc, augmentation)`` recipe drawn
    from the cartesian product of the spec's baskets, shuffled and tiled to
    ``spec.n_members``, and paired with its own feature/class permutation. The
    pool built here is the ensemble; ``spec.ensemble`` separately decides how the
    members' predictions are *aggregated*.
    """
    feature_perms = build_permutations(
        n_features, spec.feature_permutation, spec.n_members, seed
    )
    class_perms = build_class_permutations(
        n_classes or 0, spec.class_permutation, spec.n_members, seed
    )
    recipe_pool = list(
        itertools.product(spec.feature_preprocs, spec.postprocs, spec.augmentations)
    )
    random.Random(seed).shuffle(recipe_pool)
    members: list[EvaluationMember] = []
    for index in range(spec.n_members):
        feature_preproc, postproc, augmentation = recipe_pool[index % len(recipe_pool)]
        members.append(
            EvaluationMember(
                feature_preproc=feature_preproc,
                postproc=postproc,
                augmentation=augmentation,
                feature_permutation=feature_perms[index],
                class_permutation=class_perms[index],
            )
        )
    return members


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass
class TabICLEvaluator:
    spec: EvaluationSpec
    batch_size: int = 8
    outlier_threshold: float = 4.0
    members_: list[EvaluationMember] = field(default_factory=list, init=False)

    def fit_predict(self, dataset, model, device, seed, parts: tuple[str, ...] = PARTS):
        if is_classification_task(dataset.task_type):
            return self._fit_predict_classification(dataset, model, device, seed, parts)
        return self._fit_predict_regression(dataset, model, device, seed, parts)

    def _encode_parts(self, dataset, parts: tuple[str, ...]) -> dict[str, np.ndarray]:
        encoder = FeatureEncoder()
        encoded = {"train": encoder.fit_transform(dataset.x["train"])}
        for part in parts:
            if part != "train":
                encoded[part] = encoder.transform(dataset.x[part])
        return encoded

    def _fit_predict_regression(self, dataset, model, device, seed, parts):
        encoded = self._encode_parts(dataset, parts)
        self.members_ = build_members(
            self.spec,
            n_features=encoded["train"].shape[1],
            seed=seed,
        )
        per_member = {}
        uncertainty = {}
        # Fitted preprocessors are deterministic in feature_preproc (the context
        # is always the train part), so fit each one once and reuse it across
        # members and parts.
        preproc_cache: dict = {}
        for part in parts:
            per_member[part], uncertainty[part] = self._regression_member_pool(
                dataset=dataset,
                encoded=encoded,
                part=part,
                model=model,
                device=device,
                seed=seed,
                preproc_cache=preproc_cache,
            )
        predictions = aggregate_prediction_pool(
            task_type=dataset.task_type,
            ensemble_idx=self.spec.ensemble,
            per_member=per_member,
            uncertainty=uncertainty,
        )
        metrics = {
            part: regression_metrics(dataset.y_raw[part], predictions[part])
            for part in parts
        }
        return metrics, predictions

    def _iter_member_queries(
        self,
        *,
        encoded: dict[str, np.ndarray],
        part: str,
        seed: int,
        preproc_cache: dict,
    ):
        """Yield (member_index, member, preprocessor, augmented query) per member.

        The preprocessor is fit once per feature_preproc on the train context;
        the transformed query part is cached per feature_preproc and augmented
        per member (augmentations never mutate their input).
        """
        query_cache: dict = {}
        for member_index, member in enumerate(self.members_):
            preprocessor = preproc_cache.get(member.feature_preproc)
            if preprocessor is None:
                preprocessor = FEATURE_PREPROC_MAP[member.feature_preproc](
                    outlier_threshold=self.outlier_threshold,
                    random_state=seed,
                ).fit(encoded["train"])
                preproc_cache[member.feature_preproc] = preprocessor
            x_query = query_cache.get(member.feature_preproc)
            if x_query is None:
                x_query = preprocessor.transform(encoded[part])
                query_cache[member.feature_preproc] = x_query
            x_query = apply_augmentation(
                x_query,
                member.augmentation,
                seed=seed,
                part=part,
                member_index=member_index,
            )
            yield member_index, member, preprocessor, x_query

    def _regression_member_pool(
        self,
        *,
        dataset,
        encoded: dict[str, np.ndarray],
        part: str,
        model,
        device,
        seed: int,
        preproc_cache: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_members = len(self.members_)
        n_query = encoded[part].shape[0]
        predictions = np.empty((n_members, n_query), dtype=np.float32)
        spreads = np.empty((n_members, n_query), dtype=np.float32)
        member_queries = self._iter_member_queries(
            encoded=encoded, part=part, seed=seed, preproc_cache=preproc_cache
        )

        if _is_chunked(encoded["train"].shape[0], n_query, encoded["train"].shape[1]):
            # Wide/tall parts: run each member separately, chunking query rows
            # (exact -- query rows never attend to each other) and never
            # materializing all members' inputs at once.
            for member_index, member, preprocessor, x_query in member_queries:
                transform = REGRESSION_POSTPROC_MAP[member.postproc].prepare(
                    dataset, part
                )
                outputs = _run_model_chunked(
                    model=model,
                    device=device,
                    x_context=preprocessor.x_train_[
                        :, member.feature_permutation
                    ].astype(np.float32),
                    x_query=x_query[:, member.feature_permutation].astype(np.float32),
                    y_context=transform.y_context,
                )
                means, iqr = _regression_reduce(outputs)
                predictions[member_index] = transform.inverse(means)
                spreads[member_index] = (
                    np.maximum(iqr, 0.0) * transform.y_std
                ).astype(np.float32)
            return predictions, spreads

        jobs = []
        for member_index, member, preprocessor, x_query in member_queries:
            transform = REGRESSION_POSTPROC_MAP[member.postproc].prepare(dataset, part)
            x_all = np.concatenate([preprocessor.x_train_, x_query], axis=0)[
                :, member.feature_permutation
            ].astype(np.float32)
            jobs.append((member_index, x_all, transform.y_context, transform))

        for group in _group_jobs_by_shape(jobs):
            member_indices = [item[0] for item in group]
            xs = np.asarray([item[1] for item in group], dtype=np.float32)
            ys = np.asarray([item[2] for item in group], dtype=np.float32)
            outputs = _run_model_batches(
                model=model,
                device=device,
                xs=xs,
                ys=ys,
                batch_size=self.batch_size,
            )
            means, iqr = _regression_reduce(outputs)
            for local_index, member_index in enumerate(member_indices):
                transform = group[local_index][3]
                predictions[member_index] = transform.inverse(means[local_index])
                spreads[member_index] = (
                    np.maximum(iqr[local_index], 0.0) * transform.y_std
                ).astype(np.float32)
        return predictions, spreads

    def _fit_predict_classification(self, dataset, model, device, seed, parts):
        parts_to_compute = tuple(dict.fromkeys((*parts, "train")))
        encoded = self._encode_parts(dataset, parts_to_compute)
        target = ClassificationTargetEncoder.fit(dataset)
        max_classes = int(getattr(model, "max_classes", target.n_classes))
        if target.n_classes > max_classes:
            raise ValueError(
                f"Dataset has {target.n_classes} classes, but model supports only "
                f"{max_classes}."
            )
        self.members_ = build_members(
            self.spec,
            n_features=encoded["train"].shape[1],
            n_classes=target.n_classes,
            seed=seed,
        )
        # Fit each unique feature-preproc once on the train context and reuse it
        # across members and parts.
        preproc_cache: dict = {}
        logits = {
            part: self._classification_logits_pool(
                encoded=encoded,
                part=part,
                target=target,
                model=model,
                device=device,
                seed=seed,
                preproc_cache=preproc_cache,
            )
            for part in parts_to_compute
        }
        y_true = {
            part: target.transform(dataset.y[part]) for part in parts_to_compute
        }
        per_member, uncertainty = self._classification_postprocess_pool(
            logits=logits,
            y_train=y_true["train"],
            target=target,
            parts=parts,
        )
        predictions = aggregate_prediction_pool(
            task_type=dataset.task_type,
            ensemble_idx=self.spec.ensemble,
            per_member=per_member,
            n_classes=target.n_classes,
            uncertainty=uncertainty,
        )
        metrics = {
            part: classification_metrics(y_true[part], predictions[part], target.n_classes)
            for part in parts
        }
        return metrics, predictions

    def _classification_logits_pool(
        self,
        *,
        encoded: dict[str, np.ndarray],
        part: str,
        target: ClassificationTargetEncoder,
        model,
        device,
        seed: int,
        preproc_cache: dict,
    ) -> np.ndarray:
        n_members = len(self.members_)
        n_query = encoded[part].shape[0]
        logits = np.empty((n_members, n_query, target.n_classes), dtype=np.float32)
        member_queries = self._iter_member_queries(
            encoded=encoded, part=part, seed=seed, preproc_cache=preproc_cache
        )

        if _is_chunked(encoded["train"].shape[0], n_query, encoded["train"].shape[1]):
            # Wide/tall parts: run each member separately, chunking query rows
            # (exact -- query rows never attend to each other) and never
            # materializing all members' inputs at once.
            for member_index, member, preprocessor, x_query in member_queries:
                outputs = _run_model_chunked(
                    model=model,
                    device=device,
                    x_context=preprocessor.x_train_[
                        :, member.feature_permutation
                    ].astype(np.float32),
                    x_query=x_query[:, member.feature_permutation].astype(np.float32),
                    y_context=_classification_y_context(member, target),
                )[..., : target.n_classes]
                logits[member_index] = undo_class_permutation(
                    outputs, member.class_permutation
                )
            return logits

        jobs = []
        for member_index, member, preprocessor, x_query in member_queries:
            x_all = np.concatenate([preprocessor.x_train_, x_query], axis=0)[
                :, member.feature_permutation
            ].astype(np.float32)
            jobs.append(
                (member_index, x_all, _classification_y_context(member, target), member)
            )

        for group in _group_jobs_by_shape(jobs):
            member_indices = [item[0] for item in group]
            xs = np.asarray([item[1] for item in group], dtype=np.float32)
            ys = np.asarray([item[2] for item in group], dtype=np.int64)
            outputs = _run_model_batches(
                model=model,
                device=device,
                xs=xs,
                ys=ys,
                batch_size=self.batch_size,
            )[..., : target.n_classes]
            for local_index, member_index in enumerate(member_indices):
                member = group[local_index][3]
                logits[member_index] = undo_class_permutation(
                    outputs[local_index], member.class_permutation
                )
        return logits

    def _classification_postprocess_pool(
        self,
        *,
        logits: dict[str, np.ndarray],
        y_train: np.ndarray,
        target: ClassificationTargetEncoder,
        parts: tuple[str, ...],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        per_member = {
            part: np.empty_like(logits[part], dtype=np.float32) for part in parts
        }
        uncertainty = {
            part: np.empty(logits[part].shape[:2], dtype=np.float32) for part in parts
        }
        for member_index, member in enumerate(self.members_):
            postproc = CLASSIFICATION_POSTPROC_MAP[member.postproc]()
            postproc.fit(
                logits["train"][member_index],
                y_train,
                target.n_classes,
                target.prior,
            )
            for part in parts:
                proba = postproc.transform(logits[part][member_index])
                per_member[part][member_index] = proba
                clipped = np.clip(proba, 1e-12, 1.0)
                uncertainty[part][member_index] = -np.sum(
                    clipped * np.log(clipped), axis=-1
                ).astype(np.float32)
        return per_member, uncertainty


def _group_jobs_by_shape(jobs):
    groups = {}
    for job in jobs:
        key = (job[1].shape, job[2].shape)
        groups.setdefault(key, []).append(job)
    return groups.values()


# Cap members-per-forward by a (rows * cols) cell budget so wide and/or tall
# tables (e.g. 40k rows x 96 features) drop toward batch 1 and stay within GPU
# memory. Members are independent, so this only changes how they are grouped
# through the model -- predictions are identical regardless of the batch size.
_FORWARD_CELL_BUDGET = 4_000_000

# Parts whose full (context + query) sequence exceeds this cell budget are
# predicted per member in query-row chunks of at most this size. Chunking is
# exact: query rows never attend to each other in the model, so splitting the
# query across forwards cannot change any prediction. The default is anchored
# to the largest single forwards known to fit an 80GB GPU (~7.7M cells).
_CHUNK_CELL_BUDGET = int(os.environ.get("TABICL_CHUNK_CELL_BUDGET", "7500000"))


def _is_chunked(n_context: int, n_query: int, n_cols: int) -> bool:
    return (n_context + n_query) * n_cols > _CHUNK_CELL_BUDGET


def _regression_reduce(outputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce quantile-head outputs (last axis) to prediction mean and IQR."""
    sorted_outputs = np.sort(outputs, axis=-1)
    means = sorted_outputs.mean(axis=-1)
    q1_index = max(0, sorted_outputs.shape[-1] // 4)
    q3_index = min(sorted_outputs.shape[-1] - 1, (3 * sorted_outputs.shape[-1]) // 4)
    iqr = sorted_outputs[..., q3_index] - sorted_outputs[..., q1_index]
    return means, iqr


def _classification_y_context(member, target) -> np.ndarray:
    if member.class_permutation is None:
        y_context = target.y_train
    else:
        y_context = np.asarray(member.class_permutation, dtype=np.int64)[
            target.y_train
        ]
    return y_context.astype(np.int64)


def _effective_batch_size(xs: np.ndarray, batch_size: int) -> int:
    cells = int(xs.shape[1]) * int(xs.shape[2]) if xs.ndim >= 3 else int(xs.shape[1])
    if cells <= 0:
        return batch_size
    return max(1, min(batch_size, _FORWARD_CELL_BUDGET // cells))


def _run_model_chunked(
    *,
    model,
    device,
    x_context: np.ndarray,
    x_query: np.ndarray,
    y_context: np.ndarray,
) -> np.ndarray:
    n_context, n_cols = x_context.shape
    chunk_rows = max(1, _CHUNK_CELL_BUDGET // max(1, int(n_cols)) - int(n_context))
    model = model.to(device).eval()
    y = torch.from_numpy(np.ascontiguousarray(y_context))[None].to(device)
    outputs = []
    with torch.no_grad():
        for start in range(0, x_query.shape[0], chunk_rows):
            stop = min(start + chunk_rows, x_query.shape[0])
            xb = torch.from_numpy(
                np.concatenate([x_context, x_query[start:stop]], axis=0)
            )[None].to(device)
            outputs.append(model(xb, y).float().cpu().numpy()[0])
    return np.concatenate(outputs, axis=0)


def _run_model_batches(
    *,
    model,
    device,
    xs: np.ndarray,
    ys: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    eff_batch_size = _effective_batch_size(xs, batch_size)
    outputs = []
    model = model.to(device).eval()
    with torch.no_grad():
        for start in range(0, len(xs), eff_batch_size):
            stop = min(start + eff_batch_size, len(xs))
            xb = torch.from_numpy(xs[start:stop]).to(device)
            yb = torch.from_numpy(ys[start:stop]).to(device)
            outputs.append(model(xb, yb).float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def build_evaluator_from_spec(spec: EvaluationSpec, **kwargs) -> TabICLEvaluator:
    return TabICLEvaluator(spec=spec, **kwargs)


def build_evaluator_from_indices(
    task_type: str | None,
    implementation_indices: dict[str, int],
    **kwargs,
) -> TabICLEvaluator:
    spec = resolve_evaluation_spec(task_type, implementation_indices)
    return build_evaluator_from_spec(spec, **kwargs)


def run(config: dict[str, Any], *, dataset_root: str | Path | None = None,
        device: str | torch.device | None = None) -> RunResult:
    """Evaluate a saved recipe, preserving its train/train+val context choice."""
    from bin.tabicl.core import load_tabular_dataset
    from bin.tabicl.modules.nanotabicl import MODEL_REGISTRY

    recipe = config.get("run", {})
    refit_context = recipe.get("refit_context", recipe.get("context") == "train+val")
    dataset = load_tabular_dataset(
        dataset_path(config, dataset_root), split_name(config), refit_context=refit_context,
    )
    indices = recipe.get("implementation_indices", config.get("implementation_indices"))
    if indices is None:
        if recipe.get("evaluation_version", 0) != 0 or recipe.get("model_version", 0) != 0:
            raise ValueError("Only the recorded version-0 baseline recipe is supported.")
        indices = {key: 0 for key in implementation_choices(dataset.task_type)}
    spec = resolve_evaluation_spec(dataset.task_type, indices)
    device = torch.device(device if device is not None else config.get("device", "cpu"))
    checkpoint_options = {
        key: config[key] for key in ("model_path", "checkpoint_version", "allow_auto_download")
        if key in config
    }
    model = MODEL_REGISTRY[indices["model_idx"]](
        task_type=dataset.task_type, **checkpoint_options,
    ).to(device).eval()
    evaluator = build_evaluator_from_spec(spec)
    parts = tuple(recipe.get("parts", ("val", "test") if refit_context else PARTS))
    with torch.no_grad():
        metrics, predictions = evaluator.fit_predict(
            dataset, model, device, int(config.get("seed", 0)), parts=parts,
        )
    return RunResult({
        "metrics": metrics, "score_name": dataset.score_name,
        "evaluation_spec": spec.to_dict(), "implementation_indices": dict(indices),
        "refit_context": refit_context, "n_train_context": dataset.size("train"),
        "prediction_type": "labels" if dataset.task_type == "regression" else "probs",
    }, predictions, model)


__all__ = [
    "EvaluationSpec",
    "EvaluationMember",
    "TabICLEvaluator",
    "N_MEMBERS",
    "CLASS_PERMUTATION_REGISTRY",
    "CLASS_PERMUTATION_NAMES",
    "build_evaluator_from_indices",
    "build_evaluator_from_spec",
    "build_members",
    "build_permutations",
    "build_class_permutations",
    "implementation_choices",
    "ensemble_choices_for_task",
    "postproc_baskets_for_task",
    "postproc_registry_for_task",
    "is_valid_spec",
    "iter_valid_implementation_indices",
    "resolve_evaluation_spec",
    "validate_implementation_indices",
    "total_grid_size",
]
