#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterable

import numpy as np
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import bin.realmlp._lib.experiment
import bin.realmlp._lib.util
from bin.realmlp import pipeline as realmlp_agentic
from bin.realmlp.modules import (
    IMPLEMENTATION_REGISTRIES,
    STAGE_KEYS,
    registry_name,
)


SPLIT = 'default'
EVAL_BATCH_SIZE = 4096
BASE_INDICES = {key: 0 for key in STAGE_KEYS}
# Small-but-complete RealMLP `model` config. Mirrors the *keys* of
# `exp/realmlp/base/0/california/evaluation/config.toml` `[base_config.model]`
# (a guaranteed-complete schema for `realmlp_agentic.main`) with fast values so
# smoke runs stay cheap. `n_epochs`, the early-stopping keys, and
# `implementation_indices` are filled in per run by `build_config`.
BASE_MODEL: dict[str, Any] = {
    'n_hidden_layers': 2,
    'hidden_sizes': 'rectangular',
    'hidden_width': 64,
    'p_drop': 0.0,
    'act': 'mish',
    'plr_sigma': 0.1,
    'plr_lr_factor': 0.1,
    'scale_lr_factor': 6.0,
    'first_layer_lr_factor': 1.0,
    'ls_eps_sched': 'coslog4',
    'ls_eps': 0.0,
    'p_drop_sched': 'flat_cos',
    'lr': 1e-3,
    'wd': 0.0,
    'use_ls': False,
    'embedding_size': 8,
    # n_ens must be >= 3 here: with n_ens=1 every member-geometry inference
    # variant (median, trimmed mean, entropy/agreement weighting) silently
    # degenerates to v0's mean and the smoke test exercises nothing.
    'n_ens': 3,
    'ens_av_before_softmax': False,
    'one_minus_sq_mom': 0.05,
    'max_one_hot_cat_size_raw': 8.0,
    'is_large': False,
    'plr_hidden_1': 8,
    'plr_hidden_2': 4,
    'early_stopping_multiplicative_patience': 1,
}


@dataclass(frozen=True)
class Scenario:
    id: int
    task_type: str
    score_name: str
    batch_size: int
    train_rows: int
    val_rows: int
    test_rows: int
    n_num: int
    n_cat: int
    description: str

    @property
    def n_rows(self) -> int:
        return self.train_rows + self.val_rows + self.test_rows


SCENARIOS = (
    Scenario(
        id=0,
        task_type='regression',
        score_name='rmse',
        batch_size=1024,
        train_rows=2048,
        val_rows=512,
        test_rows=512,
        n_num=100,
        n_cat=100,
        description='wide mixed feature stress case',
    ),
    Scenario(
        id=1,
        task_type='binclass',
        score_name='accuracy',
        batch_size=128,
        train_rows=256,
        val_rows=64,
        test_rows=64,
        n_num=12,
        n_cat=12,
        description='categorical coverage edge case',
    ),
    Scenario(
        id=2,
        task_type='multiclass',
        score_name='accuracy',
        batch_size=128,
        train_rows=256,
        val_rows=64,
        test_rows=64,
        n_num=16,
        n_cat=0,
        description='numerical only case',
    ),
    Scenario(
        id=3,
        task_type='regression',
        score_name='rmse',
        batch_size=128,
        train_rows=256,
        val_rows=64,
        test_rows=64,
        n_num=0,
        n_cat=16,
        description='categorical only case',
    ),
)


@dataclass(frozen=True)
class RunSpec:
    scenario: Scenario
    epochs: int
    label: str


@dataclass(frozen=True)
class RunStats:
    scenario_id: int
    epochs: int
    score_name: str
    val_metric: float
    test_metric: float
    val_score: float
    wall_s: float
    train_s: float
    n_parameters: int
    rss_mb: float
    gpu_peak_allocated_mb: float | None
    gpu_peak_reserved_mb: float | None


def rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == 'darwin':
        return usage / (1024**2)
    return usage / 1024.0


def write_dataset(root: Path, scenario: Scenario, seed: int) -> Path:
    rng = np.random.default_rng(seed + scenario.id * 1009)
    dataset_root = root / f'scenario_{scenario.id}'
    split_dir = dataset_root / 'splits' / SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)

    train = np.arange(0, scenario.train_rows, dtype=np.int32)
    val = np.arange(
        scenario.train_rows,
        scenario.train_rows + scenario.val_rows,
        dtype=np.int32,
    )
    test = np.arange(
        scenario.train_rows + scenario.val_rows,
        scenario.n_rows,
        dtype=np.int32,
    )
    np.save(split_dir / 'train.npy', train)
    np.save(split_dir / 'val.npy', val)
    np.save(split_dir / 'test.npy', test)

    x_num = make_num_features(rng, scenario)
    x_cat = make_cat_features(rng, scenario)
    y = make_target(rng, scenario, x_num, x_cat)

    if x_num is not None:
        np.save(dataset_root / 'x_num.npy', x_num.astype(np.float32))
    if x_cat is not None:
        np.save(dataset_root / 'x_cat.npy', x_cat.astype(np.int64))
    np.save(dataset_root / 'y.npy', y)
    (dataset_root / 'info.json').write_text(
        json.dumps(
            {'task': {'type': scenario.task_type, 'score': scenario.score_name}},
            indent=2,
        )
        + '\n'
    )
    return dataset_root


def make_num_features(
    rng: np.random.Generator,
    scenario: Scenario,
) -> np.ndarray | None:
    if scenario.n_num == 0:
        return None
    base = rng.normal(size=(scenario.n_rows, scenario.n_num)).astype(np.float32)
    trend = np.linspace(-1.0, 1.0, scenario.n_rows, dtype=np.float32)[:, None]
    weights = np.linspace(0.05, 0.5, scenario.n_num, dtype=np.float32)[None, :]
    return base + trend * weights


def make_cat_features(
    rng: np.random.Generator,
    scenario: Scenario,
) -> np.ndarray | None:
    if scenario.n_cat == 0:
        return None
    x_cat = rng.integers(0, 6, size=(scenario.n_rows, scenario.n_cat), dtype=np.int64)
    if scenario.id != 1:
        return x_cat

    train_end = scenario.train_rows
    val_test = slice(scenario.train_rows, scenario.n_rows)

    # Eval-only categories: train sees 0..3, val/test also see category 4.
    for col in range(0, 4):
        x_cat[:train_end, col] = rng.integers(0, 4, size=train_end)
        x_cat[val_test, col] = rng.integers(
            0, 5, size=scenario.val_rows + scenario.test_rows
        )
        x_cat[train_end, col] = 4

    # Train-only categories: train sees 0..6, val/test only see 0..2.
    for col in range(4, 8):
        x_cat[:train_end, col] = rng.integers(0, 7, size=train_end)
        x_cat[0, col] = 6
        x_cat[val_test, col] = rng.integers(
            0, 3, size=scenario.val_rows + scenario.test_rows
        )

    # Normal categorical features: all splits share the same category range.
    for col in range(8, scenario.n_cat):
        x_cat[:, col] = rng.integers(0, 5, size=scenario.n_rows)
    return x_cat


def make_target(
    rng: np.random.Generator,
    scenario: Scenario,
    x_num: np.ndarray | None,
    x_cat: np.ndarray | None,
) -> np.ndarray:
    signal = np.zeros(scenario.n_rows, dtype=np.float32)
    if x_num is not None:
        n = min(x_num.shape[1], 8)
        weights = np.linspace(0.2, 1.0, n, dtype=np.float32)
        signal += (x_num[:, :n] @ weights).astype(np.float32)
    if x_cat is not None:
        n = min(x_cat.shape[1], 8)
        cat_weights = np.linspace(0.03, 0.11, n, dtype=np.float32)
        signal += ((x_cat[:, :n] % 7) @ cat_weights).astype(np.float32)
    signal += rng.normal(0.0, 0.1, size=scenario.n_rows).astype(np.float32)

    if scenario.task_type == 'regression':
        return signal.astype(np.float32)
    if scenario.task_type == 'binclass':
        threshold = float(np.median(signal[: scenario.train_rows]))
        labels = (signal > threshold).astype(np.int64)
        return ensure_class_coverage(labels, 2, scenario)
    if scenario.task_type == 'multiclass':
        cuts = np.quantile(signal[: scenario.train_rows], [1.0 / 3.0, 2.0 / 3.0])
        labels = np.digitize(signal, cuts).astype(np.int64)
        return ensure_class_coverage(labels, 3, scenario)
    raise AssertionError(f'Unknown task_type={scenario.task_type!r}')


def ensure_class_coverage(
    labels: np.ndarray,
    n_classes: int,
    scenario: Scenario,
) -> np.ndarray:
    labels = labels.copy()
    split_starts = (
        0,
        scenario.train_rows,
        scenario.train_rows + scenario.val_rows,
    )
    split_widths = (scenario.train_rows, scenario.val_rows, scenario.test_rows)
    for start, width in zip(split_starts, split_widths, strict=True):
        n_values = min(width, n_classes * 4)
        stop = start + n_values
        labels[start:stop] = np.arange(n_values, dtype=np.int64) % n_classes
    return labels


def build_config(
    scenario: Scenario,
    dataset_root: Path,
    epochs: int,
    indices: dict[str, int],
    patience: int | None,
    seed: int,
) -> dict[str, Any]:
    """Assemble a full `realmlp_agentic.Config` for one train+eval run.

    Numerical preprocessing is done inside the pipeline, so `x_num` is loaded raw
    (no `num_policy`); only `cat_policy` and — gated on `n_num > 0` so the
    categorical-only scenario doesn't trip `build_dataset` — binary extraction
    are set here.
    """
    model = dict(BASE_MODEL)
    model['n_epochs'] = epochs
    if patience is None:
        model['use_early_stopping'] = False
        model['early_stopping_additive_patience'] = epochs + 1
    else:
        model['use_early_stopping'] = True
        model['early_stopping_additive_patience'] = patience
    model['implementation_indices'] = dict(indices)

    data: dict[str, Any] = {
        'path': str(dataset_root),
        'cat_policy': 'ordinal',
        'cache': False,
        'seed': seed,
    }
    if scenario.n_num > 0:
        data['extract_bin_from_num'] = True
        data['bin_policy'] = 'convert-to-cat'

    return {
        'seed': seed,
        'batch_size': scenario.batch_size,
        'predict_batch_size': EVAL_BATCH_SIZE,
        'data': data,
        'model': model,
    }


def load_predictions(exp: Path) -> dict[str, np.ndarray]:
    with np.load(exp / 'predictions.npz') as archive:
        return {part: archive[part] for part in archive.files}


def index_signature(indices: dict[str, int]) -> str:
    changed = {key: value for key, value in indices.items() if value != 0}
    if not changed:
        return 'all_v0'
    return ','.join(f'{key}={value}' for key, value in sorted(changed.items()))


def run_once(
    *,
    dataset_root: Path,
    spec: RunSpec,
    indices: dict[str, int],
    seed: int,
    patience: int | None,
) -> RunStats:
    config = build_config(
        spec.scenario, dataset_root, spec.epochs, indices, patience, seed
    )
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='autoresearch_run_') as tmp:
        exp = bin.realmlp._lib.experiment.create(Path(tmp) / 'exp', config=config, force=True)
        report = bin.realmlp._lib.experiment.run(realmlp_agentic.main, None, exp)
        if report is None:
            raise RuntimeError('Experiment did not produce a report.')
        predictions = load_predictions(exp)
    wall_s = time.perf_counter() - t0
    assert_finite_result(report, predictions)
    metrics = report['metrics']
    score_name = spec.scenario.score_name
    gpu_alloc = gpu_reserved = None
    if cuda:
        gpu_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        gpu_reserved = torch.cuda.max_memory_reserved() / (1024**2)
    return RunStats(
        scenario_id=spec.scenario.id,
        epochs=spec.epochs,
        score_name=score_name,
        val_metric=float(metrics['val'][score_name]),
        test_metric=float(metrics['test'][score_name]),
        val_score=float(metrics['val']['score']),
        wall_s=wall_s,
        train_s=float(report['time']),
        n_parameters=int(report['n_parameters']),
        rss_mb=rss_mb(),
        gpu_peak_allocated_mb=gpu_alloc,
        gpu_peak_reserved_mb=gpu_reserved,
    )


def assert_finite_result(
    report: dict[str, Any], predictions: dict[str, np.ndarray]
) -> None:
    for part, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'Non-finite predictions for part={part!r}.')
    for part, metrics in report['metrics'].items():
        for name, value in metrics.items():
            # Classification metrics nest per-class dicts alongside scalars; only
            # the scalar numbers are meaningful to a finiteness check.
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f'Non-finite metric {part}.{name}={value!r}.')


def print_stats(
    prefix: str, spec: RunSpec, indices: dict[str, int], stats: RunStats
) -> None:
    gpu_alloc = format_optional_mb(stats.gpu_peak_allocated_mb)
    gpu_reserved = format_optional_mb(stats.gpu_peak_reserved_mb)
    print(
        f'[{prefix}] scenario={stats.scenario_id} label={spec.label} '
        f'epochs={stats.epochs} indices={index_signature(indices)} '
        f'val_{stats.score_name}={stats.val_metric:.6f} '
        f'test_{stats.score_name}={stats.test_metric:.6f} '
        f'score={stats.val_score:.6f} '
        f'train_s={stats.train_s:.3f} wall_s={stats.wall_s:.3f} '
        f'params={stats.n_parameters} rss_mb={stats.rss_mb:.1f} '
        f'gpu_alloc_mb={gpu_alloc} gpu_reserved_mb={gpu_reserved}'
    )


def print_efficiency(candidate: RunStats, baseline: RunStats) -> None:
    wall_ratio = safe_ratio(candidate.wall_s, baseline.wall_s)
    train_ratio = safe_ratio(candidate.train_s, baseline.train_s)
    param_ratio = safe_ratio(candidate.n_parameters, baseline.n_parameters)
    rss_ratio = safe_ratio(candidate.rss_mb, baseline.rss_mb)
    gpu_ratio = optional_ratio(
        candidate.gpu_peak_allocated_mb,
        baseline.gpu_peak_allocated_mb,
    )
    print(
        '[efficiency] '
        f'scenario={candidate.scenario_id} epochs={candidate.epochs} '
        f'wall_ratio={wall_ratio:.2f} train_ratio={train_ratio:.2f} '
        f'param_ratio={param_ratio:.2f} rss_ratio={rss_ratio:.2f} '
        f'gpu_alloc_ratio={format_optional_ratio(gpu_ratio)}'
    )
    notes: list[str] = []
    if wall_ratio >= 10.0:
        notes.append('wall time is at least 10x baseline')
    if param_ratio >= 10.0:
        notes.append('parameter count is at least 10x baseline')
    if rss_ratio >= 10.0:
        notes.append('process memory is at least 10x baseline')
    if gpu_ratio is not None and gpu_ratio >= 10.0:
        notes.append('GPU allocation is at least 10x baseline')
    if notes:
        print('[efficiency-note] ' + '; '.join(notes))


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return float('inf') if value else 1.0
    return float(value) / float(baseline)


def optional_ratio(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return safe_ratio(value, baseline)


def format_optional_mb(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1f}'


def format_optional_ratio(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.2f}'


def select_scenarios(value: str) -> tuple[Scenario, ...]:
    if value == 'all':
        return SCENARIOS
    scenario_id = int(value)
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return (scenario,)
    raise argparse.ArgumentTypeError(f'Unknown scenario {value!r}.')


def build_run_specs(
    scenarios: tuple[Scenario, ...],
    quick_epochs: int,
    long_epochs: int,
) -> tuple[RunSpec, ...]:
    specs = [RunSpec(scenario, quick_epochs, 'quick') for scenario in scenarios]
    if long_epochs > quick_epochs:
        long_scenario = next((s for s in scenarios if s.id == 1), scenarios[0])
        specs.append(RunSpec(long_scenario, long_epochs, 'long'))
    return tuple(specs)


def registry_choices(key: str, task_type: str) -> tuple[int, ...]:
    name = registry_name(key, task_type)
    return tuple(sorted(IMPLEMENTATION_REGISTRIES[name]))


def iter_index_configs(
    scenario: Scenario,
    overrides: dict[str, int],
    exhaustive: bool,
) -> Iterable[dict[str, int]]:
    if not exhaustive:
        yield {**BASE_INDICES, **overrides}
        return
    choices = [
        (overrides[key],)
        if key in overrides
        else registry_choices(key, scenario.task_type)
        for key in STAGE_KEYS
    ]
    for values in itertools.product(*choices):
        yield dict(zip(STAGE_KEYS, values, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Generated-data smoke and interaction tests for the modular pipeline.'
        )
    )
    parser.add_argument(
        '--scenario', default='all', choices=['all', '0', '1', '2', '3']
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick-epochs', type=int, default=2)
    parser.add_argument('--long-epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--exhaustive', action='store_true')
    for key in STAGE_KEYS:
        parser.add_argument(
            f'--{key.replace("_", "-")}-idx', dest=key, type=int, default=None
        )
    args = parser.parse_args()
    if args.quick_epochs <= 0 or args.long_epochs <= 0:
        parser.error('--quick-epochs and --long-epochs must be positive.')
    if args.patience is not None and args.patience <= 0:
        parser.error('--patience must be positive when provided.')
    return args


def collect_overrides(args: argparse.Namespace) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for key in STAGE_KEYS:
        value = getattr(args, key)
        if value is not None:
            overrides[key] = int(value)
    return overrides


def main() -> None:
    args = parse_args()
    bin.realmlp._lib.util.init()
    scenarios = select_scenarios(args.scenario)
    specs = build_run_specs(scenarios, args.quick_epochs, args.long_epochs)
    overrides = collect_overrides(args)
    print(
        f'[config] seed={args.seed} exhaustive={args.exhaustive} '
        f'overrides={overrides or "{}"}'
    )
    print(
        '[efficiency-note] Compare candidate runs to all-v0 baseline. '
        'Large slowdowns or memory growth should be justified before keeping a module.'
    )

    with tempfile.TemporaryDirectory(prefix='autoresearch_synth_') as tmp:
        root = Path(tmp)
        dataset_roots = {
            scenario.id: write_dataset(root, scenario, args.seed)
            for scenario in scenarios
        }
        baseline_cache: dict[tuple[int, int], RunStats] = {}
        tested = 0
        for spec in specs:
            baseline_key = (spec.scenario.id, spec.epochs)
            baseline_root = dataset_roots[spec.scenario.id]
            baseline = baseline_cache.get(baseline_key)
            if baseline is None:
                baseline = run_once(
                    dataset_root=baseline_root,
                    spec=spec,
                    indices=BASE_INDICES,
                    seed=args.seed,
                    patience=args.patience,
                )
                baseline_cache[baseline_key] = baseline
                print_stats('baseline', spec, BASE_INDICES, baseline)
            for indices in iter_index_configs(
                spec.scenario, overrides, args.exhaustive
            ):
                if indices == BASE_INDICES:
                    tested += 1
                    continue
                candidate = run_once(
                    dataset_root=baseline_root,
                    spec=spec,
                    indices=indices,
                    seed=args.seed,
                    patience=args.patience,
                )
                tested += 1
                print_stats('candidate', spec, indices, candidate)
                print_efficiency(candidate, baseline)
        print(f'[ok] generated-data pipeline tests completed runs={tested}')


if __name__ == '__main__':
    main()
