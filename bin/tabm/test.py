#!/usr/bin/env python
"""Generated-data smoke and interaction tests for the modular TabM pipeline.

Builds small synthetic datasets on disk (never real data), runs the all-v0
baseline and a candidate `implementation_indices` config through
`bin.tabm.pipeline.train_and_eval`, and reports correctness + efficiency vs the
baseline. Use it on each new module right after implementing it, and once more
for the final interaction sweep (`--exhaustive`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import resource
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bin.tabm.hparams import INDEX_KEYS, registry_for
from bin.tabm.pipeline import train_and_eval

SPLIT = 'default'
EVAL_BATCH_SIZE = 4096
STAGE_KEYS = tuple(key[:-4] for key in INDEX_KEYS)  # strip the trailing "_idx"
BASE_INDICES = {key: 0 for key in INDEX_KEYS}


def base_trial_params(batch_size: int, epochs: int, patience: int | None) -> dict[str, Any]:
    """A small-but-complete trial config with fast values for smoke runs.

    Mirrors the keys produced by `bin.tabm.hparams.sample_hps`; exploring these
    knobs is the job of the tuning search, not this runner.
    """
    return {
        'batch_size': batch_size,
        'gradient_clip': 1.0,
        'max_epochs': epochs,
        'patience': (epochs + 1) if patience is None else patience,
        'eval_batch_size': EVAL_BATCH_SIZE,
        'n_bins': 8,
        'lr': 1e-3,
        'weight_decay': 0.0,
        'n_blocks': 1,
        'd_block': 64,
        'dropout': 0.0,
        'd_embedding': 8,
        'k': 3,  # small TabM ensemble for fast smoke runs
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
    Scenario(0, 'regression', 'rmse', 1024, 2048, 512, 512, 100, 100, 'wide mixed feature stress case'),
    Scenario(1, 'binclass', 'accuracy', 128, 256, 64, 64, 12, 12, 'categorical coverage edge case'),
    Scenario(2, 'multiclass', 'accuracy', 128, 256, 64, 64, 16, 0, 'numerical only case'),
    Scenario(3, 'regression', 'rmse', 128, 256, 64, 64, 0, 16, 'categorical only case'),
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


def rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024**2) if sys.platform == 'darwin' else usage / 1024.0


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def write_dataset(root: Path, scenario: Scenario, seed: int) -> Path:
    rng = np.random.default_rng(seed + scenario.id * 1009)
    dataset_root = root / f'scenario_{scenario.id}'
    split_dir = dataset_root / 'splits' / SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)
    train = np.arange(0, scenario.train_rows, dtype=np.int32)
    val = np.arange(scenario.train_rows, scenario.train_rows + scenario.val_rows, dtype=np.int32)
    test = np.arange(scenario.train_rows + scenario.val_rows, scenario.n_rows, dtype=np.int32)
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
        json.dumps({'task': {'type': scenario.task_type, 'score': scenario.score_name}}, indent=2) + '\n'
    )
    return dataset_root


def make_num_features(rng: np.random.Generator, scenario: Scenario) -> np.ndarray | None:
    if scenario.n_num == 0:
        return None
    base = rng.normal(size=(scenario.n_rows, scenario.n_num)).astype(np.float32)
    trend = np.linspace(-1.0, 1.0, scenario.n_rows, dtype=np.float32)[:, None]
    weights = np.linspace(0.05, 0.5, scenario.n_num, dtype=np.float32)[None, :]
    return base + trend * weights


def make_cat_features(rng: np.random.Generator, scenario: Scenario) -> np.ndarray | None:
    if scenario.n_cat == 0:
        return None
    x_cat = rng.integers(0, 6, size=(scenario.n_rows, scenario.n_cat), dtype=np.int64)
    if scenario.id != 1:
        return x_cat
    train_end = scenario.train_rows
    val_test = slice(scenario.train_rows, scenario.n_rows)
    for col in range(0, 4):  # eval-only categories
        x_cat[:train_end, col] = rng.integers(0, 4, size=train_end)
        x_cat[val_test, col] = rng.integers(0, 5, size=scenario.val_rows + scenario.test_rows)
        x_cat[train_end, col] = 4
    for col in range(4, 8):  # train-only categories
        x_cat[:train_end, col] = rng.integers(0, 7, size=train_end)
        x_cat[0, col] = 6
        x_cat[val_test, col] = rng.integers(0, 3, size=scenario.val_rows + scenario.test_rows)
    for col in range(8, scenario.n_cat):
        x_cat[:, col] = rng.integers(0, 5, size=scenario.n_rows)
    return x_cat


def make_target(rng, scenario, x_num, x_cat) -> np.ndarray:
    signal = np.zeros(scenario.n_rows, dtype=np.float32)
    if x_num is not None:
        n = min(x_num.shape[1], 8)
        signal += (x_num[:, :n] @ np.linspace(0.2, 1.0, n, dtype=np.float32)).astype(np.float32)
    if x_cat is not None:
        n = min(x_cat.shape[1], 8)
        signal += ((x_cat[:, :n] % 7) @ np.linspace(0.03, 0.11, n, dtype=np.float32)).astype(np.float32)
    signal += rng.normal(0.0, 0.1, size=scenario.n_rows).astype(np.float32)
    if scenario.task_type == 'regression':
        return signal.astype(np.float32)
    if scenario.task_type == 'binclass':
        labels = (signal > float(np.median(signal[: scenario.train_rows]))).astype(np.int64)
        return ensure_class_coverage(labels, 2, scenario)
    cuts = np.quantile(signal[: scenario.train_rows], [1.0 / 3.0, 2.0 / 3.0])
    return ensure_class_coverage(np.digitize(signal, cuts).astype(np.int64), 3, scenario)


def ensure_class_coverage(labels: np.ndarray, n_classes: int, scenario: Scenario) -> np.ndarray:
    labels = labels.copy()
    starts = (0, scenario.train_rows, scenario.train_rows + scenario.val_rows)
    widths = (scenario.train_rows, scenario.val_rows, scenario.test_rows)
    for start, width in zip(starts, widths, strict=True):
        n_values = min(width, n_classes * 4)
        labels[start:start + n_values] = np.arange(n_values, dtype=np.int64) % n_classes
    return labels


# ---------------------------------------------------------------------------
# Run + measure
# ---------------------------------------------------------------------------


def index_signature(indices: dict[str, int]) -> str:
    changed = {key: value for key, value in indices.items() if value != 0}
    if not changed:
        return 'all_v0'
    return ','.join(f'{key[:-4]}={value}' for key, value in sorted(changed.items()))


def run_once(*, dataset_root: Path, spec: RunSpec, indices: dict[str, int], seed: int,
             patience: int | None, device: torch.device) -> RunStats:
    params = base_trial_params(spec.scenario.batch_size, spec.epochs, patience)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = train_and_eval(
        dataset_root=dataset_root, split=SPLIT, device=device, seed=seed,
        trial_params=params, implementation_indices=indices,
        max_epochs=spec.epochs, patience=params['patience'],
    )
    wall_s = time.perf_counter() - t0
    assert_finite_result(result.report, result.predictions)
    metrics = result.report['metrics']
    score_name = spec.scenario.score_name
    gpu_alloc = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else None
    return RunStats(
        scenario_id=spec.scenario.id, epochs=spec.epochs, score_name=score_name,
        val_metric=float(metrics['val'][score_name]), test_metric=float(metrics['test'][score_name]),
        val_score=float(metrics['val']['score']), wall_s=wall_s,
        train_s=float(result.report.get('time_seconds', wall_s)),
        n_parameters=int(result.report['n_parameters']), rss_mb=rss_mb(),
        gpu_peak_allocated_mb=gpu_alloc,
    )


def assert_finite_result(report: dict[str, Any], predictions: dict[str, np.ndarray]) -> None:
    for part, values in predictions.items():
        if not np.isfinite(values).all():
            raise RuntimeError(f'Non-finite predictions for part={part!r}.')
    for part, metrics in report['metrics'].items():
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(float(value)):
                raise RuntimeError(f'Non-finite metric {part}.{name}={value!r}.')


def safe_ratio(value: float, baseline: float) -> float:
    if baseline == 0:
        return float('inf') if value else 1.0
    return float(value) / float(baseline)


def fmt_opt(value: float | None) -> str:
    return 'n/a' if value is None else f'{value:.1f}'


def print_stats(prefix: str, spec: RunSpec, indices: dict[str, int], stats: RunStats) -> None:
    print(
        f'[{prefix}] scenario={stats.scenario_id} label={spec.label} epochs={stats.epochs} '
        f'indices={index_signature(indices)} val_{stats.score_name}={stats.val_metric:.6f} '
        f'test_{stats.score_name}={stats.test_metric:.6f} score={stats.val_score:.6f} '
        f'train_s={stats.train_s:.3f} wall_s={stats.wall_s:.3f} params={stats.n_parameters} '
        f'rss_mb={stats.rss_mb:.1f} gpu_alloc_mb={fmt_opt(stats.gpu_peak_allocated_mb)}'
    )


def print_efficiency(candidate: RunStats, baseline: RunStats) -> None:
    wall = safe_ratio(candidate.wall_s, baseline.wall_s)
    param = safe_ratio(candidate.n_parameters, baseline.n_parameters)
    rss = safe_ratio(candidate.rss_mb, baseline.rss_mb)
    print(
        f'[efficiency] scenario={candidate.scenario_id} epochs={candidate.epochs} '
        f'wall_ratio={wall:.2f} param_ratio={param:.2f} rss_ratio={rss:.2f}'
    )
    notes = []
    if wall >= 10.0:
        notes.append('wall time is at least 10x baseline')
    if param >= 10.0:
        notes.append('parameter count is at least 10x baseline')
    if rss >= 10.0:
        notes.append('process memory is at least 10x baseline')
    if notes:
        print('[efficiency-note] ' + '; '.join(notes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def select_scenarios(value: str) -> tuple[Scenario, ...]:
    if value == 'all':
        return SCENARIOS
    for scenario in SCENARIOS:
        if scenario.id == int(value):
            return (scenario,)
    raise argparse.ArgumentTypeError(f'Unknown scenario {value!r}.')


def build_run_specs(scenarios, quick_epochs, long_epochs) -> tuple[RunSpec, ...]:
    specs = [RunSpec(s, quick_epochs, 'quick') for s in scenarios]
    if long_epochs > quick_epochs:
        long_scenario = next((s for s in scenarios if s.id == 1), scenarios[0])
        specs.append(RunSpec(long_scenario, long_epochs, 'long'))
    return tuple(specs)


def iter_index_configs(scenario, overrides, exhaustive) -> Iterable[dict[str, int]]:
    if not exhaustive:
        yield {**BASE_INDICES, **overrides}
        return
    choices = [
        (overrides[key],) if key in overrides else tuple(sorted(registry_for(key, scenario.task_type)))
        for key in INDEX_KEYS
    ]
    for values in itertools.product(*choices):
        yield dict(zip(INDEX_KEYS, values, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generated-data smoke and interaction tests for the modular TabM pipeline.')
    parser.add_argument('--scenario', default='all', choices=['all', '0', '1', '2', '3'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quick-epochs', type=int, default=2)
    parser.add_argument('--long-epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--exhaustive', action='store_true')
    for key in INDEX_KEYS:
        parser.add_argument(f'--{key.replace("_", "-")}', dest=key, type=int, default=None)
    args = parser.parse_args()
    if args.quick_epochs <= 0 or args.long_epochs <= 0:
        parser.error('--quick-epochs and --long-epochs must be positive.')
    return args


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    scenarios = select_scenarios(args.scenario)
    specs = build_run_specs(scenarios, args.quick_epochs, args.long_epochs)
    overrides = {key: int(getattr(args, key)) for key in INDEX_KEYS if getattr(args, key) is not None}
    print(f'[config] device={device} seed={args.seed} exhaustive={args.exhaustive} overrides={overrides or "{}"}')
    print('[efficiency-note] Compare candidate runs to all-v0 baseline. '
          'Large slowdowns or memory growth should be justified before keeping a module.')
    with tempfile.TemporaryDirectory(prefix='autoresearch_synth_') as tmp:
        root = Path(tmp)
        dataset_roots = {s.id: write_dataset(root, s, args.seed) for s in scenarios}
        baseline_cache: dict[tuple[int, int], RunStats] = {}
        tested = 0
        for spec in specs:
            key = (spec.scenario.id, spec.epochs)
            baseline = baseline_cache.get(key)
            if baseline is None:
                baseline = run_once(dataset_root=dataset_roots[spec.scenario.id], spec=spec,
                                    indices=BASE_INDICES, seed=args.seed, patience=args.patience, device=device)
                baseline_cache[key] = baseline
                print_stats('baseline', spec, BASE_INDICES, baseline)
            for indices in iter_index_configs(spec.scenario, overrides, args.exhaustive):
                if indices == BASE_INDICES:
                    tested += 1
                    continue
                candidate = run_once(dataset_root=dataset_roots[spec.scenario.id], spec=spec,
                                    indices=indices, seed=args.seed, patience=args.patience, device=device)
                tested += 1
                print_stats('candidate', spec, indices, candidate)
                print_efficiency(candidate, baseline)
        print(f'[ok] generated-data pipeline tests completed runs={tested}')


if __name__ == '__main__':
    main()
