#!/usr/bin/env python
"""Generated-data smoke and interaction tests for the modular LightGBM pipeline.

Builds small synthetic datasets on disk (never real data), runs the all-identity
baseline and a candidate `feature`/`data_aug`/`postproc` config end-to-end
(feature -> data_aug -> fit LightGBM -> predict -> postproc -> metric), and
reports correctness + efficiency vs the baseline. Use it on each new module right
after implementing it, and once more for the final interaction sweep
(`--exhaustive`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import resource
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bin.lightgbm.core import seed_everything
from bin.lightgbm.pipeline import (
    REGISTRY_KEY_MAP,
    run,
    get_registries,
)

SPLIT = 'default'
SMOKE_N_ESTIMATORS = 30
INDEX_KEYS = tuple(REGISTRY_KEY_MAP)  # ('feature_idx', 'data_aug_idx', 'postproc_idx')
BASE_INDICES = {key: 0 for key in INDEX_KEYS}


@dataclass(frozen=True)
class Scenario:
    id: int
    task_type: str
    score_name: str
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
    Scenario(0, 'regression', 'rmse', 1024, 256, 256, 40, 20, 'wide mixed feature stress case'),
    Scenario(1, 'binclass', 'accuracy', 512, 128, 128, 12, 12, 'categorical coverage edge case'),
    Scenario(2, 'multiclass', 'accuracy', 512, 128, 128, 16, 0, 'numerical only case'),
    Scenario(3, 'regression', 'rmse', 512, 128, 128, 0, 16, 'categorical only case'),
)


@dataclass(frozen=True)
class RunStats:
    scenario_id: int
    score_name: str
    val_metric: float
    test_metric: float
    wall_s: float
    rss_mb: float


def rss_mb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage / (1024**2) if sys.platform == 'darwin' else usage / 1024.0


# ---------------------------------------------------------------------------
# Synthetic data (same on-disk format the real data loader expects)
# ---------------------------------------------------------------------------


def write_dataset(root: Path, scenario: Scenario, seed: int) -> Path:
    rng = np.random.default_rng(seed + scenario.id * 1009)
    dataset_root = root / f'scenario_{scenario.id}'
    split_dir = dataset_root / 'splits' / SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)
    np.save(split_dir / 'train.npy', np.arange(0, scenario.train_rows, dtype=np.int32))
    np.save(split_dir / 'val.npy', np.arange(scenario.train_rows, scenario.train_rows + scenario.val_rows, dtype=np.int32))
    np.save(split_dir / 'test.npy', np.arange(scenario.train_rows + scenario.val_rows, scenario.n_rows, dtype=np.int32))
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


def make_num_features(rng, scenario) -> np.ndarray | None:
    if scenario.n_num == 0:
        return None
    base = rng.normal(size=(scenario.n_rows, scenario.n_num)).astype(np.float32)
    trend = np.linspace(-1.0, 1.0, scenario.n_rows, dtype=np.float32)[:, None]
    return base + trend * np.linspace(0.05, 0.5, scenario.n_num, dtype=np.float32)[None, :]


def make_cat_features(rng, scenario) -> np.ndarray | None:
    if scenario.n_cat == 0:
        return None
    return rng.integers(0, 6, size=(scenario.n_rows, scenario.n_cat), dtype=np.int64)


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


def ensure_class_coverage(labels, n_classes, scenario) -> np.ndarray:
    labels = labels.copy()
    starts = (0, scenario.train_rows, scenario.train_rows + scenario.val_rows)
    widths = (scenario.train_rows, scenario.val_rows, scenario.test_rows)
    for start, width in zip(starts, widths, strict=True):
        n_values = min(width, n_classes * 4)
        labels[start:start + n_values] = np.arange(n_values, dtype=np.int64) % n_classes
    return labels


# ---------------------------------------------------------------------------
# Run one config end-to-end (composes the shipped stage functions)
# ---------------------------------------------------------------------------


def _metric(task_type: str, score_name: str, y_true: np.ndarray, preds: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    if task_type == 'regression':
        return float(np.sqrt(np.mean((preds.reshape(-1) - y_true) ** 2)))
    return float((np.asarray(preds).reshape(-1) == y_true).mean())  # accuracy


def run_once(*, dataset_root: Path, scenario: Scenario, indices: dict[str, int], seed: int) -> RunStats:
    seed_everything(seed)
    t0 = time.perf_counter()
    result = run({
        'seed': seed, 'split': SPLIT,
        'implementation_indices': {**BASE_INDICES, **indices},
        'n_estimators': SMOKE_N_ESTIMATORS, 'early_stopping_rounds': 10,
        'thread_count': 1,
    }, dataset_root=dataset_root, device='cpu')
    new_preds = result.predictions
    wall_s = time.perf_counter() - t0
    for part, arr in new_preds.items():
        if not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
            raise RuntimeError(f'Non-finite predictions for part={part!r}.')
    return RunStats(
        scenario_id=scenario.id, score_name=scenario.score_name,
        val_metric=result.report['metrics']['val'][scenario.score_name],
        test_metric=result.report['metrics']['test'][scenario.score_name],
        wall_s=wall_s, rss_mb=rss_mb(),
    )


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------


def index_signature(indices: dict[str, int]) -> str:
    changed = {key: value for key, value in indices.items() if value != 0}
    return 'all_v0' if not changed else ','.join(f'{k[:-4]}={v}' for k, v in sorted(changed.items()))


def print_stats(prefix: str, indices: dict[str, int], stats: RunStats) -> None:
    print(
        f'[{prefix}] scenario={stats.scenario_id} indices={index_signature(indices)} '
        f'val_{stats.score_name}={stats.val_metric:.6f} test_{stats.score_name}={stats.test_metric:.6f} '
        f'wall_s={stats.wall_s:.3f} rss_mb={stats.rss_mb:.1f}'
    )


def print_efficiency(candidate: RunStats, baseline: RunStats) -> None:
    wall = candidate.wall_s / baseline.wall_s if baseline.wall_s else float('inf')
    rss = candidate.rss_mb / baseline.rss_mb if baseline.rss_mb else float('inf')
    print(f'[efficiency] scenario={candidate.scenario_id} wall_ratio={wall:.2f} rss_ratio={rss:.2f}')
    notes = []
    if wall >= 10.0:
        notes.append('wall time is at least 10x baseline')
    if rss >= 10.0:
        notes.append('process memory is at least 10x baseline')
    if notes:
        print('[efficiency-note] ' + '; '.join(notes))


def select_scenarios(value: str) -> tuple[Scenario, ...]:
    if value == 'all':
        return SCENARIOS
    for scenario in SCENARIOS:
        if scenario.id == int(value):
            return (scenario,)
    raise argparse.ArgumentTypeError(f'Unknown scenario {value!r}.')


def iter_index_configs(scenario, overrides, exhaustive) -> Iterable[dict[str, int]]:
    if not exhaustive:
        yield {**BASE_INDICES, **overrides}
        return
    registries = get_registries(scenario.task_type)
    choices = [
        (overrides[key],) if key in overrides else tuple(sorted(registries[REGISTRY_KEY_MAP[key]]))
        for key in INDEX_KEYS
    ]
    for values in itertools.product(*choices):
        yield dict(zip(INDEX_KEYS, values, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generated-data smoke and interaction tests for the modular LightGBM pipeline.')
    parser.add_argument('--scenario', default='all', choices=['all', '0', '1', '2', '3'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exhaustive', action='store_true')
    for key in INDEX_KEYS:
        parser.add_argument(f'--{key.replace("_", "-")}', dest=key, type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = select_scenarios(args.scenario)
    overrides = {key: int(getattr(args, key)) for key in INDEX_KEYS if getattr(args, key) is not None}
    print(f'[config] seed={args.seed} exhaustive={args.exhaustive} overrides={overrides or "{}"}')
    print('[efficiency-note] Compare candidate runs to all-identity baseline. '
          'Large slowdowns or memory growth should be justified before keeping a module.')
    with tempfile.TemporaryDirectory(prefix='autoresearch_synth_') as tmp:
        root = Path(tmp)
        tested = 0
        for scenario in scenarios:
            dataset_root = write_dataset(root, scenario, args.seed)
            baseline = run_once(dataset_root=dataset_root, scenario=scenario, indices=BASE_INDICES, seed=args.seed)
            print_stats('baseline', BASE_INDICES, baseline)
            for indices in iter_index_configs(scenario, overrides, args.exhaustive):
                if indices == BASE_INDICES:
                    tested += 1
                    continue
                candidate = run_once(dataset_root=dataset_root, scenario=scenario, indices=indices, seed=args.seed)
                tested += 1
                print_stats('candidate', indices, candidate)
                print_efficiency(candidate, baseline)
        print(f'[ok] generated-data pipeline tests completed runs={tested}')


if __name__ == '__main__':
    main()
