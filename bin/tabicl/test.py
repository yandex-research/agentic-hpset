#!/usr/bin/env python
"""Generated-data smoke and interaction tests for the modular NanoTabICL recipe.

Builds small synthetic datasets on disk (never real data), runs the default
recipe (baseline) and a candidate recipe through the frozen NanoTabICL model
(`build_evaluator_from_indices(...).fit_predict(...)`), and reports correctness +
inference efficiency vs the baseline. There is no training — the model is
frozen — so this exercises the evaluation recipe around it. Use it on each new
recipe module right after implementing it, and once more for the final recipe
sweep (`--exhaustive`).

Requires the pretrained NanoTabICL weights (downloaded via huggingface-hub on
first use).
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
import torch

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bin.tabicl.core import load_tabular_dataset
from bin.tabicl.modules.nanotabicl import MODEL_REGISTRY
from bin.tabicl.pipeline import build_evaluator_from_indices, implementation_choices

SPLIT = 'default'
PARTS = ('train', 'val', 'test')
_MODEL_CACHE: dict[tuple[str, str], object] = {}


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


# NanoTabICL runs a transformer forward pass per ensemble member, so scenarios
# are kept small to keep smoke runs cheap.
SCENARIOS = (
    Scenario(0, 'regression', 'rmse', 96, 32, 32, 6, 4, 'small mixed feature case'),
    Scenario(1, 'binclass', 'accuracy', 96, 32, 32, 8, 0, 'binary numerical case'),
    Scenario(2, 'multiclass', 'accuracy', 96, 32, 32, 8, 0, 'multiclass numerical case'),
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


def base_indices(task_type: str) -> dict[str, int]:
    """The default recipe: the first (index-0) choice on every axis."""
    return {axis: int(choices[0]) for axis, choices in implementation_choices(task_type).items()}


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
    return rng.integers(0, 5, size=(scenario.n_rows, scenario.n_cat), dtype=np.int64)


def make_target(rng, scenario, x_num, x_cat) -> np.ndarray:
    signal = np.zeros(scenario.n_rows, dtype=np.float32)
    if x_num is not None:
        n = min(x_num.shape[1], 8)
        signal += (x_num[:, :n] @ np.linspace(0.2, 1.0, n, dtype=np.float32)).astype(np.float32)
    if x_cat is not None:
        n = min(x_cat.shape[1], 8)
        signal += ((x_cat[:, :n] % 5) @ np.linspace(0.05, 0.2, n, dtype=np.float32)).astype(np.float32)
    signal += rng.normal(0.0, 0.1, size=scenario.n_rows).astype(np.float32)
    if scenario.task_type == 'regression':
        return signal.astype(np.float32)
    n_classes = 2 if scenario.task_type == 'binclass' else 3
    cuts = np.quantile(signal[: scenario.train_rows], np.linspace(0, 1, n_classes + 1)[1:-1])
    return ensure_class_coverage(np.digitize(signal, cuts).astype(np.int64), n_classes, scenario)


def ensure_class_coverage(labels, n_classes, scenario) -> np.ndarray:
    labels = labels.copy()
    starts = (0, scenario.train_rows, scenario.train_rows + scenario.val_rows)
    widths = (scenario.train_rows, scenario.val_rows, scenario.test_rows)
    for start, width in zip(starts, widths, strict=True):
        n_values = min(width, n_classes * 4)
        labels[start:start + n_values] = np.arange(n_values, dtype=np.int64) % n_classes
    return labels


# ---------------------------------------------------------------------------
# Run one recipe end-to-end
# ---------------------------------------------------------------------------


def get_model(task_type: str, device: torch.device):
    key = (task_type, str(device))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MODEL_REGISTRY[0](task_type=task_type).to(device)
    return _MODEL_CACHE[key]


def run_once(*, dataset_root: Path, scenario: Scenario, indices: dict[str, int], seed: int,
             device: torch.device) -> RunStats:
    dataset = load_tabular_dataset(dataset_root, SPLIT)
    model = get_model(dataset.task_type, device)
    evaluator = build_evaluator_from_indices(dataset.task_type, indices)
    t0 = time.perf_counter()
    with torch.no_grad():
        metrics, predictions = evaluator.fit_predict(dataset, model, device, seed, parts=PARTS)
    wall_s = time.perf_counter() - t0
    for part, arr in predictions.items():
        if not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
            raise RuntimeError(f'Non-finite predictions for part={part!r}.')
    score = scenario.score_name
    return RunStats(
        scenario_id=scenario.id, score_name=score,
        val_metric=float(metrics['val'][score]), test_metric=float(metrics['test'][score]),
        wall_s=wall_s, rss_mb=rss_mb(),
    )


# ---------------------------------------------------------------------------
# Reporting + CLI
# ---------------------------------------------------------------------------


def index_signature(indices: dict[str, int], base: dict[str, int]) -> str:
    changed = {key: value for key, value in indices.items() if value != base.get(key)}
    return 'default' if not changed else ','.join(f'{k}={v}' for k, v in sorted(changed.items()))


def print_stats(prefix: str, indices: dict[str, int], base: dict[str, int], stats: RunStats) -> None:
    print(
        f'[{prefix}] scenario={stats.scenario_id} recipe={index_signature(indices, base)} '
        f'val_{stats.score_name}={stats.val_metric:.6f} test_{stats.score_name}={stats.test_metric:.6f} '
        f'wall_s={stats.wall_s:.3f} rss_mb={stats.rss_mb:.1f}'
    )


def print_efficiency(candidate: RunStats, baseline: RunStats) -> None:
    wall = candidate.wall_s / baseline.wall_s if baseline.wall_s else float('inf')
    print(f'[efficiency] scenario={candidate.scenario_id} wall_ratio={wall:.2f}')
    if wall >= 10.0:
        print('[efficiency-note] inference wall time is at least 10x the default recipe')


def select_scenarios(value: str) -> tuple[Scenario, ...]:
    if value == 'all':
        return SCENARIOS
    for scenario in SCENARIOS:
        if scenario.id == int(value):
            return (scenario,)
    raise argparse.ArgumentTypeError(f'Unknown scenario {value!r}.')


def axis_keys(task_type: str) -> tuple[str, ...]:
    return tuple(implementation_choices(task_type))


def iter_recipes(scenario, overrides, exhaustive) -> Iterable[dict[str, int]]:
    base = base_indices(scenario.task_type)
    if not exhaustive:
        yield {**base, **overrides}
        return
    choices_map = implementation_choices(scenario.task_type)
    axes = list(choices_map)
    choices = [(overrides[a],) if a in overrides else tuple(choices_map[a]) for a in axes]
    for values in itertools.product(*choices):
        yield dict(zip(axes, values, strict=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generated-data smoke and interaction tests for the modular NanoTabICL recipe.')
    parser.add_argument('--scenario', default='all', choices=['all', '0', '1', '2'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--exhaustive', action='store_true')
    # Union of axes across task types, so every recipe flag is accepted.
    all_axes = sorted(set().union(*(set(implementation_choices(t)) for t in ('regression', 'binclass', 'multiclass'))))
    for axis in all_axes:
        parser.add_argument(f'--{axis.replace("_", "-")}', dest=axis, type=int, default=None)
    args = parser.parse_args()
    args._axes = all_axes
    return args


def resolve_device(value: str) -> torch.device:
    if value == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(value)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    scenarios = select_scenarios(args.scenario)
    print(f'[config] device={device} seed={args.seed} exhaustive={args.exhaustive}')
    print('[efficiency-note] Compare candidate recipes to the default recipe. '
          'Large inference slowdowns should be justified before keeping a module.')
    with tempfile.TemporaryDirectory(prefix='autoresearch_synth_') as tmp:
        root = Path(tmp)
        tested = 0
        for scenario in scenarios:
            dataset_root = write_dataset(root, scenario, args.seed)
            base = base_indices(scenario.task_type)
            overrides = {a: int(getattr(args, a)) for a in axis_keys(scenario.task_type) if getattr(args, a) is not None}
            baseline = run_once(dataset_root=dataset_root, scenario=scenario, indices=base, seed=args.seed, device=device)
            print_stats('baseline', base, base, baseline)
            for indices in iter_recipes(scenario, overrides, args.exhaustive):
                if indices == base:
                    tested += 1
                    continue
                candidate = run_once(dataset_root=dataset_root, scenario=scenario, indices=indices, seed=args.seed, device=device)
                tested += 1
                print_stats('candidate', indices, base, candidate)
                print_efficiency(candidate, baseline)
        print(f'[ok] generated-data recipe tests completed runs={tested}')


if __name__ == '__main__':
    main()
