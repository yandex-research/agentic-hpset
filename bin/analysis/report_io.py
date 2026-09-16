"""Read committed single-model results from ``exp/tuned`` into tidy DataFrames.

This is the read/analysis side only. The collection code that originally imported
these reports from many git branches and external repos is not part of the
published repo — the reports are committed under ``exp/tuned`` directly.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from bin.analysis.metrics import (
    canonical_metric_name,
    metric_direction,
    metric_error,
    metric_error_std,
    metric_signed_error,
    task_type_from_metric,
)
from bin.analysis.results_config import (
    DATA_ROOT,
    DATASETS,
    MAX_BUDGET,
    SPLIT,
    TABRED_BUDGET,
    TUNED_ROOT,
    is_tabred_dataset,
    report_name_for_budget,
)


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def aggregate_experiment_metrics(report: dict[str, Any], dataset: str) -> dict[str, Any]:
    if isinstance(report.get("metrics"), dict) and isinstance(report.get("score_name"), str):
        return report
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for experiment in report.get("experiments") or []:
        metrics = ((experiment.get("report") or {}).get("metrics") or {})
        if not isinstance(metrics, dict):
            continue
        for split, split_metrics in metrics.items():
            if not isinstance(split_metrics, dict):
                continue
            for name, value in split_metrics.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    values[str(split)][str(name)].append(float(value))
    if not values:
        raise KeyError(f"No numeric metrics in report for {dataset}.")
    normalized = dict(report)
    normalized["score_name"] = normalized.get("score_name") or dataset_score_name(dataset)
    normalized["metrics"] = {
        split: {
            name: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
            for name, vals in split_values.items()
        }
        for split, split_values in values.items()
    }
    return normalized


def dataset_score_name(dataset: str) -> str:
    with (DATA_ROOT / dataset / "info.json").open() as f:
        info = json.load(f)
    score = (info.get("task") or {}).get("score")
    if not isinstance(score, str):
        raise KeyError(f"Could not infer score for {dataset}.")
    return score


def metric_block_mean_std(block: Any) -> tuple[float, float]:
    if isinstance(block, dict):
        return float(block["mean"]), float(block.get("std", float("nan")))
    return float(block), float("nan")


def extract_metric(report: dict[str, Any], *, dataset: str, split: str = SPLIT) -> tuple[str, float, float]:
    report = aggregate_experiment_metrics(report, dataset)
    metric = report.get("score_name") or dataset_score_name(dataset)
    metrics = report["metrics"][split]
    key = None
    target = canonical_metric_name(metric)
    for candidate in metrics:
        if canonical_metric_name(candidate) == target:
            key = candidate
            break
    if key is None:
        raise KeyError(f"Metric {metric!r} missing; available={sorted(metrics)}")
    mean, std = metric_block_mean_std(metrics[key])
    return canonical_metric_name(metric), mean, std


def committed_report_path(method: str, dataset: str, budget: int | str):
    return TUNED_ROOT / method / dataset / report_name_for_budget(budget)


def load_single_results(*, budget: int = MAX_BUDGET, tabred_budget: int = TABRED_BUDGET, datasets=DATASETS,
                        methods: tuple[str, ...] | None = None) -> pd.DataFrame:
    from bin.analysis.results_config import METHOD_DISPLAY, SINGLE_METHODS, TABICL_METHODS

    methods = methods or SINGLE_METHODS
    records = []
    for dataset in datasets:
        for method in methods:
            selected_budget: int | str = (
                "grid"
                if method in TABICL_METHODS
                else (tabred_budget if is_tabred_dataset(dataset) else budget)
            )
            path = committed_report_path(method, dataset, selected_budget)
            if not path.exists():
                continue
            report = json.loads(path.read_text())
            metric, mean, std = extract_metric(report, dataset=dataset)
            records.append({
                "scope": "single",
                "dataset": dataset,
                "method": method,
                "method_display": METHOD_DISPLAY.get(method, method),
                "metric": metric,
                "metric_direction": metric_direction(metric),
                "task_type": task_type_from_metric(metric),
                "score": mean,
                "score_std": std,
                "error": metric_error(mean, metric),
                "signed_error": metric_signed_error(mean, metric),
                "error_std": metric_error_std(std, metric),
                "budget": selected_budget,
                "report_path": str(path),
            })
    return pd.DataFrame(records)
