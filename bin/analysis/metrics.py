from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from bin.analysis.results_config import ELO_ANCHOR_VALUE


ELO_SCALE = 400.0 / np.log(10.0)
METRIC_ALIASES = {
    "acc": "accuracy",
    "auc": "roc_auc",
    "roc-auc": "roc_auc",
    "cross-entropy": "log_loss",
    "cross_entropy": "log_loss",
    "log-loss": "log_loss",
    "logloss": "log_loss",
    "negative-log-likelihood": "log_loss",
    "negative_log_likelihood": "log_loss",
    "nll": "log_loss",
}
LOWER_IS_BETTER = {"rmse", "mae", "mse", "log_loss"}
HIGHER_IS_BETTER = {"accuracy", "roc_auc", "f1", "r2", "score"}


def canonical_metric_name(metric: str) -> str:
    return METRIC_ALIASES.get(str(metric).strip().lower(), str(metric).strip().lower())


def task_type_from_metric(metric: str) -> str:
    return "reg" if canonical_metric_name(metric) in {"rmse", "mae", "mse", "r2"} else "clf"


def metric_direction(metric: str) -> str:
    metric = canonical_metric_name(metric)
    if metric in LOWER_IS_BETTER:
        return "lower"
    if metric in HIGHER_IS_BETTER:
        return "higher"
    raise ValueError(f"Unknown metric direction for {metric!r}.")


def metric_error(score: float, metric: str) -> float:
    if pd.isna(score):
        return float("nan")
    return float(score) if metric_direction(metric) == "lower" else float(1.0 - score)


def metric_signed_error(score: float, metric: str) -> float:
    """Order-preserving error used for rank/Elo/improvement metrics.

    Unlike ``metric_error`` (which maps higher-is-better metrics to ``1 - score``),
    this maps them to ``-score`` so relative-improvement denominators are ``|score|``
    rather than ``|1 - score|``. Regression/lower-is-better metrics are unchanged.
    """
    if pd.isna(score):
        return float("nan")
    return float(score) if metric_direction(metric) == "lower" else float(-score)


def metric_error_std(std: float, metric: str) -> float:
    return float(std) if not pd.isna(std) else float("nan")


def relative_improvement_pct(base_score: float, score: float, metric: str) -> float:
    if pd.isna(base_score) or pd.isna(score):
        return float("nan")
    base_err = metric_error(base_score, metric)
    err = metric_error(score, metric)
    if pd.isna(base_err) or base_err == 0:
        return float("nan")
    return (base_err - err) / abs(base_err) * 100.0


def pct_improvement_from_errors(base_error: float, error: float) -> float:
    if pd.isna(base_error) or pd.isna(error) or base_error == 0:
        return float("nan")
    return (float(base_error) - float(error)) / abs(float(base_error)) * 100.0


def normalized_score(error: float, best_error: float, median_error: float) -> float:
    if pd.isna(error) or pd.isna(best_error) or pd.isna(median_error):
        return float("nan")
    denom = median_error - best_error
    if denom <= 0:
        return 1.0 if error <= best_error else 0.0
    return max(0.0, (median_error - error) / denom)


def _pairwise_outcomes(rows: pd.DataFrame, method_to_idx: Mapping[str, int]):
    a_idx, b_idx, y = [], [], []
    for _, group in rows.groupby("dataset", sort=False):
        entries = []
        for row in group.itertuples(index=False):
            idx = method_to_idx.get(row.method)
            if idx is not None and not pd.isna(row.signed_error):
                entries.append((idx, float(row.signed_error)))
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                ia, ea = entries[i]
                ib, eb = entries[j]
                outcome = 1.0 if ea < eb else (0.0 if ea > eb else 0.5)
                a_idx.append(ia)
                b_idx.append(ib)
                y.append(outcome)
    return (
        np.asarray(a_idx, dtype=np.int64),
        np.asarray(b_idx, dtype=np.int64),
        np.asarray(y, dtype=np.float64),
    )


def elo_ratings(rows: pd.DataFrame, methods: Sequence[str], *, anchor_method: str | None = None,
                anchor_value: float = ELO_ANCHOR_VALUE) -> pd.Series:
    from scipy.optimize import minimize

    methods = tuple(methods)
    method_to_idx = {method: i for i, method in enumerate(methods)}
    a_idx, b_idx, y = _pairwise_outcomes(rows, method_to_idx)
    if a_idx.size == 0:
        return pd.Series(np.nan, index=methods, dtype=float)

    def loss_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        z = beta[a_idx] - beta[b_idx]
        logistic = np.where(z >= 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z)))
        loss = float(np.sum(logistic + (1.0 - y) * z))
        sigm = 1.0 / (1.0 + np.exp(-z))
        residual = sigm - y
        grad = np.zeros(len(methods))
        np.add.at(grad, a_idx, residual)
        np.add.at(grad, b_idx, -residual)
        return loss, grad

    result = minimize(loss_and_grad, np.zeros(len(methods)), jac=True, method="L-BFGS-B")
    ratings = result.x * ELO_SCALE
    if anchor_method in method_to_idx:
        ratings = ratings + (anchor_value - ratings[method_to_idx[anchor_method]])
    return pd.Series(ratings, index=methods, dtype=float)


def add_rank_metrics(rows: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    method_order = {method: i for i, method in enumerate(methods)}
    out = []
    for dataset, group in rows.groupby("dataset", sort=False):
        group = group[group["method"].isin(method_order)].copy()
        present = set(group["method"])
        if present != set(methods) or group["method"].duplicated().any():
            continue
        group = group.sort_values("method", key=lambda s: s.map(method_order)).copy()
        # Rank/Elo use the signed error (order-preserving); normalized_score keeps the
        # 1-score error below.
        group["rank"] = group["signed_error"].rank(method="average", ascending=True)
        best = float(group["error"].min())
        median = float(group["error"].median())
        group["best_error"] = best
        group["median_error"] = median
        group["normalized_score"] = [
            normalized_score(err, best, median) for err in group["error"].to_numpy(dtype=float)
        ]
        group["method_order"] = group["method"].map(method_order)
        out.append(group)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=[*rows.columns, "rank"])


def summarize_metric(rows: pd.DataFrame, methods: Sequence[str], metric: str,
                     *, anchor_method: str | None = None) -> pd.DataFrame:
    methods = tuple(methods)
    ratings = elo_ratings(rows, methods, anchor_method=anchor_method) if metric == "elo" else None
    records = []
    for order, method in enumerate(methods):
        sub = rows[rows["method"].eq(method)]
        ranks = sub["rank"].to_numpy(dtype=float) if "rank" in sub else np.array([])
        if metric == "mean_rank":
            vals = ranks
            value = float(np.nanmean(vals)) if vals.size else float("nan")
        elif metric == "normalized_score":
            vals = sub["normalized_score"].to_numpy(dtype=float)
            value = float(np.nanmean(vals)) if vals.size else float("nan")
        elif metric == "elo":
            vals = np.array([])
            value = float(ratings.loc[method]) if ratings is not None else float("nan")
        else:
            raise ValueError(metric)
        finite = vals[~np.isnan(vals)] if vals.size else np.array([])
        records.append({
            "method": method,
            "method_order": order,
            "value": value,
            "std": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
            "n_datasets": int(sub["dataset"].nunique()),
        })
    return pd.DataFrame(records)

