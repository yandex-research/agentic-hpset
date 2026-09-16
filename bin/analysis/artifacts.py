from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bin.analysis.ensemble_io import load_committed_ensemble_results
from bin.analysis.metrics import (
    add_rank_metrics,
    metric_direction,
    pct_improvement_from_errors,
    summarize_metric,
)
from bin.analysis.report_io import load_single_results, natural_key
from bin.analysis.results_config import (
    ARTIFACTS_ROOT,
    BUDGET_CURVE_METHODS,
    DATASETS,
    DATA_ROOT,
    ELO_ANCHOR_METHOD,
    ENSEMBLE_TO_MAIN_METHOD,
    FAMILY_PAIRS,
    MAX_BUDGET,
    METHOD_DISPLAY,
    TABICL_METHODS,
    TABRED_BUDGET,
    is_tabred_dataset,
)


GROUP_COLUMNS = (("clf", "small"), ("clf", "medium"), ("clf", "large"),
                 ("reg", "small"), ("reg", "medium"), ("reg", "large"))


def _npy_shape(path: Path) -> tuple[int, ...] | None:
    if not path.exists():
        return None
    return tuple(int(v) for v in np.load(path, mmap_mode="r").shape)


def dataset_size_groups(datasets=DATASETS) -> pd.DataFrame:
    # Size groups need per-dataset shapes from the raw data/ tree, which is not
    # shipped. When it is absent, fall back to the committed size table (computed
    # once from the datasets) so the analysis runs from exp/ + artifacts/ alone.
    committed = ARTIFACTS_ROOT / "dataset_size_groups.csv"
    if not DATA_ROOT.exists() and committed.exists():
        df = pd.read_csv(committed)
        df = df[df["dataset"].isin(list(datasets))].copy()
        return df.sort_values("dataset", key=lambda s: s.map(natural_key)).reset_index(drop=True)
    records = []
    for dataset in datasets:
        data_path = DATA_ROOT / dataset
        with (data_path / "info.json").open() as f:
            info = json.load(f)
        task = (info.get("task") or {}).get("type", "")
        task_type = "reg" if "reg" in str(task).lower() else "clf"
        n_features = 0
        n_rows = None
        for name in ("x_num", "x_cat", "x_bin"):
            shape = _npy_shape(data_path / f"{name}.npy")
            if shape:
                n_rows = n_rows or shape[0]
                n_features += 1 if len(shape) == 1 else shape[1]
        if n_rows is None:
            y_shape = _npy_shape(data_path / "y.npy")
            n_rows = y_shape[0] if y_shape else None
        n_cells = None if n_rows is None else n_rows * n_features
        if is_tabred_dataset(dataset):
            size_group = "large"
        elif n_cells is None:
            size_group = "unknown"
        else:
            size_group = "small" if n_cells <= 299_668.5 else "medium"
        records.append({
            "dataset": dataset,
            "task_type": task_type,
            "size_group": size_group,
            "is_large": size_group == "large",
            "n_rows": n_rows,
            "n_feature_cols": n_features,
            "n_cells": n_cells,
        })
    return pd.DataFrame(records).sort_values("dataset", key=lambda s: s.map(natural_key)).reset_index(drop=True)


def load_joint_results(*, budget: int = MAX_BUDGET) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    single = load_single_results(budget=budget)
    ensemble = load_committed_ensemble_results()
    if not ensemble.empty:
        ensemble["method"] = ensemble["method"].map(lambda method: ENSEMBLE_TO_MAIN_METHOD.get(method, method))
        ensemble = ensemble[ensemble["method"].notna()].copy()
        ensemble["method_display"] = ensemble["method"].map(METHOD_DISPLAY)
        ensemble["metric"] = ensemble["score_name"].str.replace("roc-auc", "roc_auc").str.replace("cross-entropy", "log_loss")
        ensemble["metric_direction"] = ensemble["metric"].map(metric_direction)
        ensemble["task_type"] = ensemble["task_type"].map(lambda x: "reg" if "reg" in str(x).lower() else "clf")
        ensemble["score_std"] = np.nan
        ensemble["error_std"] = np.nan
    return single, ensemble, pd.concat([single, ensemble], ignore_index=True, sort=False)


def complete_rank_rows(rows: pd.DataFrame, methods: tuple[str, ...]) -> pd.DataFrame:
    ranked = []
    groups = dataset_size_groups()
    for _, ds_row in groups.iterrows():
        dataset = ds_row["dataset"]
        sub = rows[rows["dataset"].eq(dataset)]
        rank_rows = add_rank_metrics(sub, methods)
        if rank_rows.empty:
            continue
        rank_rows["size_group"] = ds_row["size_group"]
        rank_rows["dataset_task_type"] = ds_row["task_type"]
        ranked.append(rank_rows)
    return pd.concat(ranked, ignore_index=True) if ranked else pd.DataFrame()


def write_per_dataset_tables(single: pd.DataFrame, ensemble: pd.DataFrame, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for dataset in DATASETS:
        rows = []
        for row in single[single["dataset"].eq(dataset)].itertuples(index=False):
            rows.append({
                "scope": "single",
                "method": row.method,
                "method_display": row.method_display,
                "metric": row.metric,
                "metric_direction": row.metric_direction,
                "score": row.score,
                "score_std": row.score_std,
                "error": row.error,
                "error_std": row.error_std,
                "best_model_error": np.nan,
                "selected_member_corr": np.nan,
                "budget": row.budget,
            })
        ens_rows = ensemble[ensemble["dataset"].eq(dataset)]
        for row in ens_rows.itertuples(index=False):
            rows.append({
                "scope": "ensemble",
                "method": row.method,
                "method_display": METHOD_DISPLAY.get(row.method, row.method),
                "metric": row.metric,
                "metric_direction": row.metric_direction,
                "score": row.score,
                "score_std": np.nan,
                "error": row.error,
                "error_std": np.nan,
                "best_model_error": row.best_model_error,
                "selected_member_corr": row.selected_test_corr,
                "budget": row.budget,
            })
        if rows:
            name = dataset.replace("/", "__") + ".csv"
            pd.DataFrame(rows).to_csv(output_dir / name, index=False)
            count += 1
    return count


def _improvement_over_baseline_grouped(frame: pd.DataFrame, groups: pd.DataFrame,
                                       scope: str) -> pd.DataFrame:
    """Pairwise agentic-vs-baseline-hpset % improvement per (task_type, size_group) bucket.

    Each agentic method is compared to its family's base hpset (``FAMILY_PAIRS``) on the
    datasets they share, using the signed error. This is deliberately decoupled from the
    full-coverage rank gate, so partially-covered methods (e.g. ``agentic-realmlp``)
    are included. Base methods are reported as a constant 0.0 baseline.
    """
    if frame.empty:
        return pd.DataFrame()
    size_by_dataset = groups.set_index("dataset")
    present = set(frame["method"].unique())
    # Ordered base-then-agentic method list for stable output ordering.
    method_order: dict[str, int] = {}
    for _, agentic, base in FAMILY_PAIRS:
        for method in (base, agentic):
            if method in present and method not in method_order:
                method_order[method] = len(method_order)
    base_methods = {base for _, _, base in FAMILY_PAIRS}
    pair_base = {agentic: base for _, agentic, base in FAMILY_PAIRS}
    signed = frame.dropna(subset=["signed_error"]).set_index(["dataset", "method"])["signed_error"]

    records = []
    for task_type, size_group in GROUP_COLUMNS:
        bucket = set(size_by_dataset.index[
            (size_by_dataset["task_type"] == task_type) & (size_by_dataset["size_group"] == size_group)
        ])
        for method, order in method_order.items():
            if method in base_methods:
                n = sum((ds, method) in signed.index for ds in bucket)
                value = 0.0 if n else np.nan
                std = np.nan
            else:
                base = pair_base[method]
                vals = [
                    pct_improvement_from_errors(float(signed[(ds, base)]), float(signed[(ds, method)]))
                    for ds in bucket
                    if (ds, method) in signed.index and (ds, base) in signed.index
                ]
                finite = np.asarray(vals, dtype=float)
                finite = finite[~np.isnan(finite)]
                n = int(finite.size)
                value = float(finite.mean()) if finite.size else np.nan
                std = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
            records.append({
                "scope": scope,
                "metric": "improvement_over_baseline_hpset",
                "task_type": task_type,
                "size_group": size_group,
                "method": method,
                "method_display": METHOD_DISPLAY.get(method, method),
                "method_order": order,
                "value": value,
                "std": std,
                "n_datasets": n,
            })
    return pd.DataFrame(records)


def write_improvement_tables(frame: pd.DataFrame, output_dir: Path, *, scope: str) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    long = _improvement_over_baseline_grouped(frame, dataset_size_groups(), scope)
    wide = long.pivot_table(index=["method", "method_display"], columns=["task_type", "size_group"],
                            values="value", aggfunc="first").reset_index()
    wide.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple) else str(col)
        for col in wide.columns
    ]
    long.to_csv(output_dir / f"grouped_improvement_over_baseline_hpset_{scope}_long.csv", index=False)
    wide.to_csv(output_dir / f"grouped_improvement_over_baseline_hpset_{scope}_wide.csv", index=False)
    return long


BUDGETS = tuple(range(50, MAX_BUDGET + 1, 50))
LARGE_BUDGETS = (50, TABRED_BUDGET)
# Budget-curve metrics mirror the four panels of the main figure.
BUDGET_METRICS = (
    ("mean_rank", "Mean rank", True),
    ("improvement_over_mlp", "Improvement over MLP (%)", False),
    ("normalized_score", "Normalized score", False),
    ("elo", f"Elo ({ELO_ANCHOR_METHOD} = 1000)", False),
)
MAIN_BUDGET_METRICS = ("mean_rank", "improvement_over_mlp")


def _datasets_for_scope(groups: pd.DataFrame, scope: str) -> tuple[str, ...]:
    if scope == "large":
        return tuple(groups[groups["is_large"]]["dataset"])
    if scope == "non_large":
        return tuple(groups[~groups["is_large"]]["dataset"])
    raise ValueError(scope)


def _budget_metric_summary(rank_rows: pd.DataFrame, methods: tuple[str, ...], metric: str) -> pd.DataFrame:
    if rank_rows.empty:
        return pd.DataFrame({
            "method": methods,
            "method_order": range(len(methods)),
            "value": np.nan,
            "n_datasets": 0,
        })
    if metric == "improvement_over_mlp":
        reference = rank_rows[rank_rows["method"].eq(ELO_ANCHOR_METHOD)].set_index("dataset")["signed_error"]
        records = []
        for order, method in enumerate(methods):
            sub = rank_rows[rank_rows["method"].eq(method)]
            vals = np.asarray([
                pct_improvement_from_errors(float(reference[dataset]), float(error))
                for dataset, error in zip(sub["dataset"], sub["signed_error"], strict=True)
                if dataset in reference.index
            ], dtype=float)
            vals = vals[~np.isnan(vals)]
            records.append({
                "method": method,
                "method_order": order,
                "value": float(vals.mean()) if vals.size else np.nan,
                "n_datasets": int(vals.size),
            })
        return pd.DataFrame(records)
    summary = summarize_metric(rank_rows, methods, metric, anchor_method=ELO_ANCHOR_METHOD)
    return summary[["method", "method_order", "value", "n_datasets"]]


def budget_curves(output_dir: Path) -> pd.DataFrame:
    groups = dataset_size_groups()
    frames = []
    for scope, panel_order, budgets in (
        ("non_large", 0, BUDGETS),
        ("large", 1, LARGE_BUDGETS),
    ):
        datasets = _datasets_for_scope(groups, scope)
        for budget in budgets:
            rows = load_single_results(
                budget=budget,
                tabred_budget=budget if scope == "large" else TABRED_BUDGET,
                datasets=datasets,
                methods=BUDGET_CURVE_METHODS,
            )
            rank_rows = complete_rank_rows(rows, BUDGET_CURVE_METHODS)
            common = {
                "requested_budget": budget,
                "dataset_scope": scope,
                "panel_order": panel_order,
                "candidate_datasets": len(datasets),
                "complete_datasets": int(rank_rows["dataset"].nunique()) if not rank_rows.empty else 0,
            }
            for metric_order, (metric, label, lower_is_better) in enumerate(BUDGET_METRICS):
                summary = _budget_metric_summary(rank_rows, BUDGET_CURVE_METHODS, metric).assign(
                    **common,
                    metric_order=metric_order,
                    metric_display=label,
                    lower_is_better=lower_is_better,
                )
                summary["metric"] = metric
                summary["method_display"] = summary["method"].map(lambda m: METHOD_DISPLAY.get(m, m))
                frames.append(summary)
    curve = (
        pd.concat(frames, ignore_index=True)[[
            "requested_budget", "dataset_scope", "panel_order", "metric_order",
            "metric", "metric_display", "lower_is_better",
            "method", "method_display", "method_order", "value",
            "n_datasets", "candidate_datasets", "complete_datasets",
        ]]
        .sort_values(["panel_order", "metric_order", "requested_budget", "method_order"])
        .reset_index(drop=True)
    )
    curve.to_csv(output_dir / "budget_curve.csv", index=False)
    return curve


def save_budget_curve_figure(curve: pd.DataFrame, output_path: Path, *,
                             metrics: tuple[str, ...] | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = BUDGET_CURVE_METHODS
    method_pairs = (
        ("realmlp", "agentic-realmlp"),
        ("mlp", "agentic-mlp"),
        ("lightgbm", "agentic-lightgbm"),
        ("tabm", "agentic-tabm"),
        ("tabicl", "agentic-tabicl"),
    )
    pair_idx = {method: i for i, pair in enumerate(method_pairs) for method in pair}
    cmap = plt.get_cmap("tab10")
    method_colors = {method: cmap(pair_idx[method]) for method in methods if method in pair_idx}
    tabicl_set = set(TABICL_METHODS)
    panels = (
        ("non_large", "Small and medium datasets", BUDGETS, (40, max(BUDGETS) + 30)),
        ("large", "Large datasets", LARGE_BUDGETS, (45, 108)),
    )
    metrics = metrics or tuple(dict.fromkeys(curve["metric"]))
    fig_height = 2.2 * len(metrics)
    fig, axes = plt.subplots(
        len(metrics), 2, figsize=(11.0, fig_height),
        sharex="col", squeeze=False,
        gridspec_kw={"width_ratios": [3.0, 1.0], "wspace": 0.08, "hspace": 0.22},
    )

    for row_idx, metric in enumerate(metrics):
        metric_rows = curve[curve["metric"].eq(metric)]
        metric_display = metric_rows["metric_display"].iloc[0]
        invert_y = bool(metric_rows["lower_is_better"].iloc[0])
        for ax, (scope, title, xticks, xlim) in zip(axes[row_idx], panels, strict=True):
            panel = metric_rows[metric_rows["dataset_scope"].eq(scope)]
            for method in methods:
                method_curve = panel[panel["method"].eq(method)].sort_values("requested_budget")
                if method_curve.empty:
                    continue
                is_base = not method.startswith("agentic-")
                x = method_curve["requested_budget"].to_numpy(dtype=float)
                y = method_curve["value"].to_numpy(dtype=float)
                color = method_colors.get(method, cmap(0))
                if method in tabicl_set:
                    if np.isnan(y).all():
                        continue
                    ax.plot([float(min(xticks)), float(max(xticks))], [float(np.nanmean(y))] * 2,
                            color=color, alpha=0.55, linewidth=1.7,
                            linestyle=(0, (4, 2)) if is_base else "-")
                    continue
                ax.plot(
                    x, y, marker="o",
                    linewidth=1.75 if is_base else 2.15,
                    markersize=4.2,
                    linestyle=":" if is_base else "-",
                    color=color,
                    markerfacecolor="white" if is_base else color,
                    markeredgecolor=color,
                )
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("Budget")
            ax.set_xlim(*xlim)
            ax.set_xticks(list(xticks))
            ax.grid(True, axis="y", color="0.9", linewidth=0.8)
            ax.grid(True, axis="x", color="0.94", linewidth=0.6)
            if invert_y:
                ax.invert_yaxis()
        axes[row_idx, 0].set_ylabel(metric_display)

    handles, labels = [], []
    # The legend fills column-major, so listing each (base, agentic) pair in turn puts
    # all base methods on the first row and their agentic counterparts below them.
    for method in (method for pair in method_pairs for method in pair if method in set(methods)):
        is_base = not method.startswith("agentic-")
        color = method_colors.get(method, cmap(0))
        if method in tabicl_set:
            handles.append(Line2D(
                [0], [0], color=color, alpha=0.55,
                linestyle=(0, (4, 2)) if is_base else "-",
                linewidth=1.7,
            ))
        else:
            handles.append(Line2D(
                [0], [0], color=color, marker="o",
                linestyle=":" if is_base else "-",
                linewidth=1.75 if is_base else 2.15,
                markersize=4.2,
                markerfacecolor="white" if is_base else color,
                markeredgecolor=color,
            ))
        labels.append(METHOD_DISPLAY.get(method, method))
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )
    # Reserve a fixed strip (in inches) below the axes for tick labels, the x-label,
    # and the legend, so the short main_budget_curve variant doesn't overlap them.
    fig.subplots_adjust(left=0.08, right=0.99, top=1.0 - 0.5 / fig_height,
                        bottom=0.95 / fig_height, hspace=0.24, wspace=0.08)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_all_artifacts(output_dir: Path = ARTIFACTS_ROOT) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = dataset_size_groups()
    groups.to_csv(output_dir / "dataset_size_groups.csv", index=False)
    single, ensemble, _ = load_joint_results()
    write_per_dataset_tables(single, ensemble, output_dir / "per_dataset_tables")
    write_improvement_tables(single, output_dir, scope="single")
    write_improvement_tables(ensemble, output_dir, scope="ensemble")
    curve = budget_curves(output_dir)
    save_budget_curve_figure(curve, output_dir / "budget_curve.pdf")
    save_budget_curve_figure(curve, output_dir / "main_budget_curve.pdf", metrics=MAIN_BUDGET_METRICS)
    from bin.analysis.make_main_figure import save_main_figure
    save_main_figure(output_dir=output_dir)
