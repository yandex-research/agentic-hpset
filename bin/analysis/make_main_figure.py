#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bin.analysis.ensemble_io import load_committed_ensemble_results
from bin.analysis.report_io import load_single_results
from bin.analysis.results_config import (
    ARTIFACTS_ROOT,
    BOOTSTRAP_SEED,
    DATASETS,
    ENSEMBLE_TO_MAIN_METHOD,
    MAX_BUDGET,
)


N_BOOT = 1000
ANCHOR_METHOD = "MLP|base|single"
METHOD_TO_FAMILY_SPACE = {
    "mlp": ("MLP", "base"),
    "agentic-mlp": ("MLP", "agentic"),
    "tabm": ("TabM", "base"),
    "agentic-tabm": ("TabM", "agentic"),
    "realmlp": ("RealMLP", "base"),
    "agentic-realmlp": ("RealMLP", "agentic"),
    "lightgbm": ("LightGBM", "base"),
    "agentic-lightgbm": ("LightGBM", "agentic"),
    "tabicl": ("TabICL", "base"),
    "agentic-tabicl": ("TabICL", "agentic"),
}
ENSEMBLE_MAIN_METHODS = {
    "mlp", "agentic-mlp", "tabm", "agentic-tabm", "realmlp",
    "lightgbm", "agentic-lightgbm", "agentic-realmlp",
}
SPACE_SUFFIX = {
    "base": "",
    "agentic": " (agentic)",
}
FAM_TEX = {
    "MLP": "MLP$^\\dagger$",
    "TabM": "TabM$^\\dagger$",
    "RealMLP": "RealMLP",
    "LightGBM": "LightGBM",
    "TabICL": "TabICLv2",
}


def load_joint_errors(*, budget: int = MAX_BUDGET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return two aligned dataset x method matrices: ``err`` (1-score, for the normalized
    score) and ``signed_err`` (score / -score, for rank / Elo / improvement)."""
    rows = []
    single = load_single_results(budget=budget)
    for row in single.itertuples(index=False):
        fam, space = METHOD_TO_FAMILY_SPACE[row.method]
        rows.append({
            "dataset": row.dataset,
            "method": f"{fam}|{space}|single",
            "err": row.error,
            "signed_err": row.signed_error,
        })

    ensemble = load_committed_ensemble_results()
    for _, row in ensemble.iterrows():
        main_method = row.get("output_method") or ENSEMBLE_TO_MAIN_METHOD.get(row["method"], row["method"])
        if main_method not in ENSEMBLE_MAIN_METHODS:
            continue
        fam, space = METHOD_TO_FAMILY_SPACE[main_method]
        rows.append({
            "dataset": row["dataset"],
            "method": f"{fam}|{space}|ens",
            "err": row["error"],
            "signed_err": row["signed_error"],
        })

    frame = pd.DataFrame(rows)
    err_df = frame.pivot_table(index="dataset", columns="method", values="err", aggfunc="first")
    signed_df = frame.pivot_table(index="dataset", columns="method", values="signed_err", aggfunc="first")
    return err_df, signed_df.reindex(index=err_df.index, columns=err_df.columns)


def bt_elo(err_df: pd.DataFrame, *, anchor_method: str = ANCHOR_METHOD,
           anchor: float = 1000.0) -> pd.Series:
    methods = err_df.columns.tolist()
    n_methods = len(methods)
    wins = np.zeros((n_methods, n_methods))
    for row in err_df.to_numpy(dtype=float):
        present = np.where(~np.isnan(row))[0]
        for a in present:
            for b in present:
                if a == b:
                    continue
                if row[a] < row[b]:
                    wins[a, b] += 1.0
                elif row[a] == row[b]:
                    wins[a, b] += 0.5

    def nll(theta: np.ndarray) -> float:
        diff = theta[:, None] - theta[None, :]
        return float((wins * np.logaddexp(0.0, -diff)).sum())

    def grad(theta: np.ndarray) -> np.ndarray:
        diff = theta[:, None] - theta[None, :]
        probs = 1.0 / (1.0 + np.exp(-diff))
        return -((wins - (wins + wins.T) * probs).sum(axis=1))

    result = minimize(nll, np.zeros(n_methods), jac=grad, method="L-BFGS-B")
    theta = result.x - result.x.mean()
    elo = pd.Series(theta * 400.0 / np.log(10.0), index=methods)
    if anchor_method in elo.index:
        elo = elo - elo[anchor_method] + anchor
    return elo


def normalized_scores(err_df: pd.DataFrame) -> pd.DataFrame:
    values = err_df.to_numpy(dtype=float)
    median = np.nanmedian(values, axis=1, keepdims=True)
    best = np.nanmin(values, axis=1, keepdims=True)
    denom = np.where(median > best, median - best, 1.0)
    out = np.clip((median - values) / denom, 0.0, None)
    return pd.DataFrame(out, index=err_df.index, columns=err_df.columns)


def improvement_over_mlp(err_df: pd.DataFrame, *, reference: str = ANCHOR_METHOD) -> pd.Series:
    """Mean relative error reduction (%) of each method vs the base MLP, per shared dataset."""
    if reference not in err_df.columns:
        return pd.Series(np.nan, index=err_df.columns)
    ref = err_df[reference].to_numpy(dtype=float)
    valid_ref = np.isfinite(ref) & (ref != 0.0)
    out = {}
    for method in err_df.columns:
        vals = err_df[method].to_numpy(dtype=float)
        mask = valid_ref & np.isfinite(vals)
        out[method] = float(np.mean((ref[mask] - vals[mask]) / np.abs(ref[mask]) * 100.0)) if mask.any() else float("nan")
    return pd.Series(out)


def aggregate(err_df: pd.DataFrame, signed_df: pd.DataFrame) -> pd.DataFrame:
    # Rank / Elo / improvement use the signed error; normalized score keeps 1-score.
    ranks = signed_df.rank(axis=1, method="average")
    norm = normalized_scores(err_df)
    return pd.DataFrame({
        "mean_rank": ranks.mean(),
        "improvement_over_mlp": improvement_over_mlp(signed_df),
        "norm_score": norm.mean(),
        "elo": bt_elo(signed_df),
    })


def bootstrap_ci(err_df: pd.DataFrame, signed_df: pd.DataFrame, *,
                 n_boot: int = N_BOOT) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    datasets = err_df.index.to_numpy()
    stats = {key: [] for key in ("mean_rank", "improvement_over_mlp", "norm_score", "elo")}
    for _ in range(n_boot):
        sample = rng.choice(datasets, size=len(datasets), replace=True)
        agg = aggregate(err_df.loc[sample], signed_df.loc[sample])
        for key in stats:
            stats[key].append(agg[key])
    lows, highs = {}, {}
    for key, values in stats.items():
        frame = pd.concat(values, axis=1)
        lows[key] = frame.quantile(0.025, axis=1)
        highs[key] = frame.quantile(0.975, axis=1)
    return lows, highs


def pretty(method: str) -> str:
    fam, space, mode = method.split("|")
    label = FAM_TEX[fam]
    if mode == "ens":
        label += " ens."
    label += SPACE_SUFFIX.get(space, f" ({space})")
    return label


def figure_label(method: str) -> str:
    fam, space, _mode = method.split("|")
    label = FAM_TEX[fam]
    if space != "base":
        label += " (A)"
    return label


def write_missing_datasets_report(err_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Count, for every method variant, how many of the canonical datasets are missing."""
    universe = list(DATASETS)
    full = err_df.reindex(universe)

    per_method = []
    for method in full.columns:
        missing = [d for d in universe if pd.isna(full.loc[d, method])]
        per_method.append({
            "method": method,
            "label": pretty(method),
            "n_present": len(universe) - len(missing),
            "n_missing": len(missing),
            "missing_datasets": ";".join(missing),
        })
    report = pd.DataFrame(per_method).sort_values(["n_missing", "method"], ascending=[False, True])
    report.to_csv(output_dir / "missing_datasets.csv", index=False)

    by_dataset = full.isna().sum(axis=1).reindex(universe).rename("n_models_missing")
    by_dataset = by_dataset.reset_index().rename(columns={"index": "dataset"})
    by_dataset = by_dataset.sort_values(["n_models_missing", "dataset"], ascending=[False, True])
    by_dataset.to_csv(output_dir / "missing_datasets_by_dataset.csv", index=False)

    n_models = len(full.columns)
    print(f"Missing datasets (universe={len(universe)} datasets x {n_models} model variants):")
    print(f"  total missing method-dataset cells: {int(full.isna().to_numpy().sum())}")
    for row in report.itertuples(index=False):
        if row.n_missing:
            print(f"  {row.label}: {row.n_missing} missing")
    worst = by_dataset[by_dataset["n_models_missing"] > 0]
    if not worst.empty:
        print(f"  datasets missing from >=1 model: {len(worst)} (worst: "
              + ", ".join(f"{r.dataset}={r.n_models_missing}" for r in worst.head(5).itertuples(index=False)) + ")")
    return report


def save_main_figure(*, output_dir: Path = ARTIFACTS_ROOT, budget: int = MAX_BUDGET) -> pd.DataFrame:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    errors, signed_errors = load_joint_errors(budget=budget)
    agg = aggregate(errors, signed_errors)
    ci_low, ci_high = bootstrap_ci(errors, signed_errors)
    order = agg.sort_values("mean_rank").index.tolist()

    out = agg.loc[order].copy()
    for key in ("mean_rank", "improvement_over_mlp", "norm_score", "elo"):
        out[f"{key}_ci_lo"] = ci_low[key].loc[order]
        out[f"{key}_ci_hi"] = ci_high[key].loc[order]
    out["n_datasets"] = errors.notna().sum().loc[order]
    out.insert(0, "method", out.index)
    out["label"] = [figure_label(method) for method in order]
    out.to_csv(output_dir / "main_summary_fixed_budget.csv", index=False)

    write_missing_datasets_report(errors, output_dir)

    # Two separate blocks: individual models on top, greedy ensembles below, split by a
    # dotted divider. Bars are drawn independently (no stacking) so the CIs stay readable.
    singles = sorted((k for k in agg.index if k.endswith("|single")),
                     key=lambda k: float(agg.loc[k, "mean_rank"]))
    enses = sorted((k for k in agg.index if k.endswith("|ens")),
                   key=lambda k: float(agg.loc[k, "mean_rank"]))

    n_ens = len(enses)
    n_single = len(singles)
    block_gap = 1.4
    # Best method sits at the top of each block; the individual block is stacked above.
    y_single = (np.arange(n_single)[::-1] + n_ens + block_gap).astype(float)
    y_ens = np.arange(n_ens)[::-1].astype(float)
    rows = list(zip(y_single, singles)) + list(zip(y_ens, enses))
    divider_y = n_ens + (block_gap - 1.0) / 2.0

    c_agentic = "#534AB7"
    c_base = "#B7B4AC"
    e_agentic = "#26215C"
    e_base = "#5F5E5A"

    plt.rcParams.update({
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 7.5,
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "hatch.linewidth": 0.5,
    })

    panels = [
        ("mean_rank", "Mean rank $\\downarrow$"),
        ("improvement_over_mlp", "Improvement over MLP (%) $\\uparrow$"),
        ("norm_score", "Normalized score $\\uparrow$"),
        ("elo", "Elo $\\uparrow$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 5.0), sharey=True)

    for ax, (key, title) in zip(axes, panels, strict=True):
        values, los, his = [], [], []
        for yi, mkey in rows:
            _, space, _mode = mkey.split("|")
            face = c_base if space == "base" else c_agentic
            edge = e_base if space == "base" else e_agentic
            value = float(agg.loc[mkey, key])
            lo = float(ci_low[key].loc[mkey])
            hi = float(ci_high[key].loc[mkey])
            ax.barh(yi, value, height=0.72, color=face, edgecolor=edge, linewidth=0.5, zorder=3)
            ax.errorbar(value, yi, xerr=[[value - lo], [hi - value]], fmt="none",
                        ecolor="#2C2C2A", elinewidth=0.7, capsize=1.5, capthick=0.7, zorder=4)
            values.append(value)
            los.append(lo)
            his.append(hi)
        ax.axhline(divider_y, color="#2C2C2A", linewidth=0.8, linestyle=(0, (2, 2)), zorder=5)
        ax.set_title(title, pad=4)
        ax.set_yticks([yi for yi, _ in rows])
        ax.set_ylim(-0.7, float(y_single.max()) + 0.7)
        ax.grid(axis="x", color="#DDDBD4", linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="y", length=0)
        # Robust x-limits from bar values + typical whisker, so a sparse method's huge CI
        # clips at the edge instead of rescaling the panel.
        values, los, his = np.array(values), np.array(los), np.array(his)
        upper = min(float(np.nanmax(his)), float(np.nanmax(values) + 3.0 * np.nanmedian(his - values)))
        if key in ("elo", "improvement_over_mlp"):
            lower = min(float(np.nanmin(los)),
                        float(np.nanmin(values) - 3.0 * np.nanmedian(values - los)))
            if key == "elo":
                ax.set_xlim(np.floor(max(0.0, lower) / 50) * 50, upper)
            else:
                ax.set_xlim(min(0.0, lower), upper)
                ax.axvline(0.0, color="#2C2C2A", linewidth=0.6, zorder=2)
        else:
            ax.set_xlim(0, upper)

    axes[0].set_yticklabels([figure_label(mkey) for _, mkey in rows])

    fig.subplots_adjust(left=0.13, right=0.995, top=0.94, bottom=0.085, wspace=0.08)

    # Rotated block labels in the left margin (computed in figure coords from the axis box).
    bbox = axes[0].get_position()
    ylim0, ylim1 = axes[0].get_ylim()

    def _fig_y(data_y: float) -> float:
        return bbox.y0 + (data_y - ylim0) / (ylim1 - ylim0) * bbox.height

    fig.text(0.018, _fig_y((y_single.min() + y_single.max()) / 2.0), "Individual",
             rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")
    fig.text(0.018, _fig_y((y_ens.min() + y_ens.max()) / 2.0), "Ensemble",
             rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c_base, ec=e_base, lw=0.5),
        plt.Rectangle((0, 0), 1, 1, color=c_agentic, ec=e_agentic, lw=0.5),
    ]
    fig.legend(
        handles,
        ["Base space", "Agentic space"],
        ncol=2,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        fontsize=7,
        handlelength=1.4,
        handleheight=0.9,
        columnspacing=1.6,
    )
    fig.savefig(output_dir / "main_summary_fixed_budget.pdf")
    plt.close(fig)
    return out


def main() -> None:
    save_main_figure(output_dir=ARTIFACTS_ROOT)
    print(f"Wrote {ARTIFACTS_ROOT / 'main_summary_fixed_budget.pdf'}")


if __name__ == "__main__":
    main()
