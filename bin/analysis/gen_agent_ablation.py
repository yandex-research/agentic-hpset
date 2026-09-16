"""Generate the agent-ablation tables under ``artifacts/agent_ablation``.

Compares the ten ``mlp-ablation-<agent>-<idx>`` methods (the MLP agentic protocol
re-run end-to-end by two coding agents, five independent runs each) against the
``mlp`` baseline, with the paper's ``agentic-mlp`` as a reference row. Standalone
entry point — deliberately not part of ``bin.analysis.gen_analysis`` so the main
figures and tables are unaffected.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from bin.analysis.metrics import pct_improvement_from_errors
from bin.analysis.report_io import load_single_results
from bin.analysis.results_config import (
    AGENT_ABLATION_METHODS,
    ARTIFACTS_ROOT,
    DATASETS,
)

OUTPUT_DIR = ARTIFACTS_ROOT / "agent_ablation"
BASELINE_METHOD = "mlp"
PAPER_METHOD = "agentic-mlp"


def _agent_of(method: str) -> str:
    if method == PAPER_METHOD:
        return "paper"
    return method.removeprefix("mlp-ablation-").rsplit("-", 1)[0]


def _n_eval_seeds(report_path: str) -> int:
    report = json.loads(Path(report_path).read_text())
    return len(report.get("experiments") or [])


def per_dataset_table(rows: pd.DataFrame) -> pd.DataFrame:
    base = rows[rows["method"].eq(BASELINE_METHOD)].set_index("dataset")["signed_error"]
    records = []
    for method in (PAPER_METHOD, *AGENT_ABLATION_METHODS):
        for row in rows[rows["method"].eq(method)].itertuples(index=False):
            if row.dataset not in base.index:
                continue
            improvement = pct_improvement_from_errors(
                float(base[row.dataset]), float(row.signed_error)
            )
            records.append({
                "method": method,
                "agent": _agent_of(method),
                "dataset": row.dataset,
                "metric": row.metric,
                "score": row.score,
                "signed_error": row.signed_error,
                "mlp_signed_error": float(base[row.dataset]),
                "improvement_over_mlp_pct": improvement,
                "win": bool(improvement > 0),
                "n_eval_seeds": _n_eval_seeds(row.report_path),
            })
    return pd.DataFrame(records)


def missing_table(per_dataset: pd.DataFrame) -> pd.DataFrame:
    records = []
    for method in (PAPER_METHOD, *AGENT_ABLATION_METHODS):
        covered = set(per_dataset[per_dataset["method"].eq(method)]["dataset"])
        missing = [dataset for dataset in DATASETS if dataset not in covered]
        records.append({
            "method": method,
            "agent": _agent_of(method),
            "n_present": len(covered),
            "n_missing": len(missing),
            "missing_datasets": ";".join(missing),
        })
    return pd.DataFrame(records)


def summary_table(per_dataset: pd.DataFrame) -> pd.DataFrame:
    # Coverage differs per variant, so the mean over a variant's own datasets is not
    # directly comparable across variants; the common-subset column restricts every
    # method (including the paper's agentic MLP) to datasets all of them cover.
    common = set(DATASETS)
    for method in (PAPER_METHOD, *AGENT_ABLATION_METHODS):
        common &= set(per_dataset[per_dataset["method"].eq(method)]["dataset"])
    paper = per_dataset[per_dataset["method"].eq(PAPER_METHOD)].set_index("dataset")[
        "improvement_over_mlp_pct"
    ]

    def _summarize(sub: pd.DataFrame, method: str, agent: str, order: int) -> dict:
        vals = sub["improvement_over_mlp_pct"].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        on_common = sub[sub["dataset"].isin(common)]
        common_vals = on_common["improvement_over_mlp_pct"].to_numpy(dtype=float)
        common_vals = common_vals[~np.isnan(common_vals)]
        paper_vals = paper.reindex(sorted(set(sub["dataset"]))).dropna()
        return {
            "method": method,
            "agent": agent,
            "method_order": order,
            "n_datasets": int(sub["dataset"].nunique()),
            "n_missing": len(DATASETS) - int(sub["dataset"].nunique()),
            "n_wins": int(sub["win"].sum()),
            "win_rate": float(sub["win"].mean()) if len(sub) else float("nan"),
            "mean_improvement_pct": float(vals.mean()) if vals.size else float("nan"),
            "median_improvement_pct": float(np.median(vals)) if vals.size else float("nan"),
            # Paper agentic-mlp mean restricted to this method's covered datasets, so
            # each row has a like-for-like reference despite uneven coverage.
            "paper_mean_same_datasets": float(paper_vals.mean()) if len(paper_vals) else float("nan"),
            "mean_improvement_pct_common": float(common_vals.mean()) if common_vals.size else float("nan"),
            "n_common_datasets": len(common),
        }

    records = []
    for order, method in enumerate((PAPER_METHOD, *AGENT_ABLATION_METHODS)):
        sub = per_dataset[per_dataset["method"].eq(method)]
        records.append(_summarize(sub, method, _agent_of(method), order))
    # Pooled rows treat every (variant, dataset) comparison of one agent as a sample.
    ablation = per_dataset[per_dataset["method"].ne(PAPER_METHOD)]
    for order, agent in enumerate(sorted(ablation["agent"].unique()), start=100):
        sub = ablation[ablation["agent"].eq(agent)]
        records.append(_summarize(sub, f"{agent} (pooled)", agent, order))
    return pd.DataFrame(records).sort_values("method_order").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_single_results(
        methods=(BASELINE_METHOD, PAPER_METHOD, *AGENT_ABLATION_METHODS)
    )
    per_dataset = per_dataset_table(rows)
    missing = missing_table(per_dataset)
    summary = summary_table(per_dataset)
    per_dataset.to_csv(OUTPUT_DIR / "agent_ablation_per_dataset.csv", index=False)
    missing.to_csv(OUTPUT_DIR / "agent_ablation_missing.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "agent_ablation_summary.csv", index=False)

    print(f"Wrote agent-ablation artifacts under {OUTPUT_DIR}")
    with pd.option_context("display.width", 200):
        print(summary.drop(columns=["method_order"]).to_string(index=False))
    for row in missing.itertuples(index=False):
        if row.n_missing:
            print(f"missing {row.method} ({row.n_missing}): {row.missing_datasets}")


if __name__ == "__main__":
    main()
