"""Read committed greedy-ensemble results from ``exp/ensembled``.

Each ``exp/ensembled/<method>/<dataset>/report.json`` records the selected member
indices/weights and the resulting blended test metric (metrics only — no raw
predictions). ``cfg_*.json`` alongside it holds each selected member's config.
The selection algorithm lives in :mod:`bin.analysis.ensemble_select`.
"""

from __future__ import annotations

import json

import pandas as pd

from bin.analysis.ensemble_select import score_to_signed_error
from bin.analysis.results_config import ENSEMBLED_ROOT


def load_committed_ensemble_results(output_root=ENSEMBLED_ROOT) -> pd.DataFrame:
    records = []
    for path in sorted(output_root.rglob("report.json")):
        row = json.loads(path.read_text())
        row["report_path"] = str(path)
        row["scope"] = "ensemble"
        row["score"] = row["test_score"]
        row["error"] = row["test_metric_error"]
        row["signed_error"] = row.get(
            "test_signed_error", score_to_signed_error(row["test_score"], row["score_name"])
        )
        row["best_model_error"] = row["best_member_test_metric_error"]
        records.append(row)
    return pd.DataFrame(records)
