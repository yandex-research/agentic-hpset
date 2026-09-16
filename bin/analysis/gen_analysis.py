from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bin.analysis.artifacts import build_all_artifacts
from bin.analysis.results_config import ARTIFACTS_ROOT


def main() -> None:
    build_all_artifacts(ARTIFACTS_ROOT)
    print(f"Wrote processed artifacts under {ARTIFACTS_ROOT}")


if __name__ == "__main__":
    main()
