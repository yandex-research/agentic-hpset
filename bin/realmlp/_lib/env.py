# NOTE: this file must import only modules from the standard library.

import os
from pathlib import Path

_PROJECT_DIR: None | Path = None
_PYPROJECT_FILE_NAME = 'pyproject.toml'


def get_project_dir() -> Path:
    global _PROJECT_DIR

    if _PROJECT_DIR is None:
        # Walk up from this file to the repository root (the directory that holds
        # pyproject.toml); fall back to the known layout bin/realmlp/_lib/env.py.
        path = Path(__file__).resolve().parent
        while str(path) != path.root and not (path / _PYPROJECT_FILE_NAME).exists():
            path = path.parent
        _PROJECT_DIR = (
            path
            if (path / _PYPROJECT_FILE_NAME).exists()
            else Path(__file__).resolve().parents[3]
        )
    return _PROJECT_DIR


def get_exp_dir() -> Path:
    return get_project_dir() / 'exp'


def get_cache_dir() -> Path:
    path = get_project_dir() / 'cache'
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir() -> Path:
    return get_project_dir() / 'data'


def get_local_dir() -> Path:
    return get_project_dir() / 'local'


def get_snapshot_dir() -> None | Path:
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    return Path(snapshot_dir).resolve() if snapshot_dir else None


def get_tmp_output_dir() -> None | Path:
    tmp_output_dir = os.environ.get('TMP_OUTPUT_PATH')
    return Path(tmp_output_dir).resolve() if tmp_output_dir else None


def is_local() -> bool:
    return get_snapshot_dir() is None
