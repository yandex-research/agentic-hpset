from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from bin.realmlp.modules import (
    IMPLEMENTATION_REGISTRIES,
    STAGE_KEYS,
    registry_name,
)


TASK_TYPES = ('regression', 'binclass', 'multiclass')


def main() -> None:
    print('Index flags:')
    for key in STAGE_KEYS:
        print(f'  --{key.replace("_", "-")}-idx maps to {key}')
    print()
    print('Registry choices by task type:')
    for task_type in TASK_TYPES:
        print(f'[{task_type}]')
        for key in STAGE_KEYS:
            name = registry_name(key, task_type)
            choices = sorted(IMPLEMENTATION_REGISTRIES[name])
            print(f'  {key}: {name} choices={choices}')


if __name__ == '__main__':
    main()
