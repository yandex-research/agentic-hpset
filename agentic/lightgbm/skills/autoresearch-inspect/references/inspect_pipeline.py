from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from bin.lightgbm.pipeline import REGISTRY_KEY_MAP, get_registries


TASK_TYPES = ('regression', 'binclass', 'multiclass')


def main() -> None:
    print('Index flags:')
    for arg_name in REGISTRY_KEY_MAP:
        stage = arg_name[:-4]  # strip the trailing "_idx"
        print(f'  --{stage.replace("_", "-")}-idx maps to {stage}')
    print()
    print('Registry choices by task type:')
    for task_type in TASK_TYPES:
        print(f'[{task_type}]')
        registries = get_registries(task_type)
        for arg_name, registry_name in REGISTRY_KEY_MAP.items():
            choices = sorted(registries[registry_name])
            print(f'  {arg_name[:-4]}: {registry_name} choices={choices}')


if __name__ == '__main__':
    main()
