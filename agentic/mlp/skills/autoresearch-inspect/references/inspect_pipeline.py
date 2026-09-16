from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from bin.mlp.hparams import INDEX_KEYS, registry_for


TASK_TYPES = ('regression', 'classification')


def main() -> None:
    print('Index flags:')
    for key in INDEX_KEYS:
        stage = key[:-4]  # strip the trailing "_idx"
        print(f'  --{stage.replace("_", "-")}-idx maps to {stage}')
    print()
    print('Registry choices by task type:')
    for task_type in TASK_TYPES:
        print(f'[{task_type}]')
        for key in INDEX_KEYS:
            stage = key[:-4]
            choices = sorted(registry_for(key, task_type))
            print(f'  {stage}: choices={choices}')


if __name__ == '__main__':
    main()
