from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from bin.tabicl.pipeline import implementation_choices, total_grid_size


TASK_TYPES = ('regression', 'binclass', 'multiclass')


def main() -> None:
    print('Recipe axes and choices by task type:')
    for task_type in TASK_TYPES:
        print(f'[{task_type}]  grid size = {total_grid_size(task_type)}')
        for axis, choices in implementation_choices(task_type).items():
            print(f'  {axis}: choices={list(choices)}')


if __name__ == '__main__':
    main()
