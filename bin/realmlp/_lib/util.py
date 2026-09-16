import dataclasses
import enum
import functools
import importlib
import inspect
import os
import subprocess
import sys
import types
import typing
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, NotRequired, Required

# NOTE
# This file must NOT import anything from lib except for `env`,
# because all other submodules are allowed to import `util`.
from . import env
from .types import AMPDType

try:
    _TERMINAL_SIZE = os.get_terminal_size().columns
except OSError:
    # Jupyter
    _TERMINAL_SIZE = 80
_SEPARATOR = '─' * _TERMINAL_SIZE

WORST_SCORE = -999999.0


def print_sep():
    print(_SEPARATOR)


def add_frame(text: str) -> str:
    lines = text.splitlines()
    width = max(map(len, lines))
    hline = '─' * (width + 2)
    return '\n'.join(
        [
            f'╭{hline}╮',
            *(f'│ {line}{" " * (width - len(line))} │' for line in lines),
            f'╰{hline}╯',
        ]
    )


def try_get_relative_path(path: str | Path) -> Path:
    path = Path(path).resolve()
    project_dir = env.get_project_dir()
    return path.relative_to(project_dir) if project_dir in path.parents else path


def is_typed_dict(type_) -> bool:
    return (
        issubclass(type_, dict)
        and hasattr(type_, '__required_keys__')
        and hasattr(type_, '__optional_keys__')
        and hasattr(type_, '__annotations__')
    )


def check_typed_dict[T](type_: type[T], dictionary: dict) -> T:
    if not is_typed_dict(type_):
        raise ValueError('type_ must be inherited from `typing.TypedDict`')

    presented_keys = frozenset(dictionary)
    required_keys = type_.__required_keys__  # type: ignore
    optional_keys = type_.__optional_keys__  # type: ignore

    if presented_keys < required_keys:
        raise ValueError(
            'The following required keys are missing:'
            f' {", ".join(required_keys - presented_keys)}'
        )
    if presented_keys > (required_keys | optional_keys):
        raise ValueError(
            'The following keys are unknown:'
            f' {", ".join(presented_keys - required_keys - optional_keys)}'
        )

    for key, value in dictionary.items():
        annotation = type_.__annotations__[key]
        if typing.get_origin(annotation) in (NotRequired, Required):
            annotation = typing.get_args(annotation)[0]
        while isinstance(annotation, typing.TypeAliasType):
            annotation = annotation.__value__
        if isinstance(annotation, type) and is_typed_dict(annotation):
            check_typed_dict(annotation, value)

    return typing.cast(T, dictionary)


def _parse_object_exception_handler(parse_object_fn) -> Any:
    """Add a note with the full key path to the innermost exception."""

    @functools.wraps(parse_object_fn)
    def wrapper(type_, object_, key):
        try:
            return parse_object_fn(type_, object_, key)
        except Exception as err:
            if key and not hasattr(err, '__notes__'):
                err.add_note(f'Problematic key: {".".join(key)}')
            raise err

    return wrapper


@_parse_object_exception_handler
def _parse_object(annotation: Any, object_: Any, key: tuple[str, ...]):
    # Unpack the original type.
    while isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__

    # Handle types.
    if isinstance(annotation, type):
        # Handle Any.
        if annotation is Any:
            return object_

        # Handle simple types.
        elif annotation in (
            types.NoneType,
            types.EllipsisType,
            bool,
            int,
            float,
            str,
            bytes,
        ):
            # Check the object type strictly.
            if type(object_) is not annotation:
                raise TypeError(
                    f'The expected object type is {annotation}, but the actual type is'
                    f' {type(object)}'
                )
            return object_  # type: ignore

        # Handle paths.
        elif annotation is Path:
            return annotation(object_)

        # Handle enums.
        elif issubclass(annotation, enum.Enum):
            return annotation(object_)

        # Handle dataclasses.
        elif dataclasses.is_dataclass(annotation):
            if isinstance(object_, annotation):  # type: ignore
                # The object is already an instance of the dataclass.
                return object_
            if not isinstance(object_, dict):
                raise TypeError(
                    'To be parsed as a dataclass, the object must be either an instance'
                    ' of the dataclass or a dictionary'
                )

            # Collect dataclass initialization fields.
            init_fields = {}
            for field in dataclasses.fields(annotation):
                if not field.init:
                    if field.name in object_:
                        raise ValueError(
                            f'The object contains a non-init field {field.name}'
                            f' of the dataclass, which is not allowed'
                        )
                    continue
                if field.default is dataclasses.MISSING and field.name not in object_:
                    raise ValueError(
                        f'The object is missing the required field "{field.name}"'
                        f' of the dataclass {annotation}'
                    )
                init_fields[field.name] = field

            # Parse the object's nested values.
            object_with_parsed_values = {}
            for k, v in object_.items():
                field = init_fields.get(k)
                if field is None:
                    raise ValueError(
                        f'The dataclass {annotation} does not have the field "{k}"'
                    )
                object_with_parsed_values[k] = _parse_object(field.type, v, (*key, k))
                del k, v

            # Return a dataclass instance.
            return annotation(**object_with_parsed_values)  # type: ignore

        else:
            raise ValueError(f'{annotation=} is not supported')

    # Handle type-like annotations.
    else:
        type_origin = typing.get_origin(annotation)
        type_args = typing.get_args(annotation)

        # Handle sequences and sets.
        if type_origin in (tuple, list, set, frozenset):
            if not type_args:
                raise ValueError(
                    f'The provided {annotation=} is missing type parameters'
                )
            # Handle all cases except for unnamed fixed-size tuples.
            if not issubclass(type_origin, tuple) or (
                len(type_args) == 2 and type_args[1] is ...
            ):
                return type_origin(
                    _parse_object(type_args[0], x, (*key, str(i)))
                    for i, x in enumerate(object_)
                )
            else:
                # Handle unnamed fixed-size tuples.
                if len(object_) != len(type_args):
                    raise ValueError(
                        f'The expected object size is {len(type_args)},'
                        f' but the actual size is {len(object_)}'
                    )
                return type_origin(  # type: ignore
                    _parse_object(type_args[i], x, (*key, str(i)))
                    for i, x in enumerate(object_)
                )

        # Handle dictionaries.
        elif type_origin is dict:
            if len(type_args) < 2:
                raise ValueError(
                    f'The provided {annotation=} is missing type parameters'
                )
            return type_origin(
                (
                    _parse_object(type_args[0], k, (*key, k, '<key>')),
                    _parse_object(type_args[1], v, (*key, k, '<value>')),
                )
                for k, v in object_.items()
            )

        # Handle literals.
        elif type_origin is typing.Literal:
            if object_ in typing.get_args(annotation):
                return object_
            else:
                raise ValueError(
                    f'The object {object_} is not a valid instance of {annotation}'
                )

        # Handle unions.
        elif type_origin in (
            types.UnionType,  # T1 | T2 | ...
            typing.Union,  # typing.Optional[T], typing.Union[T1, T2, ...]
        ):
            for type_candidate in type_args:
                try:
                    return _parse_object(type_candidate, object_, key)
                except Exception:
                    pass
            else:
                raise ValueError(
                    f'The object does not match any of the type variants {type_args}'
                )

        else:
            raise ValueError(f'{annotation=} is not supported')


def dataclass_from_dict[T](datacls: type[T], dict_: dict[str, Any]) -> T:
    """Create a dataclass instance from a dictionary and perform type checks.

    Supported field types:

    * Simple types: `None`, `...`, `bool`, `int`, `float`, `str`, `bytes`
    * `pathlib.Path`
    * Enums
    * Optional fields
    * Unions
    * Literals
    * Dataclasses
    * Built-in collections: `tuple`, `list`, `set`, `frozenset`, `dict`.
    * All combinations of the above field types. For example:
      `None | int | dict[str, tuple[MyDataClass, Path]]`

    NOTE
    All underlying type checks compare the content types _exactly_ against
    the expected types (i.e. using `is`, not `isinstance`).
    """
    assert dataclasses.is_dataclass(datacls), 'The first argument must be a dataclass.'

    return _parse_object(datacls, dict_, ())


def _flatten_dict(d: dict, key_prefix: str, result: dict) -> None:
    for k, v in d.items():
        new_k = f'{key_prefix}.{k}' if key_prefix else k
        if isinstance(v, dict):
            _flatten_dict(v, new_k, result)
        else:
            if result.setdefault(new_k, v) is not v:
                RuntimeError(
                    'Different parts of the dictionary resulted'
                    f' in the same flat key "{new_k}"'
                )


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    flat_d: dict[str, Any] = {}
    _flatten_dict(d, '', flat_d)
    return flat_d


def import_(fullname: str) -> Any:
    """
    Examples:

    >>> import_('bin.demo.main')
    """
    if fullname.count('.') == 0:
        raise ValueError('qualname must contain at least one dot')
    module_name, attr = fullname.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def get_function_full_name(function: Callable) -> str:
    """
    Examples:

    >>> # In the script bin/model.py
    >>> get_function_full_name(main) == 'bin.model.main'

    >>> # In the script a/b/c/foo.py
    >>> assert get_function_full_name(main) == 'a.b.c.foo.main'
    """
    module = inspect.getmodule(function)
    assert module is not None, 'Failed to locate the module of the function.'

    module_path = getattr(module, '__file__', None)
    assert module_path is not None, (
        'Failed to locate the module of the function.'
        ' This can happen if the code is running in a Jupyter notebook.'
    )

    module_path = Path(module_path).resolve()
    project_dir = env.get_project_dir()
    assert project_dir in module_path.parents, (
        'The module of the function must be located within the project directory: '
        f' {project_dir}'
    )

    module_full_name = str(
        module_path.relative_to(project_dir).with_suffix('')
    ).replace('/', '.')
    return f'{module_full_name}.{function.__name__}'  # ty:ignore[unresolved-attribute]


def get_device():  # -> torch.device
    import torch

    return torch.device(
        'cuda:0'
        if torch.cuda.is_available()
        # else 'mps:0'
        # if torch.mps.is_available()
        else 'cpu'
    )


def get_amp_dtype(
    dtype: AMPDType,
    device,  # torch.device
):  # -> torch.dtype
    import torch

    if dtype == 'bfloat16':
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError(
                f' The current {device.type.upper()} device'
                f' does not support {dtype} as the AMP data type'
            )
        return torch.bfloat16
    elif dtype == 'float16':
        return torch.float16
    else:
        raise ValueError(f'Unknown {dtype=}')


def is_oom_exception(err: RuntimeError) -> bool:
    import torch

    return isinstance(err, torch.cuda.OutOfMemoryError) or any(
        x in str(err)
        for x in [
            'CUDA out of memory',
            'CUBLAS_STATUS_ALLOC_FAILED',
            'CUDA error: out of memory',
        ]
    )


def adjust_gpu_memory_usage[**P, T](
    memory_parameter: str,
) -> Callable[[Callable[P, T]], Callable[P, tuple[T, int]]]:
    def decorator(f: Callable[P, T]) -> Callable[P, tuple[T, int]]:
        p = inspect.signature(f).parameters.get(memory_parameter)
        if p is None or p.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f'The function must have the keyword-only argument "{memory_parameter}"'
            )
        del p

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, int]:
            value: int = kwargs[memory_parameter]  # type: ignore
            if value <= 0:
                raise ValueError(f'{memory_parameter} must be a positive integer')
            while value:
                kwargs[memory_parameter] = value  # ty:ignore[invalid-assignment]
                try:
                    return f(*args, **kwargs), value
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    new_value = value // 2
                    message = (
                        f'Calling the function `{f.__name__}`'  # ty:ignore[unresolved-attribute]
                        f' with {memory_parameter}={value} triggers GPU OOM'
                    )
                    if new_value:
                        message += f'. Retrying with {memory_parameter}={new_value}'
                    import loguru

                    loguru.logger.warning(message)
                    value = new_value
            raise RuntimeError(f'Not enough memory even for {memory_parameter}=1')

        return wrapper

    return decorator


# NOTE: the following function should *not* cache its results.
def git_get_current_branch() -> str:
    return (
        subprocess.run(
            ['git', 'branch', '--show-current'], capture_output=True, check=True
        )
        .stdout.decode('utf-8')
        .strip()
    )


def configure_logging():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format='<level>{message}</level>')


def configure_torch():
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init(*, torch_: bool = True) -> None:
    configure_logging()
    if torch_:
        import torch

        if torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' not in os.environ:
            warnings.warn(
                'When CUDA is available, CUDA_VISIBLE_DEVICES should be set explicitly'
            )
        configure_torch()


_IS_NOTEBOOK = 'ipykernel_launcher' in sys.argv[0]


def is_notebook() -> bool:
    return _IS_NOTEBOOK


def init_notebook(*, torch_: bool = True) -> None:
    assert is_notebook()
    os.chdir(env.get_project_dir())
    init(torch_=torch_)


def are_valid_predictions(predictions: dict) -> bool:
    # predictions: dict[PartKey, np.ndarray]
    import numpy as np

    assert all(isinstance(x, np.ndarray) for x in predictions.values())
    return all(np.isfinite(x).all() for x in predictions.values())
