"""Small runtime helpers vendored from the original experiment harness.

Only the GPU out-of-memory retry decorator used by the inference modules is
kept here; the rest of the original ``lib.util`` was tune/eval-driver plumbing
that is not part of the published model code.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable


def is_oom_exception(err: BaseException) -> bool:
    import torch

    if isinstance(err, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(err, RuntimeError):
        return False
    return "out of memory" in str(err).lower()


def adjust_gpu_memory_usage[**P, T](
    memory_parameter: str,
) -> Callable[[Callable[P, T]], Callable[P, tuple[T, int]]]:
    """Retry ``f`` with a halved ``memory_parameter`` on GPU OOM.

    The decorated function must take ``memory_parameter`` as a keyword-only
    argument (e.g. an eval batch size). On CUDA OOM the value is halved and the
    call retried until it succeeds or reaches 0. Returns ``(result, value)``
    where ``value`` is the batch size that actually fit.
    """

    def decorator(f: Callable[P, T]) -> Callable[P, tuple[T, int]]:
        p = inspect.signature(f).parameters.get(memory_parameter)
        if p is None or p.kind != inspect.Parameter.KEYWORD_ONLY:
            raise ValueError(
                f'The function must have the keyword-only argument "{memory_parameter}"'
            )
        del p

        @functools.wraps(f)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> tuple[T, int]:
            value: int = kwargs[memory_parameter]  # type: ignore[assignment]
            if value <= 0:
                raise ValueError(f"{memory_parameter} must be a positive integer")
            while value:
                kwargs[memory_parameter] = value  # type: ignore[assignment]
                try:
                    return f(*args, **kwargs), value
                except RuntimeError as err:
                    if not is_oom_exception(err):
                        raise
                    import torch

                    torch.cuda.empty_cache()
                    new_value = value // 2
                    message = (
                        f"Calling `{f.__name__}` with "
                        f"{memory_parameter}={value} triggered GPU OOM"
                    )
                    if new_value:
                        message += f". Retrying with {memory_parameter}={new_value}"
                    from loguru import logger

                    logger.warning(message)
                    value = new_value
            raise RuntimeError(f"Not enough memory even for {memory_parameter}=1")

        return wrapper

    return decorator
