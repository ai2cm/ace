import contextlib
import os
from collections.abc import Generator

_COMPILE_ENABLED: bool = os.environ.get("FME_COMPILE", "1") == "1"


@contextlib.contextmanager
def compile_enabled(enabled: bool = True) -> Generator[None, None, None]:
    """Enable or disable ``torch.compile`` of step modules within the context.

    Outside any such context, compilation is enabled unless the ``FME_COMPILE``
    environment variable is set to ``0``, which is a kill switch for runs that
    must stay eager (e.g. to rule out a compiler bug) without editing configs.

    Args:
        enabled: If False, modules a step would compile run eagerly instead.
    """
    global _COMPILE_ENABLED
    previous = _COMPILE_ENABLED
    try:
        _COMPILE_ENABLED = enabled
        yield
    finally:
        _COMPILE_ENABLED = previous


def is_compile_enabled() -> bool:
    return _COMPILE_ENABLED
