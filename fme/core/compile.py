import contextlib
import logging
import os
import warnings
from collections.abc import Generator
from typing import Any

from fme.core.device import get_device

logger = logging.getLogger(__name__)


def _parse_compile_env(value: str | None) -> bool | None:
    if value is None:
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(f"FME_COMPILE must be '0' or '1' if set, got '{value}'.")


_COMPILE_OVERRIDE: bool | None = _parse_compile_env(os.environ.get("FME_COMPILE"))
_LOGGED_AUTO_SKIP = False


@contextlib.contextmanager
def compile_enabled(enabled: bool = True) -> Generator[None, None, None]:
    """Force ``torch.compile`` of step modules on or off within the context.

    Args:
        enabled: If True, compile on any device. If False, run eagerly.
    """
    global _COMPILE_OVERRIDE
    previous = _COMPILE_OVERRIDE
    try:
        _COMPILE_OVERRIDE = enabled
        yield
    finally:
        _COMPILE_OVERRIDE = previous


def is_compile_enabled() -> bool:
    """Whether steps should compile the modules they own.

    By default, modules are compiled only when running on CUDA: inductor on CPU
    needs a C++ toolchain and fails on some networks, and on MPS it is
    experimental. Setting the ``FME_COMPILE`` environment variable to ``0``
    disables compilation everywhere, and setting it to ``1`` forces it on any
    device, in which case any compiler failure on that device is the user's to
    resolve. :func:`compile_enabled` overrides both within a context.
    """
    global _LOGGED_AUTO_SKIP
    if _COMPILE_OVERRIDE is not None:
        return _COMPILE_OVERRIDE
    device = get_device()
    if device.type == "cuda":
        return True
    if not _LOGGED_AUTO_SKIP:
        logger.info(
            f"torch.compile skipped: device is {device.type}; "
            "set FME_COMPILE=1 to force"
        )
        _LOGGED_AUTO_SKIP = True
    return False


def remove_deprecated_compile_key(state: dict[str, Any]) -> None:
    """Drop the removed step-config ``compile`` option from a serialized config.

    Steps now always compile when :func:`is_compile_enabled`, so checkpoints
    and configs that still set the option keep loading.
    """
    if "compile" in state:
        del state["compile"]
        warnings.warn(
            "The step config 'compile' option is deprecated and ignored: steps "
            "compile automatically on CUDA. Set the FME_COMPILE environment "
            "variable to '0' to disable compilation.",
            DeprecationWarning,
        )
