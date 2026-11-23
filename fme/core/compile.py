import contextlib
import logging
import os
from collections.abc import Generator
from typing import Any

import torch
from torch import nn

from fme.core.device import get_device

logger = logging.getLogger(__name__)


def _parse_compile_env(value: str | None) -> bool | None:
    # an empty value (FME_COMPILE=) is treated as unset
    if not value:
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(
        f"FME_COMPILE must be '0' or '1' (or unset or empty), got {value!r}."
    )


DEFAULT_COMPILE_BACKEND = "inductor"  # torch.compile's own default


def _validate_backend(backend: str, source: str) -> str:
    # list_backends hides debug backends such as aot_eager unless asked for all
    available = torch._dynamo.list_backends(exclude_tags=())
    if backend not in available:
        raise ValueError(
            f"{source} must name a torch.compile backend, got {backend!r}; "
            f"available backends are {available}."
        )
    return backend


def _parse_compile_backend_env(value: str | None) -> str | None:
    # empty is unset; validating only set values keeps import cheap
    if not value:
        return None
    return _validate_backend(value, "FME_COMPILE_BACKEND")


_COMPILE_OVERRIDE: bool | None = _parse_compile_env(os.environ.get("FME_COMPILE"))
_BACKEND_OVERRIDE: str | None = _parse_compile_backend_env(
    os.environ.get("FME_COMPILE_BACKEND")
)
_LOGGED_AUTO_SKIP = False
RECOMPILE_LIMIT = 32


@contextlib.contextmanager
def compile_enabled(enabled: bool = True) -> Generator[None, None, None]:
    """Force compilation on (any device) or off within the context."""
    global _COMPILE_OVERRIDE
    previous = _COMPILE_OVERRIDE
    try:
        _COMPILE_OVERRIDE = enabled
        yield
    finally:
        _COMPILE_OVERRIDE = previous


@contextlib.contextmanager
def compile_backend(backend: str) -> Generator[None, None, None]:
    """Override ``FME_COMPILE_BACKEND`` within the context."""
    global _BACKEND_OVERRIDE
    _validate_backend(backend, "compile_backend")
    previous = _BACKEND_OVERRIDE
    try:
        _BACKEND_OVERRIDE = backend
        yield
    finally:
        _BACKEND_OVERRIDE = previous


def get_compile_backend() -> str:
    """The backend :func:`compile_module` uses.

    Inductor unless ``FME_COMPILE_BACKEND`` names another registered backend,
    e.g. ``aot_eager``, which skips kernel generation and compiles much faster.
    """
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    return DEFAULT_COMPILE_BACKEND


def is_compile_enabled() -> bool:
    """Whether steps should compile the modules they own.

    By default only on CUDA: inductor on CPU needs a C++ toolchain and fails on
    some networks, and on MPS it is experimental. ``FME_COMPILE=0`` disables
    compilation everywhere; ``FME_COMPILE=1`` forces it on any device.
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


def _raise_recompile_limit(limit: int) -> None:
    config = torch._dynamo.config
    # older torch versions name it cache_size_limit
    name = (
        "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    )
    if getattr(config, name) < limit:
        setattr(config, name, limit)


def compile_module(module: nn.Module, **kwargs: Any) -> nn.Module:
    """``torch.compile`` a module, first raising dynamo's recompile limit.

    All compiled modules share one recompile budget (default 8): they trace
    through the same ``DummyWrapper.forward`` without DDP, and on torch 2.10
    through one shared dynamo entry function even unwrapped. The train,
    validation and inference variants of several modules easily exhaust it, so
    raise it to :data:`RECOMPILE_LIMIT` (keeping larger user values); past it,
    dynamo warns and runs eagerly. An explicit ``backend`` kwarg overrides
    the configured default.
    """
    _raise_recompile_limit(RECOMPILE_LIMIT)
    kwargs.setdefault("backend", get_compile_backend())
    return torch.compile(module, **kwargs)
