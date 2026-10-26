import contextlib
import logging
import os
from collections.abc import Generator
from typing import Any

import torch
import torch_harmonics.distributed as thd
from torch import nn

from fme.core.device import get_device

logger = logging.getLogger(__name__)


def _parse_compile_env(value: str | None) -> bool | None:
    # an empty value (FME_COMPILE=) is treated as unset
    if value is None or value == "":
        return None
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError(
        f"FME_COMPILE must be '0' or '1' (or unset or empty), got {value!r}."
    )


DEFAULT_COMPILE_BACKEND = "inductor"
"""The backend modules compile with unless overridden: ``torch.compile``'s own
default, so production compiles exactly as a bare ``torch.compile`` would."""


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
    # an empty value (FME_COMPILE_BACKEND=) is treated as unset
    if value is None or value == "":
        return None
    return _validate_backend(value, "FME_COMPILE_BACKEND")


_COMPILE_OVERRIDE: bool | None = _parse_compile_env(os.environ.get("FME_COMPILE"))
_BACKEND_OVERRIDE: str | None = _parse_compile_backend_env(
    os.environ.get("FME_COMPILE_BACKEND")
)
_LOGGED_AUTO_SKIP = False
_LOGGED_LONGITUDE_SPLIT_SKIP = False
RECOMPILE_LIMIT = 32


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


@contextlib.contextmanager
def compile_backend(backend: str) -> Generator[None, None, None]:
    """Set the ``torch.compile`` backend that :func:`compile_module` uses
    within the context, overriding ``FME_COMPILE_BACKEND``.

    This selects only the backend; whether modules compile at all is still
    decided by :func:`is_compile_enabled`.

    Args:
        backend: A registered ``torch.compile`` backend, e.g. ``"inductor"``
            or ``"aot_eager"``.
    """
    global _BACKEND_OVERRIDE
    _validate_backend(backend, "compile_backend")
    previous = _BACKEND_OVERRIDE
    try:
        _BACKEND_OVERRIDE = backend
        yield
    finally:
        _BACKEND_OVERRIDE = previous


def get_compile_backend() -> str:
    """The ``torch.compile`` backend that :func:`compile_module` uses.

    This is :data:`DEFAULT_COMPILE_BACKEND` (inductor) unless the
    ``FME_COMPILE_BACKEND`` environment variable names another registered
    backend, e.g. ``aot_eager``, which traces with dynamo and AOTAutograd but
    runs the traced graph eagerly instead of generating kernels, so it compiles
    much faster. :func:`compile_backend` overrides both within a context.
    """
    if _BACKEND_OVERRIDE is not None:
        return _BACKEND_OVERRIDE
    return DEFAULT_COMPILE_BACKEND


def is_compile_enabled() -> bool:
    """Whether steps should compile the modules they own.

    By default, modules are compiled only when running on CUDA: inductor on CPU
    needs a C++ toolchain and fails on some networks, and on MPS it is
    experimental. Setting the ``FME_COMPILE`` environment variable to ``0``
    disables compilation everywhere, and setting it to ``1`` forces it on any
    device, in which case any compiler failure on that device is the user's to
    resolve. :func:`compile_enabled` overrides both within a context.

    Regardless of these settings, modules are never compiled when spatial
    parallelism splits longitude, where the compiled distributed inverse SHT
    fails.
    """
    global _LOGGED_AUTO_SKIP, _LOGGED_LONGITUDE_SPLIT_SKIP
    # TODO: remove this gate once the distributed inverse SHT compiles with
    # longitude split (https://github.com/ai2cm/ace/pull/1602)
    if thd.azimuth_group_size() > 1:
        if not _LOGGED_LONGITUDE_SPLIT_SKIP:
            logger.info(
                "torch.compile skipped: spatial parallelism splits longitude, "
                "where the compiled distributed inverse SHT fails"
            )
            _LOGGED_LONGITUDE_SPLIT_SKIP = True
        return False
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

    Without distributed data parallel every module is wrapped in the same
    ``DummyWrapper``, so all compiled modules are traced through one
    ``DummyWrapper.forward`` code object and share its recompile budget, which
    defaults to 8 and is easily used up by the train, validation and inference
    variants of several modules. This raises the process-global
    ``torch._dynamo.config.recompile_limit`` to :data:`RECOMPILE_LIMIT`,
    leaving any larger user-set value in place. When the limit is still hit,
    dynamo logs a warning on the ``torch._dynamo`` logger and runs the
    remaining variants eagerly.

    The backend is :func:`get_compile_backend` unless ``kwargs`` names one
    explicitly, in which case the explicit backend wins: a caller that asks for
    a particular backend needs that backend, while the configured one is only a
    default.

    Args:
        module: The module to compile.
        kwargs: Forwarded to ``torch.compile``.
    """
    _raise_recompile_limit(RECOMPILE_LIMIT)
    kwargs.setdefault("backend", get_compile_backend())
    return torch.compile(module, **kwargs)
