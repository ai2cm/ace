"""Helpers for tests that exercise ``torch.compile``."""

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
import torch

from fme.core.compile import compile_enabled


@contextlib.contextmanager
def compile_backend(backend: str) -> Iterator[None]:
    """Enable compilation and force every ``torch.compile`` call in the body to
    use ``backend``.

    Steps call ``torch.compile`` with its default backend, so there is no
    config-level way to pick one. The default inductor backend takes
    tens of seconds to compile even tiny networks, which does not fit the test
    suite's timeouts, while ``"aot_eager"`` still runs the full dynamo tracing
    and AOTAutograd path (so graph breaks and tracing errors still surface) in
    a fraction of the time.

    Args:
        backend: The ``torch.compile`` backend to substitute, e.g.
            ``"aot_eager"``.
    """
    real_compile = torch.compile

    def _compile(module: Any, **kwargs: Any) -> Any:
        return real_compile(module, **{**kwargs, "backend": backend})

    with compile_enabled(True), mock.patch("torch.compile", side_effect=_compile):
        yield


@pytest.fixture(autouse=True)
def compile_on() -> Iterator[None]:
    """Enable compilation for every test in a test module, on any device.

    By default tests compile only on CUDA; use this in modules where every
    test compiles, including on CPU. Elsewhere, :func:`compile_backend`
    enables compilation for the steps it builds.

    Import this name into a test module to make it autouse for that module::

        from fme.core.testing import compile_on  # noqa: F401
    """
    with compile_enabled(True):
        yield


@pytest.fixture(autouse=True)
def dynamo_hygiene() -> Iterator[None]:
    """Isolate each test's dynamo state from the rest of the suite.

    The root ``conftest.py`` imports this fixture, making it autouse for every
    test.

    Dynamo's compilation cache is keyed on code objects and is process-global,
    so compiled functions and guards leak between tests sharing an xdist
    worker; ``torch._dynamo.reset()`` before and after each test keeps a
    compile in one test from being served (or invalidated) by another. Every
    compiled module is traced through the same ``DummyWrapper.forward`` code
    object, so without a reset the recompile budget of that code object is
    shared by the whole worker, and later tests would silently run eagerly once
    it is used up.

    Within a test, hitting the recompile limit raises instead of falling back
    to eager, so that tests cannot pass while claiming to exercise compiled
    code they never ran. Tests of the production fallback patch
    ``torch._dynamo.config.fail_on_recompile_limit_hit`` back off.
    """
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
            yield
    finally:
        torch._dynamo.reset()
