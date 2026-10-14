"""Helpers for tests that exercise ``torch.compile``.

The ``dynamo_hygiene`` fixture lives here rather than in a ``conftest.py`` so
that test modules in any package (``fme/core``, ``fme/ace``, ...) can opt into
it with a single import; pytest picks up an autouse fixture from the test
module's own namespace, so importing this name into a test module makes it
autouse for that module and that module only. A ``conftest.py`` fixture would
either apply to unrelated tests in the same directory or need to be duplicated
per package.
"""

import contextlib
from collections.abc import Iterator
from typing import Any
from unittest import mock

import pytest
import torch

from fme.core.compile import compile_enabled


@contextlib.contextmanager
def compile_backend(backend: str) -> Iterator[None]:
    """Force every ``torch.compile`` call in the body to use ``backend``.

    Step and module configs only expose ``compile: bool``, so there is no
    config-level way to pick a backend. The default inductor backend takes
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

    with mock.patch("torch.compile", side_effect=_compile):
        yield


@pytest.fixture(autouse=True)
def dynamo_hygiene(compile_disabled: None) -> Iterator[None]:
    """Enable compilation and isolate a test module's dynamo state.

    The suite's ``compile_disabled`` fixture runs every test eagerly; this
    fixture depends on it so that it re-enables compilation afterwards.

    Dynamo's compilation cache is keyed on code objects and is process-global,
    so compiled functions and guards leak between tests sharing an xdist
    worker; ``torch._dynamo.reset()`` before and after each test keeps a
    compile in one test from being served (or invalidated) by another.

    Import this name into a test module to make it autouse for that module::

        from fme.core.testing import dynamo_hygiene  # noqa: F401
    """
    torch._dynamo.reset()
    try:
        with compile_enabled(True):
            yield
    finally:
        torch._dynamo.reset()
