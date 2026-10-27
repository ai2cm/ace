"""Helpers for tests that exercise ``torch.compile``."""

import os
from collections.abc import Iterator

import pytest
import torch

from fme.core.compile import compile_backend, compile_enabled

TEST_COMPILE_BACKEND = "aot_eager"
"""The backend the test suite compiles with unless ``FME_COMPILE_BACKEND`` is set.

Production compiles with inductor, whose code generation takes tens of seconds
per network, too slow for every test that builds a step. ``aot_eager`` still
runs dynamo tracing and AOTAutograd, so graph breaks, tracing errors and
recompiles surface across the whole suite, but runs the traced graph eagerly
instead of generating kernels. Inductor itself is covered by targeted tests
that select it with :func:`fme.core.compile.compile_backend`.
"""


@pytest.fixture(autouse=True, scope="session")
def session_compile_backend() -> Iterator[None]:
    """Compile with :data:`TEST_COMPILE_BACKEND` for the whole test session.

    The root ``conftest.py`` imports this fixture, making it autouse for every
    test. Whether modules compile at all is unchanged, so the suite compiles
    wherever production does (on CUDA). Setting ``FME_COMPILE_BACKEND`` runs
    the suite with that backend instead, e.g. ``FME_COMPILE_BACKEND=inductor``
    to run every test through production's backend.
    """
    if os.environ.get("FME_COMPILE_BACKEND"):
        yield
        return
    with compile_backend(TEST_COMPILE_BACKEND):
        yield


@pytest.fixture(autouse=True)
def compile_on() -> Iterator[None]:
    """Enable compilation for every test in a test module, on any device.

    By default tests compile only on CUDA; use this in modules where every
    test compiles, including on CPU. Elsewhere, wrap the code under test in
    :func:`fme.core.compile.compile_enabled`.

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
