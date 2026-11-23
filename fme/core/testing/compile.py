"""Helpers for tests that exercise ``torch.compile``."""

import os
from collections.abc import Iterator

import pytest
import torch

from fme.core.compile import compile_backend
from fme.core.device import get_device

# inductor's code generation takes tens of seconds per network; aot_eager
# still runs dynamo and AOTAutograd, so tracing errors and recompiles surface
TEST_COMPILE_BACKEND = "aot_eager"


@pytest.fixture(autouse=True, scope="session")
def session_compile_backend() -> Iterator[None]:
    """Compile with :data:`TEST_COMPILE_BACKEND` unless ``FME_COMPILE_BACKEND``
    is set, exporting it so subprocess entrypoints use it too.
    """
    if os.environ.get("FME_COMPILE_BACKEND"):
        yield
        return
    os.environ["FME_COMPILE_BACKEND"] = TEST_COMPILE_BACKEND
    try:
        with compile_backend(TEST_COMPILE_BACKEND):
            yield
    finally:
        del os.environ["FME_COMPILE_BACKEND"]


@pytest.fixture(autouse=True)
def dynamo_hygiene() -> Iterator[None]:
    """Reset dynamo around each test and raise on hitting the recompile limit.

    All compiled modules trace through one ``DummyWrapper.forward``, so tests
    sharing a worker would share its recompile budget and later ones would
    silently run eagerly.
    """
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
            yield
    finally:
        torch._dynamo.reset()


def skip_unless_cuda() -> None:
    """Skip unless on CUDA, the only device production compiles on."""
    if get_device().type != "cuda":
        pytest.skip("production compiles with inductor only on CUDA")
