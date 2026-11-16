"""Helpers for tests that exercise ``torch.compile``."""

import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch

from fme.core.compile import compile_backend, compile_fullgraph
from fme.core.device import get_device
from fme.core.step.args import StepArgs
from fme.core.step.step import StepABC

TEST_COMPILE_BACKEND = "aot_eager"
"""The backend the test suite compiles with unless ``FME_COMPILE_BACKEND`` is set.

Inductor's code generation takes tens of seconds per network, too slow for
every test that builds a step. ``aot_eager`` still runs dynamo tracing and
AOTAutograd, so graph breaks, tracing errors and recompiles surface across the
suite; targeted tests select inductor with
:func:`fme.core.compile.compile_backend`.
"""


@pytest.fixture(autouse=True, scope="session")
def session_compile_backend() -> Iterator[None]:
    """Compile with :data:`TEST_COMPILE_BACKEND` for the whole test session,
    unless ``FME_COMPILE_BACKEND`` is set (e.g. to ``inductor``).

    Applied to every test via ``pytest_plugins`` in the root ``conftest.py``.
    The backend is also exported as ``FME_COMPILE_BACKEND`` so that entrypoints
    run in a subprocess (e.g. under ``torchrun``) compile with it too.
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


@pytest.fixture(autouse=True, scope="session")
def session_compile_fullgraph() -> Iterator[None]:
    """Compile with ``fullgraph=True`` for a single-process test session,
    unless ``FME_COMPILE_FULLGRAPH`` is set.

    Production tolerates graph breaks, which cost speed but not correctness,
    so numerics tests alone would never catch one. With ``fullgraph`` on, a
    break in any module a test compiles fails that test, whatever network
    config it runs. Applied to every test via ``pytest_plugins`` in the root
    ``conftest.py``.

    Multi-process runs (``torchrun``, where ``WORLD_SIZE`` > 1) and the
    subprocesses tests launch keep the production default: distributed data
    parallel's forward graph-breaks by design before running the wrapped
    module, which ``fullgraph`` rejects. There, graph breaks are checked by
    tracing each module with ``torch._dynamo.explain`` instead (see
    :func:`assert_step_modules_have_no_graph_breaks`).
    """
    if (
        os.environ.get("FME_COMPILE_FULLGRAPH")
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    ):
        yield
        return
    with compile_fullgraph(True):
        yield


@pytest.fixture(autouse=True)
def dynamo_hygiene() -> Iterator[None]:
    """Isolate each test's dynamo state from the rest of the suite.

    Applied to every test via ``pytest_plugins`` in the root ``conftest.py``.
    Dynamo's cache is process-global, and every compiled module is traced
    through the same ``DummyWrapper.forward`` code object, so without a reset
    tests sharing an xdist worker would share its recompile budget, and later
    tests would silently run eagerly once it is used up.

    Within a test, hitting the recompile limit raises instead of falling back
    to eager, so tests cannot pass without running the compiled code they
    claim to exercise.
    """
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(fail_on_recompile_limit_hit=True):
            yield
    finally:
        torch._dynamo.reset()


def skip_unless_cuda() -> None:
    """Skip the calling test unless it runs on CUDA, where production compiles
    with inductor; inductor on CPU is opt-in and unsupported.
    """
    if get_device().type != "cuda":
        pytest.skip("production compiles with inductor only on CUDA")


def _capture_module_calls(
    step: StepABC, args: StepArgs
) -> dict[int, tuple[tuple[Any, ...], dict[str, Any]]]:
    """Run ``step`` once and return the inputs of the first call to each
    module it runs, keyed by the ``id`` of the callable that received them.
    """
    calls: dict[int, tuple[tuple[Any, ...], dict[str, Any]]] = {}

    def recording_wrapper(module: Callable[..., Any]) -> Callable[..., Any]:
        def record(*call_args: Any, **call_kwargs: Any) -> Any:
            calls.setdefault(id(module), (call_args, call_kwargs))
            return module(*call_args, **call_kwargs)

        return record

    step.step(args=args, wrapper=recording_wrapper)
    return calls


def assert_step_modules_have_no_graph_breaks(
    step: StepABC, args: StepArgs, training: bool
) -> None:
    """Assert every module ``step`` owns traces into a single graph, with the
    inputs the step actually passes it.

    ``step`` must be built with compilation disabled, so that each of its
    modules is exactly what ``Module.compile`` would hand ``torch.compile``.

    Args:
        step: The step whose modules to trace.
        args: Inputs for one call of the step.
        training: If True, trace in training mode with gradients enabled,
            otherwise in evaluation mode without (batch norm and dropout take
            different paths in each).
    """
    for module in step.modules:
        module.train(training)
    with torch.set_grad_enabled(training):
        calls = _capture_module_calls(step, args)
        for i, module in enumerate(step.modules):
            name = f"step.modules[{i}] ({type(module).__name__})"
            assert id(module) in calls, f"{name} was not called by the step"
            call_args, call_kwargs = calls[id(module)]
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(module)(*call_args, **call_kwargs)
            assert explanation.graph_break_count == 0, (
                name,
                explanation.break_reasons,
            )
            assert explanation.graph_count == 1, name
