"""Helpers for tests that exercise ``torch.compile``."""

import contextlib
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch

from fme.core.compile import compile_backend, compile_fullgraph
from fme.core.device import get_device
from fme.core.step.args import StepArgs
from fme.core.step.step import StepABC

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


@pytest.fixture(autouse=True, scope="session")
def session_compile_fullgraph() -> Iterator[None]:
    """Make any graph break fail the test, unless ``FME_COMPILE_FULLGRAPH`` is
    set or the run is multi-process.

    DDP's forward graph-breaks by design, which ``fullgraph`` rejects; there,
    :func:`assert_step_modules_have_no_graph_breaks` checks breaks instead.
    """
    if (
        os.environ.get("FME_COMPILE_FULLGRAPH")
        or int(os.environ.get("WORLD_SIZE", "1")) > 1
    ):
        yield
        return
    with compile_fullgraph(True):
        yield


ALLOW_SEEDED_NOISE_GRAPH_BREAKS = pytest.mark.allow_graph_breaks(
    reason=(
        "seeded noise is drawn from a torch.Generator (fme.core.rand), which "
        "dynamo cannot pass into a graph"
    )
)


@pytest.fixture(autouse=True)
def dynamo_hygiene(request: pytest.FixtureRequest) -> Iterator[None]:
    """Reset dynamo around each test and raise on hitting the recompile limit.

    All compiled modules trace through one ``DummyWrapper.forward``, so tests
    sharing a worker would share its recompile budget and later ones would
    silently run eagerly. ``allow_graph_breaks(reason)`` turns off fullgraph.
    """
    marker = request.node.get_closest_marker("allow_graph_breaks")
    if marker is not None and not marker.kwargs.get("reason"):
        raise ValueError("allow_graph_breaks requires a reason")
    torch._dynamo.reset()
    try:
        with (
            torch._dynamo.config.patch(fail_on_recompile_limit_hit=True),
            compile_fullgraph(False)
            if marker is not None
            else contextlib.nullcontext(),
        ):
            yield
    finally:
        torch._dynamo.reset()


def skip_unless_cuda() -> None:
    """Skip unless on CUDA, the only device production compiles on."""
    if get_device().type != "cuda":
        pytest.skip("production compiles with inductor only on CUDA")


def _capture_module_calls(
    step: StepABC, args: StepArgs
) -> dict[int, tuple[tuple[Any, ...], dict[str, Any]]]:
    """Inputs of the first call to each module ``step`` runs, by module id."""
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
    inputs the step passes it.

    ``step`` must be built with compilation disabled, so each module is what
    ``Module.compile`` would hand ``torch.compile``.
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
