import logging
import os
from unittest import mock

import pytest
import torch
from torch._inductor import cudagraph_trees

import fme.core.compile
from fme.core.compile import (
    CUDAGRAPH_COMPILE_MODE,
    RECOMPILE_LIMIT,
    _parse_compile_backend_env,
    _parse_compile_env,
    compile_backend,
    compile_enabled,
    compile_module,
    is_compile_enabled,
    uncompiled,
)
from fme.core.device import force_cpu, get_device
from fme.core.optimization import Checkpoint
from fme.core.testing import skip_unless_cuda


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("1", True), ("0", False)]
    + [(value, ValueError) for value in ["true", " ", "2"]],
)
def test_parse_compile_env(value: str | None, expected: bool | None | type):
    if expected is ValueError:
        with pytest.raises(ValueError, match="FME_COMPILE must be '0' or '1'"):
            _parse_compile_env(value)
    else:
        assert _parse_compile_env(value) is expected


@pytest.mark.parametrize(
    "value, expected",
    [(None, None), ("", None), ("aot_eager", "aot_eager"), ("inductor", "inductor")]
    + [(value, ValueError) for value in ["not_a_backend", " ", "Inductor"]],
)
def test_parse_compile_backend_env(value: str | None, expected: str | None | type):
    # a typo must fail at startup, not at the first compile well into a run
    if expected is ValueError:
        with pytest.raises(ValueError, match="FME_COMPILE_BACKEND must name"):
            _parse_compile_backend_env(value)
    else:
        assert _parse_compile_backend_env(value) == expected


def _use_auto(monkeypatch: pytest.MonkeyPatch) -> None:
    # compile_enabled deliberately cannot express "unset", so reset the
    # override directly to exercise the default rule
    monkeypatch.setattr(fme.core.compile, "_COMPILE_OVERRIDE", None)
    monkeypatch.setattr(fme.core.compile, "_LOGGED_AUTO_SKIP", False)


def test_auto_compiles_on_cuda(monkeypatch: pytest.MonkeyPatch):
    _use_auto(monkeypatch)
    monkeypatch.setattr(fme.core.compile, "get_device", lambda: torch.device("cuda", 0))
    assert is_compile_enabled()


def test_auto_skips_and_logs_once_off_cuda(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    _use_auto(monkeypatch)
    with force_cpu(), caplog.at_level(logging.INFO):
        assert not is_compile_enabled()
        assert not is_compile_enabled()
    assert caplog.text.count("torch.compile skipped") == 1


def test_override_compiles_off_cuda():
    with force_cpu(), compile_enabled(True):
        assert is_compile_enabled()


@pytest.mark.parametrize("split_dim", ["azimuth", "polar"])
def test_spatial_split_skips_compile_and_logs_once(
    split_dim: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    """Splitting longitude (azimuth) or latitude (polar) disables compilation
    even when it is forced on."""
    monkeypatch.setattr(fme.core.compile.thd, f"{split_dim}_group_size", lambda: 2)
    monkeypatch.setattr(fme.core.compile, "_LOGGED_SPATIAL_SPLIT_SKIP", False)
    with compile_enabled(True), caplog.at_level(logging.INFO):
        assert not is_compile_enabled()
        assert not is_compile_enabled()
    assert caplog.text.count("spatial parallelism splits longitude or latitude") == 1


@pytest.mark.parallel
def test_compile_skipped_only_when_spatially_split():
    """Under the parallel test matrix, forced compilation stays on unless the
    model backend splits longitude or latitude."""
    splits_space = os.environ.get("FME_DISTRIBUTED_BACKEND") == "model" and (
        int(os.environ.get("FME_DISTRIBUTED_W", "1")) > 1
        or int(os.environ.get("FME_DISTRIBUTED_H", "1")) > 1
    )
    with compile_enabled(True):
        assert is_compile_enabled() == (not splits_space)


def test_compile_module_raises_recompile_limit():
    with torch._dynamo.config.patch(recompile_limit=8):
        compile_module(torch.nn.Identity(), backend="eager")
        assert torch._dynamo.config.recompile_limit == RECOMPILE_LIMIT


def test_hitting_recompile_limit_warns_and_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
):
    """Falling back to eager at the limit costs speed, so it must be visible.
    Dynamo warns on the ``torch._dynamo`` logger, which does not propagate to
    the root logger, so caplog's handler is attached to it directly. The
    suite-wide ``dynamo_hygiene`` fixture makes hitting the limit raise, so
    this test turns that off to exercise the production fallback."""
    net = torch.nn.Linear(2, 2)
    module = compile_module(net, backend="eager")
    dynamo_logger = logging.getLogger("torch._dynamo")
    dynamo_logger.addHandler(caplog.handler)
    try:
        with torch._dynamo.config.patch(
            recompile_limit=1, fail_on_recompile_limit_hit=False
        ):
            # a change in rank always recompiles, even with dynamic shapes
            for shape in [(2,), (3, 2), (4, 3, 2)]:
                x = torch.randn(*shape)
                torch.testing.assert_close(module(x), net(x))
    finally:
        dynamo_logger.removeHandler(caplog.handler)
    assert any(
        record.levelno == logging.WARNING and "recompile_limit" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        pytest.param({}, {"backend": "eager"}, id="configured_backend"),
        pytest.param(
            {"backend": "aot_eager", "fullgraph": True},
            {"backend": "aot_eager", "fullgraph": True},
            id="explicit_backend_wins",
        ),
        pytest.param(
            {"backend": "inductor"},
            {"backend": "inductor", "mode": CUDAGRAPH_COMPILE_MODE},
            id="inductor_uses_cudagraphs",
        ),
        pytest.param(
            {"backend": "inductor", "mode": "default"},
            {"backend": "inductor", "mode": "default"},
            id="explicit_mode_wins",
        ),
    ],
)
def test_compile_module_backend(kwargs: dict, expected: dict):
    net = torch.nn.Linear(2, 2)
    with (
        compile_backend("eager"),
        mock.patch("torch.compile", side_effect=lambda m, **_: m) as mock_compile,
    ):
        compile_module(net, **kwargs)
    mock_compile.assert_called_once_with(net, **expected)


def test_compiled_module_outputs_are_copies():
    """A compiled module's outputs never share memory with the compiled graph,
    so a later CUDA graph replay cannot overwrite outputs a caller kept, while
    gradients still flow through them."""
    x = torch.randn(2, 2, requires_grad=True)
    out = compile_module(torch.nn.Identity(), backend="eager")(x)
    assert out.untyped_storage().data_ptr() != x.untyped_storage().data_ptr()
    torch.testing.assert_close(out, x)
    out.sum().backward()
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_uncompiled():
    net = torch.nn.Linear(2, 2)
    assert uncompiled(compile_module(net, backend="eager")) is net
    assert uncompiled(net) is net


class _RecordsCompiling(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.compiling: list[bool] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.compiling.append(torch.compiler.is_compiling())
        return self.linear(x)


def test_checkpointed_compiled_module_runs_eagerly():
    """Activation checkpointing recomputes the forward pass during backward,
    where calling a module compiled with CUDA graphs could free saved
    activations of other steps, so a checkpointed compiled module runs (and
    recomputes) uncompiled, with the same gradients as the plain module."""
    net = _RecordsCompiling()
    compiled = compile_module(net, backend="aot_eager")
    x = torch.randn(3, 2)
    compiled(x)
    assert net.compiling == [True]

    net.compiling.clear()
    Checkpoint({})(compiled)(x).sum().backward()
    assert net.compiling == [False, False]  # forward, then recomputation
    checkpointed_grad = net.linear.weight.grad.clone()
    net.zero_grad()
    net(x).sum().backward()
    torch.testing.assert_close(checkpointed_grad, net.linear.weight.grad)


def _rollout(
    step: torch.nn.Module,
    x: torch.Tensor,
    n_steps: int,
    checkpoint_after: int | None = None,
) -> list[torch.Tensor]:
    outputs = []
    for i in range(n_steps):
        if checkpoint_after is not None and i >= checkpoint_after:
            x = Checkpoint({})(step)(x)
        else:
            x = step(x)
        outputs.append(x)
    return outputs


@pytest.mark.timeout(300)
@pytest.mark.parametrize("checkpoint_after", [None, 1])
def test_cudagraph_rollouts_match_eager(checkpoint_after: int | None):
    """Under CUDA graphs, as in production, multi-step rollouts keep every
    step's output intact and train like eager: in inference, where each call
    starts a new CUDA graph tree generation, and in training, with and without
    activation checkpointing of later steps. Repeated iterations exercise
    both recording and replaying the graphs."""
    skip_unless_cuda()
    torch.manual_seed(0)
    device = get_device()
    net = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.GELU(), torch.nn.Linear(16, 8)
    ).to(device)
    compiled = compile_module(net, backend="inductor")
    x = torch.randn(4, 8, device=device)
    n_steps = 3
    for _ in range(3):
        with torch.no_grad():
            expected = _rollout(net, x, n_steps)
            actual = _rollout(compiled, x, n_steps)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=1e-5, rtol=1e-5)

        net.zero_grad(set_to_none=True)
        torch.stack(_rollout(net, x, n_steps)).square().sum().backward()
        expected_grads = [p.grad.clone() for p in net.parameters()]
        net.zero_grad(set_to_none=True)
        actual_outputs = _rollout(compiled, x, n_steps, checkpoint_after)
        torch.stack(actual_outputs).square().sum().backward()
        for p, e in zip(net.parameters(), expected_grads):
            torch.testing.assert_close(p.grad, e, atol=1e-5, rtol=1e-5)
    # the compiled graphs ran as CUDA graphs rather than falling back
    manager = cudagraph_trees.get_manager(
        torch.cuda.current_device(), create_if_none_exists=False
    )
    assert manager is not None and len(manager.roots) > 0
