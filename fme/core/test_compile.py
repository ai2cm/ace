import logging
from unittest import mock

import pytest
import torch

import fme.core.compile
from fme.core.compile import (
    RECOMPILE_LIMIT,
    _parse_compile_backend_env,
    _parse_compile_env,
    _parse_flag_env,
    compile_backend,
    compile_enabled,
    compile_fullgraph,
    compile_module,
    is_compile_enabled,
)
from fme.core.device import force_cpu


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


def test_parse_flag_env_names_the_variable_in_errors():
    with pytest.raises(ValueError, match="FME_COMPILE_FULLGRAPH must be '0' or '1'"):
        _parse_flag_env("FME_COMPILE_FULLGRAPH", "true")


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


@pytest.mark.parametrize("value", [None, ""])
def test_unset_compile_backend_env_skips_listing_backends(value: str | None):
    # listing backends is slow, and importing fme must not pay for it
    with mock.patch("torch._dynamo.list_backends") as mock_list_backends:
        assert _parse_compile_backend_env(value) is None
    mock_list_backends.assert_not_called()


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


@pytest.mark.parallel
def test_forced_compile_stays_on_under_parallelism():
    """Under the parallel test matrix, including spatial splits of longitude
    and latitude, forcing compilation on enables it."""
    with compile_enabled(True):
        assert is_compile_enabled()


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
    suite-wide ``dynamo_hygiene`` and ``session_compile_fullgraph`` fixtures
    make hitting the limit raise, so this test turns both off to exercise the
    production fallback."""
    net = torch.nn.Linear(2, 2)
    module = compile_module(net, backend="eager", fullgraph=False)
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
    "fullgraph, kwargs, expected",
    [
        pytest.param(
            False, {}, {"backend": "eager", "fullgraph": False}, id="configured"
        ),
        pytest.param(
            True,
            {},
            {"backend": "eager", "fullgraph": True},
            id="configured_fullgraph",
        ),
        pytest.param(
            False,
            {"backend": "aot_eager", "fullgraph": True},
            {"backend": "aot_eager", "fullgraph": True},
            id="explicit_options_win",
        ),
    ],
)
def test_compile_module_options(fullgraph: bool, kwargs: dict, expected: dict):
    net = torch.nn.Linear(2, 2)
    with (
        compile_backend("eager"),
        compile_fullgraph(fullgraph),
        mock.patch("torch.compile", side_effect=lambda m, **_: m) as mock_compile,
    ):
        compile_module(net, **kwargs)
    mock_compile.assert_called_once_with(net, **expected)


def test_test_session_compiles_with_fullgraph():
    """The suite-wide fixture makes any graph break in a compiled module fail
    the test that runs it, rather than silently splitting the graph."""

    class _BreaksGraph(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch._dynamo.graph_break()
            return x + 1

    compiled = compile_module(_BreaksGraph(), backend="eager")
    with pytest.raises(torch._dynamo.exc.Unsupported):
        compiled(torch.zeros(2))


@pytest.mark.allow_graph_breaks(reason="the module graph-breaks on purpose")
def test_allow_graph_breaks_marker_compiles_without_fullgraph():
    """A test marked with a known graph break still compiles its modules,
    without failing on the break."""

    class _BreaksGraph(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            torch._dynamo.graph_break()
            return x + 1

    compiled = compile_module(_BreaksGraph(), backend="eager")
    torch.testing.assert_close(compiled(torch.zeros(2)), torch.ones(2))
