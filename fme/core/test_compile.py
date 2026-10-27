import inspect
import logging
import os
from unittest import mock

import pytest
import torch

import fme.core.compile
from fme.core.compile import (
    DEFAULT_COMPILE_BACKEND,
    RECOMPILE_LIMIT,
    _parse_compile_backend_env,
    _parse_compile_env,
    compile_backend,
    compile_enabled,
    compile_module,
    get_compile_backend,
    is_compile_enabled,
)
from fme.core.device import force_cpu
from fme.core.testing import TEST_COMPILE_BACKEND


def test_compile_enabled_restores_previous_setting():
    previous = is_compile_enabled()
    with compile_enabled(True):
        assert is_compile_enabled()
        with compile_enabled(False):
            assert not is_compile_enabled()
        assert is_compile_enabled()
    assert is_compile_enabled() == previous


@pytest.mark.parametrize(
    "value, expected", [(None, None), ("", None), ("1", True), ("0", False)]
)
def test_parse_compile_env(value: str | None, expected: bool | None):
    assert _parse_compile_env(value) is expected


@pytest.mark.parametrize("value", ["true", " ", "2"])
def test_parse_compile_env_rejects_other_values(value: str):
    with pytest.raises(ValueError, match="FME_COMPILE must be '0' or '1'"):
        _parse_compile_env(value)


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


def test_longitude_split_skips_compile_and_logs_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Splitting longitude disables compilation even when it is forced on."""
    monkeypatch.setattr(fme.core.compile.thd, "azimuth_group_size", lambda: 2)
    monkeypatch.setattr(fme.core.compile, "_LOGGED_LONGITUDE_SPLIT_SKIP", False)
    with compile_enabled(True), caplog.at_level(logging.INFO):
        assert not is_compile_enabled()
        assert not is_compile_enabled()
    assert caplog.text.count("spatial parallelism splits longitude") == 1


@pytest.mark.parallel
def test_compile_skipped_only_when_longitude_split():
    """Under the parallel test matrix, forced compilation stays on unless the
    model backend splits longitude."""
    splits_longitude = (
        os.environ.get("FME_DISTRIBUTED_BACKEND") == "model"
        and int(os.environ.get("FME_DISTRIBUTED_W", "1")) > 1
    )
    with compile_enabled(True):
        assert is_compile_enabled() == (not splits_longitude)


def test_compile_module_raises_recompile_limit():
    with torch._dynamo.config.patch(recompile_limit=8):
        compile_module(torch.nn.Identity(), backend="eager")
        assert torch._dynamo.config.recompile_limit == RECOMPILE_LIMIT


def test_compile_module_keeps_larger_user_recompile_limit():
    with torch._dynamo.config.patch(recompile_limit=RECOMPILE_LIMIT + 1):
        compile_module(torch.nn.Identity(), backend="eager")
        assert torch._dynamo.config.recompile_limit == RECOMPILE_LIMIT + 1


def test_hitting_recompile_limit_warns_and_runs_eagerly(
    caplog: pytest.LogCaptureFixture,
):
    """Falling back to eager at the limit costs speed, so it must be visible.
    Dynamo warns on the ``torch._dynamo`` logger, which writes to stderr with
    its own handler and does not propagate to the root logger, so caplog's
    handler is attached to it directly. The suite-wide ``dynamo_hygiene``
    fixture makes hitting the limit raise, so this test turns that off to
    exercise the production fallback."""
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
    "value, expected",
    [(None, None), ("", None), ("aot_eager", "aot_eager"), ("inductor", "inductor")],
)
def test_parse_compile_backend_env(value: str | None, expected: str | None):
    assert _parse_compile_backend_env(value) == expected


@pytest.mark.parametrize("value", ["not_a_backend", " ", "Inductor"])
def test_parse_compile_backend_env_rejects_unknown_backends(value: str):
    """A typo must fail at startup rather than at the first compile, which may
    be well into a run."""
    with pytest.raises(ValueError, match="FME_COMPILE_BACKEND must name"):
        _parse_compile_backend_env(value)


def test_production_default_backend_is_torch_compile_default(
    monkeypatch: pytest.MonkeyPatch,
):
    """With nothing overriding it, modules compile with the backend a bare
    ``torch.compile`` call uses, i.e. inductor."""
    monkeypatch.setattr(fme.core.compile, "_BACKEND_OVERRIDE", None)
    torch_default = inspect.signature(torch.compile).parameters["backend"].default
    assert get_compile_backend() == DEFAULT_COMPILE_BACKEND == torch_default
    assert DEFAULT_COMPILE_BACKEND == "inductor"


def test_compile_backend_restores_previous_setting():
    previous = get_compile_backend()
    with compile_backend("eager"):
        assert get_compile_backend() == "eager"
        with compile_backend("inductor"):
            assert get_compile_backend() == "inductor"
        assert get_compile_backend() == "eager"
    assert get_compile_backend() == previous


def test_compile_backend_rejects_unknown_backend():
    previous = get_compile_backend()
    with pytest.raises(ValueError, match="compile_backend must name"):
        with compile_backend("not_a_backend"):
            pass
    assert get_compile_backend() == previous


def test_compile_module_uses_configured_backend():
    net = torch.nn.Linear(2, 2)
    with (
        compile_backend("eager"),
        mock.patch("torch.compile", side_effect=lambda m, **_: m) as mock_compile,
    ):
        compile_module(net)
    mock_compile.assert_called_once_with(net, backend="eager")


def test_compile_module_explicit_backend_wins():
    """A caller asking for a particular backend gets it; the configured backend
    is only the default."""
    net = torch.nn.Linear(2, 2)
    with (
        compile_backend("eager"),
        mock.patch("torch.compile", side_effect=lambda m, **_: m) as mock_compile,
    ):
        compile_module(net, backend="aot_eager", fullgraph=True)
    mock_compile.assert_called_once_with(net, backend="aot_eager", fullgraph=True)


def test_compile_module_runs_with_configured_backend():
    """The configured backend reaches dynamo: the compiled module runs and
    matches eager."""
    torch.manual_seed(0)
    net = torch.nn.Linear(2, 2)
    with compile_backend("aot_eager"):
        module = compile_module(net)
    x = torch.randn(3, 2)
    torch.testing.assert_close(module(x), net(x))
    assert torch._dynamo.utils.counters["stats"]["unique_graphs"] > 0


def test_test_session_compiles_with_test_backend():
    """The suite compiles with the fast ``aot_eager`` backend unless
    FME_COMPILE_BACKEND asks for another; targeted tests select inductor."""
    expected = os.environ.get("FME_COMPILE_BACKEND") or TEST_COMPILE_BACKEND
    assert TEST_COMPILE_BACKEND == "aot_eager"
    assert get_compile_backend() == expected
