import logging
import os

import pytest
import torch

import fme.core.compile
from fme.core.compile import (
    RECOMPILE_LIMIT,
    _parse_compile_env,
    compile_enabled,
    compile_module,
    is_compile_enabled,
)
from fme.core.device import force_cpu


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
