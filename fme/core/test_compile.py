import logging

import pytest
import torch

import fme.core.compile
from fme.core.compile import _parse_compile_env, compile_enabled, is_compile_enabled
from fme.core.device import force_cpu


def test_tests_run_with_compile_disabled_by_default():
    """The suite's conftest disables compilation for modules that do not opt in
    via ``fme.core.testing.dynamo_hygiene``."""
    assert not is_compile_enabled()


def test_compile_enabled_restores_previous_setting():
    with compile_enabled(True):
        assert is_compile_enabled()
        with compile_enabled(False):
            assert not is_compile_enabled()
        assert is_compile_enabled()
    assert not is_compile_enabled()


@pytest.mark.parametrize("value, expected", [(None, None), ("1", True), ("0", False)])
def test_parse_compile_env(value: str | None, expected: bool | None):
    assert _parse_compile_env(value) is expected


@pytest.mark.parametrize("value", ["true", "", "2"])
def test_parse_compile_env_rejects_other_values(value: str):
    with pytest.raises(ValueError, match="FME_COMPILE"):
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
