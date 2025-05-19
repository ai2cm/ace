import gc
import signal
from unittest import mock

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--fast",
        action="store_true",
        default=False,
        help="Skip slow tests",
    )
    parser.addoption(
        "--very-fast",
        action="store_true",
        default=False,
        help="Run only very fast tests (< 5 seconds)",
    )
    parser.addoption(
        "--no-timeout",
        action="store_true",
        default=False,
        help="Disable test timeout",
    )
    parser.addoption(
        "--meta-get-device",
        action="store_true",
        default=False,
        help=(
            "fme.get_device() returns torch.device('meta'). "
            "NOTE: This is an experimental option primarily for debugging device "
            "errors in a local (non-GPU) environment that will lead to unexpected "
            "results and failures in tests which rely on tensor values to check "
            "correctness. To work properly, get_device() must be called inside of "
            "your test function."
        ),
    )


@pytest.fixture
def skip_slow(request, very_fast_only):
    return very_fast_only or request.config.getoption("--fast")


@pytest.fixture
def very_fast_only(request):
    return request.config.getoption("--very-fast")


@pytest.fixture
def meta_get_device(request):
    return request.config.getoption("--meta-get-device")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Test took too long")


@pytest.fixture
def pdb_enabled(request):
    return request.config.getoption("--pdb")


@pytest.fixture
def no_timeout(request):
    return request.config.getoption("--no-timeout")


@pytest.fixture(autouse=True, scope="function")
def enforce_timeout(skip_slow, very_fast_only, pdb_enabled, no_timeout):
    if pdb_enabled or no_timeout:
        yield  # Do not enforce timeout if we are debugging
        return
    if very_fast_only:
        timeout_seconds = 3
    elif skip_slow:
        timeout_seconds = 30
    else:
        timeout_seconds = 90
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)  # Set the timeout for the test
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm after the test completes


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_call(item):
    try:
        yield
    except TimeoutException:
        pytest.fail("Test failed due to timeout")


@pytest.fixture(autouse=True)
def mock_gc_collect(monkeypatch):
    def mock_collect():
        pass

    monkeypatch.setattr(gc, "collect", mock_collect)


_original_cpu = torch.Tensor.cpu


def _mock_cpu(self, *args, **kwargs):
    try:
        return _original_cpu(self, *args, **kwargs)
    except NotImplementedError:
        return torch.rand_like(self, device=torch.device("cpu"))


@pytest.fixture(autouse=True)
def mock_get_device_to_meta(monkeypatch, meta_get_device):
    """
    Mocks the fme.core.device.get_device function to always return
    torch.device("meta") for all tests.
    """
    if meta_get_device:
        mock_meta_device_fn = mock.MagicMock(return_value=torch.device("meta"))

        import fme
        import fme.core.device

        monkeypatch.setattr(fme.core.device, "get_device", mock_meta_device_fn)
        monkeypatch.setattr(fme, "get_device", mock_meta_device_fn)
        monkeypatch.setattr(torch.Tensor, "cpu", _mock_cpu)
