import gc
import signal
from pathlib import Path
from unittest import mock

import fasteners
import pytest
import torch
import xdist

from fme.core.distributed.distributed import Distributed


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "parallel: mark test to work when run in parallel (e.g. torchrun)",
    )
    config.addinivalue_line(
        "markers",
        "serial: test must run without interference from any other test process",
    )


def _lock_path(config: pytest.Config) -> Path:
    # Must be a path shared by all xdist workers.
    # Using the repo root is usually simplest.
    return config.rootpath / ".pytest-serial.lock"


@pytest.fixture(scope="session")
def _rw_lock(pytestconfig: pytest.Config) -> fasteners.InterProcessReaderWriterLock:
    return fasteners.InterProcessReaderWriterLock(str(_lock_path(pytestconfig)))


@pytest.fixture(autouse=True)
def _serialize_when_needed(
    request: pytest.FixtureRequest,
    _rw_lock: fasteners.InterProcessReaderWriterLock,
):
    is_serial = request.node.get_closest_marker("serial") is not None

    if is_serial:
        with _rw_lock.write_lock():
            yield
    else:
        with _rw_lock.read_lock():
            yield


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Only the controller should remove the file.
    # In a non-xdist run, just remove it from the main process.
    is_controller = not hasattr(
        session.config, "workerinput"
    ) or xdist.is_xdist_controller(session)
    if is_controller:
        _lock_path(session.config).unlink(missing_ok=True)


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


@pytest.fixture(autouse=True, scope="session")
def distributed_context():
    with Distributed.context():
        yield


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
def reset_global_timer():
    import fme.core.timing

    fme.core.timing.singleton = None
    yield
    fme.core.timing.singleton = None


@pytest.fixture(autouse=True)
def mock_gc_collect(monkeypatch):
    def mock_collect(*args, **kwargs):
        pass

    monkeypatch.setattr(gc, "collect", mock_collect)


_original_cpu = torch.Tensor.cpu


def _mock_cpu(self, *args, **kwargs):
    try:
        return _original_cpu(self, *args, **kwargs)
    except NotImplementedError:
        return torch.rand_like(self, dtype=torch.float32, device=torch.device("cpu"))


_original_to = torch.Tensor.to


def _mock_to(self, *args, **kwargs):
    try:
        return _original_to(self, *args, **kwargs)
    except NotImplementedError:
        return torch.rand_like(self, dtype=torch.float32, device=torch.device("cpu"))


@pytest.fixture(autouse=True)
def mock_get_device_to_meta(monkeypatch, meta_get_device):
    """Mocks the fme.core.device.get_device function to always return
    torch.device("meta") for all tests if `meta_get_device == True`.

    Meta devices have metadata (e.g., a shape) but no actual data and therefore
    cannot be used in general as a replacement for CUDA devices in the tests.
    However, this mock can be useful for debugging device-related issues.

    Because tensors on the meta device have no data, certain torch.Tensor
    methods will raise an error when called on a meta tensor. For this reason,
    this fixture also mocks torch.Tensor.cpu and torch.Tensor.to to catch these
    errors and return random data, allowing the tests to proceed. As such,
    meta_get_device should only be used selectively
    for local debugging of tensor-on-wrong-device errors when only a CPU is
    available.

    See https://docs.pytorch.org/docs/stable/meta.html for more information on
    torch.device("meta").

    """
    if meta_get_device:
        mock_meta_device_fn = mock.MagicMock(return_value=torch.device("meta"))

        import fme
        import fme.core.device

        monkeypatch.setattr(fme.core.device, "get_device", mock_meta_device_fn)
        monkeypatch.setattr(fme, "get_device", mock_meta_device_fn)
        monkeypatch.setattr(torch.Tensor, "cpu", _mock_cpu)
        monkeypatch.setattr(torch.Tensor, "to", _mock_to)
