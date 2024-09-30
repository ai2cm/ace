import signal

import pytest


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


@pytest.fixture
def skip_slow(request, very_fast_only):
    return very_fast_only or request.config.getoption("--fast")


@pytest.fixture
def very_fast_only(request):
    return request.config.getoption("--very-fast")


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Test took too long")


@pytest.fixture
def pdb_enabled(request):
    return request.config.getoption("--pdb")


@pytest.fixture(autouse=True, scope="function")
def enforce_timeout(skip_slow, very_fast_only, pdb_enabled):
    if pdb_enabled:
        yield  # Do not enforce timeout if we are debugging
        return
    if very_fast_only:
        timeout_seconds = 3
    elif skip_slow:
        timeout_seconds = 30
    else:
        timeout_seconds = 60
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
