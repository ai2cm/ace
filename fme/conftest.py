import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--fast", action="store_true", default=False, help="Run only fast tests"
    )


@pytest.fixture
def skip_slow(request):
    return request.config.getoption("--fast")
