import pytest

from fme.core.distributed import Distributed


@pytest.fixture(autouse=True)
def reset_distributed_singleton():
    """Reset the Distributed singleton after each test to prevent state leakage.

    Note: we do NOT destroy the torch.distributed process group here because
    destroying mid-session prevents re-initialization for subsequent tests.
    A session-scoped fixture below will attempt a clean shutdown at the end
    of the test session to avoid leaking resources on process exit.
    """
    yield
    Distributed.reset()
