# import this before torch to avoid GLIBC error
import xarray

import pytest
import config

from modulus.distributed.manager import DistributedManager

@pytest.fixture()
def has_registry():
    if not config.MODEL_REGISTRY:
        pytest.skip("MODEL_REGISTRY not configured.")


@pytest.fixture()
def dist():
    DistributedManager.initialize()
    return DistributedManager()
