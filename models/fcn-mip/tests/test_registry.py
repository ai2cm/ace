from fcn_mip import registry
import config

import pytest


def test_list_models():
    if not config.MODEL_REGISTRY:
        pytest.skip("Model Registry not configured.")
    ans = registry.list_models()
    assert ans
    assert "/" not in ans[0], ans[0]
