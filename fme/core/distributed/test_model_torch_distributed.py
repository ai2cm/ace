"""
Note: AI was used to generate these unit tests."
"""

import os
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(
    "torch_available,h_parallel,w_parallel,expected",
    [
        # Both torch distributed and spatial parallelism enabled
        (True, "2", "1", True),
        # Torch distributed NOT available, spatial parallelism enabled
        (False, "2", "2", False),
        # Torch distributed available, spatial parallelism NOT enabled
        (True, "1", "1", False),
        # Only W_PARALLEL_SIZE > 1
        (True, "1", "3", True),
        # Only H_PARALLEL_SIZE > 1
        (True, "4", "1", True),
        # Environment variables missing (defaults to 1)
        (True, None, None, False),
        # Zero values (should be treated as disabled)
        (True, "0", "0", False),
    ],
    ids=[
        "both_true",
        "torch_false_spatial_true",
        "torch_true_spatial_false",
        "w_parallel_only",
        "h_parallel_only",
        "env_vars_missing",
        "env_vars_zero",
    ],
)
@patch("torch.distributed.is_available")
def test_is_available(
    mock_torch_available, torch_available, h_parallel, w_parallel, expected
):
    """Test ModelTorchDistributed.is_available() with various configurations."""
    mock_torch_available.return_value = torch_available

    # Build environment dict
    env_dict = {}
    if h_parallel is not None:
        env_dict["H_PARALLEL_SIZE"] = h_parallel
    if w_parallel is not None:
        env_dict["W_PARALLEL_SIZE"] = w_parallel

    clear_env = h_parallel is None and w_parallel is None

    with patch.dict(os.environ, env_dict, clear=clear_env):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

    assert result is expected
