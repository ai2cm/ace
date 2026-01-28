"""
Note: AI was used to generate these unit tests."
"""

import os
from unittest.mock import patch


@patch("torch.distributed.is_available")
def test_is_available_both_true(mock_torch_available):
    """Test when torch distributed is available and spatial parallelism is enabled."""
    mock_torch_available.return_value = True

    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "2", "W_PARALLEL_SIZE": "1"}):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is True


@patch("torch.distributed.is_available")
def test_is_available_torch_false_spatial_true(mock_torch_available):
    """Test when torch distributed is NOT available
    but spatial parallelism is enabled."""
    mock_torch_available.return_value = False

    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "2", "W_PARALLEL_SIZE": "2"}):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is False


@patch("torch.distributed.is_available")
def test_is_available_torch_true_spatial_false(mock_torch_available):
    """Test when torch distributed is available
    but spatial parallelism is NOT enabled."""
    mock_torch_available.return_value = True

    with patch.dict(
        os.environ, {"H_PARALLEL_SIZE": "1", "W_PARALLEL_SIZE": "1"}, clear=True
    ):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is False


@patch("torch.distributed.is_available")
def test_is_available_with_w_parallel_size_only(mock_torch_available):
    """Test when only W_PARALLEL_SIZE > 1."""
    mock_torch_available.return_value = True

    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "1", "W_PARALLEL_SIZE": "3"}):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is True


@patch("torch.distributed.is_available")
def test_is_available_with_h_parallel_size_only(mock_torch_available):
    """Test when only H_PARALLEL_SIZE > 1."""
    mock_torch_available.return_value = True

    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "4", "W_PARALLEL_SIZE": "1"}):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is True


@patch("torch.distributed.is_available")
def test_is_available_env_vars_missing(mock_torch_available):
    """Test when environment variables are not set (defaults to 1)."""
    mock_torch_available.return_value = True

    # Clear the environment variables
    with patch.dict(os.environ, {}, clear=True):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is False


@patch("torch.distributed.is_available")
def test_is_available_env_vars_zero(mock_torch_available):
    """Test behavior with zero values (should be treated as disabled)."""
    mock_torch_available.return_value = True

    with patch.dict(os.environ, {"H_PARALLEL_SIZE": "0", "W_PARALLEL_SIZE": "0"}):
        from fme.core.distributed.model_torch_distributed import ModelTorchDistributed

        result = ModelTorchDistributed.is_available()

        assert result is False
