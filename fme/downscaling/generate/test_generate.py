from unittest.mock import MagicMock, patch

import pytest
import torch

from fme.downscaling.data import Topography
from fme.downscaling.generate.generate import Downscaler
from fme.downscaling.generate.output import OutputTarget
from fme.downscaling.predictors import PatchPredictionConfig, PatchPredictor

# Fixtures


@pytest.fixture
def mock_model():
    """Create a mock model with coarse_shape attribute."""
    model = MagicMock()
    model.coarse_shape = (16, 16)
    return model


@pytest.fixture
def mock_output_target():
    """Create a mock OutputTarget."""
    target = MagicMock(spec=OutputTarget)
    target.name = "test_target"
    target.save_vars = ["var1", "var2"]
    target.n_ens = 4
    target.patch = PatchPredictionConfig()
    target.data = []
    return target


@pytest.fixture
def mock_topography():
    """Create a mock Topography with shape."""

    def create_topo(shape=(16, 16)):
        topo = MagicMock(spec=Topography)
        data_mock = MagicMock()
        data_mock.shape = shape
        topo.data = data_mock
        topo.coords = MagicMock()
        return topo

    return create_topo


# Tests for Downscaler initialization


def test_downscaler_initialization(mock_model, mock_output_target):
    """Test that Downscaler can be instantiated with required fields."""
    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
        output_dir="/test/output",
    )

    assert downscaler.model is mock_model
    assert downscaler.output_targets == [mock_output_target]
    assert downscaler.output_dir == "/test/output"


# Tests for Downscaler._get_generation_model


def test_get_generation_model_exact_match(
    mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model returns model unchanged when shapes match exactly.
    """
    mock_model.coarse_shape = (16, 16)
    topo = mock_topography(shape=(16, 16))

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    result = downscaler._get_generation_model(
        topography=topo,
        target=mock_output_target,
    )

    assert result is mock_model


@pytest.mark.parametrize("topo_shape", [(8, 16), (16, 8), (8, 8)])
def test_get_generation_model_raises_when_domain_too_small(
    mock_model, mock_output_target, mock_topography, topo_shape
):
    """
    Test _get_generation_model raises ValueError when domain is
    smaller than model.
    """
    mock_model.coarse_shape = (16, 16)
    topo = mock_topography(shape=topo_shape)

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    with pytest.raises(ValueError):
        downscaler._get_generation_model(
            topography=topo,
            target=mock_output_target,
        )


@patch("fme.downscaling.generate.generate.PatchPredictor")
def test_get_generation_model_creates_patch_predictor_when_needed(
    mock_patch_predictor_cls, mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model creates PatchPredictor for
    large domains with patching.
    """
    mock_model.coarse_shape = (16, 16)
    topo = mock_topography(shape=(32, 32))  # Larger than model

    patch_config = PatchPredictionConfig(
        divide_generation=True,
        composite_prediction=True,
        coarse_horizontal_overlap=2,
    )
    mock_output_target.patch = patch_config
    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    model = downscaler._get_generation_model(
        topography=topo,
        target=mock_output_target,
    )

    assert isinstance(model, PatchPredictor)
    assert model.coarse_horizontal_overlap == 2


def test_get_generation_model_raises_when_large_domain_without_patching(
    mock_model, mock_output_target, mock_topography
):
    """
    Test _get_generation_model raises when domain is large but patching
    not configured.
    """
    mock_model.coarse_shape = (16, 16)
    topo = mock_topography(shape=(32, 32))  # Larger than model
    mock_output_target.patch = PatchPredictionConfig(divide_generation=False)

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    with pytest.raises(ValueError):
        downscaler._get_generation_model(
            topography=topo,
            target=mock_output_target,
        )


def test_run_target_generation_skips_padding_items(
    mock_model, mock_output_target, mock_topography
):
    """Test run_target_generation skips writing output for padding items."""
    # Create padding work item
    mock_work_item = MagicMock()
    mock_work_item.is_padding = True
    mock_work_item.n_ens = 4
    mock_work_item.batch = MagicMock()

    mock_topo = mock_topography(shape=(16, 16))

    mock_output_target.data = [(mock_work_item, mock_topo)]
    mock_model.coarse_shape = (16, 16)

    mock_output = {
        "var1": torch.randn(1, 4, 16, 16),
        "var2": torch.randn(1, 4, 16, 16),
    }
    mock_model.generate_on_batch_no_target.return_value = mock_output

    mock_writer = MagicMock()
    mock_output_target.get_writer.return_value = mock_writer

    downscaler = Downscaler(
        model=mock_model,
        output_targets=[mock_output_target],
    )

    downscaler.run_target_generation(target=mock_output_target)

    # Verify model was still called
    mock_model.generate_on_batch_no_target.assert_called_once()

    # Verify writer did NOT record the output
    mock_writer.record_batch.assert_not_called()


# Create an integration test for the full generation process


def test_generation_main():
    pass
