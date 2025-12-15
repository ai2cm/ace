import os
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from fme.core.device import get_device
from fme.core.testing import validate_tensor

DIR = os.path.abspath(os.path.dirname(__file__))

from fme.ace.models.graphcast import GRAPHCAST_AVAIL
from fme.ace.models.graphcast.main import GraphCast
from fme.core.dataset_info import DatasetInfo
from fme.core.mask_provider import MaskProvider


def dummy_datasetinfo(height: int, width: int) -> DatasetInfo:
    """Create a dummy DatasetInfo for testing."""
    # Create dummy lat/lon coordinates
    lat = np.linspace(-90, 90, height)
    lon = np.linspace(-180, 180, width)
    lonM, latM = np.meshgrid(lon, lat)
    lonT = torch.from_numpy(lonM).float()
    latT = torch.from_numpy(latM).float()

    # Create mock horizontal coordinates
    mock_horizontal_coords = Mock()
    mock_horizontal_coords.coords = {"lat": lat, "lon": lon}
    mock_horizontal_coords.meshgrid = (latT, lonT)

    mask_provider = MaskProvider(
        masks={"mask_2d": torch.ones(height, width, dtype=torch.bool)}
    )

    # Create DatasetInfo with mocked components
    dataset_info = DatasetInfo(
        horizontal_coordinates=mock_horizontal_coords,
        mask_provider=mask_provider,
    )

    return dataset_info


@pytest.mark.skipif(not GRAPHCAST_AVAIL, reason="trimesh/rtree are not available")
@pytest.mark.parametrize("activation", ["SiLU", "ReLU", "Mish", "GELU", "Tanh"])
def test_graphcast_normalization(activation):
    # Model parameters
    input_channels = 4
    output_channels = 3
    batch_size = 2
    height = 9
    width = 18

    # Set the dataset info
    dataset_info = dummy_datasetinfo(height, width)

    # Initialize model
    model = GraphCast(
        input_channels=input_channels,
        output_channels=output_channels,
        dataset_info=dataset_info,
        latent_dimension=12,
        activation=activation,
        meshes=6,
        M0=0,
        bias=True,
        radius_fraction=1.0,
        layernorm=True,
        processor_steps=2,
        residual=True,
        is_ocean=True,
    )

    # Create dummy input
    x = torch.randn(batch_size, input_channels, height, width)

    # Forward pass
    output = model(x)

    # Check output shape
    expected_shape = (batch_size, output_channels, height, width)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"

    # Check output values
    assert not torch.isnan(output).any(), "Output contains NaN values"
    assert not torch.isinf(output).any(), "Output contains infinite values"


@pytest.mark.skipif(not GRAPHCAST_AVAIL, reason="trimesh/rtree are not available")
def test_graphcast_output_is_unchanged():
    torch.manual_seed(0)
    input_channels = 2
    output_channels = 3
    img_shape = (9, 18)
    n_samples = 4
    device = get_device()

    # Set the dataset info
    dataset_info = dummy_datasetinfo(*img_shape)

    model = GraphCast(
        input_channels=input_channels,
        output_channels=output_channels,
        dataset_info=dataset_info,
        latent_dimension=16,
        activation="SiLU",
        meshes=6,
        M0=4,
        bias=True,
        radius_fraction=1.0,
        layernorm=True,
        processor_steps=2,
        residual=True,
        is_ocean=True,
    ).to(device)

    # must initialize on CPU to get the same results on GPU
    x = torch.randn(n_samples, input_channels, *img_shape).to(device)

    with torch.no_grad():
        output = model(x)
    assert output.shape == (n_samples, output_channels, *img_shape)

    outdir = os.path.join(DIR, "testdata")
    if os.path.exists(outdir) is False:
        os.makedirs(outdir)
    reference = os.path.join(outdir, "test_graphcast_output_is_unchanged.pt")
    if os.path.exists(reference) is False:
        torch.save(output.cpu(), reference)

    validate_tensor(
        output,
        reference,
    )
