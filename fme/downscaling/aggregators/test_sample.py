import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.typing_ import FineResCoarseResPair

from .sample import PairedSampleAggregator


@pytest.mark.parametrize("n_samples", [1, 3])
def test_paired_sample_aggregator_get_dataset(n_samples):
    """get_dataset keeps the sample dimension, including for a single sample."""
    n_lat, n_lon = 4, 8
    fine = LatLonCoordinates(
        lat=torch.linspace(-30, 30, n_lat), lon=torch.linspace(0, 315, n_lon)
    )
    coarse = LatLonCoordinates(lat=torch.tensor([0.0]), lon=torch.tensor([0.0, 1.0]))
    aggregator = PairedSampleAggregator(
        target={"x": torch.zeros(1, n_lat, n_lon)},
        coarse={"x": torch.zeros(1, 1, 2)},
        latlon_coordinates=FineResCoarseResPair(fine=fine, coarse=coarse),
    )
    aggregator.record_batch({"x": torch.randn(1, n_samples, n_lat, n_lon)})
    ds = aggregator.get_dataset()
    assert ds["x_predicted"].dims == ("sample", "lat", "lon")
    assert ds["x_predicted"].shape == (n_samples, n_lat, n_lon)
    assert ds["x_target"].shape == (n_lat, n_lon)
