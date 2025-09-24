import pytest
import torch

from fme.core.coordinates import LatLonCoordinates

from .topography import Topography
from .utils import ClosedInterval


@pytest.mark.parametrize(
    "full_data_shape, expected_slices",
    [
        (
            (
                10,
                10,
            ),
            [slice(2, 6), slice(3, 8)],
        ),
        ((2, 10, 10), [slice(None), slice(2, 6), slice(3, 8)]),
        ((5, 2, 10, 10), [slice(None), slice(None), slice(2, 6), slice(3, 8)]),
    ],
)
def test_subset_latlon(full_data_shape, expected_slices):
    data = torch.randn(*full_data_shape)
    coords = LatLonCoordinates(
        lat=torch.linspace(0, 9, 10), lon=torch.linspace(0, 9, 10)
    )
    topo = Topography(data=data, coords=coords)
    lat_interval = ClosedInterval(2, 5)
    lon_interval = ClosedInterval(3, 7)
    subset_topo = topo.subset_latlon(lat_interval, lon_interval)
    expected_lats = torch.tensor([2, 3, 4, 5], dtype=coords.lat.dtype)
    expected_lons = torch.tensor([3, 4, 5, 6, 7], dtype=coords.lon.dtype)
    expected_data = data[*expected_slices]
    assert torch.equal(subset_topo.coords.lat, expected_lats)
    assert torch.equal(subset_topo.coords.lon, expected_lons)
    assert torch.allclose(subset_topo.data, expected_data)
