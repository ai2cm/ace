import pytest
import torch

from fme.core import get_device
from fme.core.coordinates import HEALPixCoordinates, LatLonCoordinates
from fme.core.distributed import Distributed

from .build_context import MetricNotSupportedError
from .utils import hemisphere_weights


def test_hemisphere_weights_partition_at_equator():
    coords = LatLonCoordinates(
        lat=torch.tensor([-45.0, 0.0, 45.0]), lon=torch.tensor([0.0, 180.0])
    )
    north = hemisphere_weights(coords, "north")
    south = hemisphere_weights(coords, "south")
    torch.testing.assert_close(north, torch.tensor([[0.0], [0.0], [1.0]]))
    torch.testing.assert_close(south, torch.tensor([[1.0], [1.0], [0.0]]))


def test_hemisphere_weights_unknown_hemisphere():
    coords = LatLonCoordinates(lat=torch.tensor([0.0]), lon=torch.tensor([0.0]))
    with pytest.raises(ValueError):
        hemisphere_weights(coords, "east")  # type: ignore[arg-type]


def test_hemisphere_weights_healpix_not_supported():
    coords = HEALPixCoordinates(
        face=torch.arange(12), height=torch.arange(4), width=torch.arange(4)
    )
    with pytest.raises(MetricNotSupportedError):
        hemisphere_weights(coords, "north")


@pytest.mark.parallel
@pytest.mark.parametrize("hemisphere", ["north", "south"])
def test_hemisphere_regional_sum_matches_global_reference(hemisphere):
    dist = Distributed.get_instance()
    nlat, nlon = 8, 16
    coords = LatLonCoordinates(
        lat=torch.linspace(-78.75, 78.75, nlat),
        lon=torch.linspace(0.0, 337.5, nlon),
    )
    global_mask = coords.lat > 0 if hemisphere == "north" else coords.lat <= 0
    gen = torch.Generator(device="cpu").manual_seed(0)
    global_data = torch.randn(2, nlat, nlon, generator=gen)
    expected = (
        global_mask.float().unsqueeze(-1) * coords.area_weights * global_data
    ).sum(dim=(-2, -1))

    h_slice, w_slice = dist.get_local_slices((nlat, nlon))
    weights = hemisphere_weights(coords, hemisphere)
    torch.testing.assert_close(weights.squeeze(-1).bool(), global_mask[h_slice])

    ops = coords.get_gridded_operations()
    local_data = global_data[..., h_slice, w_slice].to(get_device())
    result = ops.regional_area_weighted_sum(local_data, weights)
    torch.testing.assert_close(result.cpu(), expected)
