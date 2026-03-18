import pytest
import torch

from fme.core.coordinates import LatLonCoordinates

from .static import StaticInput, StaticInputs
from .utils import ClosedInterval


@pytest.mark.parametrize(
    "init_args",
    [
        pytest.param(
            [
                torch.randn((1, 2, 2)),
                LatLonCoordinates(torch.arange(2), torch.arange(2)),
            ],
            id="3d_data",
        ),
        pytest.param(
            [torch.randn((2, 2)), LatLonCoordinates(torch.arange(2), torch.arange(5))],
            id="dim_size_mismatch",
        ),
    ],
)
def test_Topography_error_cases(init_args):
    with pytest.raises(ValueError):
        StaticInput(*init_args)


def test_subset_latlon():
    full_data_shape = (10, 10)
    expected_slices = [slice(2, 6), slice(3, 8)]
    data = torch.randn(*full_data_shape)
    coords = LatLonCoordinates(
        lat=torch.linspace(0, 9, 10), lon=torch.linspace(0, 9, 10)
    )
    topo = StaticInput(data=data, coords=coords)
    lat_interval = ClosedInterval(2, 5)
    lon_interval = ClosedInterval(3, 7)
    subset_topo = topo.subset_latlon(lat_interval, lon_interval)
    expected_lats = torch.tensor([2, 3, 4, 5], dtype=coords.lat.dtype)
    expected_lons = torch.tensor([3, 4, 5, 6, 7], dtype=coords.lon.dtype)
    expected_data = data[*expected_slices]
    assert torch.equal(subset_topo.coords.lat, expected_lats)
    assert torch.equal(subset_topo.coords.lon, expected_lons)
    assert torch.allclose(subset_topo.data, expected_data)


def test_StaticInputs_serialize():
    data = torch.arange(16).reshape(4, 4)
    topography = StaticInput(
        data,
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    land_frac = StaticInput(
        data * -1.0,
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    static_inputs = StaticInputs([topography, land_frac])
    state = static_inputs.get_state()
    static_inputs_reconstructed = StaticInputs.from_state(state)
    assert static_inputs_reconstructed[0].data.equal(static_inputs[0].data)
    assert static_inputs_reconstructed[1].data.equal(static_inputs[1].data)
