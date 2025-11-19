import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data.patching import Patch, _HorizontalSlice

from .topography import StaticInput, StaticInputs, _range_to_slice
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


def test__range_to_slice():
    x = torch.arange(5)
    assert torch.equal(
        x[_range_to_slice(x, ClosedInterval(2, 4))], torch.tensor([2, 3, 4])
    )


def test_subset_latlon():
    full_data_shape = (10, 10)
    expected_slices = [slice(2, 6), slice(3, 8)]
    data = torch.randn(*full_data_shape)
    coords = LatLonCoordinates(
        lat=torch.linspace(0, 9, 10), lon=torch.linspace(0, 9, 10)
    )
    topography = StaticInput(data=data, coords=coords)
    land_frac = StaticInput(data=data * -1.0, coords=coords)

    static_inputs = StaticInputs([topography, land_frac])
    lat_interval = ClosedInterval(2, 5)
    lon_interval = ClosedInterval(3, 7)

    subset_static_inputs = static_inputs.subset_latlon(lat_interval, lon_interval)
    expected_lats = torch.tensor([2, 3, 4, 5], dtype=coords.lat.dtype)
    expected_lons = torch.tensor([3, 4, 5, 6, 7], dtype=coords.lon.dtype)

    expected_topography = data[*expected_slices]
    expected_land_frac = data[*expected_slices] * -1.0

    for i in [0, 1]:
        assert torch.equal(subset_static_inputs[i].coords.lat, expected_lats)
        assert torch.equal(subset_static_inputs[i].coords.lon, expected_lons)

    assert torch.allclose(subset_static_inputs[0].data, expected_topography)
    assert torch.allclose(subset_static_inputs[1].data, expected_land_frac)


def test_StaticInputs_generate_from_patches():
    output_slice = _HorizontalSlice(y=slice(None), x=slice(None))
    patches = [
        Patch(
            input_slice=_HorizontalSlice(y=slice(1, 3), x=slice(None, None)),
            output_slice=output_slice,
        ),
        Patch(
            input_slice=_HorizontalSlice(y=slice(0, 2), x=slice(2, 3)),
            output_slice=output_slice,
        ),
    ]
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
    static_inputs_patch_generator = static_inputs.generate_from_patches(patches)
    generated_patches = []
    for static_inputs_patch in static_inputs_patch_generator:
        generated_patches.append(static_inputs_patch)

    assert len(generated_patches) == 2

    expected_topography_patch_0 = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    expected_topography_patch_1 = torch.tensor([[2], [6]])

    # first index is the patch, second is the static input field within
    # the StaticInputs container
    assert torch.equal(generated_patches[0][0].data, expected_topography_patch_0)
    assert torch.equal(generated_patches[1][0].data, expected_topography_patch_1)

    # land_frac field values are -1 * topography
    assert torch.equal(generated_patches[0][1].data, expected_topography_patch_0 * -1.0)
    assert torch.equal(generated_patches[1][1].data, expected_topography_patch_1 * -1.0)
