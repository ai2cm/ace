import pytest
import torch

from fme.core.coordinates import LatLonCoordinates
from fme.downscaling.data.patching import Patch, _HorizontalSlice

from .topography import StaticInputs, Topography, _range_to_slice
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
        Topography(*init_args)


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


def test_Topography_generate_from_patches():
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
    topography = Topography(
        torch.arange(16).reshape(4, 4),
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    topo_patch_generator = topography.generate_from_patches(patches)
    generated_patches = []
    for topo_patch in topo_patch_generator:
        generated_patches.append(topo_patch)
    assert len(generated_patches) == 2
    assert torch.equal(
        generated_patches[0].data, torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    )
    assert torch.equal(generated_patches[1].data, torch.tensor([[2], [6]]))


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
    topography = Topography(
        data,
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    land_frac = Topography(
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


def test_StaticInputs_serialize():
    data = torch.arange(16).reshape(4, 4)
    topography = Topography(
        data,
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    land_frac = Topography(
        data * -1.0,
        LatLonCoordinates(torch.arange(4), torch.arange(4)),
    )
    static_inputs = StaticInputs([topography, land_frac])
    state = static_inputs.to_state()
    static_inputs_reconstructed = StaticInputs.from_state(state)
    assert static_inputs_reconstructed[0].data.equal(static_inputs[0].data)
    assert static_inputs_reconstructed[1].data.equal(static_inputs[1].data)


def test_StaticInputs_subset_for_coarse_coords():
    """Test that subset_for_coarse_coords correctly subsets to fine coords."""
    # Create fine-resolution static inputs (8x8)
    fine_data = torch.arange(64).reshape(8, 8).float()
    fine_coords = LatLonCoordinates(
        lat=torch.linspace(-45, 45, 8), lon=torch.linspace(-90, 90, 8)
    )
    topography = Topography(data=fine_data, coords=fine_coords)
    land_frac = Topography(data=fine_data * -1.0, coords=fine_coords)
    static_inputs = StaticInputs([topography, land_frac])

    # Create coarse coordinates (4x4) that cover a subset of the domain
    # We'll subset to the middle 2x2 coarse points
    coarse_lat = torch.linspace(-45, 45, 4)
    coarse_lon = torch.linspace(-90, 90, 4)
    # Select middle 2x2: indices 1:3 for both dimensions
    coarse_coords = LatLonCoordinates(lat=coarse_lat[1:3], lon=coarse_lon[1:3])

    downscale_factor = 2

    # Subset the static inputs
    subset = static_inputs.subset_for_coarse_coords(coarse_coords, downscale_factor)

    # With downscale_factor=2, we expect the subset to cover 4x4 fine points
    # (2 coarse points * 2 downscale factor = 4 fine points per dimension)
    assert subset.shape == (4, 4)
    assert len(subset.fields) == 2

    # Verify all fields were subset
    for i, field in enumerate(subset.fields):
        assert field.shape == (4, 4)
        # Verify coordinates are subset appropriately
        assert len(field.coords.lat) == 4
        assert len(field.coords.lon) == 4


def test_StaticInputs_subset_for_coarse_coords_full_domain():
    """Test subset_for_coarse_coords with downscale_factor=1 (no actual downscaling)."""
    # Create static inputs
    data = torch.arange(16).reshape(4, 4).float()
    coords = LatLonCoordinates(
        lat=torch.linspace(0, 9, 4), lon=torch.linspace(0, 9, 4)
    )
    topography = Topography(data=data, coords=coords)
    static_inputs = StaticInputs([topography])

    # Use the same coordinates as coarse coords with downscale_factor=1
    subset = static_inputs.subset_for_coarse_coords(coords, downscale_factor=1)

    # Should get back the same data
    assert subset.shape == (4, 4)
    assert torch.equal(subset.fields[0].data, data)
    assert torch.equal(subset.fields[0].coords.lat, coords.lat)
    assert torch.equal(subset.fields[0].coords.lon, coords.lon)


def test_StaticInputs_get_topography_for_coarse_coords():
    """Test that get_topography_for_coarse_coords returns the first field."""
    # Create fine-resolution static inputs (8x8)
    fine_data = torch.arange(64).reshape(8, 8).float()
    fine_coords = LatLonCoordinates(
        lat=torch.linspace(-45, 45, 8), lon=torch.linspace(-90, 90, 8)
    )
    topography = Topography(data=fine_data, coords=fine_coords)
    land_frac = Topography(data=fine_data * -1.0, coords=fine_coords)
    static_inputs = StaticInputs([topography, land_frac])

    # Create coarse coordinates
    coarse_lat = torch.linspace(-45, 45, 4)
    coarse_lon = torch.linspace(-90, 90, 4)
    coarse_coords = LatLonCoordinates(lat=coarse_lat[1:3], lon=coarse_lon[1:3])

    downscale_factor = 2

    # Get topography
    result = static_inputs.get_topography_for_coarse_coords(
        coarse_coords, downscale_factor
    )

    # Should return a Topography object (the first field)
    assert isinstance(result, Topography)
    assert result.shape == (4, 4)


def test_StaticInputs_get_topography_for_coarse_coords_empty():
    """Test get_topography_for_coarse_coords returns None for empty."""
    # Create empty StaticInputs
    static_inputs = StaticInputs([])

    # Create some coarse coordinates
    coarse_coords = LatLonCoordinates(
        lat=torch.tensor([0.0, 1.0]), lon=torch.tensor([0.0, 1.0])
    )

    # Should return None
    result = static_inputs.get_topography_for_coarse_coords(
        coarse_coords, downscale_factor=2
    )
    assert result is None
