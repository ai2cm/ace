import copy
import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.coordinates import (
    DimSize,
    HEALPixCoordinates,
    HorizontalCoordinates,
    LatLonCoordinates,
)

from .utils import (
    _broadcast_array_to_tensor,
    _get_indexers,
    as_broadcasted_tensor,
    decode_timestep,
    encode_timestep,
    get_horizontal_coordinates,
)

LON_DIM = "lon"
LAT_DIM = "lat"
TIME_DIM = "time"
FULL_NAME = "full"
CONSTANT_NAME = "constant"
SLICE_NONE = slice(None)


def get_sizes(
    spatial_dims: HorizontalCoordinates = LatLonCoordinates(
        lon=torch.Tensor(np.arange(6)),
        lat=torch.Tensor(np.arange(12)),
    ),
):
    spatial_sizes: list[DimSize] = copy.deepcopy(spatial_dims.loaded_sizes)
    return [DimSize(TIME_DIM, 3)] + spatial_sizes


def create_reference_dataset(
    spatial_dims: HorizontalCoordinates = LatLonCoordinates(
        lon=torch.Tensor(np.arange(6)),
        lat=torch.Tensor(np.arange(12)),
    ),
):
    dims = [TIME_DIM] + spatial_dims.dims
    dim_sizes = get_sizes(spatial_dims)
    shape = tuple(dim_size.size for dim_size in dim_sizes)
    data = np.arange(np.prod(shape)).reshape(shape)
    coords = [np.arange(size) for size in shape]
    full = xr.DataArray(data, dims=dims, coords=coords, name=FULL_NAME)
    constant = full.isel({TIME_DIM: 0}).drop_vars(TIME_DIM).rename(CONSTANT_NAME)
    return xr.merge([full, constant])


@pytest.mark.parametrize(
    "spatial_dimensions, expected_coords",
    [
        (
            "healpix",
            HEALPixCoordinates(
                face=torch.Tensor(np.arange(12)),
                width=torch.Tensor(np.arange(64)),
                height=torch.Tensor(np.arange(64)),
            ),
        ),
        (
            "latlon",
            LatLonCoordinates(
                lon=torch.Tensor(np.arange(6)),
                lat=torch.Tensor(np.arange(12)),
            ),
        ),
    ],
)
def test_get_horizontal_coordinates(spatial_dimensions, expected_coords):
    ds = create_reference_dataset(spatial_dims=expected_coords)
    result, loaded_dim_names = get_horizontal_coordinates(ds, spatial_dimensions, None)
    assert expected_coords.dims == result.dims
    for name in expected_coords.dims:
        np.testing.assert_array_equal(expected_coords.coords[name], result.coords[name])


@pytest.mark.parametrize(
    ["coordinate_type", "coord_sizes"],
    [
        pytest.param(LatLonCoordinates, {"lat": 90, "lon": 180}),
        pytest.param(HEALPixCoordinates, {"face": 12, "height": 64, "width": 64}),
    ],
)
def test_horizonal_dimension_sizes(coordinate_type, coord_sizes):
    coords = {name: torch.Tensor(np.arange(size)) for name, size in coord_sizes.items()}
    horizontal_coords = coordinate_type(**coords)
    for name, size in coord_sizes.items():
        assert len(horizontal_coords.coords[name]) == size


@pytest.mark.parametrize(
    ("variable_dims", "expected"),
    [
        ((TIME_DIM, LAT_DIM, LON_DIM), (SLICE_NONE, SLICE_NONE, SLICE_NONE)),
        ((LAT_DIM, LON_DIM), (None, SLICE_NONE, SLICE_NONE)),
        ((TIME_DIM,), (SLICE_NONE, None, None)),
        ((LAT_DIM,), (None, SLICE_NONE, None)),
        ((LON_DIM,), (None, None, SLICE_NONE)),
    ],
    ids=lambda x: f"{x}",
)
def test__get_indexers(variable_dims, expected):
    dim_sizes = get_sizes()
    shape = tuple(
        dim_size.size for dim_size in dim_sizes if dim_size.name in variable_dims
    )
    variable = xr.Variable(variable_dims, np.zeros(shape))
    dims = (TIME_DIM, LAT_DIM, LON_DIM)
    result = _get_indexers(variable, dims)
    assert result == expected


@pytest.mark.parametrize(
    "variable_dims",
    [
        (TIME_DIM, LAT_DIM, LON_DIM),
        (LAT_DIM, LON_DIM),
        (TIME_DIM,),
        (LAT_DIM,),
        (LON_DIM,),
    ],
)
def test_as_broadcasted_tensor(variable_dims):
    ds = create_reference_dataset()

    variable_shape = tuple(ds.sizes[dim] for dim in variable_dims)
    variable = xr.Variable(variable_dims, np.zeros(variable_shape))
    da = xr.DataArray(variable)

    dims = ds[FULL_NAME].dims
    shape = (ds.sizes[TIME_DIM], ds.sizes[LAT_DIM], ds.sizes[LON_DIM])
    result = as_broadcasted_tensor(variable, dims, shape)

    xarray_broadcast = da.broadcast_like(ds[FULL_NAME]).transpose(*dims)

    # Suppress read-only warning when casting to a tensor; note this is only
    # an issue following an xarray broadcast, which is used for convenience
    # in this test. The internals of the data loader convert the array to a
    # tensor prior to the broadcast, which avoids this issue.
    xarray_broadcast = xarray_broadcast.copy()

    expected = torch.as_tensor(xarray_broadcast.values)

    assert torch.equal(result, expected)


def test_encode_decode_timestep_roundtrip():
    timestep = datetime.timedelta(days=1, microseconds=1)
    encoded = encode_timestep(timestep)
    roundtripped = decode_timestep(encoded)
    assert roundtripped == timestep


def test__broadcast_array_to_tensor_no_broadcast():
    arr = np.arange(6).reshape(1, 2, 3)
    out = _broadcast_array_to_tensor(arr, (TIME_DIM, LAT_DIM, LON_DIM), (1, 2, 3))
    expected = torch.as_tensor(arr)
    torch.testing.assert_close(out, expected)


def test__broadcast_array_to_tensor_with_broadcast():
    arr = np.arange(6)
    out = _broadcast_array_to_tensor(arr, (TIME_DIM, LAT_DIM, LON_DIM), (6, 2, 3))
    expected = torch.broadcast_to(torch.as_tensor(arr)[:, None, None], (6, 2, 3))
    torch.testing.assert_close(out, expected)


def test__broadcast_array_to_tensor_raises_assertion_error():
    arr = np.zeros((1, 2))
    with pytest.raises(ValueError, match="must be 1D"):
        _broadcast_array_to_tensor(arr, (TIME_DIM, LAT_DIM, LON_DIM), (1, 2, 3))

    arr = np.zeros(3)
    with pytest.raises(ValueError, match="matching time dimension"):
        _broadcast_array_to_tensor(arr, (TIME_DIM, LAT_DIM, LON_DIM), (4, 2, 3))
