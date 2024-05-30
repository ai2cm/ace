import datetime

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.data_loading.utils import (
    _get_indexers,
    as_broadcasted_tensor,
    decode_timestep,
    encode_timestep,
    infer_horizontal_dimension_names,
)

LON_DIM = "lon"
LAT_DIM = "lat"
TIME_DIM = "time"
FULL_NAME = "full"
CONSTANT_NAME = "constant"
SLICE_NONE = slice(None)


def get_sizes(lon_dim=LON_DIM, lat_dim=LAT_DIM):
    return {lon_dim: 12, lat_dim: 6, TIME_DIM: 3}


def create_reference_dataset(lon_dim=LON_DIM, lat_dim=LAT_DIM):
    dims = (TIME_DIM, lat_dim, lon_dim)
    sizes = get_sizes(lon_dim=lon_dim, lat_dim=lat_dim)
    shape = tuple(sizes[dim] for dim in dims)
    data = np.arange(np.prod(shape)).reshape(shape)
    coords = [np.arange(size) for size in shape]
    full = xr.DataArray(data, dims=dims, coords=coords, name=FULL_NAME)
    constant = full.isel({TIME_DIM: 0}).drop_vars(TIME_DIM).rename(CONSTANT_NAME)
    return xr.merge([full, constant])


@pytest.mark.parametrize(
    ("lon_dim", "lat_dim", "warns"),
    [
        ("grid_xt", "grid_yt", False),
        ("lon", "lat", False),
        ("longitude", "latitude", False),
        ("foo", "bar", True),
    ],
)
def test_infer_horizontal_dimension_names(lon_dim, lat_dim, warns):
    ds = create_reference_dataset(lon_dim=lon_dim, lat_dim=lat_dim)
    expected = (lon_dim, lat_dim)
    if warns:
        with pytest.warns(UserWarning, match="Familiar"):
            infer_horizontal_dimension_names(ds)
    else:
        result = infer_horizontal_dimension_names(ds)
        assert result == expected


def test_infer_horizontal_dimension_names_error():
    ds = create_reference_dataset(lon_dim="foo", lat_dim="bar")
    ds = ds.isel(time=0)
    print(ds)
    with pytest.raises(ValueError, match="Could not identify"):
        infer_horizontal_dimension_names(ds)


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
    sizes = get_sizes()
    shape = tuple(sizes[dim] for dim in variable_dims)
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
