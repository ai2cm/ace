import datetime
import os

import cftime
import numpy as np
import pytest
import xarray as xr
import zarr

from fme.downscaling.writer import ZarrWriter, initialize_zarr, insert_into_zarr

NLAT, NLON = (8, 8)


def create_zarr_store(path, shape, chunks, times):
    dims = ("time", "sample", "lat", "lon")
    vars = ["var1", "var2"]
    initialize_zarr(
        path=path,
        vars=vars,
        dim_sizes=shape,
        chunks=chunks,
        dim_names=dims,
        coords={"time": times},
        dtype="f4",
    )


def test_initialize_zarr(tmp_path):
    n_times = 20
    shape = (n_times, 10, NLAT, NLON)
    chunks = {"time": 10, "sample": 5, "lat": NLAT, "lon": NLON}
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )
    path = os.path.join(tmp_path, "test.zarr")
    create_zarr_store(path, shape, chunks, times)

    ds = xr.open_zarr(
        path,
    )
    assert ds["var1"].shape == shape
    assert ds["var2"].shape == shape
    np.testing.assert_array_equal(ds.time.values, times)


@pytest.mark.parametrize(
    "time_batch_size",
    [
        pytest.param(2, id="write_size_less_than_time_chunking"),
        pytest.param(3, id="write_size_equal_to_time_chunking"),
        pytest.param(4, id="write_size_greater_than_time_chunking"),
    ],
)
def test_insert_into_zarr(tmp_path, time_batch_size):
    n_times = 10
    n_samples = 9
    time_chunk_size = 3
    sample_chunk_size = 4
    chunks = {
        "time": time_chunk_size,
        "sample": sample_chunk_size,
        "lat": NLAT,
        "lon": NLON,
    }
    shape = (n_times, n_samples, NLAT, NLON)

    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )

    path = os.path.join(tmp_path, "test.zarr")
    create_zarr_store(path, shape, chunks, times)

    data = {
        "var1": np.random.rand(n_times, n_samples, NLAT, NLON),
        "var2": np.random.rand(n_times, n_samples, NLAT, NLON),
    }

    # Data is written in slices along time and sample dims
    # Size of data along the time dimension is tested for <, =, > time_chunk_size
    for t in range(int(np.ceil(n_times / time_batch_size))):
        for s in range(int(np.ceil(n_samples / chunks["sample"]))):
            insert_slices = {
                0: slice(t * time_batch_size, time_batch_size * (t + 1)),
                1: slice(s * sample_chunk_size, sample_chunk_size * (s + 1)),
            }
            batch_data = {
                "var1": data["var1"][insert_slices[0], insert_slices[1], :, :],
                "var2": data["var2"][insert_slices[0], insert_slices[1], :, :],
            }
            insert_into_zarr(
                path,
                batch_data,
                insert_slices,
            )

    ds = xr.open_zarr(path)
    for var_name in data.keys():
        inserted_data = ds[var_name].values
        expected_data = data[var_name]
        np.testing.assert_allclose(inserted_data, expected_data, rtol=1e-4)
        assert ds[var_name].shape == (n_times, n_samples, NLAT, NLON)


def test_ZarrWriter(
    tmp_path,
):
    n_times = 8
    n_samples = 3
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )
    lat = np.linspace(-90, 90, NLAT)
    lon = np.linspace(-180, 180, NLON)

    path = os.path.join(tmp_path, "test.zarr")
    data = {
        "var": np.random.rand(n_times, n_samples, NLAT, NLON),
        "no_write_var": np.random.rand(n_times, n_samples, NLAT, NLON),
    }
    chunks = {"time": 3, "sample": 2}
    coords = {"time": times, "lat": lat, "lon": lon, "sample": np.arange(n_samples)}

    writer = ZarrWriter(
        path=path,
        coords=coords,
        dims=("time", "sample", "lat", "lon"),
        chunks=chunks,
        data_vars=["var"],
    )

    # Data is written in slices along time and sample dims
    position_slices = {
        "time": slice(3, 6),
        "sample": slice(2, 4),
    }
    batch_data = {
        v: data[v][position_slices["time"], position_slices["sample"], :, :]
        for v in data.keys()
    }
    writer.record_batch(data=batch_data, position_slices=position_slices)

    ds_write = zarr.open(writer.path, mode="r")
    np.testing.assert_array_equal(ds_write["lat"], lat)
    np.testing.assert_array_equal(ds_write["lon"], lon)
    assert ds_write["var"].shape == (n_times, n_samples, NLAT, NLON)

    np.testing.assert_allclose(
        ds_write["var"][position_slices["time"], position_slices["sample"]],
        data["var"][position_slices["time"], position_slices["sample"]],
    )

    assert ds_write["var"].chunks == (chunks["time"], chunks["sample"], NLAT, NLON)

    # Time coord check needs open with xarray to decode times
    ds_write = xr.open_zarr(writer.path)
    np.testing.assert_array_equal(ds_write["time"], times)

    assert "no_write_var" not in ds_write.variables
