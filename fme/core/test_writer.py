import datetime
import math
import os

import cftime
import numpy as np
import pytest
import xarray as xr
import zarr

from fme.core.writer import ZarrWriter, initialize_zarr, insert_into_zarr

NLAT, NLON = (8, 8)


def create_zarr_store(path, shape, chunks, times, shards=None, nondim_coords=None):
    dims = ("time", "sample", "lat", "lon")
    vars = ["var1", "var2"]
    array_attributes = {"var1": {"units": "K", "long_name": "Variable 1"}}
    initialize_zarr(
        path=path,
        vars=vars,
        dim_sizes=shape,
        chunks=chunks,
        shards=shards,
        dim_names=dims,
        coords={"time": times},
        dtype="f4",
        nondim_coords=nondim_coords,
        array_attributes=array_attributes,
        group_attributes={"description": "Test Zarr store"},
    )


def test_initialize_zarr(tmp_path):
    n_times = 20
    n_sample = 15
    shape = (n_times, n_sample, NLAT, NLON)
    chunks = {"time": 10, "sample": 5, "lat": NLAT, "lon": NLON}
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )
    path = os.path.join(tmp_path, "test.zarr")
    nondim_coords = {
        "nondimensional_coord": xr.DataArray(
            np.ones((n_times, n_sample)), dims=("time", "sample")
        )
    }
    create_zarr_store(path, shape, chunks, times, nondim_coords=nondim_coords)

    ds = xr.open_zarr(path)
    assert ds["var1"].shape == shape
    assert ds["var2"].shape == shape
    np.testing.assert_array_equal(ds.time.values, times)
    assert (
        "nondimensional_coord" in ds.coords
        and "nondimensional_coord" not in ds.data_vars
    )
    assert ds["var1"].attrs["units"] == "K"
    assert ds["var1"].attrs["long_name"] == "Variable 1"
    assert ds["var2"].attrs == {}
    assert ds.attrs["description"] == "Test Zarr store"


@pytest.mark.parametrize(
    "time_batch_size",
    [
        pytest.param(2, id="write_size_less_than_time_chunking"),
        pytest.param(3, id="write_size_equal_to_time_chunking"),
        pytest.param(4, id="write_size_greater_than_time_chunking"),
    ],
)
def test_insert_into_zarr(tmp_path, time_batch_size):
    n_times = 20
    n_samples = 20
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
                0: slice(t * time_batch_size, min(time_batch_size * (t + 1), n_times)),
                1: slice(
                    s * sample_chunk_size, min(sample_chunk_size * (s + 1), n_samples)
                ),
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
    array_attrs = {"var": {"units": "K"}}

    writer = ZarrWriter(
        path=path,
        coords=coords,
        dims=("time", "sample", "lat", "lon"),
        chunks=chunks,
        data_vars=["var"],
        array_attributes=array_attrs,
        group_attributes={"description": "Test Zarr store"},
    )

    # Data is written in slices along time and sample dims
    position_slices = {
        "time": slice(3, 6),
        "sample": slice(1, 3),
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
    assert ds_write["var"].attrs["units"] == "K"
    assert ds_write.attrs["description"] == "Test Zarr store"


def _create_writer(
    path,
    n_times,
    chunks,
    shards=None,
    overwrite_check=True,
    mode="w-",  # default to writer that errors for an existing store
    array_attributes=None,
    group_attributes=None,
):
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )
    lat = np.linspace(-90, 90, NLAT)
    lon = np.linspace(-180, 180, NLON)

    coords = {
        "time": times,
        "lat": lat,
        "lon": lon,
    }

    return ZarrWriter(
        path=path,
        coords=coords,
        dims=("time", "lat", "lon"),
        chunks=chunks,
        shards=shards,
        data_vars=["var"],
        overwrite_check=overwrite_check,
        mode=mode,
        array_attributes=array_attributes,
        group_attributes=group_attributes,
    )


def test_ZarrWriter_append_to_existing(
    tmp_path,
):
    n_times = 8

    path = os.path.join(tmp_path, "test.zarr")
    data = {
        "var": np.random.rand(n_times, NLAT, NLON),
    }

    writer_0 = _create_writer(
        n_times=n_times,
        path=path,
        chunks={"time": 3},
        array_attributes={"var": {"units": "K"}},
        group_attributes={"description": "Test Zarr store"},
    )

    writer_0.record_batch(
        data={"var": data["var"][slice(0, 4)]}, position_slices={"time": slice(0, 4)}
    )

    writer_1 = _create_writer(
        n_times=n_times,
        path=path,
        chunks={"time": 3},
        array_attributes={"var": {"units": "K"}},
        group_attributes={"description": "Test Zarr store"},
        mode="a",
    )
    writer_1.record_batch(
        data={"var": data["var"][slice(4, 8)]}, position_slices={"time": slice(4, 8)}
    )

    ds = xr.open_zarr(path)
    np.testing.assert_allclose(ds["var"].values, data["var"])
    assert ds["var"].attrs["units"] == "K"
    assert ds.attrs["description"] == "Test Zarr store"


def test_ZarrWriter_overwrite_check(tmp_path):
    writer = _create_writer(
        os.path.join(tmp_path, "test.zarr"), n_times=4, chunks={"time": 2}
    )
    batch_data = {
        "var": np.random.rand(2, NLAT, NLON),
    }
    writer.record_batch(data=batch_data, position_slices={"time": slice(0, 2)})
    with pytest.raises(RuntimeError):
        writer.record_batch(data=batch_data, position_slices={"time": slice(0, 2)})


def test_ZarrWriter_can_overwrite(tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    writer = _create_writer(path, n_times=4, chunks={"time": 2}, overwrite_check=False)
    batch_data_nonzero = {
        "var": np.random.rand(2, NLAT, NLON),
    }

    writer.record_batch(data=batch_data_nonzero, position_slices={"time": slice(0, 2)})
    ds = xr.open_zarr(path)
    assert np.all(ds["var"][:2] != 0.0)

    batch_data_zero = {"var": np.zeros((2, NLAT, NLON))}
    writer.record_batch(data=batch_data_zero, position_slices={"time": slice(0, 2)})
    ds = xr.open_zarr(path)
    assert np.all(ds["var"][:2] == 0.0)


def test_ZarrWriter_allow_existing(tmp_path):
    path = os.path.join(tmp_path, "test.zarr")
    batch_data = {
        "var": np.random.rand(2, NLAT, NLON),
    }

    writer0 = _create_writer(path, n_times=4, chunks={"time": 2})
    writer0.record_batch(data=batch_data, position_slices={"time": slice(0, 2)})

    writer1 = _create_writer(path, n_times=4, chunks={"time": 2}, mode="a")
    writer1.record_batch(data=batch_data, position_slices={"time": slice(2, 4)})


@pytest.mark.parametrize(
    "time_chunk_size, time_shard_size",
    [(1, 1), (1, 4), (1, 9), (2, 8)],
)
def test_ZarrWriter_shards_file_count(tmp_path, time_chunk_size, time_shard_size):
    n_times = 8

    path = os.path.join(tmp_path, "test.zarr")
    data = {
        "var": np.random.rand(n_times, NLAT, NLON),
    }
    writer_0 = _create_writer(
        n_times=n_times,
        path=path,
        chunks={"time": time_chunk_size},
        shards={"time": time_shard_size},
    )

    writer_0.record_batch(
        data={"var": data["var"][slice(0, 4)]}, position_slices={"time": slice(0, 4)}
    )

    writer_1 = _create_writer(
        n_times=n_times,
        path=path,
        chunks={"time": time_chunk_size},
        shards={"time": time_shard_size},
        mode="a",
    )
    writer_1.record_batch(
        data={"var": data["var"][slice(4, 8)]}, position_slices={"time": slice(4, 8)}
    )

    ds = xr.open_zarr(path)
    np.testing.assert_allclose(ds["var"].values, data["var"])
    var_dir = os.path.join(path, "var/c")
    assert len(os.listdir(var_dir)) == math.ceil(n_times / time_shard_size)


def test_ZarrWriter_shards_errors_for_wrong_chunks_size(tmp_path):
    n_times = 8
    shape = (n_times, 1, NLAT, NLON)
    path = os.path.join(tmp_path, "test.zarr")
    chunks = {"time": 2}
    shards = {"time": 1}
    times = np.array(
        [
            cftime.DatetimeJulian(2020, 1, 1, 0) + datetime.timedelta(hours=i)
            for i in range(n_times)
        ]
    )
    with pytest.raises(ValueError, match="not divisible"):
        create_zarr_store(
            path=path,
            shape=shape,
            times=times,
            chunks=chunks,
            shards=shards,
        )
