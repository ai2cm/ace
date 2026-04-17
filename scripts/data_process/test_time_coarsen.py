import json

import numpy as np
import xarray as xr
from time_coarsen import TimeCoarsenConfig, TimeSlice, coarsen, process_path_pair
from zarr.codecs import BloscCodec

from fme.ace.testing import DimSize, DimSizes, get_nd_dataset


def _assert_encoding_matches(ds_in: xr.Dataset, ds_out: xr.Dataset) -> None:
    """
    Assert that the chunking and sharding encoding of variables in the
    output dataset match the input dataset.
    """
    for name, da_out in ds_out.data_vars.items():
        da_in = ds_in[name]

        # chunks: tuple-of-tuples in xarray encoding for zarr
        in_chunks = da_in.encoding.get("chunks")
        out_chunks = da_out.encoding.get("chunks")
        assert (
            in_chunks == out_chunks
        ), f"{name}: chunks differ: {in_chunks} vs {out_chunks}"

        # shards: only present if your writer uses sharding and xarray propagates it
        in_shards = da_in.encoding.get("shards")
        out_shards = da_out.encoding.get("shards")
        assert (
            in_shards == out_shards
        ), f"{name}: shards differ: {in_shards} vs {out_shards}"


def write_v3_with_chunks(ds: xr.Dataset, path: str) -> None:
    compressor = BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")

    encoding = {}
    for name, da in ds.variables.items():  # includes coords + data_vars
        if da.ndim == 0:
            continue
        chunks = tuple(
            2 if dim == "time" else 2 if dim == "lat" else 4 if dim == "lon" else 1
            for dim in da.dims
        )
        shards = tuple(
            4 if dim == "time" else 2 if dim == "lat" else 4 if dim == "lon" else 1
            for dim in da.dims
        )
        encoding[name] = {"chunks": chunks, "compressor": compressor, "shards": shards}

    ds.to_zarr(path, mode="w", zarr_version=3, encoding=encoding)


def test_process_path_pair() -> None:
    n_input_times = 5
    nz_interface = 3
    # Create a small but non-trivial dataset (coords + attrs)
    ds = get_nd_dataset(
        dim_sizes=DimSizes(
            n_time=n_input_times,
            horizontal=[
                DimSize(name="lat", size=4),
                DimSize(name="lon", size=8),
            ],
            nz_interface=nz_interface,
        ),
        variable_names=["temp", "temp_tendency", "flag"],
        timestep_days=1.0,
        include_vertical_coordinate=True,
    )
    ds["temp"].attrs["units"] = "K"
    ds["temp_tendency"].attrs["units"] = "K/day"
    constant_names = []
    for name in ds.data_vars:
        if name.startswith("ak_") or name.startswith("bk_"):
            constant_names.append(name)
    assert len(constant_names) == 2 * nz_interface  # sanity check
    input_path = "memory://test-time-coarsen/dataset.zarr"
    output_path = "memory://test-time-coarsen/dataset_coarsened.zarr"
    ds.attrs["history"] = "Created for testing."
    ds.attrs["other_attr"] = "other_value"

    # Write to an in-memory filesystem (xarray recognizes the memory:// protocol)
    write_v3_with_chunks(ds, input_path)
    ds = xr.open_zarr(input_path)  # for comparison later, use fresh read

    config = TimeCoarsenConfig(
        factor=2,
        data_output_directory="",  # not used in this test
        stats_output_directory="",  # not used in this test
        snapshot_names=["temp"],
        window_names=["temp_tendency"],
        constant_prefixes=["ak_", "bk_"],
    )
    process_path_pair(
        input_path=input_path, output_path=output_path, config=config, dry_run=False
    )

    # Read back the coarsened dataset
    ds_coarsened = xr.open_zarr(output_path)
    _assert_encoding_matches(ds, ds_coarsened)
    # Note on each timestep, we have the tendencies which led to the current
    # snapshot alongside the snapshot itself. This means the first snapshot
    # of the coarsened dataset is not the first snapshot of the input dataset.
    assert ds_coarsened.dims["time"] == n_input_times // config.factor
    expected_snapshot_slice = slice(config.factor - 1, None, config.factor)
    np.testing.assert_array_equal(
        ds_coarsened["time"].values,
        ds["time"].isel(time=expected_snapshot_slice).values,
    )
    np.testing.assert_array_equal(
        ds_coarsened["temp"].values,
        ds["temp"].isel(time=expected_snapshot_slice).values,
    )
    np.testing.assert_array_equal(
        ds_coarsened["temp_tendency"].isel(time=0).values,
        ds["temp_tendency"].isel(time=slice(0, config.factor)).mean("time").values,
    )
    for name in constant_names:
        np.testing.assert_array_equal(ds_coarsened[name].values, ds[name].values)
    expected_history = (
        "Created for testing. Dataset coarsened by a factor of 2 by "
        "scripts/data_process/time_coarsen.py."
    )
    assert ds_coarsened.attrs["history"] == expected_history
    assert ds_coarsened.attrs["other_attr"] == "other_value"
    assert json.loads(ds_coarsened.attrs["snapshot_names"]) == config.snapshot_names
    assert json.loads(ds_coarsened.attrs["window_names"]) == config.window_names
    assert (
        json.loads(ds_coarsened.attrs["constant_prefixes"]) == config.constant_prefixes
    )
    assert ds_coarsened.attrs["coarsen_factor"] == config.factor
    assert len(ds_coarsened.attrs) == 6  # no unexpected attrs


def test_coarsen_time_slice() -> None:
    times = xr.date_range("2000-01-01", periods=6, freq="1D")
    ds = xr.Dataset(
        {
            "temp": xr.DataArray(np.arange(6.0), dims=["time"]),
            "temp_tendency": xr.DataArray(np.arange(6.0), dims=["time"]),
        },
        coords={"time": times},
    )
    config = TimeCoarsenConfig(
        factor=2,
        data_output_directory="",
        stats_output_directory="",
        snapshot_names=["temp"],
        window_names=["temp_tendency"],
        constant_prefixes=[],
        time_slice=TimeSlice(start="2000-01-02"),
    )
    ds_coarsened = coarsen(ds, config)
    # Slicing to start at Jan 2 gives [Jan 2..6] (5 times).
    # factor=2 with trim → 2 output snapshots at indices 1 and 3 of the sliced
    # dataset, i.e. Jan 3 and Jan 5.
    xr.testing.assert_equal(
        ds_coarsened["time"],
        xr.DataArray(times[2::2], dims=["time"]),  # Jan 3, Jan 5
    )
    assert ds_coarsened.attrs["time_slice.start"] == "2000-01-02"
    assert ds_coarsened.attrs["time_slice.stop"] == "None"
