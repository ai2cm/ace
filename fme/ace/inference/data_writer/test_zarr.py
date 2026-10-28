import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from fme.ace.inference.data_writer.test_data_writer import get_initial_condition_times
from fme.ace.inference.data_writer.zarr import (
    SeparateICZarrWriterAdapter,
    ZarrWriterAdapter,
    _get_ace_time_coords,
    ensure_numpy_coords,
)
from fme.core.timing import GlobalTimer


def get_batch_time(n_batch_times, n_initial_conditions, calendar="julian"):
    times = xr.date_range(
        "2020-01-01",
        freq="6h",
        periods=n_batch_times,
        calendar=calendar,
        use_cftime=True,
    )
    batch_time = xr.DataArray(times, dims="time")
    return xr.concat(
        [
            batch_time + i * datetime.timedelta(hours=6)
            for i in range(n_initial_conditions)
        ],
        dim="sample",
    )


CALENDARS = {
    "julian": cftime.DatetimeJulian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
}


@pytest.mark.parametrize("calendar", ["julian", "proleptic_gregorian", "noleap"])
def test__get_ace_time_coords(calendar):
    n_batch_times = 3
    n_timesteps = 6
    n_samples = 2
    timestep = datetime.timedelta(hours=6)
    start_time = (2020, 1, 1, 0, 0, 0)
    initial_condition_times = get_initial_condition_times(
        start_time, calendar, n_samples, separation_timedelta=timestep
    )
    batch_time = get_batch_time(
        n_batch_times=n_batch_times, n_initial_conditions=n_samples, calendar=calendar
    )
    lead_times_coord, init_times_coord, valid_times_coord = _get_ace_time_coords(
        initial_condition_times, batch_time, timestep, n_timesteps=n_timesteps
    )

    assert lead_times_coord.dims == ("time",)
    assert lead_times_coord.size == n_timesteps
    assert lead_times_coord.dtype == np.int64

    assert init_times_coord.dims == ("sample",)
    assert init_times_coord.size == n_samples
    assert init_times_coord.dtype == np.int64

    assert valid_times_coord.dims == ("sample", "time")
    assert valid_times_coord.shape == (n_samples, n_timesteps)
    assert valid_times_coord.dtype == np.int64

    start = CALENDARS[calendar](2020, 1, 1, 0)
    np.testing.assert_array_equal(
        lead_times_coord.values, np.arange(1, n_timesteps + 1) * 6 * 3600 * 1e6
    )
    np.testing.assert_array_equal(
        init_times_coord.values,
        cftime.date2num(
            initial_condition_times,
            units="microseconds since 1970-01-01",
            calendar=calendar,
        ),
    )

    reference = CALENDARS[calendar](1970, 1, 1)

    # check the first sample which is 6h init after sample 0
    expected_valid_times = []
    for i in range(n_timesteps):
        dt = (start + (i + 1) * datetime.timedelta(hours=6)) - reference
        expected_valid_times.append(
            dt.days * 86_400_000_000 + dt.seconds * 1_000_000 + dt.microseconds
        )
    np.testing.assert_array_equal(
        valid_times_coord.isel(sample=1), expected_valid_times
    )


def get_adapter_args(tmpdir, writer_cls) -> dict:
    return dict(
        path=str(tmpdir / "test.zarr"),
        dims=("sample", "time", "lat", "lon")
        if writer_cls == ZarrWriterAdapter
        else ("time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
                "ak": xr.DataArray([0, 1], dims=["z_interface"]),
            }
        ),
        timestep=datetime.timedelta(days=1),
        n_timesteps=2,
        initial_condition_times=np.array([cftime.datetime(2019, 12, 31)]),
    )


def get_two_step_batch() -> tuple[dict[str, torch.Tensor], xr.DataArray]:
    data = {"foo": torch.zeros((1, 2, 2, 2))}
    time = xr.DataArray(
        [[cftime.datetime(2020, 1, 1), cftime.datetime(2020, 1, 2)]],
        dims=("sample", "time"),
    )
    return data, time


@pytest.mark.parametrize("writer_cls", [ZarrWriterAdapter, SeparateICZarrWriterAdapter])
def test_zarr_adapter_can_overwrite(tmpdir, writer_cls):
    data, time = get_two_step_batch()
    args = get_adapter_args(tmpdir, writer_cls)
    adapter = writer_cls(**args)  # type: ignore
    adapter.append_batch(data, time)
    adapter = writer_cls(**args)  # type: ignore
    adapter.append_batch(data, time)


@pytest.mark.parametrize("writer_cls", [ZarrWriterAdapter, SeparateICZarrWriterAdapter])
def test_zarr_adapters_record_data_writer_io(tmpdir, writer_cls):
    data, time = get_two_step_batch()
    adapter = writer_cls(**get_adapter_args(tmpdir, writer_cls))  # type: ignore
    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        adapter.append_batch(data, time)
        durations = timer.get_durations()
    assert durations["data_writer_io"] > 0.0


@pytest.mark.parametrize("calendar", ["julian", "proleptic_gregorian", "noleap"])
def test_separate_ic_writer_preserves_time_calendar(tmpdir, calendar):
    """A downstream reader must decode the written ``time`` to the source dates.

    ``SeparateICZarrWriterAdapter`` encodes the time values with the source
    calendar, so the stored ``calendar`` attribute must name that same calendar.
    A mismatched label (e.g. ``"julian"`` for non-julian data) leaves the values
    inconsistent with the label, so decoding shifts every timestamp by the leap
    days accumulated between the 1970 encoding epoch and the data dates -- a shift
    present even when the coordinate does not span a leap day (~12 days for 2020).
    """
    n_timesteps = 2
    data = {"foo": torch.zeros((1, n_timesteps, 2, 2))}
    timestep = datetime.timedelta(days=1)
    initial_condition_times = get_initial_condition_times(
        (2019, 12, 31, 0, 0, 0), calendar, n_initial_conditions=1
    )
    # Daily batch times so the timestep-derived lead times reproduce them exactly.
    batch_time = xr.DataArray(
        xr.date_range(
            "2020-01-01",
            freq="1D",
            periods=n_timesteps,
            calendar=calendar,
            use_cftime=True,
        ).values[None, :],
        dims=("sample", "time"),
    )
    adapter = SeparateICZarrWriterAdapter(
        path=str(tmpdir / "test.zarr"),
        dims=("time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
            }
        ),
        timestep=timestep,
        n_timesteps=n_timesteps,
        initial_condition_times=initial_condition_times,
    )
    adapter.append_batch(data, batch_time)
    adapter.finalize()

    output = str(tmpdir / "test_ic0000.zarr")
    # A downstream reader uses xarray's default decoding, which honors the stored
    # ``calendar`` attribute. The decoded calendar dates must match the source
    # (compare day strings to ignore the cftime subclass and isolate the dates).
    decoded = xr.open_zarr(output)
    np.testing.assert_array_equal(
        decoded.time.dt.strftime("%Y-%m-%d").values,
        batch_time.isel(sample=0).dt.strftime("%Y-%m-%d").values,
    )
    # The stored attribute itself must also name the true calendar.
    raw = xr.open_zarr(output, decode_times=False)
    assert raw.time.attrs["calendar"] == calendar


def test_zarr_adapter_sample_region_writes(tmpdir):
    """With sample_offset the adapter writes data into the correct region.

    A store is sized for 2 samples (global_ic_times has 2 entries).
    An adapter with sample_offset=1 writes 1 local sample into the
    second slot.  The first sample region stays at the fill value.
    """
    n_timesteps = 2
    timestep = datetime.timedelta(days=1)
    global_ic_times = np.array(
        [cftime.datetime(2019, 12, 31), cftime.datetime(2020, 1, 1)]
    )

    # Local data for "rank 1" (1 of the 2 samples)
    data = {"foo": torch.ones((1, n_timesteps, 2, 2)) * 42.0}
    batch_time = xr.DataArray(
        [[cftime.datetime(2020, 1, 2), cftime.datetime(2020, 1, 3)]],
        dims=("sample", "time"),
    )

    adapter = ZarrWriterAdapter(
        path=str(tmpdir / "test.zarr"),
        dims=("sample", "time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
            }
        ),
        timestep=timestep,
        n_timesteps=n_timesteps,
        initial_condition_times=global_ic_times,
        sample_offset=1,
    )
    adapter.append_batch(data, batch_time)

    ds = xr.open_zarr(str(tmpdir / "test.zarr"))
    # Store has 2 samples; only sample index 1 was written
    assert ds.foo.shape == (2, n_timesteps, 2, 2)
    np.testing.assert_array_equal(ds.foo.values[1], 42.0)


def test_zarr_adapter_sample_shard_equals_local_samples(tmpdir):
    """The sample shard must equal the local sample count, not the global count.

    Under multi-GPU each rank writes a contiguous block of the sample
    dimension.  If the shard spans more samples than a single rank owns,
    two ranks will contend on the same storage object.
    """
    n_timesteps = 2
    timestep = datetime.timedelta(days=1)
    n_global = 4
    n_local = 2
    global_ic_times = np.array(
        [cftime.datetime(2020, 1, 1 + i) for i in range(n_global)]
    )
    data = {"foo": torch.zeros((n_local, n_timesteps, 2, 2))}
    batch_time = xr.DataArray(
        [
            [cftime.datetime(2020, 2, 1 + s, t) for t in range(n_timesteps)]
            for s in range(n_local)
        ],
        dims=("sample", "time"),
    )
    adapter = ZarrWriterAdapter(
        path=str(tmpdir / "test.zarr"),
        dims=("sample", "time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
            }
        ),
        timestep=timestep,
        n_timesteps=n_timesteps,
        initial_condition_times=global_ic_times,
        sample_offset=0,
    )
    adapter.append_batch(data, batch_time)

    ds = zarr.open_group(str(tmpdir / "test.zarr"), mode="r")
    sample_dim_index = 0  # dims = (sample, time, lat, lon)
    sample_shard = ds["foo"].metadata.shards[sample_dim_index]
    assert sample_shard == n_local, (
        f"sample shard {sample_shard} should equal local sample count {n_local}, "
        f"not global count {n_global}"
    )


def test_separate_ic_adapter_sample_offset_naming(tmpdir):
    """With sample_offset the per-IC stores are named with global indices."""
    n_timesteps = 2
    timestep = datetime.timedelta(days=1)
    local_ic_times = np.array([cftime.datetime(2020, 1, 1)])
    data = {"foo": torch.zeros((1, n_timesteps, 2, 2))}
    time = xr.DataArray(
        [[cftime.datetime(2020, 1, 2), cftime.datetime(2020, 1, 3)]],
        dims=("sample", "time"),
    )
    adapter = SeparateICZarrWriterAdapter(
        path=str(tmpdir / "test.zarr"),
        dims=("time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
            }
        ),
        timestep=timestep,
        n_timesteps=n_timesteps,
        initial_condition_times=local_ic_times,
        sample_offset=3,
    )
    adapter.append_batch(data, time)
    # The file should be named with global index 3, not local index 0
    assert (tmpdir / "test_ic0003.zarr").exists()
    assert not (tmpdir / "test_ic0000.zarr").exists()


def test_zarr_adapter_single_timestep_data(
    tmpdir,
):
    data = {"foo": torch.zeros((1, 1, 2, 2))}
    timestep = datetime.timedelta(days=1)
    initial_condition_times = np.array([cftime.datetime(2019, 12, 31)])
    time = xr.DataArray(
        [
            [
                cftime.datetime(2020, 1, 1),
            ]
        ],
        dims=("sample", "time"),
    )
    args = dict(
        path=str(tmpdir / "test.zarr"),
        dims=("sample", "time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
                "ak": xr.DataArray([0, 1], dims=["z_interface"]),
            }
        ),
        timestep=timestep,
        n_timesteps=1,
        initial_condition_times=initial_condition_times,
    )
    adapter = ZarrWriterAdapter(**args)  # type: ignore
    adapter.append_batch(data, time)

    ds = xr.open_zarr(str(tmpdir / "test.zarr"))
    assert ds.time.size == 1


@pytest.mark.parametrize(
    "bad_chunks",
    [{"sample": 2}, {"time": 2}, {"sample": 4, "time": 1}],
)
def test_zarr_adapter_rejects_non_unit_sample_or_time_chunks(bad_chunks):
    """Chunk sizes of sample=1 and time=1 are required for parallel-safe writes.

    Each data-parallel rank writes to its own contiguous block of samples.
    If a chunk spans more than one sample, two ranks could contend on the same
    chunk.
    """
    with pytest.raises(ValueError, match="must be 1"):
        ZarrWriterAdapter(
            path="/unused",
            dims=("sample", "time", "lat", "lon"),
            data_coords=ensure_numpy_coords(
                {
                    "lat": xr.DataArray([0, 1], dims=["lat"]),
                    "lon": xr.DataArray([0, 1], dims=["lon"]),
                }
            ),
            timestep=datetime.timedelta(days=1),
            n_timesteps=2,
            initial_condition_times=np.array([cftime.datetime(2020, 1, 1)]),
            chunks=bad_chunks,
        )


@pytest.mark.parallel
def test_zarr_adapter_parallel_region_writes(tmp_path):
    """Each data-parallel rank writes its sample region; combined store matches
    a serial reference."""
    from fme.core.distributed import Distributed

    dist = Distributed.get_instance()
    if dist.has_spatial_parallelism:
        pytest.skip("zarr region writes use data parallelism only")
    rank = dist.data_parallel_rank
    n_ranks = dist.total_data_parallel_ranks
    if n_ranks < 2:
        pytest.skip("needs multiple data-parallel ranks")

    # tmp_path differs per rank under torchrun; broadcast rank 0's path
    store_path = dist.data_parallel_broadcast_object(
        str(tmp_path / "parallel_test.zarr") if rank == 0 else None
    )

    n_global_samples = n_ranks * 2
    n_local = n_global_samples // n_ranks
    sample_offset = rank * n_local
    n_timesteps = 2
    timestep = datetime.timedelta(days=1)

    global_ic_times = np.array(
        [cftime.datetime(2020, 1, 1 + i) for i in range(n_global_samples)]
    )
    local_data = {
        "foo": torch.full((n_local, n_timesteps, 2, 2), float(rank)),
    }
    local_time = xr.DataArray(
        [
            [cftime.datetime(2020, 2, 1 + s, t) for t in range(n_timesteps)]
            for s in range(n_local)
        ],
        dims=("sample", "time"),
    )
    adapter = ZarrWriterAdapter(
        path=store_path,
        dims=("sample", "time", "lat", "lon"),
        data_coords=ensure_numpy_coords(
            {
                "lat": xr.DataArray([0, 1], dims=["lat"]),
                "lon": xr.DataArray([0, 1], dims=["lon"]),
            }
        ),
        timestep=timestep,
        n_timesteps=n_timesteps,
        initial_condition_times=global_ic_times,
        sample_offset=sample_offset,
    )
    adapter.append_batch(local_data, local_time)
    dist.barrier()

    ds = xr.open_zarr(store_path)
    assert ds.foo.shape == (n_global_samples, n_timesteps, 2, 2)
    for r in range(n_ranks):
        offset = r * n_local
        np.testing.assert_array_equal(
            ds.foo.values[offset : offset + n_local], float(r)
        )
