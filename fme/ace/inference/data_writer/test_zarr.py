import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.inference.data_writer.test_data_writer import get_initial_condition_times
from fme.ace.inference.data_writer.zarr import (
    SeparateICZarrWriterAdapter,
    ZarrWriterAdapter,
    _get_ace_time_coords,
    ensure_numpy_coords,
)


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


@pytest.mark.parametrize("writer_cls", [ZarrWriterAdapter, SeparateICZarrWriterAdapter])
def test_zarr_adapter_can_overwrite(tmpdir, writer_cls):
    data = {"foo": torch.zeros((1, 2, 2, 2))}
    timestep = datetime.timedelta(days=1)
    initial_condition_times = np.array([cftime.datetime(2019, 12, 31)])
    time = xr.DataArray(
        [[cftime.datetime(2020, 1, 1), cftime.datetime(2020, 1, 2)]],
        dims=("sample", "time"),
    )
    args = dict(
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
        timestep=timestep,
        n_timesteps=2,
        initial_condition_times=initial_condition_times,
    )
    adapter = writer_cls(**args)  # type: ignore
    adapter.append_batch(data, time)
    adapter = writer_cls(**args)  # type: ignore
    adapter.append_batch(data, time)


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
