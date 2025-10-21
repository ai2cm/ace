import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.inference.data_writer.zarr import ZarrWriterAdapter, _get_ace_time_coords


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
    batch_time = get_batch_time(
        n_batch_times=n_batch_times, n_initial_conditions=n_samples, calendar=calendar
    )
    lead_times_coord, init_times_coord, valid_times_coord = _get_ace_time_coords(
        batch_time, n_timesteps=n_timesteps
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
        lead_times_coord.values, np.arange(n_timesteps) * 6 * 3600 * 1e6
    )
    expected_init_times = [
        start + datetime.timedelta(hours=6) * i for i in range(n_samples)
    ]
    np.testing.assert_array_equal(
        init_times_coord.values,
        cftime.date2num(
            expected_init_times,
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


def test_zarr_adapter_can_overwrite(tmpdir):
    data = {"foo": torch.zeros((1, 2, 2, 2))}
    time = xr.DataArray(
        [[cftime.datetime(2020, 1, 1), cftime.datetime(2020, 1, 2)]],
        dims=("sample", "time"),
    )
    args = dict(
        path=str(tmpdir / "test.zarr"),
        dims=("sample", "time", "lat", "lon"),
        data_coords={
            "lat": xr.DataArray([0, 1], dims=["lat"]),
            "lon": xr.DataArray([0, 1], dims=["lon"]),
        },
        n_timesteps=2,
        n_initial_conditions=1,
    )
    adapter = ZarrWriterAdapter(**args)  # type: ignore
    adapter.append_batch(data, 0, time)
    adapter = ZarrWriterAdapter(**args)  # type: ignore
    adapter.append_batch(data, 0, time)
