import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from xarray.coders import CFDatetimeCoder

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly import (
    MonthlyDataWriter,
    add_data,
    find_boundary,
    get_days_since_reference,
)
from fme.core.dataset.data_typing import VariableMetadata

TIMESTEP = datetime.timedelta(hours=6)


@pytest.mark.parametrize(
    "window_size",
    [
        1,
        2,
    ],
)
@pytest.mark.parametrize(
    "n_writes",
    [1, 2],
)
def test_monthly_data_writer(tmpdir, window_size: int, n_writes: int):
    n_samples = 2
    n_lat, n_lon = 8, 16
    writer = MonthlyDataWriter(
        path=str(tmpdir),
        label="monthly_mean_predictions",
        n_samples=n_samples,
        save_names=None,
        variable_metadata={"x": VariableMetadata(units="m", long_name="x_name")},
        coords={},
        dataset_metadata=DatasetMetadata(source={"inference_version": "1.0"}),
    )
    month_values = []
    for year in range(2020, 2022):
        for month in range(1, 13):
            x = torch.rand(n_samples, 1, n_lat, n_lon) + 1.0
            month_values.append(x)
            # repeat x along axis 1 to simulate a window_size > 1
            x_window = torch.cat([x] * window_size, dim=1)
            month_data = {"x": x_window}
            initial_time = cftime.DatetimeProlepticGregorian(year, month, 1, 0, 0, 0)
            for i_write in range(n_writes):
                time = xr.DataArray(
                    [
                        [
                            initial_time + datetime.timedelta(hours=6 * i_write)
                            for _ in range(window_size)
                        ]
                        for _ in range(n_samples)
                    ],
                    dims=["sample", "time"],
                )
                assert time.shape == (n_samples, window_size)
                writer.append_batch(data=month_data, batch_time=time)
    writer.finalize()
    written = xr.open_dataset(
        str(tmpdir / "monthly_mean_predictions.nc"),
        decode_timedelta=False,
        decode_times=CFDatetimeCoder(use_cftime=True),
    )
    assert written["x"].shape == (n_samples, 24, n_lat, n_lon)
    assert np.sum(written["x"].values != 0) > 0, "No non-zero values written"
    assert (
        np.sum(written["x"].values == 0.0) == 0
    ), "Some values are zero (were not added to)"
    np.testing.assert_array_equal(written["counts"].values, window_size * n_writes)
    np.testing.assert_allclose(
        written["x"],
        torch.cat(month_values, dim=1).cpu().numpy(),
    )
    assert "counts" in written.coords
    assert "counts" in written.x.coords
    assert "counts" in written.valid_time.coords
    assert written.attrs["title"] == "ACE monthly mean predictions data file"
    assert written.attrs["source.inference_version"] == "1.0"

    # Validate time coordinates.
    expected_init_time = xr.DataArray(
        [
            cftime.DatetimeProlepticGregorian(2020, 1, 1),
            cftime.DatetimeProlepticGregorian(2020, 1, 1),
        ],
        dims=["sample"],
        name="init_time",
    )
    expected_init_time = expected_init_time.assign_coords(init_time=expected_init_time)
    expected_time = np.arange(24)
    expected_counts = (["sample", "time"], np.full((2, 24), window_size * n_writes))
    valid_times = (
        xr.date_range(
            "2020",
            periods=24,
            freq="MS",
            calendar="proleptic_gregorian",
            use_cftime=True,
        )
        .shift(14, "D")
        .tolist()
    )
    valid_times = [valid_times, valid_times]
    expected_valid_time = xr.DataArray(
        valid_times, dims=["sample", "time"], name="valid_time"
    )
    expected_valid_time = expected_valid_time.assign_coords(
        init_time=expected_init_time, time=expected_time, counts=expected_counts
    )
    expected_valid_time = expected_valid_time.assign_coords(
        valid_time=expected_valid_time
    )
    xr.testing.assert_equal(written.init_time, expected_init_time)
    xr.testing.assert_equal(written.valid_time, expected_valid_time)
    xr.testing.assert_equal(written.time, expected_valid_time.time)
    xr.testing.assert_equal(written.counts, expected_valid_time.counts)
    assert written.init_time.dt.calendar == "proleptic_gregorian"
    assert written.valid_time.dt.calendar == "proleptic_gregorian"
    assert written.time.attrs["units"] == "months"


@pytest.mark.parametrize("num_years", [2, 500])
@pytest.mark.parametrize("calendar", ["proleptic_gregorian", "noleap"])
def test_get_days_since_reference(num_years, calendar):
    first_year = 2020
    final_year = first_year + num_years - 1
    years = np.array([i for i in range(first_year, final_year + 1)])
    months = np.zeros((num_years,), dtype=int)
    # For last year set month to 1
    months[-1] = 1
    if calendar == "proleptic_gregorian":
        reference_date = cftime.DatetimeProlepticGregorian(2020, 1, 1)
    else:
        reference_date = cftime.DatetimeNoLeap(2020, 1, 1)
    n_months = 3
    days = get_days_since_reference(years, months, reference_date, n_months, calendar)
    assert days.shape == (num_years, 3)
    # 2020 is a leap year in proleptic_gregorian
    if calendar == "proleptic_gregorian":
        assert days[0, 0] == 0
        assert days[0, 1] == 31
        assert days[0, 2] == 31 + 29
        if num_years == 2:
            assert days[1, 0] == 366 + 31
            assert days[1, 1] == 366 + 31 + 28
            assert days[1, 2] == 366 + 31 + 28 + 31
        if num_years == 500:
            # 121 is number of leap days
            assert days[499, 0] == 182135 + 121 + 31
            assert days[499, 1] == 182135 + 121 + 31 + 28
    if calendar == "noleap":
        assert days[0, 0] == 0
        assert days[0, 1] == 31
        assert days[0, 2] == 31 + 28
        if num_years == 2:
            assert days[1, 0] == 365 + 31
            assert days[1, 1] == 365 + 31 + 28
            assert days[1, 2] == 365 + 31 + 28 + 31
        if num_years == 500:
            assert days[499, 0] == 182135 + 31
            assert days[499, 1] == 182135 + 31 + 28


def test_days_since_reference_with_month_offset():
    calendar = "noleap"
    month_offset = 2
    offset_n_months = 3
    n_months = month_offset + offset_n_months

    years = np.array([2020, 2021])
    months = np.zeros((2,), dtype=int)
    reference_date = cftime.DatetimeNoLeap(2020, 1, 1)

    full = get_days_since_reference(years, months, reference_date, n_months, calendar)
    expected = full[:, month_offset:]
    result = get_days_since_reference(
        years,
        months,
        reference_date,
        offset_n_months,
        calendar,
        month_offset=month_offset,
    )
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "month_array, expected",
    [
        pytest.param([1, 2, 3, 4, 5, 6], 1, id="linear"),
        pytest.param([1, 1, 2], 2, id="after two steps"),
        pytest.param([1], 1, id="one value"),
        pytest.param([1, 1, 1, 1], 4, id="all the same"),
        pytest.param([0] * 50 + [1] * (23), 50, id="long array case"),
    ],
)
def test_find_boundary(month_array, expected):
    assert (
        find_boundary(np.asarray(month_array), start_month=month_array[0]) == expected
    )


def test_add_data_one_first_month():
    target = np.zeros((2, 3))
    target_start_counts = np.zeros((2, 3), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32)
    expected = np.zeros((2, 3))
    expected[:, 0] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)
    np.testing.assert_array_equal(target_start_counts, 0)


def test_add_data_one_first_month_averaging():
    target = np.zeros((2, 3))
    target[0, 0] = 2.0
    target_start_counts = np.zeros((2, 3), dtype=np.int32)
    target_start_counts[0, 0] = 1
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32)
    expected = np.zeros((2, 3))
    expected[0, 0] = (2 + 5) / 6
    expected[1, 0] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def test_add_data_one_later_month():
    target = np.zeros((2, 4))
    target_start_counts = np.zeros((2, 4), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32) + 2
    expected = np.zeros((2, 4))
    expected[:, 2] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def test_add_data_two_later_months():
    target = np.zeros((2, 4))
    target_start_counts = np.zeros((2, 4), dtype=np.int32)
    source = np.ones((2, 5))
    months_elapsed = np.zeros((2, 5), dtype=np.int32) + 2
    months_elapsed[0, 2:] = 3
    months_elapsed[1, 3:] = 3
    expected = np.zeros((2, 4))
    expected[0, 2] = 1
    expected[0, 3] = 1
    expected[1, 2] = 1
    expected[1, 3] = 1

    add_data(
        target=target,
        target_start_counts=target_start_counts,
        source=source,
        months_elapsed=months_elapsed,
    )
    np.testing.assert_array_equal(target, expected)


def test_monthly_data_writer_long_run(tmpdir):
    # Regression test for GitHub issue #1246. Using a 360-day calendar with
    # 82 timesteps and a timestep length of 360 days is close to the
    # fastest possible test we can construct for this issue. It takes less
    # than 0.2s on a laptop.
    n_timesteps = 82
    n_samples = 1
    n_lat, n_lon = 1, 1
    timestep = datetime.timedelta(days=360)
    writer = MonthlyDataWriter(
        path=str(tmpdir),
        label="monthly_mean_predictions",
        n_samples=n_samples,
        save_names=None,
        variable_metadata={"x": VariableMetadata(units="m", long_name="x_name")},
        coords={},
        dataset_metadata=DatasetMetadata(source={"inference_version": "1.0"}),
    )
    time = cftime.Datetime360Day(2020, 1, 1)
    for _ in range(n_timesteps):
        x = torch.ones((n_samples, 1, n_lat, n_lon))
        month_data = {"x": x}
        batch_time = xr.DataArray([[time]], dims=["sample", "time"])
        writer.append_batch(data=month_data, batch_time=batch_time)
        time = time + timestep
    writer.finalize()
