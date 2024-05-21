import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.inference.data_writer.monthly import (
    MonthlyDataWriter,
    add_data,
    find_boundary,
    get_days_since_reference,
    months_for_timesteps,
)
from fme.core.data_loading.data_typing import VariableMetadata

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
        label="predictions",
        n_samples=n_samples,
        n_months=24,
        save_names=None,
        metadata={"x": VariableMetadata(units="m", long_name="x_name")},
        coords={},
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
                times = xr.DataArray(
                    [
                        [
                            initial_time + datetime.timedelta(hours=6 * i_write)
                            for _ in range(window_size)
                        ]
                        for _ in range(n_samples)
                    ],
                    dims=["sample", "time"],
                )
                assert times.shape == (n_samples, window_size)
                writer.append_batch(
                    data=month_data, start_timestep=0, batch_times=times
                )
    writer.flush()
    written = xr.open_dataset(str(tmpdir / "monthly_mean_predictions.nc"))
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


@pytest.mark.parametrize(
    "n_timesteps, min_expected",
    [
        (1, 1),
        (2, 2),  # 2 timesteps can cross a month boundary
        (3, 2),  # 3 timesteps can't cross an additional boundary
        (4 * 28, 2),  # 28 days can only cross one month boundary
        (4 * 28 + 1, 3),  # 29 days can cross two month boundaries
    ],
)
def test_months_for_timesteps(n_timesteps: int, min_expected: int):
    assert months_for_timesteps(n_timesteps, TIMESTEP) >= min_expected


def test_get_days_since_reference():
    years = np.array([2020, 2021])
    months = np.array([0, 1])  # expects zero-indexed months
    reference_date = cftime.DatetimeProlepticGregorian(2020, 1, 1)
    n_months = 3
    calendar = "proleptic_gregorian"
    days = get_days_since_reference(years, months, reference_date, n_months, calendar)
    assert days.shape == (2, 3)
    assert days[0, 0] == 0
    assert days[0, 1] == 31
    assert days[0, 2] == 31 + 29
    # 2020 is a leap year
    assert days[1, 0] == 366 + 31
    assert days[1, 1] == 366 + 31 + 28
    assert days[1, 2] == 366 + 31 + 28 + 31


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
