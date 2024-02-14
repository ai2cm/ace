import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.data_loading.data_typing import VariableMetadata
from fme.fcn_training.inference.data_writer.monthly import (
    MonthlyDataWriter,
    add_at,
    get_days_since_reference,
    months_for_timesteps,
)


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
            for i_day in range(n_writes):
                times = xr.DataArray(
                    [
                        [
                            cftime.DatetimeProlepticGregorian(
                                year, month, 1, i_day, 0, 0
                            )
                            for _ in range(window_size)
                        ]
                        for _ in range(n_samples)
                    ],
                    dims=["sample", "time"],
                )
                assert times.shape == (n_samples, window_size)
                writer.append_batch(
                    data=month_data,
                    start_sample=0,
                    batch_times=times,
                )
    writer.flush()
    written = xr.open_dataset(str(tmpdir / "monthly_binned_predictions.nc"))
    assert written["x"].shape == (n_samples, 24, n_lat, n_lon)
    assert np.sum(written["x"].values != 0) > 0, "No non-zero values written"
    assert (
        np.sum(written["x"].values == 0.0) == 0
    ), "Some values are zero (were not added to)"
    np.testing.assert_array_equal(written["counts"].values, window_size * n_writes)
    np.testing.assert_allclose(
        written["x"],
        torch.cat(month_values, dim=1).cpu().numpy() * window_size * n_writes,
    )


def test_add_at():
    m = 2
    n_samples = 3
    n_time = 5
    n_lat = 4
    n_lon = 8
    values = np.random.uniform(size=(m, n_lat, n_lon))
    indices = np.zeros((m,), dtype=int) + 2
    target = np.zeros((n_samples, n_time, n_lat, n_lon))
    assert np.sum(target) == 0.0
    add_at(target, (indices, indices), values)
    assert np.sum(target) > 0.0
    np.testing.assert_allclose(target[2, 2], np.sum(values, axis=0))
    np.testing.assert_allclose(np.sum(target), np.sum(values))


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
    assert months_for_timesteps(n_timesteps) >= min_expected


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
