import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from xarray.coders import CFDatetimeCoder

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly import MonthlyDataWriter
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
    initial_condition_times = np.array(
        [
            cftime.DatetimeProlepticGregorian(2020, 1, 1),
            cftime.DatetimeProlepticGregorian(2020, 1, 2),
        ]
    )
    n_samples = len(initial_condition_times)
    n_lat, n_lon = 8, 16
    writer = MonthlyDataWriter(
        path=str(tmpdir),
        label="monthly_mean_predictions",
        initial_condition_times=initial_condition_times,
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
            cftime.DatetimeProlepticGregorian(2020, 1, 2),
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


def test_monthly_data_writer_long_run(tmpdir):
    # Regression test for GitHub issue #1246. Using a 360-day calendar with
    # 82 timesteps and a timestep length of 360 days is close to the
    # fastest possible test we can construct for this issue. It takes less
    # than 0.2s on a laptop.
    n_timesteps = 82
    initial_condition_times = np.array([cftime.Datetime360Day(2019, 1, 1)])
    n_samples = len(initial_condition_times)
    n_lat, n_lon = 1, 1
    timestep = datetime.timedelta(days=360)
    writer = MonthlyDataWriter(
        path=str(tmpdir),
        label="monthly_mean_predictions",
        initial_condition_times=initial_condition_times,
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
