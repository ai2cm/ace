import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly import MonthlyDataWriter
from fme.ace.inference.data_writer.monthly_zarr import MonthlyZarrWriter
from fme.core.dataset.data_typing import VariableMetadata

TIMESTEP = datetime.timedelta(days=5)
COORDS = {"lat": np.array([0.0, 1.0]), "lon": np.array([0.0, 1.0, 2.0])}


def _writer(path, initial_condition_times, n_timesteps, **kwargs) -> MonthlyZarrWriter:
    return MonthlyZarrWriter(
        path=str(path),
        initial_condition_times=initial_condition_times,
        n_timesteps=n_timesteps,
        timestep=TIMESTEP,
        save_names=kwargs.pop("save_names", None),
        variable_metadata=kwargs.pop(
            "variable_metadata", {"foo": VariableMetadata(units="m", long_name="foo")}
        ),
        coords=COORDS,
        dataset_metadata=DatasetMetadata(source={"inference_version": "1.0"}),
        **kwargs,
    )


# spatially varying, so a wrong lat/lon layout shows up in values, not just shapes
SPATIAL_PATTERN = 1.0 + torch.arange(
    len(COORDS["lat"]) * len(COORDS["lon"]), dtype=torch.float32
).reshape(len(COORDS["lat"]), len(COORDS["lon"]))


def _batch(values, times):
    """A batch of shape [sample, time, lat, lon]."""
    data = torch.tensor(values, dtype=torch.float32)[..., None, None] * SPATIAL_PATTERN
    return {"foo": data}, xr.DataArray(times, dims=["sample", "time"])


def test_monthly_zarr_writer_accumulates_across_batches(tmp_path):
    """5-day steps from mid-January span two months; the stored means and counts
    must aggregate by calendar month across separate appends."""
    initial_condition_times = np.array([cftime.DatetimeProlepticGregorian(2020, 1, 17)])
    # forward times: Jan 22, Jan 27, Feb 1, Feb 6
    writer = _writer(tmp_path / "monthly.zarr", initial_condition_times, n_timesteps=4)
    writer.append_batch(
        *_batch(
            [[1.0, 2.0]],
            [
                [
                    cftime.DatetimeProlepticGregorian(2020, 1, 22),
                    cftime.DatetimeProlepticGregorian(2020, 1, 27),
                ]
            ],
        )
    )
    writer.append_batch(
        *_batch(
            [[3.0, 5.0]],
            [
                [
                    cftime.DatetimeProlepticGregorian(2020, 2, 1),
                    cftime.DatetimeProlepticGregorian(2020, 2, 6),
                ]
            ],
        )
    )
    writer.finalize()

    ds = xr.open_zarr(str(tmp_path / "monthly.zarr"), decode_timedelta=False)
    assert dict(ds.sizes) == {"sample": 1, "time": 2, "lat": 2, "lon": 3}
    np.testing.assert_array_equal(ds["counts"].values, [[2, 2]])
    np.testing.assert_allclose(ds["foo"].isel(sample=0, lat=0, lon=0), [1.5, 4.0])
    np.testing.assert_allclose(
        ds["foo"].isel(sample=0, lat=1, lon=2),
        np.array([1.5, 4.0]) * SPATIAL_PATTERN[1, 2].item(),
        rtol=1e-6,
    )
    assert ds["foo"].attrs["units"] == "m"
    assert "counts" in ds.coords
    assert "counts" in ds["foo"].coords
    assert ds["time"].attrs["units"] == "months"
    np.testing.assert_array_equal(ds["time"].values, [0, 1])
    np.testing.assert_array_equal(
        ds["valid_time"].isel(sample=0).values,
        np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]"),
    )
    assert ds["init_time"].isel(sample=0).values == np.datetime64("2020-01-17")
    assert ds.attrs["title"] == "ACE monthly data file"
    assert ds.attrs["source.inference_version"] == "1.0"


def test_monthly_zarr_writer_matches_netcdf_writer(tmp_path):
    """The zarr and netCDF monthly writers are two representations of the same
    aggregation, so their means and counts must agree."""
    initial_condition_times = np.array(
        [
            cftime.DatetimeProlepticGregorian(2020, 1, 1),
            cftime.DatetimeProlepticGregorian(2020, 2, 20),
        ]
    )
    n_samples, n_timesteps = 2, 30
    torch.manual_seed(0)
    zarr_writer = _writer(
        tmp_path / "monthly_mean.zarr", initial_condition_times, n_timesteps
    )
    netcdf_writer = MonthlyDataWriter(
        path=str(tmp_path),
        label="monthly_mean",
        initial_condition_times=initial_condition_times,
        timestep=TIMESTEP,
        save_names=None,
        variable_metadata={"foo": VariableMetadata(units="m", long_name="foo")},
        coords=COORDS,
        dataset_metadata=DatasetMetadata(),
    )
    window = 5
    for i_window in range(n_timesteps // window):
        values = torch.rand(n_samples, window).tolist()
        times = [
            [
                initial_condition_times[i_sample]
                + (i_window * window + i_step + 1) * TIMESTEP
                for i_step in range(window)
            ]
            for i_sample in range(n_samples)
        ]
        data, batch_time = _batch(values, times)
        zarr_writer.append_batch(data, batch_time)
        netcdf_writer.append_batch(data, batch_time)
    zarr_writer.finalize()
    netcdf_writer.finalize()

    from_zarr = xr.open_zarr(str(tmp_path / "monthly_mean.zarr"), decode_times=False)
    from_netcdf = xr.open_dataset(
        str(tmp_path / "monthly_mean.nc"), decode_times=False, decode_timedelta=False
    )
    for name in ["foo", "counts", "valid_time", "init_time"]:
        np.testing.assert_allclose(
            from_zarr[name].values, from_netcdf[name].values, rtol=1e-6
        )


def test_monthly_zarr_writer_saves_only_requested_names(tmp_path):
    writer = _writer(
        tmp_path / "monthly.zarr",
        np.array([cftime.DatetimeProlepticGregorian(2020, 1, 1)]),
        n_timesteps=1,
        save_names=["foo"],
    )
    data, batch_time = _batch(
        [[1.0]], [[cftime.DatetimeProlepticGregorian(2020, 1, 6)]]
    )
    data["bar"] = data["foo"].clone()
    writer.append_batch(data, batch_time)
    ds = xr.open_zarr(str(tmp_path / "monthly.zarr"), decode_timedelta=False)
    assert set(ds.data_vars) == {"foo"}


def test_monthly_zarr_writer_rejects_times_outside_the_run(tmp_path):
    writer = _writer(
        tmp_path / "monthly.zarr",
        np.array([cftime.DatetimeProlepticGregorian(2020, 1, 1)]),
        n_timesteps=2,
    )
    data, batch_time = _batch(
        [[1.0]], [[cftime.DatetimeProlepticGregorian(2020, 6, 1)]]
    )
    with pytest.raises(ValueError, match="pre-allocated"):
        writer.append_batch(data, batch_time)


@pytest.mark.parametrize("dim", ["time", "sample"])
def test_monthly_zarr_writer_rejects_chunked_time_or_sample(tmp_path, dim: str):
    with pytest.raises(ValueError, match="must be 1"):
        _writer(
            tmp_path / "monthly.zarr",
            np.array([cftime.DatetimeProlepticGregorian(2020, 1, 1)]),
            n_timesteps=1,
            chunks={dim: 2},
        )


def test_monthly_zarr_writer_writes_to_non_local_filesystem():
    writer = _writer(
        "memory://experiment_dir/monthly.zarr",
        np.array([cftime.DatetimeProlepticGregorian(2020, 1, 1)]),
        n_timesteps=1,
    )
    writer.append_batch(
        *_batch([[1.0]], [[cftime.DatetimeProlepticGregorian(2020, 1, 6)]])
    )
    writer.finalize()
    ds = xr.open_zarr("memory://experiment_dir/monthly.zarr", decode_timedelta=False)
    np.testing.assert_allclose(ds["foo"].isel(sample=0, time=0, lat=0, lon=0), 1.0)


def test_monthly_zarr_writer_month_zero_follows_the_first_output_time(tmp_path):
    """An IC late in a month has its first output time in the next month, so
    month 0 of the lead-time axis is the later month, not the IC's."""
    initial_condition_times = np.array([cftime.DatetimeProlepticGregorian(2020, 1, 30)])
    # forward times: Feb 4, Feb 9 -- the run never writes anything in January
    writer = _writer(tmp_path / "monthly.zarr", initial_condition_times, n_timesteps=2)
    writer.append_batch(
        *_batch(
            [[1.0, 3.0]],
            [
                [
                    cftime.DatetimeProlepticGregorian(2020, 2, 4),
                    cftime.DatetimeProlepticGregorian(2020, 2, 9),
                ]
            ],
        )
    )
    writer.finalize()

    ds = xr.open_zarr(str(tmp_path / "monthly.zarr"), decode_timedelta=False)
    # one month, February, rather than a January origin with an empty month 0
    assert dict(ds.sizes) == {"sample": 1, "time": 1, "lat": 2, "lon": 3}
    np.testing.assert_array_equal(ds["counts"].values, [[2]])
    np.testing.assert_allclose(ds["foo"].isel(sample=0, lat=0, lon=0), [2.0])
    np.testing.assert_array_equal(
        ds["valid_time"].isel(sample=0).values,
        np.array(["2020-02-15"], dtype="datetime64[ns]"),
    )
    assert ds["init_time"].isel(sample=0).values == np.datetime64("2020-01-30")
