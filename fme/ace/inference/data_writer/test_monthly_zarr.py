import datetime

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly_zarr import MonthlyZarrWriter

TIMESTEP = datetime.timedelta(days=5)


def _writer(
    tmp_path,
    initial_condition_times: np.ndarray,
    n_timesteps: int,
    snapshot_path: str | None = None,
    restore_path: str | None = None,
) -> MonthlyZarrWriter:
    return MonthlyZarrWriter(
        path=str(tmp_path / "monthly.zarr"),
        initial_condition_times=initial_condition_times,
        n_timesteps=n_timesteps,
        timestep=TIMESTEP,
        save_names=None,
        variable_metadata={},
        coords={"lat": np.array([0.0, 1.0]), "lon": np.array([0.0, 1.0])},
        dataset_metadata=DatasetMetadata(),
        snapshot_path=snapshot_path,
        restore_path=restore_path,
    )


def _batch(values: list[float], times: list[cftime.datetime]):
    data = {
        "foo": torch.tensor(values, dtype=torch.float32).reshape(1, len(values), 1, 1)
        * torch.ones(1, len(values), 2, 2)
    }
    batch_time = xr.DataArray([times], dims=("sample", "time"))
    return data, batch_time


def test_monthly_zarr_accumulates_means_and_counts(tmp_path):
    """5-day steps starting late January span two months; the stored means and
    counts must aggregate by calendar month across multiple appends."""
    initial_condition_times = np.array([cftime.datetime(2020, 1, 17)])
    # forward times: Jan 22, Jan 27, Feb 1, Feb 6
    writer = _writer(tmp_path, initial_condition_times, n_timesteps=4)
    data, batch_time = _batch(
        [1.0, 2.0],
        [cftime.datetime(2020, 1, 22), cftime.datetime(2020, 1, 27)],
    )
    writer.append_batch(data, batch_time)
    data, batch_time = _batch(
        [3.0, 5.0],
        [cftime.datetime(2020, 2, 1), cftime.datetime(2020, 2, 6)],
    )
    writer.append_batch(data, batch_time)

    ds = xr.open_zarr(str(tmp_path / "monthly.zarr"), decode_timedelta=False)
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["counts"].values, [[2, 2]])
    np.testing.assert_allclose(
        ds["foo"].isel(sample=0).mean(["lat", "lon"]).values, [1.5, 4.0]
    )


def test_monthly_zarr_snapshot_restore_is_idempotent(tmp_path):
    """Re-running a segment after restoring the previous segment's snapshot
    yields the same result as running each segment once, and matches an
    unsegmented run."""
    initial_condition_times = np.array([cftime.datetime(2020, 1, 17)])
    segment_times = [
        [cftime.datetime(2020, 1, 22), cftime.datetime(2020, 1, 27)],
        [cftime.datetime(2020, 2, 1), cftime.datetime(2020, 2, 6)],
    ]
    segment_values = [[1.0, 2.0], [3.0, 5.0]]
    snapshot_path = str(tmp_path / "segment_0000" / "monthly.nc")

    single_dir = tmp_path / "single"
    single_dir.mkdir()
    single = _writer(single_dir, initial_condition_times, n_timesteps=4)
    for values, times in zip(segment_values, segment_times):
        single.append_batch(*_batch(values, times))
    ds_single = xr.open_zarr(
        str(single_dir / "monthly.zarr"), decode_timedelta=False
    ).load()

    # the writer is always constructed with the whole run's length, so the
    # month axis covers all segments
    segment_0 = _writer(
        tmp_path,
        initial_condition_times,
        n_timesteps=4,
        snapshot_path=snapshot_path,
    )
    segment_0.append_batch(*_batch(segment_values[0], segment_times[0]))
    segment_0.finalize()

    # segment 1 is constructed with its restart time as the initial condition
    # time, which must not redefine the run's month origin
    restart_times = np.array([cftime.datetime(2020, 1, 27)])

    def run_segment_1():
        writer = _writer(
            tmp_path,
            restart_times,
            n_timesteps=4,
            restore_path=snapshot_path,
        )
        writer.append_batch(*_batch(segment_values[1], segment_times[1]))

    run_segment_1()
    # simulate a preempted segment 1 being re-run from the snapshot
    run_segment_1()

    ds_segmented = xr.open_zarr(
        str(tmp_path / "monthly.zarr"), decode_timedelta=False
    ).load()
    xr.testing.assert_allclose(ds_segmented.drop_attrs(), ds_single.drop_attrs())
    np.testing.assert_array_equal(ds_segmented["counts"].values, [[2, 2]])


def test_monthly_zarr_restore_requires_snapshot(tmp_path):
    with pytest.raises(RuntimeError, match="does not exist"):
        _writer(
            tmp_path,
            np.array([cftime.datetime(2020, 1, 17)]),
            n_timesteps=2,
            restore_path=str(tmp_path / "missing.nc"),
        )


def test_monthly_zarr_rejects_times_outside_run(tmp_path):
    writer = _writer(tmp_path, np.array([cftime.datetime(2020, 1, 17)]), n_timesteps=2)
    data, batch_time = _batch([1.0], [cftime.datetime(2020, 6, 1)])
    with pytest.raises(ValueError, match="pre-allocated"):
        writer.append_batch(data, batch_time)
