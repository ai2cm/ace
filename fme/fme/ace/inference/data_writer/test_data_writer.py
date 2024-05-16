from typing import NamedTuple

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.ace.inference.data_writer.main import (
    DataWriter,
    DataWriterConfig,
    PairedDataWriter,
)
from fme.ace.inference.data_writer.raw import get_batch_lead_times_microseconds
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig

CALENDAR_CFTIME = {
    "julian": cftime.DatetimeJulian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
}

SECONDS_PER_HOUR = 3600
MICROSECONDS_PER_SECOND = 1_000_000


def test_data_writer_config_save_names():
    variable_names = ["temp", "humidity"]
    kwargs = dict(
        save_prediction_files=False,
        save_monthly_files=False,
        save_histogram_files=False,
    )
    with pytest.warns():
        DataWriterConfig(save_raw_prediction_names=variable_names, **kwargs)
    for save_writer in [
        "save_prediction_files",
        "save_monthly_files",
        "save_histogram_files",
    ]:
        kwargs_copy = kwargs.copy()
        kwargs_copy.update({save_writer: True})
        DataWriterConfig(save_raw_prediction_names=variable_names, **kwargs_copy)


class TestDataWriter:
    class VariableMetadata(NamedTuple):
        units: str
        long_name: str

    @pytest.fixture(params=["julian", "proleptic_gregorian", "noleap"])
    def calendar(self, request):
        """
        These are the calendars for the datasets we tend to use: 'julian'
        for FV3GFS, 'noleap' for E3SM, and 'proleptic_gregorian' for generic
        datetimes in testing.

        Check that output datasets written with each calendar for their
        time coordinate re-load correctly.
        """
        return request.param

    def get_batch_times(self, start_time, end_time, freq, n_samples, calendar="julian"):
        datetime_class = CALENDAR_CFTIME[calendar]
        start_time = datetime_class(*start_time)
        end_time = datetime_class(*end_time)
        batch_times = xr.DataArray(
            xr.cftime_range(
                start_time,
                end_time,
                freq=freq,
                calendar=calendar,
            ).values,
            dims="time",
        )
        return xr.concat([batch_times for _ in range(n_samples)], dim="sample")

    @pytest.fixture
    def sample_metadata(self):
        return {
            "temp": self.VariableMetadata(units="K", long_name="Temperature"),
            "humidity": self.VariableMetadata(units="%", long_name="Relative Humidity"),
        }

    @pytest.fixture
    def sample_target_data(self):
        data = {
            # sample, time, lat, lon
            "temp": torch.rand((2, 3, 4, 5)),
            "humidity": torch.rand((2, 3, 4, 5)),  # input-only variable
            "pressure": torch.rand(
                (2, 3, 4, 5)
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand((2, 3, 4, 5)),  # optionally saved
        }
        return data

    @pytest.fixture
    def sample_prediction_data(self):
        data = {
            # sample, time, lat, lon
            "temp": torch.rand((2, 3, 4, 5)),
            "pressure": torch.rand(
                (2, 3, 4, 5)
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand((2, 3, 4, 5)),  # optionally saved
        }
        return data

    def test_append_batch(
        self,
        sample_metadata,
        sample_target_data,
        sample_prediction_data,
        tmp_path,
        calendar,
    ):
        n_samples = 2
        n_timesteps = 6
        writer = PairedDataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            enable_histogram_netcdfs=True,
            save_names=None,
            prognostic_names=[],
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            batch_times=batch_times,
        )
        start_time_2 = (2020, 1, 1, 18, 0, 0)
        end_time_2 = (2020, 1, 2, 6, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time_2,
            end_time=end_time_2,
            freq="6H",
            n_samples=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=3,
            batch_times=batch_times,
        )
        writer.flush()

        # Open the file and check the data
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        assert dataset["time"].units == "microseconds"
        assert dataset["init_time"].units == "microseconds since 1970-01-01 00:00:00"
        assert dataset["init_time"].calendar == calendar
        for var_name in set(sample_prediction_data.keys()):
            var_data = dataset.variables[var_name][:]
            assert var_data.shape == (
                n_samples,
                n_timesteps,
                4,
                5,
            )  # sample, time, lat, lon
            assert not np.isnan(var_data).any(), "unexpected NaNs in prediction data"
            if var_name in sample_metadata:
                assert (
                    dataset.variables[var_name].units == sample_metadata[var_name].units
                )
                assert (
                    dataset.variables[var_name].long_name
                    == sample_metadata[var_name].long_name
                )
            else:
                assert not hasattr(dataset.variables[var_name], "units")
                assert not hasattr(dataset.variables[var_name], "long_name")

        dataset.close()

        # Open the target output file and do smaller set of checks
        dataset = Dataset(tmp_path / "autoregressive_target.nc", "r")
        coord_names = {"time", "init_time", "valid_time", "lat", "lon"}
        assert set(dataset.variables) == set(sample_target_data) | coord_names

        # Open the file again with xarray and check the time coordinates,
        # since the values opened this way depend on calendar/units
        with xr.open_dataset(
            tmp_path / "autoregressive_predictions.nc",
            use_cftime=True,
            decode_timedelta=False,  # prefer handling lead times in ints not timedelta
        ) as ds:
            expected_lead_times = xr.DataArray(
                [
                    MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                    for i in np.arange(0, 31, 6)
                ],
                dims="time",
            ).assign_coords(
                {
                    "time": [
                        MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                        for i in np.arange(0, 31, 6)
                    ]
                }
            )
            xr.testing.assert_equal(ds["time"], expected_lead_times)
            expected_init_times = xr.DataArray(
                [CALENDAR_CFTIME[calendar](*start_time) for _ in range(n_samples)],
                dims=["sample"],
            )
            expected_init_times = expected_init_times.assign_coords(
                {"init_time": expected_init_times}
            )
            xr.testing.assert_equal(ds["init_time"], expected_init_times)
            for var_name in set(sample_prediction_data.keys()):
                assert "valid_time" in ds[var_name].coords
                assert "init_time" in ds[var_name].coords

        histograms = xr.open_dataset(tmp_path / "histograms.nc")
        actual_var_names = sorted([str(k) for k in histograms.keys()])
        assert len(actual_var_names) == 8
        assert "humidity" in actual_var_names
        assert histograms.data_vars["humidity"].attrs["units"] == "count"
        assert "humidity_bin_edges" in actual_var_names
        assert histograms.data_vars["humidity_bin_edges"].attrs["units"] == "%"
        counts_per_timestep = histograms["humidity"].sum(dim=["bin", "source"])
        same_count_each_timestep = np.all(
            counts_per_timestep.values == counts_per_timestep.values[0]
        )
        assert same_count_each_timestep

        with xr.open_dataset(tmp_path / "monthly_mean_predictions.nc") as ds:
            assert ds.counts.sum() == n_samples * n_timesteps
            assert np.sum(np.isnan(ds["precipitation"])) == 0
            assert np.sum(np.isnan(ds["temp"])) == 0
            assert np.sum(np.isnan(ds["pressure"])) == 0
            assert np.all(ds.init_time.dt.year.values > 0)
            assert np.all(ds.init_time.dt.year.values >= 0)
            assert np.all(ds.valid_time.dt.month.values >= 0)

    @pytest.mark.parametrize(
        ["save_names"],
        [
            pytest.param(None, id="None"),
            pytest.param(["temp", "humidity", "pressure"], id="subset"),
        ],
    )
    def test_append_batch_save_names(
        self,
        sample_metadata,
        sample_target_data,
        sample_prediction_data,
        tmp_path,
        save_names,
    ):
        n_samples = 2
        writer = PairedDataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=4,  # unused
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            save_names=save_names,
            enable_histogram_netcdfs=True,
            prognostic_names=save_names or [],
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            batch_times=batch_times,
        )
        writer.flush()
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        expected_variables = (
            set(save_names).intersection(sample_prediction_data)
            if save_names is not None
            else set(sample_prediction_data.keys())
        )
        assert set(dataset.variables.keys()) == expected_variables.union(
            {"init_time", "time", "lat", "lon", "valid_time"}
        )
        expected_prediction_variables = set(sample_prediction_data.keys())
        if save_names is not None:
            expected_prediction_variables = expected_prediction_variables.intersection(
                save_names
            )
        dataset = Dataset(tmp_path / "monthly_mean_predictions.nc", "r")
        expected_variables = (
            set(save_names)
            if save_names is not None
            else set(sample_target_data.keys())
        )
        assert set(dataset.variables.keys()) == expected_prediction_variables.union(
            {
                "init_time",
                "time",
                "valid_time",
                "counts",
                "lat",
                "lon",
            }
        )
        histograms = xr.open_dataset(tmp_path / "histograms.nc")
        if save_names is None:
            expected_names = set(sample_target_data)
        else:
            expected_names = set(save_names)
        expected_bin_edge_names = {f"{name}_bin_edges" for name in expected_names}
        assert set(histograms) == expected_names.union(expected_bin_edge_names)

    def test_append_batch_data_time_mismatch(
        self, sample_metadata, sample_target_data, sample_prediction_data, tmp_path
    ):
        n_samples = 2
        writer = PairedDataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=3,
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            save_names=None,
            enable_histogram_netcdfs=True,
            prognostic_names=[],
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples + 1,
        )
        with pytest.raises(ValueError):
            writer.append_batch(
                sample_target_data,
                sample_prediction_data,
                start_timestep=0,
                batch_times=batch_times,
            )

    def test_prediction_only_append_batch(self, sample_metadata, tmp_path, calendar):
        n_samples = 2
        n_timesteps = 8
        coarsen_factor = 2
        prediction_data = {
            "temp": torch.rand((n_samples, n_timesteps // coarsen_factor, 4, 5)),
            "pressure": torch.rand((n_samples, n_timesteps // coarsen_factor, 4, 5)),
        }
        writer = DataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=n_timesteps,
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_monthly_netcdfs=True,
            save_names=None,
            prognostic_names=["temp"],
            time_coarsen=TimeCoarsenConfig(coarsen_factor),
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 18, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            prediction_data,
            start_timestep=0,
            batch_times=batch_times,
        )
        start_time_2 = (2020, 1, 2, 0, 0, 0)
        end_time_2 = (2020, 1, 2, 18, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time_2,
            end_time=end_time_2,
            freq="6H",
            n_samples=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            prediction_data,
            start_timestep=4,
            batch_times=batch_times,
        )
        writer.flush()

        with xr.open_dataset(tmp_path / "autoregressive_predictions.nc") as ds:
            assert "temp" in ds
            expected_shape = (n_samples, n_timesteps // coarsen_factor, 4, 5)
            assert ds.temp.shape == expected_shape
            assert ds.valid_time.shape == expected_shape[:2]

        with xr.open_dataset(tmp_path / "monthly_mean_predictions.nc") as ds:
            assert ds.counts.sum() == n_samples * n_timesteps
            assert np.sum(np.isnan(ds["temp"])) == 0
            assert np.sum(np.isnan(ds["pressure"])) == 0
            assert np.all(ds.init_time.dt.year.values > 0)
            assert np.all(ds.init_time.dt.year.values >= 0)
            assert np.all(ds.valid_time.dt.month.values >= 0)

        xr.open_dataset(tmp_path / "restart.nc")


@pytest.mark.parametrize(
    ["init_times", "batch_times", "expected"],
    [
        pytest.param(
            np.array([cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0) for _ in range(3)]),
            np.array(
                [
                    xr.cftime_range(
                        cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0),
                        freq="6H",
                        periods=3,
                    ).values
                    for _ in range(3)
                ]
            ),
            np.array(
                [
                    MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                    for i in range(0, 18, 6)
                ]
            ),
            id="init_same",
        ),
        pytest.param(
            np.array(
                [cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0) for i in range(3)]
            ),
            np.array(
                [
                    xr.cftime_range(
                        cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0),
                        freq="6H",
                        periods=3,
                    )
                    for i in range(3)
                ]
            ),
            np.array(
                [
                    MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                    for i in range(0, 18, 6)
                ]
            ),
            id="init_different",
        ),
        pytest.param(
            np.array(
                [cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0) for i in range(3)]
            ),
            np.array(
                [
                    xr.cftime_range(
                        cftime.DatetimeJulian(2020, 1, 2, 6 * i, 0, 0),
                        freq="6H",
                        periods=3,
                    )
                    for i in range(3)
                ]
            ),
            np.array(
                [
                    MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                    for i in range(24, 42, 6)
                ]
            ),
            id="not_initial_time_window",
        ),
    ],
)
def test_get_batch_lead_times_microseconds(init_times, batch_times, expected):
    lead_time_seconds = get_batch_lead_times_microseconds(init_times, batch_times)
    assert lead_time_seconds.shape == expected.shape
    np.testing.assert_equal(lead_time_seconds, expected)


def test_get_batch_lead_times_microseconds_length_mismatch():
    init_times = np.array(
        [cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0) for i in range(3)]
    )
    batch_times = np.array(
        [
            xr.cftime_range(
                cftime.DatetimeJulian(2020, 1, 2, 6 * i, 0, 0),
                freq="6H",
                periods=3,
            ).values
            for i in range(2)
        ],
    )
    with pytest.raises(ValueError):
        get_batch_lead_times_microseconds(init_times, batch_times)


def test_get_batch_lead_times_microseconds_inconsistent_samples():
    init_times = np.array(
        [cftime.DatetimeJulian(2020, 1, 1, 6, 0, 0) for _ in range(2)]
    )
    batch_times = np.array(
        [
            xr.cftime_range(
                cftime.DatetimeJulian(2020, 1, 1, 6, 0, 0),
                freq="6H",
                periods=3,
            ),
            xr.cftime_range(
                cftime.DatetimeJulian(2020, 1, 1, 12, 0, 0),
                freq="6H",
                periods=3,
            ),
        ]
    )
    with pytest.raises(ValueError):
        get_batch_lead_times_microseconds(init_times, batch_times)


@pytest.mark.parametrize(
    ["years_ahead", "overflow"],
    [
        pytest.param(1e2, False, id="100_years_OK"),
        pytest.param(1e4, False, id="10_000_years_OK"),
        pytest.param(1e6, True, id="1_000_000_years_fails"),
    ],
)
def test_get_batch_lead_times_microseconds_overflow(years_ahead, overflow):
    init_times = np.array([cftime.DatetimeNoLeap(2020, 1, 1)])
    batch_times = np.array([cftime.DatetimeNoLeap(2020 + years_ahead, 1, 1)])[:, None]
    days_per_year_noleap = 365
    seconds_per_day = 86400
    expected_lead_time_microseconds = (
        MICROSECONDS_PER_SECOND * seconds_per_day * days_per_year_noleap * years_ahead
    )
    if not overflow:
        lead_time = get_batch_lead_times_microseconds(init_times, batch_times)
        assert lead_time.item() == expected_lead_time_microseconds
    else:
        with pytest.raises(OverflowError):
            get_batch_lead_times_microseconds(init_times, batch_times)
