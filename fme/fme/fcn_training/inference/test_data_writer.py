from typing import NamedTuple

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from netCDF4 import Dataset

from fme.fcn_training.inference.data_writer import DataWriter, DataWriterConfig
from fme.fcn_training.inference.data_writer.prediction import (
    get_batch_lead_times_microseconds,
)

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
        n_samples = 4
        n_timesteps = 6
        writer = DataWriter(
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
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples // 2,
            calendar=calendar,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            start_sample=0,
            batch_times=batch_times,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            start_sample=2,
            batch_times=batch_times,
        )
        start_time_2 = (2020, 1, 1, 18, 0, 0)
        end_time_2 = (2020, 1, 2, 6, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time_2,
            end_time=end_time_2,
            freq="6H",
            n_samples=n_samples // 2,
            calendar=calendar,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=3,
            start_sample=0,
            batch_times=batch_times,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=3,
            start_sample=2,
            batch_times=batch_times,
        )
        writer.flush()

        # Open the file and check the data
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        assert dataset["lead"].units == "microseconds"
        assert dataset["init"].units == "microseconds since 1970-01-01 00:00:00"
        assert dataset["init"].calendar == calendar
        for var_name in set(sample_target_data.keys()):
            var_data = dataset.variables[var_name][:]
            assert var_data.shape == (
                2,
                n_samples,
                n_timesteps,
                4,
                5,
            )  # source, sample, time, lat, lon
            assert not np.isnan(var_data[0, :]).any(), "unexpected NaNs in target data"
            if var_name in sample_prediction_data:
                assert not np.isnan(
                    var_data[1, :]
                ).any(), "unexpected NaNs in prediction data"

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
                dims="lead",
            ).assign_coords(
                {
                    "lead": [
                        MICROSECONDS_PER_SECOND * SECONDS_PER_HOUR * i
                        for i in np.arange(0, 31, 6)
                    ]
                }
            )
            xr.testing.assert_equal(ds["lead"], expected_lead_times)
            expected_init_times = xr.DataArray(
                [CALENDAR_CFTIME[calendar](*start_time) for _ in range(n_samples)],
                dims=["sample"],
            )
            expected_init_times = expected_init_times.assign_coords(
                {"init": expected_init_times}
            )
            xr.testing.assert_equal(ds["init"], expected_init_times)
            for var_name in set(sample_target_data.keys()):
                assert "valid_time" in ds[var_name].coords
                assert "init" in ds[var_name].coords

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

        with xr.open_dataset(tmp_path / "monthly_binned_predictions.nc") as ds:
            assert ds.counts.sum() == n_samples * n_timesteps
            assert np.sum(np.isnan(ds["precipitation"])) == 0
            assert np.sum(np.isnan(ds["temp"])) == 0
            assert np.sum(np.isnan(ds["pressure"])) == 0
            assert np.all(ds.init.dt.year.values > 0)
            assert np.all(ds.init.dt.year.values >= 0)
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
        writer = DataWriter(
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
            start_sample=0,
            batch_times=batch_times,
        )
        writer.flush()
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        expected_variables = (
            set(save_names)
            if save_names is not None
            else set(sample_target_data.keys())
        )
        assert set(dataset.variables.keys()) == expected_variables.union(
            {"source", "init", "lead", "lat", "lon", "valid_time"}
        )
        expected_prediction_variables = set(sample_prediction_data.keys())
        if save_names is not None:
            expected_prediction_variables = expected_prediction_variables.intersection(
                save_names
            )
        dataset = Dataset(tmp_path / "monthly_binned_predictions.nc", "r")
        expected_variables = (
            set(save_names)
            if save_names is not None
            else set(sample_target_data.keys())
        )
        assert set(dataset.variables.keys()) == expected_prediction_variables.union(
            {
                "init",
                "lead",
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

    def test_append_batch_past_end_of_samples(
        self, sample_metadata, sample_target_data, sample_prediction_data, tmp_path
    ):
        n_samples = 4
        writer = DataWriter(
            str(tmp_path),
            n_samples=n_samples,
            n_timesteps=4,  # unused
            metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            save_names=None,
            enable_histogram_netcdfs=True,
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_times = self.get_batch_times(
            start_time=start_time,
            end_time=end_time,
            freq="6H",
            n_samples=n_samples // 2,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            start_sample=0,
            batch_times=batch_times,
        )
        writer.append_batch(
            sample_target_data,
            sample_prediction_data,
            start_timestep=0,
            start_sample=2,
            batch_times=batch_times,
        )
        with pytest.raises(ValueError):
            writer.append_batch(
                sample_target_data,
                sample_prediction_data,
                start_timestep=0,
                start_sample=4,
                batch_times=batch_times,
            )

    def test_append_batch_data_time_mismatch(
        self, sample_metadata, sample_target_data, sample_prediction_data, tmp_path
    ):
        n_samples = 2
        writer = DataWriter(
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
                start_sample=0,
                batch_times=batch_times,
            )


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
