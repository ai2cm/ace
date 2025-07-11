import datetime
from typing import NamedTuple

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
from netCDF4 import Dataset
from xarray.coding.times import CFDatetimeCoder

from fme.ace.data_loading.batch_data import PairedData
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.main import (
    DataWriter,
    DataWriterConfig,
    PairedDataWriter,
)
from fme.ace.inference.data_writer.raw import get_batch_lead_time_microseconds
from fme.ace.inference.data_writer.time_coarsen import TimeCoarsenConfig
from fme.core.device import get_device

CALENDAR_CFTIME = {
    "julian": cftime.DatetimeJulian,
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
}

SECONDS_PER_HOUR = 3600
MICROSECONDS_PER_SECOND = 1_000_000
TIMESTEP = datetime.timedelta(hours=6)


def test_data_writer_config_save_names():
    variable_names = ["temp", "humidity"]
    kwargs = dict(
        save_prediction_files=False,
        save_monthly_files=False,
        save_histogram_files=False,
    )
    for save_writer in [
        "save_prediction_files",
        "save_monthly_files",
        "save_histogram_files",
    ]:
        kwargs_copy = kwargs.copy()
        kwargs_copy.update({save_writer: True})
        DataWriterConfig(names=variable_names, **kwargs_copy)  # type: ignore


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

    def get_batch_time(
        self, start_time, end_time, freq, n_initial_conditions, calendar="julian"
    ):
        datetime_class = CALENDAR_CFTIME[calendar]
        start_time = datetime_class(*start_time)
        end_time = datetime_class(*end_time)
        batch_time = xr.DataArray(
            xr.date_range(
                start_time,
                end_time,
                freq=freq,
                calendar=calendar,
                use_cftime=True,
            ).values,
            dims="time",
        )
        return xr.concat(
            [batch_time for _ in range(n_initial_conditions)], dim="sample"
        )

    @pytest.fixture
    def sample_metadata(self):
        return {
            "temp": self.VariableMetadata(units="K", long_name="Temperature"),
            "humidity": self.VariableMetadata(units="%", long_name="Relative Humidity"),
        }

    @pytest.fixture
    def sample_target_data(self, request):
        shape = request.param
        data = {
            # sample, time, *horizontal_dims
            "temp": torch.rand(shape),
            "humidity": torch.rand(shape),  # input-only variable
            "pressure": torch.rand(
                shape
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand(shape),  # optionally saved
        }
        return data

    @pytest.fixture
    def sample_prediction_data(self, request):
        shape = request.param
        data = {
            # sample, time, lat, lon
            "temp": torch.rand(shape),
            "pressure": torch.rand(
                shape
            ),  # Extra variable for which there's no metadata
            "precipitation": torch.rand(shape),  # optionally saved
        }
        return data

    @pytest.fixture
    def coords(self, request):
        return request.param

    @pytest.mark.parametrize(
        "sample_target_data, sample_prediction_data, coords",
        [
            pytest.param(
                (2, 3, 4, 5),
                (2, 3, 4, 5),
                {"lat": np.arange(4), "lon": np.arange(5)},
                id="LatLon",
            ),
            pytest.param(
                (2, 3, 6, 4, 5),
                (2, 3, 6, 4, 5),
                {"face": np.arange(6), "height": np.arange(4), "width": np.arange(5)},
                id="HEALPix",
            ),
        ],
        indirect=True,
    )
    def test_append_batch(
        self,
        sample_metadata,
        sample_target_data,
        sample_prediction_data,
        tmp_path,
        calendar,
        coords,
    ):
        n_initial_conditions = 2
        n_timesteps = 6
        writer = PairedDataWriter(
            str(tmp_path),
            n_initial_conditions=n_initial_conditions,
            n_timesteps=n_timesteps,
            timestep=TIMESTEP,
            variable_metadata=sample_metadata,
            coords=coords,
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            enable_histogram_netcdfs=True,
            save_names=None,
            dataset_metadata=DatasetMetadata(source={"inference_version": "1.0"}),
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time,
            end_time=end_time,
            freq="6h",
            n_initial_conditions=n_initial_conditions,
            calendar=calendar,
        )
        writer.append_batch(
            batch=PairedData(
                prediction=sample_prediction_data,
                reference=sample_target_data,
                time=batch_time,
            ),
        )
        start_time_2 = (2020, 1, 1, 18, 0, 0)
        end_time_2 = (2020, 1, 2, 6, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time_2,
            end_time=end_time_2,
            freq="6h",
            n_initial_conditions=n_initial_conditions,
            calendar=calendar,
        )
        writer.append_batch(
            batch=PairedData(
                prediction=sample_prediction_data,
                reference=sample_target_data,
                time=batch_time,
            ),
        )
        writer.flush()

        # Open the file and check the data
        dataset = Dataset(tmp_path / "autoregressive_predictions.nc", "r")
        assert dataset["time"].units == "microseconds"
        assert dataset["init_time"].units == "microseconds since 1970-01-01 00:00:00"
        assert dataset["init_time"].calendar == calendar
        assert "source.inference_version" in dataset.ncattrs()
        assert dataset.getncattr("source.inference_version") == "1.0"
        assert dataset.getncattr("title") == "ACE autoregressive predictions data file"
        horizontal_shape = (4, 5) if "lat" in coords else (6, 4, 5)
        for var_name in set(sample_prediction_data.keys()):
            var_data = dataset.variables[var_name][:]
            assert var_data.shape == (
                n_initial_conditions,
                n_timesteps,
                *horizontal_shape,
            )
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
        coord_names = {"time", "init_time", "valid_time", *set(coords)}
        assert set(dataset.variables) == set(sample_target_data) | coord_names
        assert "source.inference_version" in dataset.ncattrs()
        assert dataset.getncattr("source.inference_version") == "1.0"
        assert dataset.getncattr("title") == "ACE autoregressive target data file"

        # Open the file again with xarray and check the time coordinates,
        # since the values opened this way depend on calendar/units
        with xr.open_dataset(
            tmp_path / "autoregressive_predictions.nc",
            decode_times=CFDatetimeCoder(use_cftime=True),
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
                [
                    CALENDAR_CFTIME[calendar](*start_time)
                    for _ in range(n_initial_conditions)
                ],
                dims=["sample"],
            )
            expected_init_times = expected_init_times.assign_coords(
                {"init_time": expected_init_times}
            )
            xr.testing.assert_equal(ds["init_time"], expected_init_times)
            for var_name in set(sample_prediction_data.keys()):
                assert "valid_time" in ds[var_name].coords
                assert "init_time" in ds[var_name].coords
        for source in ["target", "prediction"]:
            histograms = xr.open_dataset(
                tmp_path / f"histograms_{source}.nc", decode_timedelta=False
            )
            actual_var_names = sorted([str(k) for k in histograms.keys()])
            assert histograms.data_vars["temp"].attrs["units"] == "count"
            assert "temp_bin_edges" in actual_var_names
            assert histograms.data_vars["temp_bin_edges"].attrs["units"] == "K"
            counts_per_timestep = histograms["temp"].sum(dim=["bin"])
            same_count_each_timestep = np.all(
                counts_per_timestep.values == counts_per_timestep.values[0]
            )
            assert same_count_each_timestep

        with xr.open_dataset(
            tmp_path / "monthly_mean_predictions.nc", decode_timedelta=False
        ) as ds:
            assert ds.counts.sum() == n_initial_conditions * n_timesteps
            assert np.sum(np.isnan(ds["precipitation"])) == 0
            assert np.sum(np.isnan(ds["temp"])) == 0
            assert np.sum(np.isnan(ds["pressure"])) == 0
            assert np.all(ds.init_time.dt.year.values > 0)
            assert np.all(ds.init_time.dt.year.values >= 0)
            assert np.all(ds.valid_time.dt.month.values >= 0)
            assert ds.attrs["title"] == "ACE monthly predictions data file"
            assert ds.attrs["source.inference_version"] == "1.0"

    @pytest.mark.parametrize(
        "sample_target_data, sample_prediction_data",
        [pytest.param((2, 3, 4, 5), (2, 3, 4, 5), id="LatLon")],
        indirect=True,
    )
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
            n_initial_conditions=n_samples,
            n_timesteps=4,  # unused
            timestep=TIMESTEP,
            variable_metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            save_names=save_names,
            enable_histogram_netcdfs=True,
            dataset_metadata=DatasetMetadata(),
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time,
            end_time=end_time,
            freq="6h",
            n_initial_conditions=n_samples,
        )
        writer.append_batch(
            batch=PairedData(
                prediction=sample_prediction_data,
                reference=sample_target_data,
                time=batch_time,
            ),
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

    @pytest.mark.parametrize(
        "sample_target_data, sample_prediction_data",
        [pytest.param((2, 3, 4, 5), (2, 3, 4, 5), id="LatLon")],
        indirect=True,
    )
    def test_append_batch_data_time_mismatch(
        self, sample_metadata, sample_target_data, sample_prediction_data, tmp_path
    ):
        n_samples = 2
        writer = PairedDataWriter(
            str(tmp_path),
            n_initial_conditions=n_samples,
            n_timesteps=3,
            timestep=TIMESTEP,
            variable_metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            enable_prediction_netcdfs=True,
            enable_video_netcdfs=False,
            enable_monthly_netcdfs=True,
            save_names=None,
            enable_histogram_netcdfs=True,
            dataset_metadata=DatasetMetadata(),
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 12, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time,
            end_time=end_time,
            freq="6h",
            n_initial_conditions=n_samples + 1,
        )
        with pytest.raises(ValueError):
            writer.append_batch(
                batch=PairedData(
                    prediction=sample_prediction_data,
                    reference=sample_target_data,
                    time=batch_time,
                ),
            )

    def test_prediction_only_append_batch(self, sample_metadata, tmp_path, calendar):
        n_samples = 2
        n_timesteps = 8
        coarsen_factor = 2
        device = get_device()
        prediction_data = {
            "temp": torch.rand(
                (n_samples, n_timesteps // coarsen_factor, 4, 5), device=device
            ),
            "pressure": torch.rand(
                (n_samples, n_timesteps // coarsen_factor, 4, 5), device=device
            ),
        }
        reference_data = {
            "insolation": torch.rand(
                (n_samples, n_timesteps // coarsen_factor, 4, 5), device=device
            ),
        }
        writer = DataWriter(
            str(tmp_path),
            n_initial_conditions=n_samples,
            n_timesteps=n_timesteps,
            variable_metadata=sample_metadata,
            coords={"lat": np.arange(4), "lon": np.arange(5)},
            timestep=TIMESTEP,
            enable_prediction_netcdfs=True,
            enable_monthly_netcdfs=True,
            save_names=None,
            time_coarsen=TimeCoarsenConfig(coarsen_factor),
            dataset_metadata=DatasetMetadata(source={"inference_version": "1.0"}),
        )
        start_time = (2020, 1, 1, 0, 0, 0)
        end_time = (2020, 1, 1, 18, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time,
            end_time=end_time,
            freq="6h",
            n_initial_conditions=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            batch=PairedData(
                prediction=prediction_data,
                reference=reference_data,
                time=batch_time,
            ),
        )
        start_time_2 = (2020, 1, 2, 0, 0, 0)
        end_time_2 = (2020, 1, 2, 18, 0, 0)
        batch_time = self.get_batch_time(
            start_time=start_time_2,
            end_time=end_time_2,
            freq="6h",
            n_initial_conditions=n_samples,
            calendar=calendar,
        )
        writer.append_batch(
            batch=PairedData(
                prediction=prediction_data,
                reference=reference_data,
                time=batch_time,
            ),
        )
        writer.flush()

        with xr.open_dataset(
            tmp_path / "autoregressive_predictions.nc",
            decode_timedelta=False,
        ) as ds:
            assert "temp" in ds
            expected_shape = (n_samples, n_timesteps // coarsen_factor, 4, 5)
            assert ds.temp.shape == expected_shape
            assert ds.valid_time.shape == expected_shape[:2]
            assert set(ds.data_vars) == {"temp", "pressure", "insolation"}
            assert ds.attrs["title"] == "ACE autoregressive predictions data file"
            assert ds.attrs["source.inference_version"] == "1.0"

        with xr.open_dataset(
            tmp_path / "monthly_mean_predictions.nc", decode_timedelta=False
        ) as ds:
            assert ds.counts.sum() == n_samples * n_timesteps
            assert np.sum(np.isnan(ds["temp"])) == 0
            assert np.sum(np.isnan(ds["pressure"])) == 0
            assert np.all(ds.init_time.dt.year.values > 0)
            assert np.all(ds.init_time.dt.year.values >= 0)
            assert np.all(ds.valid_time.dt.month.values >= 0)
            assert set(ds.data_vars) == {"temp", "pressure", "insolation"}
            assert set(ds.coords) == {
                "init_time",
                "valid_time",
                "counts",
                "lat",
                "lon",
                "time",
            }
            assert ds.attrs["title"] == "ACE monthly predictions data file"
            assert ds.attrs["source.inference_version"] == "1.0"


@pytest.mark.parametrize(
    ["init_times", "batch_time", "expected"],
    [
        pytest.param(
            np.array([cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0) for _ in range(3)]),
            np.array(
                [
                    xr.date_range(
                        cftime.DatetimeJulian(2020, 1, 1, 0, 0, 0),
                        freq="6h",
                        periods=3,
                        use_cftime=True,
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
                    xr.date_range(
                        cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0),
                        freq="6h",
                        periods=3,
                        use_cftime=True,
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
                    xr.date_range(
                        cftime.DatetimeJulian(2020, 1, 2, 6 * i, 0, 0),
                        freq="6h",
                        periods=3,
                        use_cftime=True,
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
def test_get_batch_lead_times_microseconds(init_times, batch_time, expected):
    lead_time_seconds = get_batch_lead_time_microseconds(init_times, batch_time)
    assert lead_time_seconds.shape == expected.shape
    np.testing.assert_equal(lead_time_seconds, expected)


def test_get_batch_lead_time_microseconds_length_mismatch():
    init_times = np.array(
        [cftime.DatetimeJulian(2020, 1, 1, 6 * i, 0, 0) for i in range(3)]
    )
    batch_time = np.array(
        [
            xr.date_range(
                cftime.DatetimeJulian(2020, 1, 2, 6 * i, 0, 0),
                freq="6h",
                periods=3,
                use_cftime=True,
            ).values
            for i in range(2)
        ],
    )
    with pytest.raises(ValueError):
        get_batch_lead_time_microseconds(init_times, batch_time)


def test_get_batch_lead_time_microseconds_inconsistent_samples():
    init_times = np.array(
        [cftime.DatetimeJulian(2020, 1, 1, 6, 0, 0) for _ in range(2)]
    )
    batch_time = np.array(
        [
            xr.date_range(
                cftime.DatetimeJulian(2020, 1, 1, 6, 0, 0),
                freq="6h",
                periods=3,
                use_cftime=True,
            ),
            xr.date_range(
                cftime.DatetimeJulian(2020, 1, 1, 12, 0, 0),
                freq="6h",
                periods=3,
                use_cftime=True,
            ),
        ]
    )
    with pytest.raises(ValueError):
        get_batch_lead_time_microseconds(init_times, batch_time)


@pytest.mark.parametrize(
    ["years_ahead", "overflow"],
    [
        pytest.param(1e2, False, id="100_years_OK"),
        pytest.param(1e4, False, id="10_000_years_OK"),
        pytest.param(1e6, True, id="1_000_000_years_fails"),
    ],
)
def test_get_batch_lead_time_microseconds_overflow(years_ahead, overflow):
    init_times = np.array([cftime.DatetimeNoLeap(2020, 1, 1)])
    batch_time = np.array([cftime.DatetimeNoLeap(2020 + years_ahead, 1, 1)])[:, None]
    days_per_year_noleap = 365
    seconds_per_day = 86400
    expected_lead_time_microseconds = (
        MICROSECONDS_PER_SECOND * seconds_per_day * days_per_year_noleap * years_ahead
    )
    if not overflow:
        lead_time = get_batch_lead_time_microseconds(init_times, batch_time)
        assert lead_time.item() == expected_lead_time_microseconds
    else:
        with pytest.raises(OverflowError):
            get_batch_lead_time_microseconds(init_times, batch_time)
