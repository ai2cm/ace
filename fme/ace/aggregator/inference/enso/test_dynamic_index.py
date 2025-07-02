from datetime import timedelta

import cftime
import pytest
import torch
import xarray as xr
from matplotlib import pyplot as plt

from fme import get_device
from fme.core.coordinates import LatLonCoordinates
from fme.core.testing import mock_distributed
from fme.core.typing_ import TensorMapping

from .dynamic_index import (
    LatLonRegion,
    PairedRegionalIndexAggregator,
    RegionalIndexAggregator,
    anomalies_from_monthly_climo,
    running_monthly_mean,
)


def _get_windowed_data(
    n_samples: int,
    n_times: int,
    n_lat: int,
    n_lon: int,
    i_start: int = 0,
    sst_name: str = "surface_temperature",
    time_varying: bool = True,
) -> TensorMapping:
    lat_coord = torch.linspace(0.0, 10.0, n_lat)
    lon_coord = torch.linspace(0.0, 20.0, n_lon)
    sst_data = (
        (lat_coord.unsqueeze(-1) + lon_coord.unsqueeze(0))
        .unsqueeze(0)
        .unsqueeze(0)
        .expand(n_samples, n_times, n_lat, n_lon)
    )
    if time_varying:
        sst_data = (
            sst_data
            + torch.arange(i_start, i_start + n_times, 1.0)[None, :, None, None]
        )
    return {
        sst_name: sst_data.to(device=get_device()),
        "lat": lat_coord,
        "lon": lon_coord,
    }


def _get_windowed_times(
    start_time: tuple[int, ...],
    n_samples: int,
    n_times: int,
    i_start: int = 0,
    freq="6h",
    calendar: str = "noleap",
) -> xr.DataArray:
    start_time = cftime.datetime(*start_time, calendar=calendar) + timedelta(
        hours=6 * i_start
    )
    sample_time_array = xr.DataArray(
        data=xr.date_range(
            start=start_time,
            periods=n_times,
            freq=freq,
            use_cftime=True,
        ).values,
        dims=("time",),
    )
    return xr.concat(
        [sample_time_array for i in range(n_samples)],
        dim="sample",
    )


@pytest.mark.parametrize(
    "lat_bounds, lon_bounds, case",
    [
        pytest.param((0.0, 10.0), (0.0, 20.0), "original_domain"),
        pytest.param((20.0, 30.0), (25.0, 35.0), "null_domain"),
        pytest.param((0.0, 5.0), (0.0, 20.0), "first_half_lat"),
        pytest.param((0.0, 10.0), (0.0, 10.0), "first_half_lon"),
        pytest.param((0.0, 5.0), (0.0, 10.0), "first_half_both"),
    ],
)
def test_lat_lon_region(lat_bounds, lon_bounds, case):
    n_lat, n_lon = 3, 5
    lat_coord = torch.linspace(0.0, 10.0, n_lat)
    lon_coord = torch.linspace(0.0, 20.0, n_lon)
    region = LatLonRegion(
        lat=lat_coord,
        lon=lon_coord,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
    )
    regional_weights = region.regional_weights
    assert regional_weights.shape == (n_lat, n_lon)
    if case == "original_domain":
        assert torch.allclose(regional_weights, torch.ones_like(regional_weights))
    elif case == "null_domain":
        assert torch.allclose(regional_weights, torch.zeros_like(regional_weights))
    elif case == "first_half_lat":
        expected = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(regional_weights, expected)
    elif case == "first_half_lon":
        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(regional_weights, expected)
    else:
        expected = torch.tensor(
            [
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float32,
        )
        assert torch.allclose(regional_weights, expected)


def test_regional__raw_index():
    n_samples, n_times, n_lat, n_lon = 2, 10, 11, 21
    start_date = (2000, 1, 1, 0, 0, 0)
    i_start = 0
    first_data = _get_windowed_data(
        n_samples,
        n_times // 2,
        n_lat,
        n_lon,
        i_start,
    )
    lat_lon_coordinates = LatLonCoordinates(
        lat=first_data["lat"],
        lon=first_data["lon"],
    )
    region = LatLonRegion(
        lat=lat_lon_coordinates.lat,
        lon=lat_lon_coordinates.lon,
        lat_bounds=(4.5, 6.5),
        lon_bounds=(9.5, 11.5),
    )
    # overwrite the area weights with ones for testing
    gridded_operations = lat_lon_coordinates.get_gridded_operations()
    gridded_operations._cpu_area = torch.ones(n_lat, n_lon)
    gridded_operations._device_area = torch.ones(n_lat, n_lon).to(device=get_device())
    agg = RegionalIndexAggregator(
        regional_weights=region.regional_weights,
        regional_mean=gridded_operations.regional_area_weighted_mean,
    )
    first_times = _get_windowed_times(start_date, n_samples, n_times // 2, i_start)
    agg.record_batch(first_times, first_data)
    i_start += n_times // 2
    second_data = _get_windowed_data(
        n_samples,
        n_times // 2,
        n_lat,
        n_lon,
        i_start,
    )
    second_times = _get_windowed_times(start_date, n_samples, n_times // 2, i_start)
    agg.record_batch(second_times, second_data)

    expected_values = torch.arange(16.0, 16.0 + n_times, 1.0).to(device=get_device())
    raw_indices: torch.Tensor = agg._raw_indices
    for raw_index in raw_indices.values():
        assert raw_index.shape == (n_samples, n_times)
        assert torch.allclose(raw_index, expected_values)
    raw_times: xr.DataArray = agg._raw_index_times
    assert raw_times.sizes["sample"] == n_samples
    assert raw_times.sizes["time"] == n_times
    expected_times = xr.concat(
        [
            xr.DataArray(
                data=xr.date_range(
                    start=cftime.DatetimeNoLeap(*start_date),
                    periods=n_times,
                    freq="6h",
                    use_cftime=True,
                ).values,
                dims=("time"),
            )
            for _ in range(n_samples)
        ],
        dim="sample",
    )
    xr.testing.assert_allclose(raw_times, expected_times)


def test_anomalies_from_monthly_climo():
    n_months = 24
    n_samples = 2
    # data starts at 0, increases by 1.0 every month
    data = _get_windowed_data(
        n_samples=n_samples,
        n_times=n_months,
        n_lat=1,
        n_lon=1,
    )["surface_temperature"].squeeze()
    times = _get_windowed_times(
        start_time=(2000, 1, 1, 0, 0, 0), n_samples=2, n_times=n_months, freq="MS"
    )
    anomalies = anomalies_from_monthly_climo(data, times)
    assert anomalies.shape == data.shape
    assert torch.allclose(
        anomalies.mean(dim=1), torch.zeros(n_samples).to(device=get_device())
    )
    expected = torch.concat(
        [
            torch.full((n_samples, 12), fill_value=-6.0),
            torch.full((n_samples, 12), fill_value=6.0),
        ],
        dim=1,
    ).to(device=get_device())
    assert torch.allclose(anomalies, expected)


def test_running_monthly_mean_monthly_averaging():
    n_times = 1440  # 12 months of 6 hourly, 30-day months
    n_samples = 2
    # data starts at 0, increases by 1.0 every 6 hours
    data = _get_windowed_data(
        n_samples=n_samples,
        n_times=n_times,
        n_lat=1,
        n_lon=1,
    )["surface_temperature"].squeeze()
    times = _get_windowed_times(
        start_time=(2000, 1, 1, 0, 0, 0),
        n_samples=2,
        n_times=n_times,
        freq="6h",
        calendar="360_day",
    )
    running_mean, _ = running_monthly_mean(data, times, n_months=1)
    assert running_mean.shape == (n_samples, 12)
    expected = (  # monthly means for 30-day months
        torch.arange(0.0, n_times, 1.0)
        .reshape(12, 120)
        .mean(dim=1)
        .unsqueeze(0)
        .expand(n_samples, -1)
    ).to(device=get_device())
    assert torch.allclose(running_mean, expected)


@pytest.mark.parametrize(
    ["n_months"], [pytest.param(1, id="1-month"), pytest.param(5, id="5-month")]
)
def test_running_monthly_mean_window_averaging(n_months: int):
    n_times = 24
    n_samples = 2
    # data starts at 0, increases by 1.0 every month
    data = _get_windowed_data(
        n_samples=n_samples,
        n_times=n_times,
        n_lat=1,
        n_lon=1,
    )["surface_temperature"].squeeze()
    times = _get_windowed_times(
        start_time=(2000, 1, 1, 0, 0, 0), n_samples=2, n_times=n_times, freq="MS"
    )
    running_mean, _ = running_monthly_mean(data, times, n_months=n_months)
    assert running_mean.shape == data.shape
    expected = torch.cat(
        [  # first n_months - 1 months are nan
            torch.full((n_samples, n_months - 1), fill_value=float("nan")),
            # remaining months are the running mean of n_months months
            torch.arange(n_months / 2.0 - 0.5, n_times - n_months / 2.0 + 0.5, 1.0)
            .unsqueeze(0)
            .expand(n_samples, -1),
        ],
        dim=1,
    ).to(device=get_device())
    assert torch.allclose(running_mean, expected, equal_nan=True)


@pytest.mark.parametrize(
    "use_mock_distributed",
    [pytest.param(False, id="single_process"), pytest.param(True, id="distributed")],
)
def test_regional_index__get_gathered_indices(use_mock_distributed):
    n_samples, n_times, n_lat, n_lon = 1, 1440, 11, 21
    start_date = (2000, 1, 1, 0, 0, 0)
    first_sample_data = _get_windowed_data(n_samples, n_times, n_lat, n_lon)
    second_sample_data = _get_windowed_data(
        n_samples, n_times, n_lat, n_lon, i_start=n_times
    )
    sample_data = {
        "surface_temperature": torch.concat(
            [
                first_sample_data["surface_temperature"],
                second_sample_data["surface_temperature"],
            ],
            dim=0,
        ),
    }
    first_sample_times = _get_windowed_times(
        start_date, n_samples, n_times, i_start=0
    )  # first sample's inference period spans year 2000
    second_sample_times = _get_windowed_times(
        start_date, n_samples, n_times, i_start=n_times
    )  # second sample's inference period spans year 2001
    sample_times = xr.concat([first_sample_times, second_sample_times], dim="sample")
    lat_lon_coordinates = LatLonCoordinates(
        lat=first_sample_data["lat"],
        lon=first_sample_data["lon"],
    )
    region = LatLonRegion(
        lat=lat_lon_coordinates.lat,
        lon=lat_lon_coordinates.lon,
        lat_bounds=(4.5, 6.5),
        lon_bounds=(9.5, 11.5),
    )
    agg = RegionalIndexAggregator(
        regional_weights=region.regional_weights,
        regional_mean=lat_lon_coordinates.get_gridded_operations().regional_area_weighted_mean,
    )
    agg.record_batch(sample_times, sample_data)
    if use_mock_distributed:
        world_size = 2
        with mock_distributed(world_size=world_size):
            indices_dataset = agg.get_dataset()
            assert indices_dataset is not None
    else:
        world_size = 1
        indices_dataset = agg.get_dataset()
        assert indices_dataset is not None
    assert set(indices_dataset.dims) == {"sample", "time"}
    assert set(indices_dataset.time.dt.year.values) == {2000, 2001}
    assert indices_dataset.sizes["sample"] == 2 * n_samples * world_size
    assert indices_dataset.sizes["time"] == 2 * 12  # 2 years of 12 months each


@pytest.mark.parametrize(
    "variable_name",
    [
        pytest.param("surface_temperature", id="surface_temperature"),
        pytest.param("sst", id="sst"),
        pytest.param("TS", id="TS"),
    ],
)
def test_regional_index_aggregator(variable_name):
    n_lat = 10
    n_lon = 20
    n_sample = 2
    n_times = 365 * 4  # one year of data for monthly averaging
    data = _get_windowed_data(n_sample, n_times, n_lat, n_lon, sst_name=variable_name)
    time = _get_windowed_times((2000, 1, 1, 0, 0, 0), n_sample, n_times)
    lat_lon_coordinates = LatLonCoordinates(
        lat=data["lat"],
        lon=data["lon"],
    )
    region = LatLonRegion(
        lat=lat_lon_coordinates.lat,
        lon=lat_lon_coordinates.lon,
        lat_bounds=(4.5, 6.5),
        lon_bounds=(9.5, 11.5),
    )
    agg = RegionalIndexAggregator(
        regional_weights=region.regional_weights,
        regional_mean=lat_lon_coordinates.get_gridded_operations().regional_area_weighted_mean,
    )
    agg.record_batch(time=time, data=data)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    metric_name = f"test/{variable_name}_nino34_index"
    assert metric_name in logs
    assert isinstance(logs[metric_name], plt.Figure)

    metric_name = f"test/{variable_name}_nino34_index_power_spectrum"
    assert metric_name in logs
    assert isinstance(logs[metric_name], plt.Figure)


@pytest.mark.parametrize(
    "variable_name",
    [
        pytest.param("surface_temperature", id="surface_temperature"),
        pytest.param("sst", id="sst"),
        pytest.param("TS", id="TS"),
    ],
)
def test_paired_regional_index_aggregator(variable_name):
    n_lat = 10
    n_lon = 20
    n_sample = 2
    n_times = 365 * 4  # one year of data for monthly averaging
    target_data = _get_windowed_data(
        n_sample, n_times, n_lat, n_lon, sst_name=variable_name
    )
    prediction_data = _get_windowed_data(
        n_sample, n_times, n_lat, n_lon, time_varying=False, sst_name=variable_name
    )
    time = _get_windowed_times((2000, 1, 1, 0, 0, 0), n_sample, n_times)
    lat_lon_coordinates = LatLonCoordinates(
        lat=target_data["lat"],
        lon=target_data["lon"],
    )
    region = LatLonRegion(
        lat=lat_lon_coordinates.lat,
        lon=lat_lon_coordinates.lon,
        lat_bounds=(4.5, 6.5),
        lon_bounds=(9.5, 11.5),
    )
    agg = PairedRegionalIndexAggregator(
        target_aggregator=RegionalIndexAggregator(
            regional_weights=region.regional_weights,
            regional_mean=lat_lon_coordinates.get_gridded_operations().regional_area_weighted_mean,
        ),
        prediction_aggregator=RegionalIndexAggregator(
            regional_weights=region.regional_weights,
            regional_mean=lat_lon_coordinates.get_gridded_operations().regional_area_weighted_mean,
        ),
    )
    agg.record_batch(time=time, target_data=target_data, gen_data=prediction_data)
    logs = agg.get_logs(label="test")
    assert len(logs) > 0
    metric_name = f"test/{variable_name}_nino34_index"
    assert metric_name in logs
    assert isinstance(logs[metric_name], plt.Figure)

    metric_name = f"test/{variable_name}_nino34_index_power_spectrum"
    assert metric_name in logs
    assert isinstance(logs[metric_name], plt.Figure)
