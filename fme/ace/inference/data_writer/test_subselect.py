from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.dataset.time import TimeSlice
from fme.core.typing_ import Slice

from .subselect import (
    MonthSelector,
    SubselectWriter,
    SubselectWriterConfig,
    _get_time_mask,
    _month_string_to_int,
    _select_time,
)


def test_subselect_writer_missing_selection():
    with pytest.raises(ValueError):
        SubselectWriterConfig(
            label="test_region",
            names=["temperature", "humidity"],
        )


def test_subselect_writer_config_build():
    config = SubselectWriterConfig(
        label="test_region",
        names=["temperature", "humidity"],
        lat_extent=(-10.0, 10.0),
        lon_extent=(-20.0, 20.0),
    )

    with patch("fme.ace.inference.data_writer.subselect.RawDataWriter") as mock_writer:
        mock_writer.return_value = MagicMock()
        writer = config.build(
            experiment_dir="test_experiment",
            n_initial_conditions=1,
            n_timesteps=10,
            variable_metadata={},
            coords={
                "latitude": np.array([-20, -10.0, 0.0, 10.0, 20.0]),
                "longitude": np.array([-30.0, -20.0, 0.0, 20.0, 30.0]),
            },
            dataset_metadata=MagicMock(),
        )
        assert isinstance(writer, SubselectWriter)


@pytest.mark.parametrize(
    "lat_extent, lon_extent",
    [
        ([0, 1], [0]),  # longitude extent missing a value
        ([0, 1], [0, 1, 2]),  # Invalid longitude extent
        ([0], [0, 1]),  # Invalid latitude extent
        ([0, 1, 2], [0, 1]),
    ],
)
def test_subselect_writer_config_invalid_lat_lon_extent(lat_extent, lon_extent):
    with pytest.raises(ValueError):
        SubselectWriterConfig(
            label="test_region",
            names=["temperature", "humidity"],
            lat_extent=lat_extent,
            lon_extent=lon_extent,
        )


@pytest.mark.parametrize(
    "lat_name, lon_name",
    [
        ("lat", "longitude"),  # Missing longitude
        ("latitude", "lon"),  # Missing latitude
    ],
)
def test_subselect_writer_config_build_missing_coords(lat_name, lon_name):
    coords = {
        "lat": np.array([-20, -10.0, 0.0, 10.0, 20.0]),
        "lon": np.array([-30.0, -20.0, 0.0, 20.0, 30.0]),
    }
    config = SubselectWriterConfig(
        label="test_region",
        names=["temperature", "humidity"],
        lat_extent=(-10.0, 10.0),
        lon_extent=(-20.0, 20.0),
        latitude_name=lat_name,
        longitude_name=lon_name,
    )

    with pytest.raises(ValueError):
        config.build(
            experiment_dir="test_experiment",
            n_initial_conditions=1,
            n_timesteps=10,
            variable_metadata={},
            coords=coords,
            dataset_metadata=MagicMock(),
        )


@pytest.mark.parametrize("extent", [None, []])
def test_subselect_writer_config_build_missing_extent(extent):
    config = SubselectWriterConfig(
        label="test_region",
        names=["temperature", "humidity"],
        lat_extent=extent,
        lon_extent=extent,
        time_selection=MonthSelector(months=["January"]),
    )
    assert config.lat_slice == slice(None)
    assert config.lon_slice == slice(None)


def test__month_str_to_int():
    assert _month_string_to_int("jan")
    assert _month_string_to_int("January") == 1

    with pytest.raises(ValueError, match="Invalid month string: "):
        _month_string_to_int("invalid")


def test__get_time_mask_empty():
    with pytest.raises(ValueError, match="Cannot build a mask*"):
        _get_time_mask(
            xr.DataArray(xr.date_range("2020-01-01", periods=7, freq="2MS")), []
        )


@pytest.mark.parametrize(
    "selection, expected_month_mask",
    [
        (["January"], [True, False, False, False, False, False, True]),
        (["Feb"], [False, False, False, False, False, False, False]),
        (["MAM"], [False, True, True, False, False, False, False]),
        (["January", "Jan", "DJF"], [True, False, False, False, False, False, True]),
        (
            ["january", "JANUARY", "Jan", "JAN", "jan"],
            [True, False, False, False, False, False, True],
        ),
    ],
)
def test__get_time_mask(selection, expected_month_mask):
    time = xr.DataArray(xr.date_range("2020-01-01", periods=7, freq="2MS"), dims="time")
    mask = _get_time_mask(time, selection)
    assert mask.values.tolist() == expected_month_mask


def test__get_time_mask_non_dt_dataarray():
    time = xr.DataArray(np.arange(7), dims="time")
    with pytest.raises(
        ValueError, match="Input does not contain datetime data with 'dt' accessor."
    ):
        _get_time_mask(time, ["January"])


def test_month_selector_empty_selection():
    selector = MonthSelector(months=[])
    data = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.random.rand(7, 3, 3)),
            "time": xr.date_range(
                "2020-01-01", periods=7, freq="2MS"
            ),  # every 2 months
        }
    )
    selected_data = selector.select(data)
    assert selected_data == data


def test_month_selector_select():
    selector = MonthSelector(months=["January", "MAM"])
    data = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.random.rand(7, 3, 3)),
            "time": xr.date_range(
                "2020-01-01", periods=7, freq="2MS"
            ),  # every 2 months
        }
    )
    selected_data = selector.select(data)
    assert "temperature" in selected_data
    assert len(selected_data.time) < len(data.time)
    # Jan, Mar, May, Jan from above dataset
    expected = np.concatenate(
        [data.temperature.values[:3], data.temperature.values[-1:]], axis=0
    )
    np.testing.assert_array_equal(selected_data["temperature"].values, expected)


def test__select_time():
    data = xr.Dataset(
        {
            "temperature": (("time", "lat", "lon"), np.random.rand(7, 3, 3)),
            "time": xr.date_range(
                "2020-01-01", periods=7, freq="2MS"
            ),  # every 2 months
        }
    )

    # Test with TimeSlice
    time_slice = TimeSlice(start_time="2020-01", stop_time="2020-04")
    selected_data = _select_time(data, time_slice)
    assert len(selected_data.time) == 2
    np.testing.assert_array_equal(
        data.temperature[0:2, :, :], selected_data.temperature
    )

    # Test with MonthSelector string
    month_selector = MonthSelector(months=["Jan"])
    selected_data = _select_time(data, month_selector)
    assert len(selected_data.time) == 2
    expected = np.concatenate(
        [data.temperature[0:1, :, :], data.temperature[-1:, :, :]], axis=0
    )
    np.testing.assert_array_equal(expected, selected_data.temperature)

    # Test with Slice index
    selected_data = _select_time(data, Slice(start=0, stop=5, step=2))
    assert len(selected_data.time) == 3
    np.testing.assert_array_equal(
        data.temperature[:5:2, :, :], selected_data.temperature
    )

    # Test with None
    selected_data = _select_time(data, None)
    assert len(selected_data.time) == 7
    np.testing.assert_array_equal(data.temperature, selected_data.temperature)

    with pytest.raises(ValueError):
        _select_time(data, 0.1)  # type: ignore


def get_subselect_writer(**kwarg_updates):
    if kwarg_updates is None:
        kwarg_updates = {}

    kwargs = {
        "label": "test_region",
        "names": ["temperature", "humidity"],
        "lat_extent": (-10.0, 10.0),
        "lon_extent": (-20.0, 20.0),
        "time_selection": None,
        **kwarg_updates,
    }

    config = SubselectWriterConfig(**kwargs)
    full_coords = {
        "latitude": np.array([-20, -10.0, 0.0, 10.0, 20.0]),
        "longitude": np.array([-30.0, -20.0, 0.0, 20.0, 30.0]),
    }
    raw_writer = MagicMock()
    return SubselectWriter(config, raw_writer, full_coords=full_coords)


def test_subselect_writer__subselect_data_limits_variables():
    subselect_writer = get_subselect_writer()

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=3, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
        "pressure": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))

    subselected_data = subselect_writer._subselect_data(data, batch_time)

    assert "pressure" not in subselected_data
    assert set(subselected_data.keys()) == {"temperature", "humidity"}


def test_subselect_writer__subselect_data_empty_time():
    subselect_writer = get_subselect_writer(
        time_selection=TimeSlice(start_time="2020-01", stop_time="2020-02"),
    )

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=3, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
    }
    # times out of the selection range
    batch_time = xr.DataArray(
        xr.date_range("2021-01-01", periods=10, freq="D"), dims="time"
    )

    subselected_data = subselect_writer._subselect_data(data, batch_time)

    assert (
        not subselected_data
    )  # Should return empty dict if no time steps are selected


def test_subselect_writer_no_names_specified_saves_all():
    subselect_writer = get_subselect_writer(names=None)

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=3, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
        "pressure": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))

    subselected_data = subselect_writer._subselect_data(data, batch_time)

    assert set(subselected_data.keys()) == {"temperature", "humidity", "pressure"}


def test_subselect_writer_append_batch():
    subselect_writer = get_subselect_writer()

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=5, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))
    subselect_writer.append_batch(data, start_timestep=0, batch_time=batch_time)

    # Check if the data was subselected correctly
    expected_temperature = data["temperature"][
        :, :, 1:4, 1:4
    ]  # lat -10 to 10, lon -20 to 20
    writer: MagicMock = cast(MagicMock, subselect_writer.writer)
    writer.append_batch.assert_called_once()
    args, kwargs = writer.append_batch.call_args
    torch.testing.assert_close(kwargs["data"]["temperature"], expected_temperature)
