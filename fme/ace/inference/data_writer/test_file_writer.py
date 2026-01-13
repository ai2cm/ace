import datetime
from typing import cast
from unittest.mock import MagicMock, patch

import cftime
import numpy as np
import pytest
import torch
import xarray as xr

from fme.ace.data_loading.batch_data import BatchData
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.core.dataset.time import TimeSlice
from fme.core.typing_ import Slice

from .file_writer import (
    FileWriter,
    FileWriterConfig,
    MonthSelector,
    _get_time_mask,
    _month_string_to_int,
    _select_time,
)
from .raw import NetCDFWriterConfig
from .time_coarsen import MonthlyCoarsenConfig, TimeCoarsen
from .zarr import ZarrWriterConfig


def test_file_writer_config_build():
    config = FileWriterConfig(
        label="test_region",
        names=["temperature", "humidity"],
        lat_extent=(-10.0, 10.0),
        lon_extent=(-20.0, 20.0),
    )

    with patch(
        "fme.ace.inference.data_writer.file_writer.RawDataWriter"
    ) as mock_writer:
        mock_writer.return_value = MagicMock()
        writer = config.build(
            experiment_dir="test_experiment",
            n_initial_conditions=1,
            n_timesteps=10,
            variable_metadata={},
            timestep=datetime.timedelta(days=1),
            coords={
                "lat": np.array([-20, -10.0, 0.0, 10.0, 20.0]),
                "lon": np.array([-30.0, -20.0, 0.0, 20.0, 30.0]),
            },
            dataset_metadata=MagicMock(),
        )
        assert isinstance(writer, FileWriter)


@pytest.mark.parametrize(
    "lat_extent, lon_extent",
    [
        ([0, 1], [0]),  # longitude extent missing a value
        ([0, 1], [0, 1, 2]),  # Invalid longitude extent
        ([0], [0, 1]),  # Invalid latitude extent
        ([0, 1, 2], [0, 1]),
    ],
)
def test_file_writer_config_invalid_lat_lon_extent(lat_extent, lon_extent):
    with pytest.raises(ValueError):
        FileWriterConfig(
            label="test_region",
            names=["temperature", "humidity"],
            lat_extent=lat_extent,
            lon_extent=lon_extent,
        )


def test_file_writer_config_build_missing_coords():
    coords = {"face": np.array([-20, -10.0, 0.0, 10.0, 20.0])}
    config = FileWriterConfig(
        label="test_region",
        names=["temperature", "humidity"],
        lat_extent=(-10.0, 10.0),
        lon_extent=(-20.0, 20.0),
    )

    with pytest.raises(ValueError):
        config.build(
            experiment_dir="test_experiment",
            n_initial_conditions=1,
            n_timesteps=10,
            timestep=datetime.timedelta(days=1),
            variable_metadata={},
            coords=coords,
            dataset_metadata=MagicMock(),
        )


@pytest.mark.parametrize("extent", [None, []])
def test_file_writer_config_build_missing_extent(extent):
    config = FileWriterConfig(
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


@pytest.mark.parametrize(
    "time_selection, expected_time_indices",
    [
        pytest.param(
            TimeSlice(start_time="2020-01-01", stop_time="2020-01-02"),
            slice(0, 2),
            id="TimeSlice",
        ),
        pytest.param(MonthSelector(months=["Jan"]), slice(0, 31), id="MonthSelector"),
        pytest.param(Slice(start=0, stop=3, step=2), slice(0, 3, 2), id="Slice"),
        pytest.param(Slice(start=5, stop=15), slice(5, 15), id="Slice_non_initial"),
        pytest.param(Slice(start=50, stop=55), slice(50, 55), id="Slice_second_batch"),
        pytest.param(None, slice(None), id="None"),
    ],
)
def test__select_time_file_writer_single_sample(
    time_selection, expected_time_indices, tmpdir
):
    n_samples = 1
    n_timesteps = 40
    freq = "1D"
    batch_data = BatchData.new_for_testing(
        names=["temperature"],
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        t_initial=cftime.datetime(2020, 1, 1),
        freq=freq,
    )
    second_batch_data = BatchData.new_for_testing(
        names=["temperature"],
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        t_initial=cftime.datetime(2020, 1, 1) + datetime.timedelta(days=n_timesteps),
        freq=freq,
    )
    all_time = xr.concat([batch_data.time, second_batch_data.time], dim="time")
    all_temperature = torch.cat(
        [batch_data.data["temperature"], second_batch_data.data["temperature"]], dim=1
    )
    file_writer_config = FileWriterConfig(
        label="test_writer",
        names=["temperature"],
        format=NetCDFWriterConfig(name="netcdf"),
        time_selection=time_selection,
    )
    file_writer = file_writer_config.build(
        experiment_dir=str(tmpdir),
        n_initial_conditions=n_samples,
        n_timesteps=-1,  # unused for netcdf
        timestep=datetime.timedelta(days=1),
        coords={},
        variable_metadata={},
        dataset_metadata=DatasetMetadata(),
    )
    file_writer.append_batch(
        data=dict(batch_data.data),
        batch_time=batch_data.time,
    )
    file_writer.append_batch(
        data=dict(second_batch_data.data),
        batch_time=second_batch_data.time,
    )
    file_writer.finalize()

    with xr.open_dataset(
        tmpdir / "test_writer.nc", decode_times=False
    ) as subselected_data:
        assert len(subselected_data.sample) == n_samples
        expected_time_length = all_time.isel(time=expected_time_indices).sizes["time"]
        assert len(subselected_data.time) == expected_time_length
        np.testing.assert_almost_equal(
            all_temperature[0, expected_time_indices].cpu().numpy(),
            subselected_data.squeeze().temperature,
        )


@pytest.mark.parametrize(
    ["time_selection"],
    [
        pytest.param(
            TimeSlice(start_time="2020-01-01", stop_time="2020-01-02"), id="TimeSlice"
        ),
        pytest.param(MonthSelector(months=["Jan"]), id="MonthSelector"),
        pytest.param(Slice(start=0, stop=3, step=2), id="Slice"),
        pytest.param(None, id="None"),
    ],
)
def test__select_time_file_writer_multiple_samples(time_selection, tmpdir):
    n_samples = 2
    n_timesteps = 40
    batch_data = BatchData.new_for_testing(
        names=["temperature"],
        n_samples=n_samples,
        n_timesteps=n_timesteps,
        t_initial="2020-01-01T00:00:00",
        freq="1D",
        increment_times=True,
    )
    file_writer_config = FileWriterConfig(
        label="test_writer",
        names=["temperature"],
        format=NetCDFWriterConfig(name="netcdf"),
        time_selection=time_selection,
    )

    if isinstance(time_selection, TimeSlice) or isinstance(
        time_selection, MonthSelector
    ):
        with pytest.raises(NotImplementedError):
            file_writer = file_writer_config.build(
                experiment_dir=str(tmpdir),
                n_initial_conditions=n_samples,
                n_timesteps=-1,  # unused for netcdf
                timestep=datetime.timedelta(days=1),
                coords={},
                variable_metadata={},
                dataset_metadata=DatasetMetadata(),
            )
        return

    file_writer = file_writer_config.build(
        experiment_dir=str(tmpdir),
        n_initial_conditions=n_samples,
        n_timesteps=-1,  # unused for netcdf
        timestep=datetime.timedelta(days=1),
        coords={},
        variable_metadata={},
        dataset_metadata=DatasetMetadata(),
    )
    file_writer.append_batch(
        data=dict(batch_data.data),
        batch_time=batch_data.time,
    )
    file_writer.finalize()

    with xr.open_dataset(
        tmpdir / "test_writer.nc", decode_timedelta=False
    ) as subselected_data:
        assert len(subselected_data.sample) == n_samples
        if isinstance(time_selection, Slice):
            assert len(subselected_data.time) == 2
            np.testing.assert_almost_equal(
                batch_data.data["temperature"][:, 0:3:2, :].cpu().numpy(),
                subselected_data.temperature,
            )
        else:  # time_selection is None
            assert len(subselected_data.time) == n_timesteps
            np.testing.assert_almost_equal(
                batch_data.data["temperature"].cpu().numpy(),
                subselected_data.temperature,
            )


def test_subset_time_invalid_time_selection():
    with pytest.raises(ValueError):
        _select_time(xr.Dataset(), 0.1)  # type: ignore


def get_file_writer(**kwarg_updates):
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

    config = FileWriterConfig(**kwargs)
    full_coords = {
        "lat": np.array([-20, -10.0, 0.0, 10.0, 20.0]),
        "lon": np.array([-30.0, -20.0, 0.0, 20.0, 30.0]),
    }
    raw_writer = MagicMock()
    return FileWriter(config, raw_writer, full_coords=full_coords)


def test_file_writer__subselect_data_limits_variables():
    writer = get_file_writer()

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=3, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
        "pressure": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))

    subselected_data, _ = writer._subselect_data(data, batch_time)

    assert "pressure" not in subselected_data
    assert set(subselected_data.keys()) == {"temperature", "humidity"}


def test_file_writer__subselect_data_empty_time():
    writer = get_file_writer(
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

    subselected_data, subselected_time = writer._subselect_data(data, batch_time)

    assert (
        not subselected_data
    )  # Should return empty dict if no time steps are selected
    assert subselected_time.sizes["time"] == 0


def test_file_writer_no_names_specified_saves_all():
    file_writer = get_file_writer(names=None)

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=3, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
        "pressure": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))

    subselected_data, _ = file_writer._subselect_data(data, batch_time)

    assert set(subselected_data.keys()) == {"temperature", "humidity", "pressure"}


def test_file_writer_append_batch():
    file_writer = get_file_writer()

    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=5, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(xr.date_range("2020-01-01", periods=10, freq="D"))
    file_writer.append_batch(data, batch_time=batch_time)

    # Check if the data was subselected correctly
    expected_temperature = data["temperature"][
        :, :, 1:4, 1:4
    ]  # lat -10 to 10, lon -20 to 20
    writer: MagicMock = cast(MagicMock, file_writer.writer)
    writer.append_batch.assert_called_once()
    args, kwargs = writer.append_batch.call_args
    torch.testing.assert_close(kwargs["data"]["temperature"], expected_temperature)


def test_file_writer_with_healpix_data_and_zarr(tmpdir):
    config = FileWriterConfig("filename", format=ZarrWriterConfig())
    n_samples = 2
    n_timesteps = 6
    shape = (12, 4, 4)
    coords = {"face": np.arange(12), "height": np.arange(4), "width": np.arange(4)}
    writer = config.build(
        experiment_dir=str(tmpdir),
        n_initial_conditions=n_samples,
        n_timesteps=n_timesteps,
        timestep=datetime.timedelta(days=1),
        variable_metadata={},
        coords=coords,
        dataset_metadata=MagicMock(),
    )
    data = {"temperature": torch.rand(n_samples, n_timesteps, *shape)}
    data_first_half = {k: v[:, :3] for k, v in data.items()}
    data_second_half = {k: v[:, 3:] for k, v in data.items()}
    batch_time_single_sample = xr.DataArray(
        xr.date_range("2020-01-01", periods=n_timesteps, freq="D", use_cftime=True),
        dims="time",
    )
    batch_time = xr.concat([batch_time_single_sample] * n_samples, dim="sample")
    batch_time_first_half = batch_time.isel(time=slice(0, 3))
    batch_time_second_half = batch_time.isel(time=slice(3, None))
    writer.append_batch(data_first_half, batch_time=batch_time_first_half)
    writer.append_batch(data_second_half, batch_time=batch_time_second_half)
    writer.finalize()
    zarr_data = xr.open_zarr(tmpdir / "filename.zarr", decode_timedelta=False)
    assert dict(zarr_data.sizes) == {
        "sample": n_samples,
        "time": n_timesteps,
        "face": 12,
        "height": 4,
        "width": 4,
    }
    assert zarr_data.temperature.dims == ("sample", "time", "face", "height", "width")
    np.testing.assert_allclose(zarr_data.temperature, data["temperature"].numpy())
    # drop coordinates which are not expected to agree
    xr.testing.assert_allclose(
        batch_time.astype("datetime64[ns]").drop_vars("time"),
        zarr_data.valid_time.drop_vars(("time", "valid_time", "sample", "init_time")),
    )


def test_file_writer_append_batch_time_coarsened():
    file_writer = get_file_writer()
    coarsen_factor = 2
    time_coarsen_writer = TimeCoarsen(
        data_writer=file_writer, coarsen_factor=coarsen_factor
    )
    data = {
        "temperature": torch.rand(3, 10, 5, 5),  # batch_size=5, time=10, lat=5, lon=5
        "humidity": torch.rand(3, 10, 5, 5),
    }
    batch_time = xr.DataArray(
        xr.date_range("2020-01-01", periods=10, freq="D"), dims=["time"]
    )

    time_coarsen_writer.append_batch(data, batch_time=batch_time)

    subselected_temperature = data["temperature"][:, :, 1:4, 1:4]

    coarsened_time_blocks = []
    for i in range(5):
        coarsened_time_blocks.append(
            subselected_temperature[
                :, coarsen_factor * i : coarsen_factor * (i + 1), :, :
            ]
            .mean(axis=1)
            .unsqueeze(1)
        )
    time_coarsened_temperature = torch.concat(coarsened_time_blocks, axis=1)
    coarsened_times = batch_time.coarsen(time=coarsen_factor).mean()

    writer: MagicMock = cast(MagicMock, file_writer.writer)
    writer.append_batch.assert_called_once()
    _, kwargs = writer.append_batch.call_args
    torch.testing.assert_close(
        kwargs["data"]["temperature"], time_coarsened_temperature
    )
    xr.testing.assert_equal(kwargs["batch_time"], coarsened_times)


def test_file_writer_monthly(tmpdir):
    label = "monthly_mean_output"
    config = FileWriterConfig(label=label, time_coarsen=MonthlyCoarsenConfig())
    n_timesteps = 24
    n_samples = 1
    writer = config.build(
        experiment_dir=str(tmpdir),
        n_initial_conditions=n_samples,
        n_timesteps=n_timesteps,
        timestep=datetime.timedelta(days=5),
        variable_metadata={},
        coords={"lat": np.linspace(-90, 90, 5), "lon": np.linspace(-180, 180, 5)},
        dataset_metadata=MagicMock(),
    )
    data = {"foo": torch.rand(n_samples, n_timesteps, 5, 5)}
    batch_time = xr.DataArray(
        xr.date_range("2020-01-01", periods=n_timesteps, freq="5D"), dims=["time"]
    )
    batch_time = xr.concat([batch_time] * n_samples, dim="sample")
    writer.append_batch(data, batch_time=batch_time)
    ds = xr.open_dataset(tmpdir / f"{label}.nc")
    assert "counts" in ds.coords
    assert "foo" in ds.data_vars
    assert dict(ds.sizes) == {"sample": 1, "time": 4, "lat": 5, "lon": 5}
    assert ds.valid_time.isel(sample=0, time=0).values == np.datetime64("2020-01-15")
    assert ds.valid_time.isel(sample=0, time=1).values == np.datetime64("2020-02-15")


@pytest.mark.parametrize("save_reference", [True, False])
def test_file_writer_paired_save_reference(tmpdir, save_reference: bool):
    config = FileWriterConfig(
        label="test_writer",
        names=["temperature"],
        save_reference=save_reference,
    )
    writer = config.build_paired(
        experiment_dir=str(tmpdir),
        n_initial_conditions=1,
        n_timesteps=10,
        timestep=datetime.timedelta(days=1),
        variable_metadata={},
        coords={},
        dataset_metadata=DatasetMetadata(),
    )
    writer.finalize()
    if save_reference:
        expected_filenames = {"test_writer_predictions.nc", "test_writer_target.nc"}
        for expected_filename in expected_filenames:
            assert (tmpdir / expected_filename).exists()
    else:
        expected_filename = "test_writer.nc"
        assert (tmpdir / expected_filename).exists()
