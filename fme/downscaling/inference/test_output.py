from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data import ClosedInterval
from fme.downscaling.inference.output import (
    DownscalingOutput,
    DownscalingOutputConfig,
    EventConfig,
    TimeRangeConfig,
    WriterParams,
)
from fme.downscaling.predictors import PatchPredictionConfig
from fme.downscaling.requirements import DataRequirements


def _make_downscaling_output(zarr_chunks_override=None, zarr_shards_override=None):
    mock_data = MagicMock()
    mock_data.max_output_shape = (2, 4)
    mock_data.dtype = torch.float32
    mock_data.all_times.to_numpy.return_value = np.zeros(2)
    return DownscalingOutput(
        name="test",
        save_vars=None,
        n_ens=4,
        max_samples_per_gpu=4,
        data=mock_data,
        patch=MagicMock(),
        zarr_chunks_override=zarr_chunks_override,
        zarr_shards_override=zarr_shards_override,
    )


def _make_latlon(lat_size=10, lon_size=20):
    latlon = MagicMock()
    latlon.lat = torch.zeros(lat_size)
    latlon.lon = torch.zeros(lon_size)
    return latlon


def test_build_writer_params_default_chunks_and_shards():
    output = _make_downscaling_output()
    latlon = _make_latlon(lat_size=10, lon_size=20)
    params = output._build_writer_params(latlon)
    assert isinstance(params, WriterParams)
    assert params.shards == {"time": 2, "ensemble": 4, "latitude": 10, "longitude": 20}
    assert params.chunks["time"] == 1
    assert params.chunks["ensemble"] == 1


def test_build_writer_params_override_chunks_and_shards():
    zarr_chunks_override = {"time": 5, "ensemble": 5, "latitude": 5, "longitude": 5}
    zarr_shards_override = {"time": 10, "ensemble": 10, "latitude": 10, "longitude": 10}
    output = _make_downscaling_output(
        zarr_chunks_override=zarr_chunks_override,
        zarr_shards_override=zarr_shards_override,
    )
    latlon = _make_latlon(lat_size=10, lon_size=20)
    params = output._build_writer_params(latlon)
    assert params.chunks == zarr_chunks_override
    assert params.shards == zarr_shards_override


# Tests for OutputTargetConfig validation


def test_single_xarray_config_accepts_single_config():
    """Test that _single_xarray_config accepts a single XarrayDataConfig."""
    xarray_config = XarrayDataConfig(
        data_path="/path/to/data", file_pattern="*.nc", engine="netcdf4"
    )
    result = DownscalingOutputConfig._single_xarray_config([xarray_config])
    assert result == [xarray_config]


def test_single_xarray_config_rejects_multiple_configs():
    """Test that _single_xarray_config rejects multiple configs."""
    config1 = XarrayDataConfig(
        data_path="/path1", file_pattern="*.nc", engine="netcdf4"
    )
    config2 = XarrayDataConfig(
        data_path="/path2", file_pattern="*.nc", engine="netcdf4"
    )

    with pytest.raises(NotImplementedError, match="single XarrayDataConfig"):
        DownscalingOutputConfig._single_xarray_config([config1, config2])


def test_single_xarray_config_rejects_non_xarray_config():
    """Test that _single_xarray_config rejects non-XarrayDataConfig objects."""
    mock_config = MagicMock()

    with pytest.raises(NotImplementedError, match="XarrayDataConfig objects"):
        DownscalingOutputConfig._single_xarray_config([mock_config])


# Tests for EventConfig instantiation and validation


def test_event_config_requires_event_time():
    """Test that EventConfig raises ValueError without event_time."""
    with pytest.raises(ValueError, match="event_time must be specified"):
        EventConfig(name="test", n_ens=8, save_vars=["var1"])


# Tests for RegionConfig instantiation and validation


def test_region_config_requires_time_range():
    """Test that RegionConfig raises ValueError without time_range."""
    with pytest.raises(ValueError, match="time_range must be specified"):
        TimeRangeConfig(name="test", n_ens=8, save_vars=["var1"])


# Integration test fixtures and helpers


@pytest.fixture
def requirements():
    """Create DataRequirements for generation."""
    return DataRequirements(
        coarse_names=["var0", "var1"],
        fine_names=["var0", "var1"],
        n_timesteps=1,
        use_fine_topography=True,
    )


@pytest.fixture
def patch_config():
    """Create PatchPredictionConfig."""
    return PatchPredictionConfig()


# Integration tests for Config.build()


@pytest.mark.parametrize("loader_config", [True], indirect=True)
def test_event_config_build_creates_output_target_with_single_time(
    loader_config, requirements, patch_config
):
    """Test EventConfig.build() creates OutputTarget with single timestep."""
    config = EventConfig(
        name="test_event",
        event_time="2000-01-01T00:00:00",
        n_ens=4,
        save_vars=["var0", "var1"],
        lat_extent=ClosedInterval(0.0, 6.0),
        lon_extent=ClosedInterval(0.0, 6.0),
    )
    output_target = config.build(loader_config, requirements, patch_config)

    # Verify OutputTarget was created
    assert isinstance(output_target, DownscalingOutput)
    assert output_target.name == "test_event"
    assert output_target.save_vars == ["var0", "var1"]
    assert output_target.n_ens == 4

    # Verify time dimension - should have exactly 1 timestep
    assert len(output_target.data.all_times) == 1
    assert output_target.data is not None


@pytest.mark.parametrize("loader_config", [True], indirect=True)
def test_region_config_build_creates_output_target_with_time_range(
    loader_config, requirements, patch_config
):
    """Test RegionConfig.build() creates OutputTarget with time range."""
    config = TimeRangeConfig(
        name="test_region",
        time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-02T00:00:00"),
        n_ens=4,
        save_vars=["var0", "var1"],
    )
    output_target = config.build(loader_config, requirements, patch_config)

    # Verify OutputTarget was created
    assert isinstance(output_target, DownscalingOutput)
    assert output_target.name == "test_region"
    assert output_target.n_ens == 4
    assert len(output_target.data.all_times) == 2

    assert output_target.data is not None


def test_time_range_config_raise_error_invalid_lat_extent():
    with pytest.raises(ValueError):
        TimeRangeConfig(
            name="test_region",
            time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-02T00:00:00"),
            n_ens=4,
            save_vars=["var0", "var1"],
            lat_extent=ClosedInterval(-90, 90),
        )


def test_event_config_raise_error_invalid_lat_extent():
    with pytest.raises(ValueError):
        EventConfig(
            name="test_event",
            event_time="2000-01-01T00:00:00",
            n_ens=4,
            save_vars=["var0", "var1"],
            lat_extent=ClosedInterval(-90, 90),
        )
