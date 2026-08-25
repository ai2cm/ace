from unittest.mock import MagicMock

import numpy as np
import pytest

from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data import ClosedInterval
from fme.downscaling.inference.output import (
    DownscalingOutput,
    DownscalingOutputConfig,
    EventConfig,
    TimeRangeConfig,
)
from fme.downscaling.predictors import PatchPredictionConfig
from fme.downscaling.requirements import DataRequirements

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
    assert output_target.chunks is not None
    assert tuple(output_target.chunks.values())[:2] == (1, 1)
    assert output_target.shards is not None
    assert tuple(output_target.shards.values()) == output_target.data.max_output_shape


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

    # Verify chunks dict structure
    assert output_target.data is not None
    assert output_target.chunks is not None
    assert tuple(output_target.chunks.values())[:2] == (1, 1)
    assert output_target.shards is not None
    assert tuple(output_target.shards.values()) == output_target.data.max_output_shape


def test_downscaling_output_config_overwrite_defaults_false():
    config = TimeRangeConfig(
        name="test",
        n_ens=4,
        time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-02T00:00:00"),
    )
    assert config.overwrite is False


@pytest.mark.parametrize("loader_config", [True], indirect=True)
def test_region_config_build_propagates_overwrite_to_output_target(
    loader_config, requirements, patch_config
):
    config = TimeRangeConfig(
        name="test_region",
        time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-02T00:00:00"),
        n_ens=4,
        save_vars=["var0", "var1"],
        overwrite=True,
    )
    output_target = config.build(loader_config, requirements, patch_config)
    assert output_target.overwrite is True


@pytest.mark.parametrize("overwrite,expected_mode", [(False, "w-"), (True, "w")])
def test_downscaling_output_get_writer_mode_matches_overwrite(overwrite, expected_mode):
    output_target = DownscalingOutput(
        name="test",
        save_vars=["var0"],
        n_ens=1,
        max_samples_per_gpu=1,
        data=MagicMock(),
        patch=PatchPredictionConfig(),
        chunks={"time": 1, "ensemble": 1, "latitude": 1, "longitude": 1},
        shards={"time": 1, "ensemble": 1, "latitude": 1, "longitude": 1},
        overwrite=overwrite,
    )
    latlon_coords = MagicMock()
    latlon_coords.lat.cpu().numpy.return_value = np.array([0.0])
    latlon_coords.lon.cpu().numpy.return_value = np.array([0.0])
    output_target.data.all_times.to_numpy.return_value = np.array([0])

    writer = output_target.get_writer(latlon_coords=latlon_coords, output_dir="/tmp")
    assert writer._mode == expected_mode


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
