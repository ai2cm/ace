import datetime
from pathlib import Path
from unittest.mock import MagicMock

import cftime
import numpy as np
import pytest
import xarray as xr

from fme.core.dataset.time import TimeSlice
from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data import ClosedInterval, DataLoaderConfig
from fme.downscaling.generate.output import (
    EventConfig,
    OutputTarget,
    OutputTargetConfig,
    RegionConfig,
)
from fme.downscaling.predictors import PatchPredictionConfig
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.test_train import data_paths_helper

# Fixtures for unit tests


@pytest.fixture
def mock_loader():
    """Create a mock DataLoader."""
    return MagicMock()


@pytest.fixture
def mock_all_times():
    """Create a mock time coordinate DataArray."""
    times = np.array(["2021-01-01", "2021-01-02"], dtype="datetime64[ns]")
    return xr.DataArray(times, dims=["time"])


@pytest.fixture
def mock_patch_config():
    """Create a mock PatchPredictionConfig."""
    patch_config = MagicMock()
    patch_config.needs_patch_predictor = False
    return patch_config


@pytest.fixture
def mock_latlon_coords():
    """Create mock LatLonCoordinates."""
    coords = MagicMock()
    coords.lat = MagicMock()
    coords.lat.numpy.return_value = np.linspace(-90, 90, 180)
    coords.lon = MagicMock()
    coords.lon.numpy.return_value = np.linspace(0, 360, 360)
    return coords


# Tests for OutputTargetConfig validation


def test_single_xarray_config_accepts_single_config():
    """Test that _single_xarray_config accepts a single XarrayDataConfig."""
    xarray_config = XarrayDataConfig(
        data_path="/path/to/data", file_pattern="*.nc", engine="netcdf4"
    )
    result = OutputTargetConfig._single_xarray_config([xarray_config])
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
        OutputTargetConfig._single_xarray_config([config1, config2])


def test_single_xarray_config_rejects_non_xarray_config():
    """Test that _single_xarray_config rejects non-XarrayDataConfig objects."""
    mock_config = MagicMock()

    with pytest.raises(NotImplementedError, match="XarrayDataConfig objects"):
        OutputTargetConfig._single_xarray_config([mock_config])


# Tests for EventConfig instantiation and validation


def test_event_config_requires_event_time():
    """Test that EventConfig raises ValueError without event_time."""
    with pytest.raises(ValueError, match="event_time must be specified"):
        EventConfig(name="test", n_ens=8, save_vars=["var1"])


# Tests for RegionConfig instantiation and validation


def test_region_config_requires_time_range():
    """Test that RegionConfig raises ValueError without time_range."""
    with pytest.raises(ValueError, match="time_range must be specified"):
        RegionConfig(name="test", n_ens=8, save_vars=["var1"])


# Integration test fixtures and helpers


def _midpoints_from_count(start, end, n_mid):
    """Generate midpoints for grid coordinates."""
    width = (end - start) / n_mid
    return np.linspace(start + width / 2, end - width / 2, n_mid, dtype=np.float32)


def create_test_coarse_data(
    filename: Path, n_times: int = 4, n_lat: int = 8, n_lon: int = 8
) -> Path:
    """Create test coarse data NetCDF file for integration tests."""
    # Create time coordinates
    time_coord = [
        cftime.DatetimeProlepticGregorian(2000, 1, 1) + datetime.timedelta(days=i)
        for i in range(n_times)
    ]

    # Create spatial coordinates as midpoints
    lat = _midpoints_from_count(0, 8, n_lat)
    lon = _midpoints_from_count(0, 8, n_lon)

    # Create data variables
    variable_names = ["x", "y", "HGTsfc"]
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(n_times, n_lat, n_lon).astype(np.float32)
        data_vars[name] = xr.DataArray(
            data,
            dims=["time", "lat", "lon"],
            attrs={"units": "m", "long_name": name},
        )

    # Add ak, bk scalar variables
    for i in range(7):
        data_vars[f"ak_{i}"] = float(i)
        data_vars[f"bk_{i}"] = float(i + 1)

    # Create dataset
    coords = {"time": time_coord, "lat": lat, "lon": lon}
    ds = xr.Dataset(data_vars=data_vars, coords=coords)

    # Save to file
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")
    return filename


def create_test_topography(filename: Path, n_lat: int = 16, n_lon: int = 16) -> Path:
    """Create test topography data at fine resolution."""
    # Create spatial coordinates at fine resolution (2x coarse)
    lat = _midpoints_from_count(0, 8, n_lat)
    lon = _midpoints_from_count(0, 8, n_lon)

    # Create topography data
    data = np.random.randn(n_lat, n_lon).astype(np.float32)
    topo = xr.DataArray(
        data,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={"units": "m", "long_name": "surface_altitude"},
    )

    ds = xr.Dataset({"HGTsfc": topo})
    ds.to_netcdf(filename, format="NETCDF4_CLASSIC")
    return filename


@pytest.fixture
def loader_config(tmp_path):
    """Create DataLoaderConfig with test data."""
    path = tmp_path / "test_data"
    path.mkdir()
    # TODO: should probably consolidate cross imported
    # .      data path helpers to a single file instead
    # .      of importing from test_train in each location
    test_data_path = data_paths_helper(path)

    return DataLoaderConfig(
        coarse=[
            XarrayDataConfig(
                data_path=str(test_data_path.coarse),
                file_pattern="*.nc",
                engine="netcdf4",
            )
        ],
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
        topography=f"{test_data_path.fine}/data.nc",
    )


@pytest.fixture
def requirements():
    """Create DataRequirements for generation."""
    return DataRequirements(
        coarse_names=["x", "y"],
        fine_names=["x", "y"],
        n_timesteps=1,
        use_fine_topography=True,
    )


@pytest.fixture
def patch_config():
    """Create PatchPredictionConfig."""
    return PatchPredictionConfig()


# Integration tests for Config.build()


def test_event_config_build_creates_output_target_with_single_time(
    loader_config, requirements, patch_config
):
    """Test EventConfig.build() creates OutputTarget with single timestep."""
    config = EventConfig(
        name="test_event",
        event_time="2000-01-01T00:00:00",
        n_ens=4,
        save_vars=["x", "y"],
        lat_extent=ClosedInterval(2.0, 6.0),
        lon_extent=ClosedInterval(2.0, 6.0),
    )

    output_target = config.build(loader_config, requirements, patch_config)

    # Verify OutputTarget was created
    assert isinstance(output_target, OutputTarget)
    assert output_target.name == "test_event"
    assert output_target.save_vars == ["x", "y"]
    assert output_target.n_ens == 4

    # Verify time dimension - should have exactly 1 timestep
    assert len(output_target.all_times) == 1
    assert output_target.data is not None
    assert output_target.chunks is not None


def test_region_config_build_creates_output_target_with_time_range(
    loader_config, requirements, patch_config
):
    """Test RegionConfig.build() creates OutputTarget with time range."""
    config = RegionConfig(
        name="test_region",
        time_range=TimeSlice("2000-01-01T00:00:00", "2000-01-02T00:00:00"),
        n_ens=4,
        save_vars=["x", "y"],
    )

    output_target = config.build(loader_config, requirements, patch_config)

    # Verify OutputTarget was created
    assert isinstance(output_target, OutputTarget)
    assert output_target.name == "test_region"
    assert output_target.n_ens == 4
    assert len(output_target.all_times) == 2

    # Verify chunks dict structure
    assert output_target.data is not None
    assert output_target.chunks is not None


def test_config_build_uses_custom_patch_config(loader_config, requirements):
    """Test that custom PatchPredictionConfig is used when provided."""
    custom_patch = PatchPredictionConfig(divide_generation=True)

    config = EventConfig(
        name="test_custom_patch",
        event_time="2000-01-01T00:00:00",
        n_ens=2,
        save_vars=["x"],
        patch=custom_patch,
    )

    default_patch = PatchPredictionConfig(divide_generation=False)
    output_target = config.build(loader_config, requirements, default_patch)

    # Verify custom patch config was used, not default
    assert output_target.patch is custom_patch
    assert output_target.patch.divide_generation is True


def test_config_build_uses_default_patch_when_none_provided(
    loader_config, requirements, patch_config
):
    """Test that default PatchPredictionConfig is used when custom is None."""
    config = EventConfig(
        name="test_default_patch",
        event_time="2000-01-01T00:00:00",
        n_ens=2,
        save_vars=["x"],
        patch=None,
    )

    output_target = config.build(loader_config, requirements, patch_config)

    # Verify default patch config was used
    assert output_target.patch is patch_config
