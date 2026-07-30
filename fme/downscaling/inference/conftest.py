import pytest

from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data import DataLoaderConfig
from fme.downscaling.test_utils import data_paths_helper


@pytest.fixture
def data_paths(tmp_path):
    """Create test data paths."""
    # TODO: should probably consolidate cross imported
    # .      data path helpers to a single file instead
    # .      of importing from test_train in each location
    # NOTE: data_paths_helper creates fine and coarse data
    #   including the static input topography field
    path = tmp_path / "test_data"
    path.mkdir()
    return data_paths_helper(path)


@pytest.fixture
def loader_config(data_paths):
    """Create DataLoaderConfig with test data."""
    return DataLoaderConfig(
        coarse=[
            XarrayDataConfig(
                data_path=str(data_paths.coarse),
                file_pattern="*.nc",
                engine="netcdf4",
            )
        ],
        batch_size=2,
        num_data_workers=0,
        strict_ensemble=False,
    )
