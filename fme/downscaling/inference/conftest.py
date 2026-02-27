import pytest

from fme.core.dataset.xarray import XarrayDataConfig
from fme.downscaling.data import DataLoaderConfig
from fme.downscaling.test_utils import data_paths_helper


@pytest.fixture
def loader_config(tmp_path, request):
    """Create DataLoaderConfig with test data."""
    path = tmp_path / "test_data"
    path.mkdir()
    add_topography_path = request.param
    # TODO: should probably consolidate cross imported
    # .      data path helpers to a single file instead
    # .      of importing from test_train in each location
    test_data_path = data_paths_helper(path)
    topography_path = (
        f"{test_data_path.fine}/data.nc" if add_topography_path is True else None
    )

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
        topography=topography_path,
    )
