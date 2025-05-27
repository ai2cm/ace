import datetime

import numpy as np
import xarray as xr

from fme.core import logging_utils
from fme.core.dataset.config import XarrayDataConfig
from fme.core.testing.wandb import mock_wandb

from .benchmark import BenchmarkConfig, benchmark
from .config import ConcatDatasetConfig, DataLoaderConfig


def get_test_data(path):
    """Create a test dataset for benchmarking."""
    n_lat = 2
    n_lon = 4
    n_time = 10
    ds = xr.Dataset(
        {
            "a": (("time", "lat", "lon"), np.random.rand(n_time, n_lat, n_lon)),
            "b": (("time", "lat", "lon"), np.random.rand(n_time, n_lat, n_lon)),
        },
        coords={
            "time": (
                "time",
                [datetime.datetime(2023, 1, i + 1) for i in range(n_time)],
            ),
            "lat": ("lat", np.arange(n_lat)),
            "lon": ("lon", np.arange(n_lon)),
        },
    )
    ds.to_netcdf(path)
    return ds


def test_benchmark(tmp_path):
    """Ensure that the benchmark runs without error."""
    ds = get_test_data(tmp_path / "test.nc")
    dataset_config = XarrayDataConfig(data_path=str(tmp_path))
    loader_config = DataLoaderConfig(
        dataset=ConcatDatasetConfig(concat=[dataset_config]), batch_size=4
    )
    logging_config = logging_utils.LoggingConfig(project="test", entity="test")
    config = BenchmarkConfig(
        loader=loader_config,
        logging=logging_config,
        names=list(ds.data_vars),
        n_timesteps=3,
        sleep=0,
    )
    with mock_wandb() as wandb:  # noqa: F841
        benchmark(config)
