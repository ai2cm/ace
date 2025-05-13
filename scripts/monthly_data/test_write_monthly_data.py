import pathlib
from typing import List

import pytest
import xarray as xr
from write_monthly_data import Config, run

from fme.ace.data_loading.config import ConcatDatasetConfig, DataLoaderConfig
from fme.ace.testing import DimSize, DimSizes
from fme.ace.testing.fv3gfs_data import save_nd_netcdf
from fme.core.dataset.config import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig


def write_ensemble_dataset(
    path: pathlib.Path, n_members: int, names: List[str], dim_sizes: DimSizes
):
    if not path.exists():
        path.mkdir(parents=True)
    for i in range(n_members):
        ensemble_dir = path / f"ic_{i:04d}"
        ensemble_dir.mkdir(exist_ok=True)
        save_nd_netcdf(
            ensemble_dir / "data.nc",
            dim_sizes,
            names,
            timestep_days=5,
        )


def test_write_monthly_data(very_fast_only: bool, tmp_path: pathlib.Path):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    all_names = ["a", "b"]
    horizontal = [DimSize("lat", 8), DimSize("lon", 4)]
    dim_sizes = DimSizes(
        n_time=4 * 60,
        horizontal=horizontal,
        nz_interface=2,
    )
    n_members = 3
    write_ensemble_dataset(tmp_path / "data", n_members, all_names, dim_sizes)
    dataset = [
        XarrayDataConfig(data_path=str(tmp_path / "data" / f"ic_{i:04}"))
        for i in range(n_members)
    ]
    config = Config(
        experiment_dir=str(tmp_path),
        data_loader=DataLoaderConfig(
            dataset=ConcatDatasetConfig(concat=dataset),
            batch_size=1,
            num_data_workers=0,
        ),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=False,
        ),
        variable_names=all_names,
    )
    run(config)
    xr.open_dataset(tmp_path / "monthly_mean_data.nc", decode_timedelta=False)
