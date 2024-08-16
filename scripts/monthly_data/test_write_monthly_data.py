import pathlib
from typing import List

import xarray as xr
from write_monthly_data import Config, run

from fme.core.data_loading.config import DataLoaderConfig, XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.testing import DimSize, DimSizes
from fme.core.testing.fv3gfs_data import save_nd_netcdf


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


def test_write_monthly_data(tmp_path: pathlib.Path):
    all_names = ["a", "b"]
    horizontal = [DimSize("grid_yt", 8), DimSize("grid_xt", 4)]
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
            dataset=dataset,
            batch_size=1,
            num_data_workers=1,
        ),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=False,
        ),
        variable_names=all_names,
    )
    run(config)
    xr.open_dataset(tmp_path / "monthly_mean_data.nc")
