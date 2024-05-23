import pathlib

import xarray as xr
from write_monthly_data import Config, run, write_ensemble_dataset

from fme.ace.train_config import LoggingConfig
from fme.core.data_loading.config import DataLoaderConfig, XarrayDataConfig
from fme.core.testing import DimSizes


def test_write_monthly_data(tmp_path: pathlib.Path):
    all_names = ["a", "b"]
    dim_sizes = DimSizes(
        n_time=4 * 60,
        n_lat=4,
        n_lon=8,
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
