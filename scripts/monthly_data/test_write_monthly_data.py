import pathlib

import xarray as xr
from write_monthly_data import Config, run, write_ensemble_dataset

from fme.core.data_loading.config import DataLoaderConfig, XarrayDataConfig
from fme.core.testing import DimSizes
from fme.fcn_training.train_config import LoggingConfig


def test_write_monthly_data(tmp_path: pathlib.Path):
    all_names = ["a", "b"]
    dim_sizes = DimSizes(
        n_time=4 * 60,
        n_lat=4,
        n_lon=8,
        nz_interface=2,
    )
    write_ensemble_dataset(tmp_path / "data", 3, all_names, dim_sizes)
    config = Config(
        experiment_dir=str(tmp_path),
        data_loader=DataLoaderConfig(
            dataset=XarrayDataConfig(data_path=str(tmp_path / "data")),
            batch_size=1,
            num_data_workers=1,
            data_type="ensemble_xarray",
        ),
        logging=LoggingConfig(
            log_to_screen=True,
            log_to_file=False,
            log_to_wandb=False,
        ),
        variable_names=all_names,
    )
    run(config)
    xr.open_dataset(tmp_path / "monthly_binned_data.nc")
