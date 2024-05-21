import argparse
import dataclasses
import datetime
import logging
import os
import pathlib
from typing import List

import dacite
import torch.utils.data
import xarray as xr
import yaml

from fme.ace.inference.data_writer.monthly import (
    MonthlyDataWriter,
    months_for_timesteps,
)
from fme.ace.inference.derived_variables import compute_derived_quantities
from fme.ace.train_config import LoggingConfig
from fme.ace.utils import logging_utils
from fme.core.data_loading._xarray import XarrayDataset, get_datasets_at_path
from fme.core.data_loading.config import DataLoaderConfig
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.testing import DimSizes, save_2d_netcdf


def get_data_loaders(
    config: DataLoaderConfig, requirements: DataRequirements
) -> List[torch.utils.data.DataLoader]:
    dist = Distributed.get_instance()
    if dist.world_size > 1:
        raise RuntimeError(
            "Data loading for write_monthly_data.py is not "
            "supported in distributed mode."
        )
    if config.data_type == "xarray":
        datasets = [XarrayDataset(config.dataset, requirements)]
    elif config.data_type == "ensemble_xarray":
        datasets = get_datasets_at_path(
            config.dataset, requirements, subset=config.subset
        )

    data_loaders = []
    for dataset in datasets:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_data_workers,
            shuffle=False,
            sampler=None,
            drop_last=True,
            pin_memory=using_gpu(),
            collate_fn=BatchData.from_sample_tuples,
        )
        data_loaders.append(dataloader)
    return data_loaders


def get_timesteps(data_loaders: List[torch.utils.data.DataLoader]) -> int:
    timesteps_per_dataset = len(data_loaders[0].dataset)
    if not all(len(loader.dataset) == timesteps_per_dataset for loader in data_loaders):
        raise ValueError("All datasets must have the same number of timesteps.")
    return timesteps_per_dataset


@dataclasses.dataclass
class Config:
    """
    Configuration for applying the MonthlyDataWriter to a dataset.

    Attributes:
        experiment_dir: Directory to save results to.
        dataset: Configuration for the dataset to load.
        num_data_workers: Number of parallel workers to use for data loading.
        logging: Configuration for logging.
        variable_names: Names of the variables to write to disk.
    """

    experiment_dir: str
    data_loader: DataLoaderConfig
    variable_names: List[str]
    logging: LoggingConfig = dataclasses.field(
        default_factory=lambda: LoggingConfig(
            log_to_file=True, log_to_screen=False, log_to_wandb=False
        )
    )

    def __post_init__(self):
        if self.data_loader.batch_size != 1:
            raise ValueError("Batch size must be 1 to write dataset using writer.")

    def get_data(self) -> "Data":
        data_loaders = get_data_loaders(
            config=self.data_loader,
            requirements=DataRequirements(
                names=self.variable_names,
                n_timesteps=1,  # 1 to count steps properly, avoid window overlap
            ),
        )
        n_timesteps = get_timesteps(data_loaders=data_loaders)
        return Data(
            loaders=data_loaders,
            sigma_coordinates=data_loaders[0].dataset.sigma_coordinates,
            timestep=data_loaders[0].dataset.timestep,
            n_timesteps=n_timesteps,
        )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def get_data_writer(self, data: "Data") -> MonthlyDataWriter:
        n_months = months_for_timesteps(data.n_timesteps, data.timestep)
        coords = {
            **data.loaders[0].dataset.horizontal_coordinates.coords,
            **data.loaders[0].dataset.sigma_coordinates.coords,
        }
        return MonthlyDataWriter(
            path=self.experiment_dir,
            label="data",
            save_names=None,  # save all data given
            n_samples=self.data_loader.batch_size * len(data.loaders),
            n_months=n_months,
            metadata=data.loaders[0].dataset.metadata,
            coords=coords,
        )


@dataclasses.dataclass
class Data:
    loaders: List[torch.utils.data.DataLoader]
    sigma_coordinates: SigmaCoordinates
    timestep: datetime.timedelta
    n_timesteps: int


def merge_loaders(loaders: List[torch.utils.data.DataLoader]):
    window_batch_data_list: List[BatchData]
    for window_batch_data_list in zip(*loaders):
        tensors = [item.data for item in window_batch_data_list]
        times = [item.times for item in window_batch_data_list]
        window_batch_data = {
            k: torch.concat([d[k] for d in tensors]) for k in tensors[0].keys()
        }
        times = xr.concat(times, dim="sample")
        yield BatchData(data=window_batch_data, times=times)


def run(config: Config):
    config.configure_logging(log_filename="write_monthly_data_out.log")
    logging_utils.log_versions()

    data = config.get_data()
    writer = config.get_data_writer(data)

    n_batches = len(data.loaders[0].dataset) // config.data_loader.batch_size
    for i, window_batch_data in enumerate(merge_loaders(data.loaders)):
        # no need to trim initial conditions because
        # we set n_timesteps to 1 in the DataRequirements
        assert list(window_batch_data.data.values())[0].shape[1] == 1

        window_batch_data.data = compute_derived_quantities(
            window_batch_data.data, data.sigma_coordinates, data.timestep
        )
        writer.append_batch(
            data=window_batch_data.data,
            start_timestep=-1,  # ignored
            batch_times=window_batch_data.times,
        )
        if i % 10 == 0:
            logging.info(f"Writing batch {i+1} of {n_batches}.")
            writer.flush()

    writer.flush()
    logging.info("Finished writing data.")


def main(
    yaml_config: str,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    config = dacite.from_dict(
        data_class=Config,
        data=data,
        config=dacite.Config(strict=True),
    )
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    run(config)


def write_ensemble_dataset(
    path: pathlib.Path, n_members: int, names: List[str], dim_sizes: DimSizes
):
    if not path.exists():
        path.mkdir(parents=True)
    for i in range(n_members):
        ensemble_dir = path / f"ic_{i:04d}"
        ensemble_dir.mkdir(exist_ok=True)
        save_2d_netcdf(
            ensemble_dir / "data.nc",
            dim_sizes,
            names,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
