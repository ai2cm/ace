import argparse
import dataclasses
import logging
import os
from typing import List, Sequence, Tuple

import dacite
import torch.utils.data
import xarray as xr
import yaml

import fme.core.logging_utils as logging_utils
from fme.ace.data_loading.batch_data import BatchData, default_collate
from fme.ace.data_loading.config import (
    ConcatDatasetConfig,
    DataLoaderConfig,
    MergeDatasetConfig,
)
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.monthly import (
    MonthlyDataWriter,
    months_for_timesteps,
)
from fme.ace.requirements import DataRequirements
from fme.core.coordinates import (
    AtmosphericDeriveFn,
    OptionalHybridSigmaPressureCoordinate,
)
from fme.core.dataset.getters import get_datasets, get_merged_datasets
from fme.core.dataset.properties import DatasetProperties
from fme.core.device import using_gpu
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class CollateFn:
    horizontal_dims: List[str]

    def __call__(
        self, samples: Sequence[Tuple[TensorMapping, xr.DataArray]]
    ) -> "BatchData":
        sample_data, sample_time = zip(*samples)
        batch_data = default_collate(sample_data)
        batch_time = xr.concat(sample_time, dim="sample")
        return BatchData(
            data=batch_data,
            time=batch_time,
            horizontal_dims=self.horizontal_dims,
        )


def get_data_loaders(
    config: DataLoaderConfig, requirements: DataRequirements
) -> Tuple[List[torch.utils.data.DataLoader], DatasetProperties]:
    dist = Distributed.get_instance()
    if dist.world_size > 1:
        raise RuntimeError(
            "Data loading for write_monthly_data.py is not "
            "supported in distributed mode."
        )
    datasets: torch.utils.data.Dataset
    if isinstance(config.dataset, ConcatDatasetConfig):
        datasets, properties = get_datasets(
            config.dataset.concat, requirements.names, requirements.n_timesteps
        )
    elif isinstance(config.dataset, MergeDatasetConfig):
        datasets, properties = get_merged_datasets(
            config.dataset,
            requirements.names,
            requirements.n_timesteps,
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
            collate_fn=CollateFn(
                horizontal_dims=list(properties.horizontal_coordinates.dims),
            ),
        )
        data_loaders.append(dataloader)
    return data_loaders, properties


def get_timesteps(data_loaders: List[torch.utils.data.DataLoader]) -> int:
    timesteps_per_dataset = len(data_loaders[0].dataset)
    if not all(len(loader.dataset) == timesteps_per_dataset for loader in data_loaders):
        raise ValueError("All datasets must have the same number of timesteps.")
    return timesteps_per_dataset


@dataclasses.dataclass
class Config:
    """
    Configuration for applying the MonthlyDataWriter to a dataset.

    Parameters:
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
        data_loaders, properties = get_data_loaders(
            config=self.data_loader,
            requirements=DataRequirements(
                names=self.variable_names,
                n_timesteps=1,  # 1 to count steps properly, avoid window overlap
            ),
        )
        n_timesteps = get_timesteps(data_loaders=data_loaders)
        return Data(
            loaders=data_loaders,
            properties=properties,
            n_timesteps=n_timesteps,
        )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def get_data_writer(self, data: "Data") -> MonthlyDataWriter:
        n_months = months_for_timesteps(data.n_timesteps, data.properties.timestep)
        coords = {
            **data.properties.horizontal_coordinates.coords,
            **data.properties.vertical_coordinate.coords,
        }
        return MonthlyDataWriter(
            path=self.experiment_dir,
            label="data",
            save_names=None,  # save all data given
            n_samples=self.data_loader.batch_size * len(data.loaders),
            n_months=n_months,
            variable_metadata=data.properties.variable_metadata,
            coords=coords,
            dataset_metadata=DatasetMetadata.from_env(),
        )


@dataclasses.dataclass
class Data:
    loaders: List[torch.utils.data.DataLoader]
    properties: DatasetProperties
    n_timesteps: int


def merge_loaders(loaders: List[torch.utils.data.DataLoader]):
    for window_batch_data_list in zip(*loaders):
        tensors = [item.data for item in window_batch_data_list]
        time = [item.time for item in window_batch_data_list]
        window_batch_data = {
            k: torch.concat([d[k] for d in tensors]) for k in tensors[0].keys()
        }
        time = xr.concat(time, dim="sample")
        yield BatchData(
            data=window_batch_data,
            time=time,
        )


def run(config: Config):
    config.configure_logging(log_filename="write_monthly_data_out.log")
    logging_utils.log_versions()

    data = config.get_data()
    writer = config.get_data_writer(data)

    assert isinstance(
        data.properties.vertical_coordinate, OptionalHybridSigmaPressureCoordinate
    )
    derive_func = AtmosphericDeriveFn(
        vertical_coordinate=data.properties.vertical_coordinate,
        timestep=data.properties.timestep,
    )

    n_batches = len(data.loaders[0].dataset) // config.data_loader.batch_size
    for i, window_batch_data in enumerate(merge_loaders(data.loaders)):
        # no need to trim initial conditions because
        # we set n_timesteps to 1 in the DataRequirements
        assert list(window_batch_data.data.values())[0].shape[1] == 1

        window_batch_data = window_batch_data.compute_derived_variables(
            derive_func=derive_func,
            forcing_data=window_batch_data,
        )
        writer.append_batch(
            data=window_batch_data.data,
            start_timestep=-1,  # ignored
            batch_time=window_batch_data.time,
        )
        if i % 10 == 0:
            logging.info(f"Writing batch {i + 1} of {n_batches}.")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_config", type=str)

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
    )
