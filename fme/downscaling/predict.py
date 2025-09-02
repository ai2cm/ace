import argparse
import dataclasses
import logging
from datetime import datetime, timedelta

import dacite
import torch
import xarray as xr
import yaml

import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_directory
from fme.core.dataset.time import TimeSlice
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators.no_target import NoTargetAggregator
from fme.downscaling.data import ClosedInterval, DataLoaderConfig, GriddedData
from fme.downscaling.evaluator import CheckpointModelConfig
from fme.downscaling.models import DiffusionModel, Model
from fme.downscaling.patching import (
    MultipatchConfig,
    PatchPredictor,
    patch_generator_from_loader,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.train import count_parameters


@dataclasses.dataclass
class EventConfig:
    name: str
    date: str
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-90.0, 90.0)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(float("-inf"), float("inf"))
    )
    n_samples: int = 64
    date_format: str = "%Y-%m-%dT%H:%M"
    save_generated_samples: bool = False

    @property
    def _time_selection_slice(self) -> TimeSlice:
        """Returns a TimeSlice containing only the first 6h time(s).
        Event evaluation only load the first snapshot.
        Filling the slice stop isn't necessary but guards against
        future code trying to iterate over the entire dataloader.
        """
        _stop = (
            datetime.strptime(self.date, self.date_format) + timedelta(hours=6)
        ).strftime(self.date_format)
        return TimeSlice(self.date, _stop)

    def get_gridded_data(
        self, base_data_config: DataLoaderConfig, requirements: DataRequirements
    ) -> GriddedData:
        event_coarse = dataclasses.replace(
            base_data_config.full_config[0], subset=self._time_selection_slice
        )
        n_processes = Distributed.get_instance().world_size
        event_data_config = dataclasses.replace(
            base_data_config,
            coarse=[event_coarse],
            repeat=n_processes,
            batch_size=n_processes,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )
        return event_data_config.build(requirements=requirements)


class Downscaler:
    def __init__(
        self,
        data: GriddedData,
        model: DiffusionModel,
        experiment_dir: str,
        n_samples: int,
        patch: MultipatchConfig = MultipatchConfig(
            divide_generation=True,
            composite_prediction=True,
            coarse_horizontal_overlap=1,
        ),
    ) -> None:
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()
        self.patch = patch

    @property
    def generation_model(self):
        if self.patch.needs_patch_predictor:
            return PatchPredictor(
                self.model,
                self.data.shape,
                coarse_horizontal_overlap=self.patch.coarse_horizontal_overlap,
            )
        else:
            return self.model

    @property
    def batch_generator(self):
        if self.patch.needs_patch_data_generator:
            return patch_generator_from_loader(
                self.data.loader,
                yx_extent=self.data.shape,
                yx_patch_extents=self.model.coarse_shape,
                overlap=self.patch.coarse_horizontal_overlap,
                drop_partial_patches=False,
            )
        else:
            return self.data.loader

    def save_netcdf_data(self, ds: xr.Dataset):
        if self.dist.is_root():
            # no slashes allowed in netcdf variable names
            ds = ds.rename({k: k.replace("/", "_") for k in ds.data_vars})
            ds.to_netcdf(
                f"{self.experiment_dir}/generated_maps_and_metrics.nc", mode="w"
            )

    def run(self):
        aggregator = NoTargetAggregator()
        for i, batch in enumerate(self.batch_generator):
            with torch.no_grad():
                logging.info(f"Generating predictions on batch {i + 1}")
                prediction = self.generation_model.generate_on_batch_no_target(
                    batch=batch,
                    n_samples=self.n_samples,
                )
                logging.info("Recording diagnostics to aggregator")
                # Add sample dimension to coarse values for generation comparison
                coarse = {k: v.unsqueeze(1) for k, v in batch.data.items()}
                aggregator.record_batch(prediction, coarse)
        logs = aggregator.get_wandb()
        wandb = WandB.get_instance()
        wandb.log(logs, step=0)

        self.save_netcdf_data(aggregator.get_dataset())


@dataclasses.dataclass
class DownscalerConfig:
    model: CheckpointModelConfig
    experiment_dir: str
    data: DataLoaderConfig
    logging: LoggingConfig
    n_samples: int = 4
    patch: MultipatchConfig = dataclasses.field(default_factory=MultipatchConfig)
    """
    This class is used to configure the downscaling model generation.
    Fine-resolution outputs are generated from coarse-resolution inputs.
    In contrast to the Evaluator, there is no fine-resolution target data
    to compare the generated outputs against.

    TODO: add event generation.
    """

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def build(self) -> Downscaler:
        dataset = self.data.build(
            requirements=self.model.data_requirements,
        )
        model = self.model.build()
        if isinstance(model, Model):
            raise NotImplementedError(
                "No-target generation is only enabled for DiffusionModel, not Model"
            )

        return Downscaler(
            data=dataset,
            model=model,
            experiment_dir=self.experiment_dir,
            n_samples=self.n_samples,
        )


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    predictor_config: DownscalerConfig = dacite.from_dict(
        data_class=DownscalerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(predictor_config.experiment_dir, config)

    predictor_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    predictor_config.configure_wandb(resumable=True, notes=beaker_url)

    logging.info("Starting downscaling model generation...")
    predictor = predictor_config.build()
    logging.info(f"Number of parameters: {count_parameters(predictor.model.modules)}")
    predictor.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
