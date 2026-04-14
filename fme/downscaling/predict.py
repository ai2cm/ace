import argparse
import dataclasses
import logging
from datetime import datetime, timedelta

import dacite
import torch
import xarray as xr
import yaml

from fme.core.cli import prepare_directory
from fme.core.dataset.time import TimeSlice
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import NoTargetAggregator, SampleAggregator
from fme.downscaling.data import (
    ClosedInterval,
    DataLoaderConfig,
    GriddedData,
    enforce_lat_bounds,
)
from fme.downscaling.models import CheckpointModelConfig, DiffusionModel
from fme.downscaling.predictors import (
    DenoisingMoECheckpointConfig,
    DenoisingScheduleSequentialPredictor,
    PatchPredictionConfig,
    PatchPredictor,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class EventConfig:
    name: str
    date: str
    lat_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-66.0, 70.0)
    )
    lon_extent: ClosedInterval = dataclasses.field(
        default_factory=lambda: ClosedInterval(-180.0, 360.0)
    )
    n_samples: int = 64
    date_format: str = "%Y-%m-%dT%H:%M"
    save_generated_samples: bool = False
    patch: PatchPredictionConfig = dataclasses.field(
        default_factory=PatchPredictionConfig
    )

    @property
    def _time_selection_slice(self) -> TimeSlice:
        """Returns a TimeSlice containing only the first 6h time(s).
        Event evaluation only load the first snapshot.
        Filling the slice stop isn't necessary but guards against
        future code trying to iterate over the entire dataloader.
        """
        _stop = (
            datetime.strptime(self.date, self.date_format) + timedelta(hours=12)
        ).strftime(self.date_format)
        return TimeSlice(self.date, _stop)

    def get_gridded_data(
        self,
        base_data_config: DataLoaderConfig,
        requirements: DataRequirements,
    ) -> GriddedData:
        enforce_lat_bounds(self.lat_extent)
        event_coarse = dataclasses.replace(base_data_config.full_config[0])
        event_coarse.update_subset(self._time_selection_slice)
        n_processes = Distributed.get_instance().world_size
        event_data_config = dataclasses.replace(
            base_data_config,
            coarse=[event_coarse],
            repeat=n_processes,
            batch_size=n_processes,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )
        return event_data_config.build(
            requirements=requirements,
        )


class EventDownscaler:
    def __init__(
        self,
        event_name: str,
        data: GriddedData,
        model: DiffusionModel | DenoisingScheduleSequentialPredictor,
        experiment_dir: str,
        n_samples: int,
        patch: PatchPredictionConfig = PatchPredictionConfig(
            divide_generation=True,
            composite_prediction=True,
            coarse_horizontal_overlap=1,
        ),
        save_generated_samples: bool = False,
    ):
        self.event_name = event_name
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()
        self.patch = patch
        self._max_sample_group = 8
        self.save_generated_samples = save_generated_samples

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

    def run(self):
        logging.info(f"Running {self.event_name} event downscaling...")
        batch = next(iter(self.data.get_generator()))
        coarse_coords = batch[0].latlon_coordinates
        fine_coords = self.model.get_fine_coords_for_batch(batch)
        sample_agg = SampleAggregator(
            coarse=batch[0].data,
            latlon_coordinates=FineResCoarseResPair(
                fine=fine_coords,
                coarse=coarse_coords,
            ),
        )
        # Sample generation is split up into chunks for GPU parallelism
        # since there is no batch parallelism in event evaluation.
        total_samples = self.dist.local_batch_size(self.n_samples)
        for start_idx in range(0, total_samples, self._max_sample_group):
            end_idx = min(start_idx + self._max_sample_group, total_samples)
            logging.info(
                f"Generating samples {start_idx} to {end_idx} "
                f"for event {self.event_name}"
            )
            outputs = self.model.generate_on_batch_no_target(
                batch, n_samples=end_idx - start_idx
            )
            sample_agg.record_batch(outputs)
        to_log = sample_agg.get_wandb()
        wandb = WandB.get_instance()
        wandb.log({f"{self.event_name}/{k}": v for k, v in to_log.items()}, step=0)

        if self.save_generated_samples:
            ds = sample_agg.get_dataset()
            if self.dist.is_root():
                # no slashes allowed in netcdf variable names
                ds = ds.rename({k: k.replace("/", "_") for k in ds.data_vars})
                ds.to_netcdf(f"{self.experiment_dir}/{self.event_name}.nc", mode="w")
            logging.info(
                f"{self.n_samples} generated samples saved for {self.event_name}"
            )
        torch.cuda.empty_cache()


class Downscaler:
    def __init__(
        self,
        data: GriddedData,
        model: DiffusionModel | DenoisingScheduleSequentialPredictor,
        experiment_dir: str,
        n_samples: int,
        patch: PatchPredictionConfig = PatchPredictionConfig(
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
                model=self.model,
                coarse_yx_patch_extent=self.model.coarse_shape,
                coarse_horizontal_overlap=self.patch.coarse_horizontal_overlap,
            )
        else:
            return self.model

    @property
    def batch_generator(self):
        if self.patch.needs_patch_data_generator:
            return self.data.get_patched_generator(
                yx_patch_extent=self.model.coarse_shape,
                overlap=self.patch.coarse_horizontal_overlap,
                drop_partial_patches=False,
            )
        else:
            return self.data.get_generator()

    def save_netcdf_data(self, ds: xr.Dataset):
        if self.dist.is_root():
            # no slashes allowed in netcdf variable names
            ds = ds.rename({k: k.replace("/", "_") for k in ds.data_vars})
            ds.to_netcdf(
                f"{self.experiment_dir}/generated_maps_and_metrics.nc", mode="w"
            )

    def run(self):
        aggregator: NoTargetAggregator | None = None
        for i, batch in enumerate(self.batch_generator):
            if aggregator is None:
                fine_coords = self.model.get_fine_coords_for_batch(batch)
                aggregator = NoTargetAggregator(
                    downscale_factor=self.model.downscale_factor,
                    latlon_coordinates=fine_coords,
                )
            with torch.no_grad():
                logging.info(f"Generating predictions on batch {i + 1}")
                prediction = self.generation_model.generate_on_batch_no_target(
                    batch=batch,
                    n_samples=self.n_samples,
                )
                logging.info("Recording diagnostics to aggregator")
                # Add sample dimension to coarse values for generation comparison
                coarse = {k: v.unsqueeze(1) for k, v in batch.data.items()}
                aggregator.record_batch(prediction, coarse, batch.time)

        # dataset build ensures non-empty batch_generator
        assert aggregator is not None
        logs = aggregator.get_wandb()
        wandb = WandB.get_instance()
        wandb.log(logs, step=0)

        self.save_netcdf_data(aggregator.get_dataset())


@dataclasses.dataclass
class DownscalerConfig:
    model: DenoisingMoECheckpointConfig | CheckpointModelConfig
    experiment_dir: str
    data: DataLoaderConfig
    logging: LoggingConfig
    n_samples: int = 4
    patch: PatchPredictionConfig = dataclasses.field(
        default_factory=PatchPredictionConfig
    )
    events: list[EventConfig] | None = None
    """
    This class is used to configure the downscaling model generation.
    Fine-resolution outputs are generated from coarse-resolution inputs.
    In contrast to the Evaluator, there is no fine-resolution target data
    to compare the generated outputs against.
    """

    def configure_logging(self, log_filename: str):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )

    def build(self) -> list[Downscaler | EventDownscaler]:
        model = self.model.build()
        dataset = self.data.build(
            requirements=self.model.data_requirements,
        )
        downscaler = Downscaler(
            data=dataset,
            model=model,
            experiment_dir=self.experiment_dir,
            n_samples=self.n_samples,
        )

        event_downscalers = []
        for event_config in self.events or []:
            event_dataset = event_config.get_gridded_data(
                base_data_config=self.data,
                requirements=self.model.data_requirements,
            )
            event_downscalers.append(
                EventDownscaler(
                    event_name=event_config.name,
                    data=event_dataset,
                    model=model,
                    experiment_dir=self.experiment_dir,
                    n_samples=event_config.n_samples,
                    patch=event_config.patch,
                    save_generated_samples=event_config.save_generated_samples,
                )
            )
        return [downscaler] + event_downscalers


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    downscaler_config: DownscalerConfig = dacite.from_dict(
        data_class=DownscalerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(downscaler_config.experiment_dir, config)

    downscaler_config.configure_logging(log_filename="out.log")

    logging.info("Starting downscaling model generation...")
    downscalers = downscaler_config.build()
    logging.info(
        f"Number of parameters: {count_parameters(downscalers[0].model.modules)}"
    )
    for downscaler in downscalers:
        downscaler.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling generation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Distributed.context():
        main(args.config_path)
