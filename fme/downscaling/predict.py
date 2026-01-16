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
from fme.core.coordinates import LatLonCoordinates
from fme.core.dataset.time import TimeSlice
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import NoTargetAggregator, SampleAggregator
from fme.downscaling.data import (
    ClosedInterval,
    DataLoaderConfig,
    GriddedData,
    StaticInputs,
    enforce_lat_bounds,
)
from fme.downscaling.models import CheckpointModelConfig, DiffusionModel
from fme.downscaling.predictors import (
    CascadePredictor,
    CascadePredictorConfig,
    PatchPredictionConfig,
    PatchPredictor,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.train import count_parameters
from fme.downscaling.typing_ import FineResCoarseResPair


def _downscale_coord(coord: torch.tensor, downscale_factor: int):
    """
    This is a bandaid fix for the issue where BatchData does not
    contain coords for the topography, which is fine-res in the no-target
    generation case. The SampleAggregator requires the fine-res coords
    for the predictions.

    TODO: remove after topography refactors to have its own data container.
    """
    if len(coord.shape) != 1:
        raise ValueError("coord tensor to downscale must be 1d")
    spacing = coord[1] - coord[0]
    # Compute edges from midpoints
    first_edge = coord[0] - spacing / 2
    last_edge = coord[-1] + spacing / 2

    # Subdivide edges
    step = spacing / downscale_factor
    new_edges = torch.arange(first_edge, last_edge + step / 2, step)

    # Compute new midpoints
    coord_new = (new_edges[:-1] + new_edges[1:]) / 2
    return coord_new.to(device=coord.device, dtype=coord.dtype)


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
        static_inputs_from_checkpoint: StaticInputs | None = None,
    ) -> GriddedData:
        enforce_lat_bounds(self.lat_extent)
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
        return event_data_config.build(
            requirements=requirements,
            static_inputs_from_checkpoint=static_inputs_from_checkpoint,
        )


class EventDownscaler:
    def __init__(
        self,
        event_name: str,
        data: GriddedData,
        model: DiffusionModel | CascadePredictor,
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
        batch, topography = next(iter(self.data.get_generator()))
        coarse_coords = batch[0].latlon_coordinates
        fine_coords = LatLonCoordinates(
            lat=_downscale_coord(coarse_coords.lat, self.model.downscale_factor),
            lon=_downscale_coord(coarse_coords.lon, self.model.downscale_factor),
        )
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
                batch, topography=topography, n_samples=end_idx - start_idx
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
        model: DiffusionModel | CascadePredictor,
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

    @property
    def _fine_latlon_coordinates(self) -> LatLonCoordinates | None:
        if self.data.topography is not None:
            return self.data.topography.coords
        else:
            return None

    def run(self):
        aggregator = NoTargetAggregator(
            downscale_factor=self.model.downscale_factor,
            latlon_coordinates=self._fine_latlon_coordinates,
        )
        for i, (batch, topography) in enumerate(self.batch_generator):
            with torch.no_grad():
                logging.info(f"Generating predictions on batch {i + 1}")
                prediction = self.generation_model.generate_on_batch_no_target(
                    batch=batch,
                    topography=topography,
                    n_samples=self.n_samples,
                )
                logging.info("Recording diagnostics to aggregator")
                # Add sample dimension to coarse values for generation comparison
                coarse = {k: v.unsqueeze(1) for k, v in batch.data.items()}
                aggregator.record_batch(prediction, coarse, batch.time)
        logs = aggregator.get_wandb()
        wandb = WandB.get_instance()
        wandb.log(logs, step=0)

        self.save_netcdf_data(aggregator.get_dataset())


@dataclasses.dataclass
class DownscalerConfig:
    model: CheckpointModelConfig | CascadePredictorConfig
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
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def build(self) -> list[Downscaler | EventDownscaler]:
        model = self.model.build()
        dataset = self.data.build(
            requirements=self.model.data_requirements,
            static_inputs_from_checkpoint=model.static_inputs,
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
                static_inputs_from_checkpoint=model.static_inputs,
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
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    downscaler_config.configure_wandb(resumable=True, notes=beaker_url)

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
    main(args.config_path)
