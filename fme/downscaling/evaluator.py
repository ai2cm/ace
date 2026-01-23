import argparse
import dataclasses
import logging

import dacite
import torch
import yaml

import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_directory
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import GenerationAggregator, PairedSampleAggregator
from fme.downscaling.data import (
    PairedDataLoaderConfig,
    PairedGriddedData,
    StaticInputs,
    enforce_lat_bounds,
)
from fme.downscaling.models import CheckpointModelConfig, DiffusionModel
from fme.downscaling.predict import EventConfig
from fme.downscaling.predictors import (
    CascadePredictorConfig,
    PatchPredictionConfig,
    PatchPredictor,
)
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.train import count_parameters
from fme.downscaling.typing_ import FineResCoarseResPair


class Evaluator:
    def __init__(
        self,
        data: PairedGriddedData,
        model: DiffusionModel | PatchPredictor,
        experiment_dir: str,
        n_samples: int,
        patch_data: bool = False,
    ) -> None:
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()
        self.patch_data = patch_data

    def run(self):
        aggregator = GenerationAggregator(
            self.data.dims,
            self.model.downscale_factor,
            include_positional_comparisons=False if self.patch_data else True,
            percentiles=[99.99, 99.9999],
        )

        if self.patch_data:
            batch_generator = self.data.get_patched_generator(
                coarse_yx_patch_extent=self.model.coarse_shape,
            )
        else:
            batch_generator = self.data.get_generator()

        for i, (batch, topography) in enumerate(batch_generator):
            with torch.no_grad():
                logging.info(f"Generating predictions on batch {i + 1}")
                outputs = self.model.generate_on_batch(
                    batch, topography, n_samples=self.n_samples
                )
                logging.info("Recording diagnostics to aggregator")
                # Add sample dimension to coarse values for generation comparison
                coarse = {k: v.unsqueeze(1) for k, v in batch.coarse.data.items()}

                aggregator.record_batch(
                    outputs=outputs,
                    coarse=coarse,
                    batch=batch,
                )

        logs = aggregator.get_wandb()
        wandb = WandB.get_instance()
        wandb.log(logs, step=0)

        ds = aggregator.get_dataset()
        if self.dist.is_root():
            # no slashes allowed in netcdf variable names
            ds = ds.rename({k: k.replace("/", "_") for k in ds.data_vars})
            ds.to_netcdf(
                f"{self.experiment_dir}/evaluator_maps_and_metrics.nc", mode="w"
            )
        logging.info(f"Evaluation complete. Results saved to {self.experiment_dir}.")


class EventEvaluator:
    def __init__(
        self,
        event_name: str,
        data: PairedGriddedData,
        model: DiffusionModel | PatchPredictor,
        experiment_dir: str,
        n_samples: int,
        save_generated_samples: bool = False,
    ) -> None:
        self.event_name = event_name
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()
        self._max_sample_group = 8
        self.save_generated_samples = save_generated_samples

    def run(self):
        logging.info(f"Running {self.event_name} event evaluation")
        batch, topography = next(iter(self.data.get_generator()))
        sample_agg = PairedSampleAggregator(
            target=batch[0].fine.data,
            coarse=batch[0].coarse.data,
            latlon_coordinates=FineResCoarseResPair(
                fine=batch[0].fine.latlon_coordinates,
                coarse=batch[0].coarse.latlon_coordinates,
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
            outputs = self.model.generate_on_batch(
                batch, topography, n_samples=end_idx - start_idx
            )
            sample_agg.record_batch(outputs.prediction)

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


@dataclasses.dataclass
class PairedEventConfig(EventConfig):
    def get_paired_gridded_data(
        self,
        base_data_config: PairedDataLoaderConfig,
        requirements: DataRequirements,
        static_inputs_from_checkpoint: StaticInputs | None = None,
    ) -> PairedGriddedData:
        enforce_lat_bounds(self.lat_extent)
        time_slice = self._time_selection_slice
        event_fine = dataclasses.replace(base_data_config.fine[0], subset=time_slice)
        event_coarse = dataclasses.replace(
            base_data_config.coarse_full_config[0], subset=time_slice
        )
        n_processes = Distributed.get_instance().world_size
        event_data_config = dataclasses.replace(
            base_data_config,
            fine=[event_fine],
            coarse=[event_coarse],
            repeat=n_processes,
            batch_size=n_processes,
            lat_extent=self.lat_extent,
            lon_extent=self.lon_extent,
        )
        return event_data_config.build(
            train=False,
            requirements=requirements,
            static_inputs_from_checkpoint=static_inputs_from_checkpoint,
        )


@dataclasses.dataclass
class EvaluatorConfig:
    model: CheckpointModelConfig | CascadePredictorConfig
    experiment_dir: str
    data: PairedDataLoaderConfig
    logging: LoggingConfig
    n_samples: int = 4
    patch: PatchPredictionConfig = dataclasses.field(
        default_factory=PatchPredictionConfig
    )
    events: list[PairedEventConfig] | None = None

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def _build_default_evaluator(self) -> Evaluator:
        model = self.model.build()
        dataset = self.data.build(
            train=False,
            requirements=self.model.data_requirements,
            static_inputs_from_checkpoint=model.static_inputs,
        )
        evaluator_model: DiffusionModel | PatchPredictor
        if self.patch.divide_generation and self.patch.composite_prediction:
            evaluator_model = PatchPredictor(
                model,
                coarse_yx_patch_extent=model.coarse_shape,
                coarse_horizontal_overlap=self.patch.coarse_horizontal_overlap,
            )
        else:
            evaluator_model = model

        if self.patch.divide_generation and not self.patch.composite_prediction:
            # Subdivide evaluation into patches, do not composite them together
            # No maps will be saved for this configuration.
            patch_data = True
        else:
            patch_data = False

        return Evaluator(
            data=dataset,
            model=evaluator_model,
            experiment_dir=self.experiment_dir,
            n_samples=self.n_samples,
            patch_data=patch_data,
        )

    def _build_event_evaluator(
        self,
        event_config: PairedEventConfig,
    ) -> EventEvaluator:
        model = self.model.build()
        evaluator_model: DiffusionModel | PatchPredictor

        dataset = event_config.get_paired_gridded_data(
            base_data_config=self.data, requirements=self.model.data_requirements
        )

        if (dataset.coarse_shape[0] > model.coarse_shape[0]) or (
            dataset.coarse_shape[1] > model.coarse_shape[1]
        ):
            evaluator_model = PatchPredictor(
                model=model,
                coarse_yx_patch_extent=model.coarse_shape,
                coarse_horizontal_overlap=self.patch.coarse_horizontal_overlap,
            )
        else:
            evaluator_model = model

        return EventEvaluator(
            event_name=event_config.name,
            data=dataset,
            model=evaluator_model,
            experiment_dir=self.experiment_dir,
            n_samples=event_config.n_samples,
            save_generated_samples=event_config.save_generated_samples,
        )

    def build(self) -> list[Evaluator | EventEvaluator]:
        default_evaluator = self._build_default_evaluator()
        event_evaluators = []
        for event_config in self.events or []:
            event_evaluator = self._build_event_evaluator(event_config)
            event_evaluators.append(event_evaluator)
        return [default_evaluator] + event_evaluators


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    evaluator_config: EvaluatorConfig = dacite.from_dict(
        data_class=EvaluatorConfig,
        data=config,
        config=dacite.Config(strict=True),
    )
    prepare_directory(evaluator_config.experiment_dir, config)

    evaluator_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    evaluator_config.configure_wandb(resumable=True, notes=beaker_url)

    logging.info("Starting downscaling model evaluation")
    evaluators = evaluator_config.build()
    logging.info(
        f"Number of parameters: {count_parameters(evaluators[0].model.modules)}"
    )
    for evaluator in evaluators:
        evaluator.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling evaluation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
