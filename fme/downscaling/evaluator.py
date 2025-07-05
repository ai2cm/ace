import argparse
import dataclasses
import logging
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

import dacite
import torch
import yaml

import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_directory
from fme.core.dataset.config import TimeSlice
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import GenerationAggregator, SampleAggregator
from fme.downscaling.datasets import (
    ClosedInterval,
    PairedBatchData,
    PairedDataLoaderConfig,
    PairedGriddedData,
)
from fme.downscaling.models import (
    DiffusionModel,
    DiffusionModelConfig,
    DownscalingModelConfig,
    Model,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.patching import PatchPredictor, paired_patch_generator_from_loader
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.train import count_parameters
from fme.downscaling.typing_ import FineResCoarseResPair


@dataclasses.dataclass
class InterpolateModelConfig:
    mode: Literal["bicubic", "nearest"]
    downscale_factor: int
    in_names: list[str]
    out_names: list[str]

    def build(
        self,
    ) -> Model:
        module = ModuleRegistrySelector(type="interpolate", config={"mode": self.mode})
        var_names = list(set(self.in_names).union(set(self.out_names)))
        normalization_config = PairedNormalizationConfig(
            NormalizationConfig(
                means={var_name: 0.0 for var_name in var_names},
                stds={var_name: 1.0 for var_name in var_names},
            ),
            NormalizationConfig(
                means={var_name: 0.0 for var_name in var_names},
                stds={var_name: 1.0 for var_name in var_names},
            ),
        )

        return DownscalingModelConfig(
            module,
            LossConfig("NaN"),
            self.in_names,
            self.out_names,
            normalization_config,
        ).build(
            (-1, -1),
            self.downscale_factor,
        )

    @property
    def data_requirements(self) -> DataRequirements:
        return DataRequirements(
            fine_names=self.out_names,
            coarse_names=list(set(self.in_names).union(self.out_names)),
            n_timesteps=1,
        )


@dataclasses.dataclass
class _CheckpointModelConfigSelector:
    wrapper: DownscalingModelConfig | DiffusionModelConfig

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any]
    ) -> DownscalingModelConfig | DiffusionModelConfig:
        return dacite.from_dict(
            data={"wrapper": state}, data_class=cls, config=dacite.Config(strict=True)
        ).wrapper


@dataclasses.dataclass
class CheckpointModelConfig:
    checkpoint_path: str

    def __post_init__(self) -> None:
        # For config validation testing, we don't want to load immediately
        # so we defer until build or properties are accessed.
        self._checkpoint_is_loaded = False

    @property
    def _checkpoint(self) -> Mapping[str, Any]:
        if not self._checkpoint_is_loaded:
            self._checkpoint_data = torch.load(self.checkpoint_path, weights_only=False)
            self._checkpoint_is_loaded = True
        return self._checkpoint_data

    def build(
        self,
    ) -> Model | DiffusionModel:
        model = _CheckpointModelConfigSelector.from_state(
            self._checkpoint["model"]["config"]
        ).build(
            coarse_shape=self._checkpoint["model"]["coarse_shape"],
            downscale_factor=self._checkpoint["model"]["downscale_factor"],
        )
        model.module.load_state_dict(self._checkpoint["model"]["module"])
        return model

    @property
    def data_requirements(self) -> DataRequirements:
        in_names = self._checkpoint["model"]["config"]["in_names"]
        out_names = self._checkpoint["model"]["config"]["out_names"]
        return DataRequirements(
            fine_names=out_names,
            coarse_names=list(set(in_names).union(out_names)),
            n_timesteps=1,
            use_fine_topography=self._checkpoint["model"]["config"][
                "use_fine_topography"
            ],
        )


class Evaluator:
    def __init__(
        self,
        data: PairedGriddedData,
        model: Model | DiffusionModel | PatchPredictor,
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
        )

        if self.patch_data:
            batch_generator = paired_patch_generator_from_loader(
                loader=self.data.loader,
                coarse_yx_extent=self.data.coarse_shape,
                coarse_yx_patch_extents=self.model.coarse_shape,
                downscale_factor=self.model.downscale_factor,
            )
        else:
            batch_generator = self.data.loader

        for i, batch in enumerate(batch_generator):
            with torch.no_grad():
                logging.info(f"Generating predictions on batch {i + 1}")
                outputs = self.model.generate_on_batch(batch, n_samples=self.n_samples)
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


class EventEvaluator:
    def __init__(
        self,
        event_name: str,
        data: PairedGriddedData,
        model: Model | DiffusionModel | PatchPredictor,
        experiment_dir: str,
        n_samples: int,
    ) -> None:
        self.event_name = event_name
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()
        self._max_sample_group = 8

    def run(self):
        logging.info(f"Running {self.event_name} event evaluation")
        batch: PairedBatchData = next(iter(self.data.loader))

        sample_agg = SampleAggregator(
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
            outputs = self.model.generate_on_batch(batch, n_samples=end_idx - start_idx)
            sample_agg.record_batch(outputs.prediction)

        to_log = sample_agg.get_wandb()
        wandb = WandB.get_instance()
        wandb.log({f"{self.event_name}/{k}": v for k, v in to_log.items()}, step=0)
        torch.cuda.empty_cache()


@dataclasses.dataclass
class MultipatchConfig:
    """
    Configuration to enable predictions on multiple patches for evaluation.

    Args:
        divide_evaluation: enables the patched prediction of the full
            input data extent for evaluation.
        composite_prediction: if True, recombines the smaller prediction
            regions into the original full region as a single sample.
        coarse_horizontal_overlap: number of pixels to overlap in the
            coarse data.
    """

    divide_evaluation: bool = False
    composite_prediction: bool = False
    coarse_horizontal_overlap: int = 1


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

    def get_gridded_data(
        self, base_data_config: PairedDataLoaderConfig, requirements: DataRequirements
    ) -> PairedGriddedData:
        # Event evaluation only load the first snapshot.
        # Filling the slice stop isn't necessary but guards against
        # future code trying to iterate over the entire dataloader.
        _stop = (
            datetime.strptime(self.date, self.date_format) + timedelta(hours=6)
        ).strftime(self.date_format)

        time_slice = TimeSlice(self.date, _stop)

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
        return event_data_config.build(train=False, requirements=requirements)


@dataclasses.dataclass
class EvaluatorConfig:
    model: CheckpointModelConfig
    experiment_dir: str
    data: PairedDataLoaderConfig
    logging: LoggingConfig
    n_samples: int = 4
    patch: MultipatchConfig = dataclasses.field(default_factory=MultipatchConfig)
    events: list[EventConfig] | None = None

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def _build_default_evaluator(self) -> Evaluator:
        dataset = self.data.build(
            train=False, requirements=self.model.data_requirements
        )

        model = self.model.build()
        evaluator_model: Model | DiffusionModel | PatchPredictor
        if self.patch.divide_evaluation and self.patch.composite_prediction:
            evaluator_model = PatchPredictor(
                model,
                dataset.coarse_shape,
                coarse_horizontal_overlap=self.patch.coarse_horizontal_overlap,
            )
        else:
            evaluator_model = model

        if self.patch.divide_evaluation and not self.patch.composite_prediction:
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
        event_config: EventConfig,
    ) -> EventEvaluator:
        model = self.model.build()
        evaluator_model: Model | DiffusionModel | PatchPredictor

        dataset = event_config.get_gridded_data(
            base_data_config=self.data, requirements=self.model.data_requirements
        )

        if (dataset.coarse_shape[0] > model.coarse_shape[0]) or (
            dataset.coarse_shape[1] > model.coarse_shape[1]
        ):
            evaluator_model = PatchPredictor(
                model,
                dataset.coarse_shape,
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
