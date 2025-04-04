import argparse
import dataclasses
import logging
from typing import Any, List, Literal, Mapping, Union

import dacite
import torch
import yaml

import fme.core.logging_utils as logging_utils
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import GenerationAggregator
from fme.downscaling.datasets_new import DataLoaderConfig, GriddedData, PairedBatchData
from fme.downscaling.models import (
    DiffusionModel,
    DiffusionModelConfig,
    DownscalingModelConfig,
    Model,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.requirements import DataRequirements
from fme.downscaling.train import count_parameters


@dataclasses.dataclass
class InterpolateModelConfig:
    mode: Literal["bicubic", "nearest"]
    downscale_factor: int
    in_names: List[str]
    out_names: List[str]

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
    wrapper: Union[DownscalingModelConfig, DiffusionModelConfig]

    @classmethod
    def from_state(
        cls, state: Mapping[str, Any]
    ) -> Union[DownscalingModelConfig, DiffusionModelConfig]:
        return dacite.from_dict(
            data={"wrapper": state}, data_class=cls, config=dacite.Config(strict=True)
        ).wrapper


@dataclasses.dataclass
class CheckpointModelConfig:
    checkpoint: str

    def __post_init__(self) -> None:
        self.checkpoint_dict: Mapping[str, Any] = torch.load(
            self.checkpoint, weights_only=False
        )

    def build(
        self,
    ) -> Union[Model, DiffusionModel]:
        model = _CheckpointModelConfigSelector.from_state(
            self.checkpoint_dict["model"]["config"]
        ).build(
            coarse_shape=self.checkpoint_dict["model"]["coarse_shape"],
            downscale_factor=self.checkpoint_dict["model"]["downscale_factor"],
        )
        model.module.load_state_dict(self.checkpoint_dict["model"]["module"])
        return model

    @property
    def data_requirements(self) -> DataRequirements:
        in_names = self.checkpoint_dict["model"]["config"]["in_names"]
        out_names = self.checkpoint_dict["model"]["config"]["out_names"]
        return DataRequirements(
            fine_names=out_names,
            coarse_names=list(set(in_names).union(out_names)),
            n_timesteps=1,
        )


class Evaluator:
    def __init__(
        self,
        data: GriddedData,
        experiment_dir: str,
        n_samples: int,
        model: Union[Model, DiffusionModel],
    ) -> None:
        self.data = data
        self.model = model
        self.experiment_dir = experiment_dir
        self.n_samples = n_samples
        self.dist = Distributed.get_instance()

    def run(self):
        aggregator = GenerationAggregator(
            self.data.dims,
            self.model.downscale_factor,
        )

        batch: PairedBatchData
        for batch_idx, batch in enumerate(self.data.loader):
            logging.info(f"Processing batch {batch_idx} of {len(self.data.loader)}")
            with torch.no_grad():
                logging.info("Generating predictions")
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


@dataclasses.dataclass
class EvaluatorConfig:
    model: CheckpointModelConfig
    experiment_dir: str
    data: DataLoaderConfig
    logging: LoggingConfig
    n_samples: int = 4

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = False, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )

    def build(self) -> Evaluator:
        dataset = self.data.build(
            train=False, requirements=self.model.data_requirements
        )

        model = self.model.build()
        return Evaluator(
            data=dataset,
            experiment_dir=self.experiment_dir,
            n_samples=self.n_samples,
            model=model,
        )


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    evaluator_config: EvaluatorConfig = dacite.from_dict(
        data_class=EvaluatorConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    evaluator_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    evaluator_config.configure_wandb(resumable=True, notes=beaker_url)

    logging.info("Starting downscaling model evaluation")
    evaluator = evaluator_config.build()
    logging.info(f"Number of parameters: {count_parameters(evaluator.model.modules)}")
    evaluator.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling evaluation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
