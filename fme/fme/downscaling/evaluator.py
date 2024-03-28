import abc
import argparse
import dataclasses
import logging
from typing import List, Literal, Optional, Union

import dacite
import torch
import yaml

from fme.core.dicts import to_flat_dict
from fme.core.loss import LossConfig
from fme.core.normalizer import NormalizationConfig
from fme.core.optimization import NullOptimization
from fme.core.wandb import WandB
from fme.downscaling import train
from fme.downscaling.aggregators import Aggregator
from fme.downscaling.datasets import DataLoaderConfig, GriddedData
from fme.downscaling.models import (
    DownscalingModelConfig,
    Model,
    PairedNormalizationConfig,
)
from fme.downscaling.modules.registry import ModuleRegistrySelector
from fme.downscaling.typing_ import HighResLowResPair
from fme.fcn_training.train_config import LoggingConfig
from fme.fcn_training.utils import logging_utils


class Evaluator:
    def __init__(
        self, data: GriddedData, model: Model, experiment_dir: Optional[str]
    ) -> None:
        self.data = data
        self.model = model
        self.optimization = NullOptimization()
        self.experiment_dir = experiment_dir

    def run(self):
        aggregator = Aggregator(
            self.data.area_weights.highres,
            self.data.horizontal_coordinates.highres.lat.cpu(),
        )

        for batch in self.data.loader:
            inputs = HighResLowResPair(
                train.squeeze_time_dim(batch.highres),
                train.squeeze_time_dim(batch.lowres),
            )
            with torch.no_grad():
                outputs = self.model.run_on_batch(inputs, self.optimization)
                aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )

        logs = aggregator.get_wandb()
        wandb = WandB.get_instance()
        wandb.log(logs, step=0)

        datasets = aggregator.get_datasets()
        for ds_name in datasets:
            datasets[ds_name].to_netcdf(
                f"{self.experiment_dir}/{ds_name}_diagnostics.nc"
            )


class Config(abc.ABC):
    @abc.abstractmethod
    def build(self) -> Model:
        pass


@dataclasses.dataclass
class InterpolateModelConfig(Config):
    mode: Literal["bicubic", "nearest"]
    downscale_factor: int
    in_names: List[str]
    out_names: List[str]

    def build(self) -> Model:
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

        area_weights = HighResLowResPair(torch.tensor(1.0), torch.tensor(1.0))

        return DownscalingModelConfig(
            module,
            LossConfig("NaN"),
            self.in_names,
            self.out_names,
            normalization_config,
        ).build((-1, -1), self.downscale_factor, area_weights)


@dataclasses.dataclass
class CheckpointModelConfig(Config):
    checkpoint: str

    def build(self) -> Model:
        raise NotImplementedError(
            "Evaluating a checkpointed model is not yet implemented."
        )


@dataclasses.dataclass
class EvaluatorConfig:
    model: Union[InterpolateModelConfig, CheckpointModelConfig]
    experiment_dir: str
    data: DataLoaderConfig
    logging: LoggingConfig

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        if "environment" in config:
            logging.warning("Not recording env vars since 'environment' is in config.")
        elif env_vars is not None:
            config["environment"] = env_vars
        self.logging.configure_wandb(config=config, **kwargs)

    def build(self) -> Evaluator:
        model = self.model.build()
        in_names = model.in_packer.names
        out_names = model.out_packer.names
        var_names = list(set(in_names).union(set(out_names)))
        return Evaluator(
            self.data.build(False, var_names, None), model, self.experiment_dir
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
    evaluator_config.configure_wandb(resume=True, notes=beaker_url)

    logging.info("Starting downscaling model evaluation")
    evaluator = evaluator_config.build()
    logging.info(f"Number of parameters: {evaluator.model.count_parameters()}")
    evaluator.run()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling evaluation script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
