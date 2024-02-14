import dataclasses
import logging
import sys

import dacite
import torch
import yaml

from fme.core.dicts import to_flat_dict
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator
from fme.downscaling.datasets import BatchData, DataLoaderParams, DownscalingDataLoader
from fme.downscaling.models import DownscalingModelConfig, Model
from fme.downscaling.typing_ import HighResLowResPair
from fme.fcn_training.train_config import LoggingConfig
from fme.fcn_training.utils import logging_utils


def _squeeze_time_dim(x: TensorMapping) -> TensorMapping:
    return {k: v.squeeze(dim=-3) for k, v in x.items()}  # (b, t=1, h, w) -> (b, h, w)


class Trainer:
    def __init__(
        self,
        model: Model,
        optimization: Optimization,
        train_loader: DownscalingDataLoader,
        validation_loader: DownscalingDataLoader,
        num_epochs: int,
    ) -> None:
        self.model = model

        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.num_epochs = num_epochs
        self.area_weights = self.train_loader.area_weights.highres.cpu()
        self.latitudes = self.validation_loader.horizontal_coordinates.highres.lat.cpu()
        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)
        self._num_batches_seen = 0

    def train_one_epoch(self) -> None:
        train_aggregator = Aggregator(self.area_weights, self.latitudes)
        batch: BatchData
        for batch in self.train_loader.loader:
            inputs = HighResLowResPair(
                _squeeze_time_dim(batch.highres), _squeeze_time_dim(batch.lowres)
            )
            outputs = self.model.run_on_batch(inputs, self.optimization)
            with torch.no_grad():
                train_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )
                self._num_batches_seen += 1
                wandb = WandB.get_instance()
                wandb.log(
                    train_aggregator.get_wandb(prefix="train"),
                    step=self._num_batches_seen,
                )

    def valid_one_epoch(self) -> None:
        with torch.no_grad():
            validation_aggregator = Aggregator(self.area_weights, self.latitudes)
            batch: BatchData
            for batch in self.validation_loader.loader:
                inputs = HighResLowResPair(
                    _squeeze_time_dim(batch.highres),
                    _squeeze_time_dim(batch.lowres),
                )
                outputs = self.model.run_on_batch(inputs, self.null_optimization)
                validation_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                ),
        wandb = WandB.get_instance()
        wandb.log(
            validation_aggregator.get_wandb(prefix="validation"),
            self._num_batches_seen,
        )

    def train(self) -> None:
        for epoch in range(self.num_epochs):
            logging.info(f"Epoch: {epoch+1}")
            self.valid_one_epoch()
            self.train_one_epoch()

        self.valid_one_epoch()


@dataclasses.dataclass
class TrainerConfig:
    model: DownscalingModelConfig
    optimization: OptimizationConfig
    train_data: DataLoaderParams
    validation_data: DataLoaderParams
    num_epochs: int
    experiment_dir: str
    logging: LoggingConfig

    def build(self) -> Trainer:
        all_names = list(set(self.model.in_names).union(set(self.model.out_names)))
        train_data_loader: DownscalingDataLoader = self.train_data.build(
            train=True, var_names=all_names
        )
        valid_data_loader: DownscalingDataLoader = self.validation_data.build(
            train=False, var_names=all_names
        )

        downscaling_model = self.model.build(
            train_data_loader.img_shape.lowres,
            train_data_loader.downscale_factor,
        )

        optimization = self.optimization.build(
            downscaling_model.module.parameters(), self.num_epochs
        )

        return Trainer(
            downscaling_model,
            optimization,
            train_data_loader,
            valid_data_loader,
            self.num_epochs,
        )

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


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_config: TrainerConfig = dacite.from_dict(
        data_class=TrainerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    train_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    train_config.configure_wandb(resume=True, notes=beaker_url)

    logging.info("Starting training")
    trainer = train_config.build()
    trainer.train()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_path>")
        sys.exit(-1)
    main(sys.argv[1])
