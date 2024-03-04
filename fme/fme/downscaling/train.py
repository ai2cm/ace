import argparse
import dataclasses
import logging

import dacite
import torch
import yaml

from fme.core.dicts import to_flat_dict
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator
from fme.downscaling.datasets import BatchData, DataLoaderConfig, GriddedData
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
        train_data: GriddedData,
        validation_data: GriddedData,
        num_epochs: int,
    ) -> None:
        self.model = model

        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.num_epochs = num_epochs
        self.area_weights = self.train_data.area_weights.highres.cpu()
        self.latitudes = self.train_data.horizontal_coordinates.highres.lat.cpu()
        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)
        self._num_batches_seen = 0

    def train_one_epoch(self) -> None:
        train_aggregator = Aggregator(self.area_weights, self.latitudes)
        batch: BatchData
        wandb = WandB.get_instance()
        for batch in self.train_data.loader:
            inputs = HighResLowResPair(
                _squeeze_time_dim(batch.highres), _squeeze_time_dim(batch.lowres)
            )
            outputs = self.model.run_on_batch(inputs, self.optimization)
            self._num_batches_seen += 1
            with torch.no_grad():
                train_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )
                wandb.log(
                    {"train/batch_loss": outputs.loss.detach().cpu().numpy()},
                    step=self._num_batches_seen,
                )

        with torch.no_grad():
            wandb.log(
                train_aggregator.get_wandb(prefix="train"),
                step=self._num_batches_seen,
            )

    def valid_one_epoch(self) -> None:
        with torch.no_grad():
            validation_aggregator = Aggregator(self.area_weights, self.latitudes)
            batch: BatchData
            for batch in self.validation_data.loader:
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
        logging.info("Running metrics on validation data.")
        self.valid_one_epoch()
        for epoch in range(self.num_epochs):
            logging.info(f"Training epoch: {epoch+1}")
            self.train_one_epoch()
            logging.info("Running metrics on validation data.")
            self.valid_one_epoch()


@dataclasses.dataclass
class TrainerConfig:
    model: DownscalingModelConfig
    optimization: OptimizationConfig
    train_data: DataLoaderConfig
    validation_data: DataLoaderConfig
    num_epochs: int
    experiment_dir: str
    logging: LoggingConfig

    def build(self) -> Trainer:
        all_names = list(set(self.model.in_names).union(set(self.model.out_names)))
        train_data: GriddedData = self.train_data.build(train=True, var_names=all_names)
        validation_data: GriddedData = self.validation_data.build(
            train=False, var_names=all_names
        )

        downscaling_model = self.model.build(
            train_data.img_shape.lowres,
            train_data.downscale_factor,
        )

        optimization = self.optimization.build(
            downscaling_model.module.parameters(), self.num_epochs
        )

        return Trainer(
            downscaling_model,
            optimization,
            train_data,
            validation_data,
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
    logging.info(f"Number of parameters: {trainer.model.count_parameters()}")
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling train script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
