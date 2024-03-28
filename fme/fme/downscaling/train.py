import argparse
import dataclasses
import logging
import os
from typing import Optional

import dacite
import torch
import yaml

from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.typing_ import TensorMapping
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator
from fme.downscaling.datasets import BatchData, DataLoaderConfig, GriddedData
from fme.downscaling.models import DownscalingModelConfig, Model
from fme.downscaling.typing_ import HighResLowResPair
from fme.fcn_training.train_config import LoggingConfig
from fme.fcn_training.utils import logging_utils


def squeeze_time_dim(x: TensorMapping) -> TensorMapping:
    return {k: v.squeeze(dim=-3) for k, v in x.items()}  # (b, t=1, h, w) -> (b, h, w)


class Trainer:
    def __init__(
        self,
        model: Model,
        optimization: Optimization,
        train_data: GriddedData,
        validation_data: GriddedData,
        num_epochs: int,
        checkpoint_dir: Optional[str],
    ) -> None:
        self.model = model
        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.area_weights = self.train_data.area_weights.highres.cpu()
        self.latitudes = self.train_data.horizontal_coordinates.highres.lat.cpu()
        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)
        self._num_batches_seen = 0

        self._best_valid_loss = float("inf")
        self.epoch_checkpoint_path: Optional[str] = None
        self.best_checkpoint_path: Optional[str] = None
        if self.checkpoint_dir is not None:
            self.epoch_checkpoint_path = os.path.join(
                self.checkpoint_dir, "latest.ckpt"
            )
            self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.ckpt")

    def train_one_epoch(self) -> None:
        train_aggregator = Aggregator(self.area_weights, self.latitudes)
        batch: BatchData
        wandb = WandB.get_instance()
        for batch in self.train_data.loader:
            inputs = HighResLowResPair(
                squeeze_time_dim(batch.highres), squeeze_time_dim(batch.lowres)
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

    def valid_one_epoch(self) -> float:
        with torch.no_grad():
            validation_aggregator = Aggregator(self.area_weights, self.latitudes)
            batch: BatchData
            for batch in self.validation_data.loader:
                inputs = HighResLowResPair(
                    squeeze_time_dim(batch.highres),
                    squeeze_time_dim(batch.lowres),
                )
                outputs = self.model.run_on_batch(inputs, self.null_optimization)
                validation_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )
        wandb = WandB.get_instance()
        metrics = validation_aggregator.get_wandb(prefix="validation")
        wandb.log(
            metrics,
            self._num_batches_seen,
        )
        return metrics["validation/loss"]

    def save_checkpoint(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.get_state(),
                "num_batches_seen": self._num_batches_seen,
                "num_epochs": self.num_epochs,
                "optimization": self.optimization.get_state(),
            },
            path,
        )

    def save_checkpoints(self, valid_loss: float) -> None:
        if (
            self.epoch_checkpoint_path is not None
            and self.best_checkpoint_path is not None
        ):
            logging.info(f"Saving latest checkpoint")
            self.save_checkpoint(self.epoch_checkpoint_path)
            if valid_loss < self._best_valid_loss:
                logging.info(f"Saving best checkpoint")
                self._best_valid_loss = valid_loss
                self.save_checkpoint(self.best_checkpoint_path)

    def train(self) -> None:
        logging.info("Running metrics on validation data.")
        self.valid_one_epoch()
        for epoch in range(self.num_epochs):
            logging.info(f"Training epoch: {epoch+1}")
            self.train_one_epoch()
            logging.info("Running metrics on validation data.")
            valid_loss = self.valid_one_epoch()

            dist = Distributed.get_instance()
            if dist.is_root():
                self.save_checkpoints(valid_loss)


@dataclasses.dataclass
class TrainerConfig:
    model: DownscalingModelConfig
    optimization: OptimizationConfig
    train_data: DataLoaderConfig
    validation_data: DataLoaderConfig
    num_epochs: int
    experiment_dir: str
    save_checkpoints: bool
    logging: LoggingConfig

    @property
    def checkpoint_dir(self) -> Optional[str]:
        if self.save_checkpoints:
            return os.path.join(self.experiment_dir, "checkpoints")
        else:
            return None

    def build(self) -> Trainer:
        all_names = list(set(self.model.in_names).union(set(self.model.out_names)))
        train_data: GriddedData = self.train_data.build(train=True, var_names=all_names)
        validation_data: GriddedData = self.validation_data.build(
            train=False, var_names=all_names
        )

        downscaling_model = self.model.build(
            train_data.img_shape.lowres,
            train_data.downscale_factor,
            train_data.area_weights,
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
            self.checkpoint_dir,
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

    dist = Distributed.get_instance()
    if dist.is_root():
        if not os.path.isdir(train_config.experiment_dir):
            os.makedirs(train_config.experiment_dir)
        if train_config.checkpoint_dir is not None and not os.path.isdir(
            train_config.checkpoint_dir
        ):
            os.makedirs(train_config.checkpoint_dir)

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
