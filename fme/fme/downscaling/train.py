import argparse
import dataclasses
import logging
import os
from typing import Optional, Union

import dacite
import torch
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator
from fme.downscaling.datasets import BatchData, DataLoaderConfig, GriddedData
from fme.downscaling.models import (
    DiffusionModel,
    DiffusionModelConfig,
    DownscalingModelConfig,
    Model,
)
from fme.downscaling.typing_ import FineResCoarseResPair


def count_parameters(modules: torch.nn.ModuleList) -> int:
    parameters = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters += parameter.numel()
    return parameters


def save_checkpoint(trainer: "Trainer", path: str) -> None:
    torch.save(
        {
            "model": trainer.model.get_state(),
            "optimization": trainer.optimization.get_state(),
            "num_batches_seen": trainer.num_batches_seen,
            "startEpoch": trainer.startEpoch,
            "best_valid_loss": trainer.best_valid_loss,
        },
        path,
    )


def restore_checkpoint(trainer: "Trainer", checkpoint_path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
    trainer.model = trainer.model.from_state(
        checkpoint["model"], trainer.area_weights, trainer.fine_topography
    )
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["startEpoch"]
    trainer.best_valid_loss = checkpoint["best_valid_loss"]


class Trainer:
    def __init__(
        self,
        model: Union[Model, DiffusionModel],
        optimization: Optimization,
        train_data: GriddedData,
        validation_data: GriddedData,
        max_epochs: int,
        segment_epochs: Optional[int],
        checkpoint_dir: Optional[str],
    ) -> None:
        self.model = model
        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.area_weights = self.train_data.area_weights
        self.latitudes = self.train_data.horizontal_coordinates.fine.lat.cpu()
        self.fine_topography = self.train_data.fine_topography
        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)
        self.num_batches_seen = 0

        self.startEpoch = 0
        self.segment_epochs = segment_epochs

        self.best_valid_loss = float("inf")
        self.epoch_checkpoint_path: Optional[str] = None
        self.best_checkpoint_path: Optional[str] = None
        if self.checkpoint_dir is not None:
            self.epoch_checkpoint_path = os.path.join(
                self.checkpoint_dir, "latest.ckpt"
            )
            self.best_checkpoint_path = os.path.join(self.checkpoint_dir, "best.ckpt")

    def train_one_epoch(self) -> None:
        train_aggregator = Aggregator(self.area_weights.fine.cpu(), self.latitudes)
        batch: BatchData
        wandb = WandB.get_instance()
        for batch in self.train_data.loader:
            inputs = FineResCoarseResPair(batch.fine, batch.coarse)
            outputs = self.model.train_on_batch(inputs, self.optimization)
            self.num_batches_seen += 1
            with torch.no_grad():
                train_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )
                wandb.log(
                    {"train/batch_loss": outputs.loss.detach().cpu().numpy()},
                    step=self.num_batches_seen,
                )

        with torch.no_grad():
            wandb.log(
                train_aggregator.get_wandb(prefix="train"),
                step=self.num_batches_seen,
            )

    def valid_one_epoch(self) -> float:
        with torch.no_grad():
            validation_aggregator = Aggregator(
                self.area_weights.fine.cpu(), self.latitudes
            )
            batch: BatchData
            for batch in self.validation_data.loader:
                inputs = FineResCoarseResPair(
                    batch.fine,
                    batch.coarse,
                )
                outputs = self.model.generate_on_batch(inputs)
                validation_aggregator.record_batch(
                    outputs.loss, outputs.target, outputs.prediction
                )
        wandb = WandB.get_instance()
        metrics = validation_aggregator.get_wandb(prefix="validation")
        wandb.log(
            metrics,
            self.num_batches_seen,
        )
        return metrics["validation/loss"]

    @property
    def resuming(self) -> bool:
        if self.epoch_checkpoint_path is None:
            return False
        return os.path.isfile(self.epoch_checkpoint_path)

    def save_checkpoints(self, valid_loss: float) -> None:
        if (
            self.epoch_checkpoint_path is not None
            and self.best_checkpoint_path is not None
        ):
            logging.info(f"Saving latest checkpoint")
            save_checkpoint(self, self.epoch_checkpoint_path)
            if valid_loss < self.best_valid_loss:
                logging.info(f"Saving best checkpoint")
                self.best_valid_loss = valid_loss
                save_checkpoint(self, self.best_checkpoint_path)

    def train(self) -> None:
        logging.info("Running metrics on validation data.")
        self.valid_one_epoch()
        wandb = WandB.get_instance()

        if self.segment_epochs is None:
            segment_max_epochs = self.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.segment_epochs, self.max_epochs
            )

        for epoch in range(self.startEpoch, segment_max_epochs):
            self.startEpoch = epoch
            logging.info(f"Training epoch: {epoch+1}")
            self.train_one_epoch()
            logging.info("Running metrics on validation data.")
            valid_loss = self.valid_one_epoch()
            wandb.log({"epoch": epoch}, step=self.num_batches_seen)

            dist = Distributed.get_instance()
            if dist.is_root():
                self.save_checkpoints(valid_loss)


@dataclasses.dataclass
class TrainerConfig:
    model: Union[DownscalingModelConfig, DiffusionModelConfig]
    optimization: OptimizationConfig
    train_data: DataLoaderConfig
    validation_data: DataLoaderConfig
    max_epochs: int
    experiment_dir: str
    save_checkpoints: bool
    logging: LoggingConfig
    segment_epochs: Optional[int] = None

    @property
    def checkpoint_dir(self) -> Optional[str]:
        if self.save_checkpoints:
            return os.path.join(self.experiment_dir, "checkpoints")
        else:
            return None

    def build(self) -> Trainer:
        train_data: GriddedData = self.train_data.build(
            train=True, requirements=self.model.data_requirements
        )
        validation_data: GriddedData = self.validation_data.build(
            train=False, requirements=self.model.data_requirements
        )

        downscaling_model = self.model.build(
            train_data.img_shape.coarse,
            train_data.downscale_factor,
            train_data.area_weights,
            train_data.fine_topography,
        )

        optimization = self.optimization.build(
            downscaling_model.module.parameters(), self.max_epochs
        )

        return Trainer(
            downscaling_model,
            optimization,
            train_data,
            validation_data,
            self.max_epochs,
            self.segment_epochs,
            self.checkpoint_dir,
        )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(config=config, env_vars=env_vars, **kwargs)


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

    if trainer.resuming:
        logging.info(f"Resuming training from {trainer.epoch_checkpoint_path}")
        restore_checkpoint(trainer, trainer.epoch_checkpoint_path)

    logging.info(f"Number of parameters: {count_parameters(trainer.model.modules)}")
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling train script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
