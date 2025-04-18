import argparse
import contextlib
import dataclasses
import logging
import os
import warnings
from typing import Optional, Union

import dacite
import torch
import yaml

import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_directory
from fme.core.device import get_device
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator, GenerationAggregator
from fme.downscaling.datasets import DataLoaderConfig, GriddedData, PairedBatchData
from fme.downscaling.models import (
    DiffusionModel,
    DiffusionModelConfig,
    DownscalingModelConfig,
    Model,
)
from fme.downscaling.patching import paired_patch_generator_from_loader


def count_parameters(modules: torch.nn.ModuleList) -> int:
    parameters = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters += parameter.numel()
    return parameters


def _save_checkpoint(trainer: "Trainer", path: str) -> None:
    torch.save(
        {
            "model": trainer.model.get_state(),
            "ema": trainer.ema.get_state(),
            "optimization": trainer.optimization.get_state(),
            "num_batches_seen": trainer.num_batches_seen,
            "startEpoch": trainer.startEpoch,
            "best_valid_loss": trainer.best_valid_loss,
            "validate_using_ema": trainer.validate_using_ema,
        },
        path,
    )


def restore_checkpoint(trainer: "Trainer") -> None:
    if trainer.epoch_checkpoint_path is None:
        raise ValueError("Cannot restore checkpoint without a checkpoint path")

    checkpoint = torch.load(
        trainer.epoch_checkpoint_path, map_location=get_device(), weights_only=False
    )
    trainer.model = trainer.model.from_state(checkpoint["model"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["startEpoch"]
    trainer.best_valid_loss = checkpoint["best_valid_loss"]

    trainer.validate_using_ema = checkpoint["validate_using_ema"]
    ema_checkpoint = torch.load(
        trainer.ema_checkpoint_path, map_location=get_device(), weights_only=False
    )
    ema_model = trainer.model.from_state(ema_checkpoint["model"])
    trainer.ema = EMATracker.from_state(ema_checkpoint["ema"], ema_model.modules)


class Trainer:
    def __init__(
        self,
        model: Union[Model, DiffusionModel],
        optimization: Optimization,
        train_data: GriddedData,
        validation_data: GriddedData,
        config: "TrainerConfig",
    ) -> None:
        self.model = model
        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.ema = config.ema.build(self.model.modules)
        self.validate_using_ema = config.validate_using_ema
        self.dims = self.train_data.dims
        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)
        self.num_batches_seen = 0
        self.config = config
        self.patch_data = (
            True
            if (config.coarse_patch_extent_lat and config.coarse_patch_extent_lon)
            else False
        )

        self.startEpoch = 0
        self.segment_epochs = self.config.segment_epochs

        dist = Distributed.get_instance()
        if dist.is_root():
            if not os.path.isdir(self.config.experiment_dir):
                os.makedirs(self.config.experiment_dir)
            if self.config.checkpoint_dir is not None and not os.path.isdir(
                self.config.checkpoint_dir
            ):
                os.makedirs(self.config.checkpoint_dir)

        self.best_valid_loss = float("inf")
        self.epoch_checkpoint_path: Optional[str] = None
        self.best_checkpoint_path: Optional[str] = None
        if self.config.checkpoint_dir is not None:
            self.epoch_checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "latest.ckpt"
            )
            self.best_checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "best.ckpt"
            )
            self.ema_checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "ema_ckpt.tar"
            )

    def _get_batch_generator(self, data: GriddedData, random_offset: bool):
        if self.patch_data:
            batch_generator = paired_patch_generator_from_loader(
                loader=data.loader,
                coarse_yx_extent=data.coarse_shape,
                coarse_yx_patch_extents=self.model.coarse_shape,
                downscale_factor=self.model.downscale_factor,
                random_offset=random_offset,
            )
        else:
            batch_generator = data.loader
        return batch_generator

    def train_one_epoch(self) -> None:
        self.model.module.train()
        include_positional_comparisons = False if self.patch_data else True

        train_aggregator = Aggregator(
            self.dims,
            self.model.downscale_factor,
            include_positional_comparisons=include_positional_comparisons,
        )
        batch: PairedBatchData
        wandb = WandB.get_instance()
        train_batch_generator = self._get_batch_generator(
            self.train_data, random_offset=True
        )
        outputs = None
        for i, batch in enumerate(train_batch_generator):
            self.num_batches_seen += 1
            logging.info(f"Training on batch {i+1}")
            outputs = self.model.train_on_batch(batch, self.optimization)
            self.ema(self.model.modules)
            with torch.no_grad():
                train_aggregator.record_batch(
                    outputs=outputs,
                    coarse=batch.coarse.data,
                    batch=batch,
                )
                wandb.log(
                    {"train/batch_loss": outputs.loss.detach().cpu().numpy()},
                    step=self.num_batches_seen,
                )
        if outputs is None:
            raise RuntimeError("Empty training batch generator")
        self.optimization.step_scheduler(outputs.loss.item())
        with torch.no_grad():
            wandb.log(
                train_aggregator.get_wandb(prefix="train"),
                step=self.num_batches_seen,
            )

    @contextlib.contextmanager
    def _ema_context(self):
        """
        A context where the model uses the EMA model.
        """
        self.ema.store(parameters=self.model.modules.parameters())
        self.ema.copy_to(model=self.model.modules)  # type: ignore
        try:
            yield
        finally:
            self.ema.restore(parameters=self.model.modules.parameters())

    @contextlib.contextmanager
    def _validation_context(self):
        """
        The context for running validation.

        In this context, the model uses the EMA model if
        `self.config.validate_using_ema` is True.
        """
        if self.validate_using_ema:
            with self._ema_context():
                yield
        else:
            yield

    def valid_one_epoch(self) -> float:
        self.model.module.eval()
        if (
            self.patch_data
            and self.validation_data.coarse_shape != self.model.coarse_shape
        ):
            include_positional_comparisons = False
        else:
            include_positional_comparisons = True
        with torch.no_grad(), self._validation_context():
            validation_aggregator = Aggregator(
                self.dims,
                self.model.downscale_factor,
                include_positional_comparisons=include_positional_comparisons,
            )
            generation_aggregator = GenerationAggregator(
                self.dims,
                self.model.downscale_factor,
                include_positional_comparisons=include_positional_comparisons,
            )
            batch: PairedBatchData
            validation_batch_generator = self._get_batch_generator(
                self.validation_data, random_offset=False
            )
            for batch in validation_batch_generator:
                outputs = self.model.train_on_batch(batch, self.null_optimization)
                validation_aggregator.record_batch(
                    outputs=outputs,
                    coarse=batch.coarse.data,
                    batch=batch,
                )
                generated_outputs = self.model.generate_on_batch(
                    batch, n_samples=self.config.generate_n_samples
                )
                # Add sample dimension to coarse values for generation comparison
                coarse = {k: v.unsqueeze(1) for k, v in batch.coarse.data.items()}
                generation_aggregator.record_batch(
                    outputs=generated_outputs,
                    coarse=coarse,
                    batch=batch,
                )

        wandb = WandB.get_instance()
        validation_metrics = validation_aggregator.get_wandb(prefix="validation")
        generation_metrics = generation_aggregator.get_wandb(prefix="generation")

        wandb.log(
            {**generation_metrics, **validation_metrics},
            self.num_batches_seen,
        )
        channel_mean_crps = _get_crps(generation_metrics)

        return channel_mean_crps

    @property
    def resuming(self) -> bool:
        if self.epoch_checkpoint_path is None:
            return False
        return os.path.isfile(self.epoch_checkpoint_path)

    def save_all_checkpoints(self, valid_loss: float) -> None:
        if (
            self.epoch_checkpoint_path is not None
            and self.best_checkpoint_path is not None
        ):
            logging.info(f"Saving latest checkpoint")
            if self.validate_using_ema:
                best_checkpoint_context = self._ema_context
            else:
                best_checkpoint_context = contextlib.nullcontext  # type: ignore

            if valid_loss < self.best_valid_loss:
                logging.info(f"Saving best checkpoint")
                self.best_valid_loss = valid_loss
                with best_checkpoint_context():
                    _save_checkpoint(self, self.best_checkpoint_path)

            _save_checkpoint(self, self.epoch_checkpoint_path)

            with self._ema_context():
                _save_checkpoint(self, self.ema_checkpoint_path)

    def _validate_current_epoch(self, epoch: int) -> bool:
        valid_frequency = self.config.validate_interval
        if epoch % valid_frequency == 0:
            return True
        else:
            return False

    def train(self) -> None:
        logging.info("Running metrics on validation data.")
        self.valid_one_epoch()
        wandb = WandB.get_instance()

        if self.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.segment_epochs, self.config.max_epochs
            )

        for epoch in range(self.startEpoch, segment_max_epochs):
            self.startEpoch = epoch
            logging.info(f"Training epoch: {epoch + 1}")
            self.train_one_epoch()
            if self._validate_current_epoch(epoch):
                logging.info("Running metrics on validation data.")
                valid_loss = self.valid_one_epoch()
                wandb.log({"epoch": epoch}, step=self.num_batches_seen)

                dist = Distributed.get_instance()
                if dist.is_root():
                    self.save_all_checkpoints(valid_loss)


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
    ema: EMAConfig = dataclasses.field(default_factory=EMAConfig)
    validate_using_ema: bool = False
    generate_n_samples: int = 1
    segment_epochs: Optional[int] = None
    validate_interval: int = 1
    coarse_patch_extent_lat: Optional[int] = None
    coarse_patch_extent_lon: Optional[int] = None

    def __post_init__(self):
        if (
            self.coarse_patch_extent_lat is not None
            and self.coarse_patch_extent_lon is None
        ) or (
            self.coarse_patch_extent_lat is None
            and self.coarse_patch_extent_lon is not None
        ):
            raise ValueError(
                "Either none or both of coarse_patch_extent_lat and "
                "coarse_patch_extent_lon must be set."
            )

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_dir, "checkpoints")

    def build(self) -> Trainer:
        train_data: GriddedData = self.train_data.build(
            train=True, requirements=self.model.data_requirements
        )
        validation_data: GriddedData = self.validation_data.build(
            train=False, requirements=self.model.data_requirements
        )
        if self.coarse_patch_extent_lat and self.coarse_patch_extent_lon:
            model_coarse_shape = (
                self.coarse_patch_extent_lat,
                self.coarse_patch_extent_lon,
            )
        else:
            model_coarse_shape = train_data.coarse_shape
        downscaling_model = self.model.build(
            model_coarse_shape,
            train_data.downscale_factor,
        )

        optimization = self.optimization.build(
            modules=[downscaling_model.module],
            max_epochs=self.max_epochs,
        )

        return Trainer(
            downscaling_model,
            optimization,
            train_data,
            validation_data,
            self,
        )

    def configure_logging(self, log_filename: str):
        self.logging.configure_logging(self.experiment_dir, log_filename)

    def configure_wandb(self, resumable: bool = True, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        env_vars = logging_utils.retrieve_env_vars()
        self.logging.configure_wandb(
            config=config, env_vars=env_vars, resumable=resumable, **kwargs
        )


def _get_crps(metrics, prefix="generation/metrics/crps"):
    channel_crps = [v for k, v in metrics.items() if k.startswith(prefix)]

    if len(channel_crps) != 1:
        warnings.warn(
            "CRPS metric used for checkpoint selection is computed on "
            "denormalized outputs. If multiple outputs are present, "
            "the mean CRPS will not be evenly weighted across outputs."
        )
    if len(channel_crps) == 0:
        return float("inf")
    else:
        return sum(channel_crps) / len(channel_crps)


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    train_config: TrainerConfig = dacite.from_dict(
        data_class=TrainerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    prepare_directory(train_config.experiment_dir, config)
    train_config.configure_logging(log_filename="out.log")
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    train_config.configure_wandb(notes=beaker_url)

    logging.info("Starting training")
    trainer = train_config.build()

    if trainer.resuming:
        logging.info(f"Resuming training from {trainer.epoch_checkpoint_path}")
        restore_checkpoint(trainer)

    logging.info(f"Number of parameters: {count_parameters(trainer.model.modules)}")
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Downscaling train script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.config_path)
