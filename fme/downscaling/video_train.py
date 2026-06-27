"""Training entry point for the endpoint-conditioned video diffusion model.

Mirrors ``fme/downscaling/train.py`` (EMA, preemption-safe checkpointing,
distributed-aware, W&B logging) but drives the temporal-interpolation
``VideoDiffusionModel`` over ``PairedVideoGriddedData`` clips. Validation reports
the denoising loss plus an interior-frame generation MAE.
"""

import argparse
import contextlib
import dataclasses
import logging
import os
import shutil
import time
import uuid

import dacite
import torch
import yaml

from fme.core.cli import prepare_directory, remove_stale_tmp_checkpoints
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, OptimizationConfig
from fme.core.wandb import WandB
from fme.downscaling.data import PairedDataLoaderConfig, PairedVideoBatchData
from fme.downscaling.video_models import VideoDiffusionModelConfig


def _save_checkpoint(trainer: "VideoTrainer", path: str) -> None:
    temporary_location = os.path.join(os.path.dirname(path), f".{uuid.uuid4()}.tmp")
    try:
        torch.save(
            {
                "module": trainer.model.module.state_dict(),
                "ema": trainer.ema.get_state(),
                "optimization": trainer.optimization.get_state(),
                "num_batches_seen": trainer.num_batches_seen,
                "startEpoch": trainer.startEpoch,
                "best_valid_loss": trainer.best_valid_loss,
            },
            temporary_location,
        )
        os.replace(temporary_location, path)
    finally:
        if os.path.exists(temporary_location):
            os.remove(temporary_location)


def restore_checkpoint(trainer: "VideoTrainer") -> None:
    checkpoint = torch.load(
        trainer.epoch_checkpoint_path, map_location="cpu", weights_only=False
    )
    trainer.model.module.load_state_dict(checkpoint["module"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["startEpoch"]
    trainer.best_valid_loss = checkpoint["best_valid_loss"]
    trainer.ema = EMATracker.from_state(checkpoint["ema"], trainer.model.modules)


@dataclasses.dataclass
class VideoTrainerConfig:
    """Configuration for the video diffusion Trainer.

    ``train_data``/``validation_data`` must set ``n_timesteps`` equal to the
    model's ``n_timesteps`` (the clip length). For temporal-only interpolation
    the fine and coarse dataset entries point at the same (single-resolution)
    store -- coarse is loaded but ignored by the model.
    """

    model: VideoDiffusionModelConfig
    optimization: OptimizationConfig
    train_data: PairedDataLoaderConfig
    validation_data: PairedDataLoaderConfig
    max_epochs: int
    experiment_dir: str
    logging: LoggingConfig
    save_checkpoints: bool = True
    ema: EMAConfig = dataclasses.field(default_factory=EMAConfig)
    validate_using_ema: bool = False
    generate_n_samples: int = 1
    segment_epochs: int | None = None
    validate_interval: int = 1
    resume_results_dir: str | None = None

    def __post_init__(self):
        for name, data in (
            ("train_data", self.train_data),
            ("validation_data", self.validation_data),
        ):
            if data.n_timesteps != self.model.n_timesteps:
                raise ValueError(
                    f"{name}.n_timesteps ({data.n_timesteps}) must equal "
                    f"model.n_timesteps ({self.model.n_timesteps})."
                )

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.experiment_dir, "checkpoints")

    def build(self) -> "VideoTrainer":
        requirements = self.model.data_requirements
        train_data = self.train_data.build_video(train=True, requirements=requirements)
        validation_data = self.validation_data.build_video(
            train=False, requirements=requirements
        )
        model = self.model.build()
        optimization = self.optimization.build(
            modules=[model.module], max_epochs=self.max_epochs
        )
        return VideoTrainer(model, optimization, train_data, validation_data, self)

    def configure_logging(self, log_filename: str):
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )


class VideoTrainer:
    def __init__(self, model, optimization, train_data, validation_data, config):
        self.model = model
        self.optimization = optimization
        self.null_optimization = NullOptimization()
        self.train_data = train_data
        self.validation_data = validation_data
        self.config = config
        self.ema = config.ema.build(self.model.modules)
        self.validate_using_ema = config.validate_using_ema
        self.num_batches_seen = 0
        self.startEpoch = 0
        self.segment_epochs = config.segment_epochs
        self.best_valid_loss = float("inf")

        wandb = WandB.get_instance()
        wandb.watch(self.model.modules)

        dist = Distributed.get_instance()
        self.epoch_checkpoint_path: str | None = None
        self.best_checkpoint_path: str | None = None
        if config.save_checkpoints:
            if dist.is_root():
                os.makedirs(config.checkpoint_dir, exist_ok=True)
                remove_stale_tmp_checkpoints(config.checkpoint_dir)
            self.epoch_checkpoint_path = os.path.join(
                config.checkpoint_dir, "latest.ckpt"
            )
            self.best_checkpoint_path = os.path.join(config.checkpoint_dir, "best.ckpt")

    @property
    def resuming(self) -> bool:
        return self.epoch_checkpoint_path is not None and os.path.isfile(
            self.epoch_checkpoint_path
        )

    @contextlib.contextmanager
    def _ema_context(self):
        self.ema.store(parameters=self.model.modules.parameters())
        self.ema.copy_to(model=self.model.modules)
        try:
            yield
        finally:
            self.ema.restore(parameters=self.model.modules.parameters())

    @contextlib.contextmanager
    def _validation_context(self):
        if self.validate_using_ema:
            with self._ema_context():
                yield
        else:
            yield

    def train_one_epoch(self) -> None:
        self.model.module.train()
        wandb = WandB.get_instance()
        batch: PairedVideoBatchData
        epoch_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(self.train_data.loader):
            self.num_batches_seen += 1
            outputs = self.model.train_on_batch(batch, self.optimization)
            self.ema(self.model.modules)
            batch_loss = outputs.loss.detach().cpu().item()
            epoch_loss += batch_loss
            n_batches += 1
            if i % 10 == 0:
                logging.info(f"Training batch {i + 1}, loss {batch_loss:.4f}")
            wandb.log({"train/batch_loss": batch_loss}, step=self.num_batches_seen)
        if n_batches == 0:
            raise RuntimeError("Empty training batch generator")
        self.optimization.step_scheduler(epoch_loss / n_batches)
        wandb.log(
            {"train/epoch_loss": epoch_loss / n_batches}, step=self.num_batches_seen
        )

    @torch.no_grad()
    def valid_one_epoch(self) -> dict[str, float]:
        self.model.module.eval()
        total_loss = 0.0
        total_gen_mae = 0.0
        n_batches = 0
        with self._validation_context():
            for batch in self.validation_data.loader:
                outputs = self.model.train_on_batch(batch, self.null_optimization)
                total_loss += outputs.loss.detach().cpu().item()
                total_gen_mae += self._interior_generation_mae(batch)
                n_batches += 1
        if n_batches == 0:
            raise RuntimeError("Empty validation batch generator")
        summary = {
            "validation/loss": total_loss / n_batches,
            "validation/interior_mae": total_gen_mae / n_batches,
        }
        WandB.get_instance().log(summary, step=self.num_batches_seen)
        return summary

    def _interior_generation_mae(self, batch: PairedVideoBatchData) -> float:
        """Ensemble-mean MAE over the generated interior frames."""
        generated = self.model.generate(batch, n_samples=self.config.generate_n_samples)
        n_times = self.model.n_timesteps
        errors = []
        for name, samples in generated.items():
            ens_mean = samples.mean(dim=1)  # (B, T, H, W)
            truth = batch.fine.data[name].to(ens_mean.device)
            interior = slice(1, n_times - 1)
            errors.append(
                (ens_mean[:, interior] - truth[:, interior]).abs().mean().item()
            )
        return sum(errors) / len(errors)

    def save_best_checkpoint(self, summary: dict[str, float]) -> None:
        if self.best_checkpoint_path is None:
            return
        context = self._ema_context if self.validate_using_ema else contextlib.nullcontext
        if summary["validation/loss"] < self.best_valid_loss:
            logging.info("Saving best checkpoint")
            self.best_valid_loss = summary["validation/loss"]
            with context():
                _save_checkpoint(self, self.best_checkpoint_path)

    def save_epoch_checkpoint(self) -> None:
        if self.epoch_checkpoint_path is not None:
            _save_checkpoint(self, self.epoch_checkpoint_path)

    def train(self) -> None:
        logging.info("Running initial validation.")
        self.valid_one_epoch()
        wandb = WandB.get_instance()
        dist = Distributed.get_instance()

        if self.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.segment_epochs, self.config.max_epochs
            )

        for epoch in range(self.startEpoch, segment_max_epochs):
            logging.info(f"Training epoch {epoch + 1}")
            start = time.time()
            self.train_one_epoch()
            self.startEpoch = epoch + 1
            wandb.log({"epoch": epoch}, step=self.num_batches_seen)
            if epoch % self.config.validate_interval == 0:
                summary = self.valid_one_epoch()
                if dist.is_root() and self.config.save_checkpoints:
                    self.save_best_checkpoint(summary)
            if dist.is_root() and self.config.save_checkpoints:
                self.save_epoch_checkpoint()
            wandb.log(
                {"epoch_seconds": time.time() - start}, step=self.num_batches_seen
            )


def _resume_from_results_dir_if_not_preempted(experiment_dir, resume_results_dir):
    resuming_from_preempt = os.path.isfile(
        os.path.join(experiment_dir, "checkpoints/latest.ckpt")
    )
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir, exist_ok=True)
    if resume_results_dir is not None and not resuming_from_preempt:
        if not os.path.isdir(resume_results_dir):
            raise ValueError(
                f"Existing results directory {resume_results_dir} does not exist."
            )
        shutil.copytree(resume_results_dir, experiment_dir, dirs_exist_ok=True)
        remove_stale_tmp_checkpoints(os.path.join(experiment_dir, "checkpoints"))


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_config: VideoTrainerConfig = dacite.from_dict(
        data_class=VideoTrainerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    if train_config.resume_results_dir is not None:
        _resume_from_results_dir_if_not_preempted(
            experiment_dir=train_config.experiment_dir,
            resume_results_dir=train_config.resume_results_dir,
        )
    prepare_directory(train_config.experiment_dir, config)
    train_config.configure_logging(log_filename="out.log")
    logging.info("Starting video diffusion training")
    trainer = train_config.build()
    if trainer.resuming:
        logging.info(f"Resuming training from {trainer.epoch_checkpoint_path}")
        restore_checkpoint(trainer)
    logging.info(f"Number of parameters: {count_parameters(trainer.model.modules)}")
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Video downscaling train script")
    parser.add_argument("config_path", type=str, help="Path to the config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with Distributed.context():
        main(args.config_path)
