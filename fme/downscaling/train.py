import argparse
import contextlib
import dataclasses
import logging
import os
import re
import shutil
import time
import uuid

import dacite
import torch
import yaml

from fme.core.cli import prepare_directory
from fme.core.device import get_device
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMAConfig, EMATracker
from fme.core.generics.trainer import count_parameters
from fme.core.logging_utils import LoggingConfig
from fme.core.optimization import NullOptimization, Optimization, OptimizationConfig
from fme.core.wandb import WandB
from fme.downscaling.aggregators import Aggregator, GenerationAggregator
from fme.downscaling.data import (
    PairedBatchData,
    PairedDataLoaderConfig,
    PairedGriddedData,
    load_static_inputs,
)
from fme.downscaling.models import DiffusionModel, DiffusionModelConfig


def _save_checkpoint(trainer: "Trainer", path: str) -> None:
    # save to a temporary file in case we get pre-empted during save
    temporary_location = os.path.join(os.path.dirname(path), f".{uuid.uuid4()}.tmp")
    try:
        torch.save(
            {
                "model": trainer.model.get_state(),
                "ema": trainer.ema.get_state(),
                "optimization": trainer.optimization.get_state(),
                "num_batches_seen": trainer.num_batches_seen,
                "startEpoch": trainer.startEpoch,
                "best_valid_loss": trainer.best_valid_loss,
                "best_histogram_tail_metric": trainer.best_histogram_tail_metric,
                "validate_using_ema": trainer.validate_using_ema,
            },
            temporary_location,
        )
        os.replace(temporary_location, path)
    finally:
        if os.path.exists(temporary_location):
            os.remove(temporary_location)


def restore_checkpoint(trainer: "Trainer") -> None:
    if trainer.epoch_checkpoint_path is None:
        raise ValueError("Cannot restore checkpoint without a checkpoint path")

    checkpoint = torch.load(
        trainer.epoch_checkpoint_path, map_location=get_device(), weights_only=False
    )
    trainer.model.module.load_state_dict(checkpoint["model"]["module"])
    trainer.optimization.load_state(checkpoint["optimization"])

    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["startEpoch"]
    trainer.best_valid_loss = checkpoint["best_valid_loss"]
    trainer.best_histogram_tail_metric = checkpoint.get(
        "best_histogram_tail_metric", float("inf")
    )

    trainer.validate_using_ema = checkpoint["validate_using_ema"]
    ema_checkpoint = torch.load(
        trainer.ema_checkpoint_path, map_location=get_device(), weights_only=False
    )
    ema_model = trainer.model.from_state(ema_checkpoint["model"])
    trainer.ema = EMATracker.from_state(ema_checkpoint["ema"], ema_model.modules)


class Trainer:
    def __init__(
        self,
        model: DiffusionModel,
        optimization: Optimization,
        train_data: PairedGriddedData,
        validation_data: PairedGriddedData,
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
        self.resume_results_dir = config.resume_results_dir

        dist = Distributed.get_instance()
        if dist.is_root():
            if not os.path.isdir(self.config.experiment_dir):
                os.makedirs(self.config.experiment_dir)
            if self.config.checkpoint_dir is not None and not os.path.isdir(
                self.config.checkpoint_dir
            ):
                os.makedirs(self.config.checkpoint_dir)

        self.epoch_checkpoint_path: str | None = None

        self.best_valid_loss = float("inf")
        self.best_checkpoint_path: str | None = None
        self.best_histogram_tail_metric = float("inf")
        self.best_histogram_tail_checkpoint_path: str | None = None

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
            self.best_histogram_tail_checkpoint_path = os.path.join(
                self.config.checkpoint_dir, "best_histogram_tail.ckpt"
            )

        self._best_valid_loss_name = "generation/metrics/relative_crps_bicubic"
        self._best_histogram_tail_name = (
            "generation/histogram/prediction_frac_of_target/99.9999th-percentile"
        )

    def _get_batch_generator(
        self, data: PairedGriddedData, random_offset: bool, shuffle: bool
    ):
        if self.patch_data:
            batch_generator = data.get_patched_generator(
                coarse_yx_patch_extent=self.model.coarse_shape,
                overlap=0,
                drop_partial_patches=True,
                random_offset=random_offset,
                shuffle=shuffle,
            )
        else:
            batch_generator = data.get_generator()
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
            self.train_data, random_offset=True, shuffle=True
        )
        outputs = None
        for i, batch in enumerate(train_batch_generator):
            self.num_batches_seen += 1
            if i % 10 == 0:
                logging.info(f"Training on batch {i + 1}")
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

    def valid_one_epoch(self) -> dict[str, float]:
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
                percentiles=[99.99, 99.9999],
                include_positional_comparisons=include_positional_comparisons,
            )
            batch: PairedBatchData
            validation_batch_generator = self._get_batch_generator(
                self.validation_data, random_offset=False, shuffle=False
            )
            for batch in validation_batch_generator:
                outputs = self.model.train_on_batch(batch, self.null_optimization)
                validation_aggregator.record_batch(
                    outputs=outputs,
                    coarse=batch.coarse.data,
                    batch=batch,
                )
                generated_outputs = self.model.generate_on_batch(
                    batch,
                    n_samples=self.config.generate_n_samples,
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
        channel_mean_checkpoint_metrics = {
            prefix: _get_channel_mean_scalar_metric(generation_metrics, prefix)
            for prefix in [
                self._best_valid_loss_name,
                self._best_histogram_tail_name,
            ]
        }
        return channel_mean_checkpoint_metrics

    @property
    def resuming(self) -> bool:
        if self.epoch_checkpoint_path is None:
            return False
        return os.path.isfile(self.epoch_checkpoint_path)

    def save_best_checkpoint(self, valid_metrics: dict[str, float]) -> None:
        if self.best_checkpoint_path is not None:
            if self.validate_using_ema:
                best_checkpoint_context = self._ema_context
            else:
                best_checkpoint_context = contextlib.nullcontext  # type: ignore
            # Best checkpoint is hard coded to use validation CRPS channel mean
            if valid_metrics[self._best_valid_loss_name] < self.best_valid_loss:
                logging.info("Saving best checkpoint")
                self.best_valid_loss = valid_metrics[self._best_valid_loss_name]
                with best_checkpoint_context():
                    _save_checkpoint(self, self.best_checkpoint_path)
            else:
                logging.info(
                    "Validation loss did not improve, will not overwrite "
                    "best checkpoint."
                )
        if self.best_histogram_tail_checkpoint_path is not None:
            if (
                valid_metrics[self._best_histogram_tail_name]
                < self.best_histogram_tail_metric
            ):
                logging.info("Saving checkpoint for best histogram tail.")
                self.best_histogram_tail_metric = valid_metrics[
                    self._best_histogram_tail_name
                ]
                with best_checkpoint_context():
                    _save_checkpoint(self, self.best_histogram_tail_checkpoint_path)
            else:
                logging.info(
                    "Histogram tail metric did not improve, will not overwrite "
                    "best histogram tail checkpoint."
                )
        else:
            raise ValueError("Best checkpoint path is not set")

    def save_epoch_checkpoints(self) -> None:
        if self.epoch_checkpoint_path is not None:
            logging.info(f"Saving latest checkpoint")
            _save_checkpoint(self, self.epoch_checkpoint_path)
            with self._ema_context():
                _save_checkpoint(self, self.ema_checkpoint_path)
        else:
            raise ValueError("Latest checkpoint path is not set")

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
        dist = Distributed.get_instance()

        if self.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.segment_epochs, self.config.max_epochs
            )
        for epoch in range(self.startEpoch, segment_max_epochs):
            logging.info(f"Training epoch: {epoch + 1}")
            start_time = time.time()
            self.train_one_epoch()
            train_end = time.time()

            self.startEpoch = epoch + 1
            wandb.log({"epoch": epoch}, step=self.num_batches_seen)
            if self._validate_current_epoch(epoch):
                logging.info("Running metrics on validation data.")
                valid_metrics = self.valid_one_epoch()
                valid_end = time.time()
                if dist.is_root():
                    self.save_best_checkpoint(valid_metrics)
            else:
                valid_end = train_end
            if dist.is_root():
                self.save_epoch_checkpoints()
            epoch_end = time.time()
            timings = {
                "epoch_train_seconds": train_end - start_time,
                "epoch_valid_seconds": valid_end - train_end,
                "epoch_total_seconds": epoch_end - start_time,
            }
            wandb.log(timings, step=self.num_batches_seen)


@dataclasses.dataclass
class TrainerConfig:
    model: DiffusionModelConfig
    optimization: OptimizationConfig
    train_data: PairedDataLoaderConfig
    validation_data: PairedDataLoaderConfig
    max_epochs: int
    experiment_dir: str
    save_checkpoints: bool
    logging: LoggingConfig
    static_inputs: dict[str, str] = dataclasses.field(default_factory=dict)
    ema: EMAConfig = dataclasses.field(default_factory=EMAConfig)
    validate_using_ema: bool = False
    generate_n_samples: int = 1
    segment_epochs: int | None = None
    validate_interval: int = 1
    coarse_patch_extent_lat: int | None = None
    coarse_patch_extent_lon: int | None = None
    resume_results_dir: str | None = None

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
        static_inputs = (
            load_static_inputs(self.static_inputs) if self.static_inputs else None
        )

        train_data: PairedGriddedData = self.train_data.build(
            train=True,
            requirements=self.model.data_requirements,
        )
        validation_data: PairedGriddedData = self.validation_data.build(
            train=False,
            requirements=self.model.data_requirements,
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
            full_fine_coords=train_data.fine_coords,
            static_inputs=static_inputs,
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
        config = to_flat_dict(dataclasses.asdict(self))
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=True
        )


def _get_complement_percentile_prefix(prefix):
    """
    Given a prefix containing a percentile value, return a prefix with
    100 minus that percentile. Returns None if no percentile pattern is found.
    Ex. "prediction_frac_of_target/99.9999th-percentile"
        -> "prediction_frac_of_target/0.0001th-percentile", or
        "some_var/percentile/99.9999" -> "some_var/percentile/0.0001".
    """
    match = re.search(
        r"(\d+(?:\.\d+)?)(?:th)?[-_/]percentile|percentile[-_/](\d+(?:\.\d+)?)",
        prefix,
    )
    if match is None:
        return None
    if match.group(1) is not None:
        num_str = match.group(1)
        num_start, num_end = match.start(1), match.end(1)
    else:
        num_str = match.group(2)
        num_start, num_end = match.start(2), match.end(2)
    complement = 100 - float(num_str)
    if "." in num_str:
        decimal_places = len(num_str.split(".")[1])
        complement_str = f"{complement:.{decimal_places}f}"
    else:
        complement_str = str(int(complement))
    return prefix[:num_start] + complement_str + prefix[num_end:]


def _get_channel_mean_scalar_metric(
    metrics, prefix="generation/metrics/relative_crps_bicubic"
):
    prefixes = [prefix]
    if "percentile" in prefix:
        complement = _get_complement_percentile_prefix(prefix)
        if complement is not None and complement != prefix:
            prefixes.append(complement)
    channel_metric = [
        v for k, v in metrics.items() if any(k.startswith(p) for p in prefixes)
    ]
    if len(channel_metric) == 0:
        return float("inf")
    else:
        return sum(channel_metric) / len(channel_metric)


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


def main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_config: TrainerConfig = dacite.from_dict(
        data_class=TrainerConfig,
        data=config,
        config=dacite.Config(strict=True),
    )

    if train_config.resume_results_dir is not None:
        _resume_from_results_dir_if_not_preempted(
            experiment_dir=train_config.experiment_dir,
            resume_results_dir=train_config.resume_results_dir,
        )
    # Calling this after resuming from results dir so that the submitted config is saved
    prepare_directory(train_config.experiment_dir, config)

    train_config.configure_logging(log_filename="out.log")
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
    with Distributed.context():
        main(args.config_path)
