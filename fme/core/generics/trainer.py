# This module is derived from the train.py module in the following repository:
# https://github.com/NVlabs/FourCastNet. The corresponding license is
# provided below.

# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import abc
import contextlib
import gc
import logging
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Generic, Protocol, TypeVar

import torch

import fme
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC, InferenceAggregatorABC
from fme.core.generics.data import GriddedDataABC, InferenceDataABC
from fme.core.generics.inference import run_inference
from fme.core.generics.metrics_aggregator import MetricsAggregator
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.optimization import NullOptimization, Optimization
from fme.core.timing import GlobalTimer
from fme.core.training_history import TrainingJob
from fme.core.typing_ import Slice
from fme.core.wandb import WandB


class EndOfBatchCallback(Protocol):
    def __call__(self) -> None: ...


class EndOfEpochCallback(Protocol):
    def __call__(self, epoch: int) -> Mapping[str, Any]: ...


def null_end_of_epoch_callback(epoch: int) -> Mapping[str, Any]:
    return {}


class TrainConfigProtocol(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Any]]

    @property
    def experiment_dir(self) -> str: ...

    @property
    def output_dir(self) -> str: ...

    @property
    def checkpoint_dir(self) -> str: ...

    @property
    def max_epochs(self) -> int: ...

    @property
    def save_checkpoint(self) -> bool: ...

    @property
    def validate_using_ema(self) -> bool: ...

    @property
    def log_train_every_n_batches(self) -> int: ...

    @property
    def checkpoint_every_n_batches(self) -> int: ...

    @property
    def segment_epochs(self) -> int | None: ...

    @property
    def checkpoint_save_epochs(self) -> Slice | None: ...

    @property
    def ema_checkpoint_save_epochs(self) -> Slice | None: ...

    @property
    def evaluate_before_training(self) -> bool: ...

    @property
    def save_best_inference_epoch_checkpoints(self) -> bool: ...

    def get_inference_epochs(self) -> list[int]: ...


PS = TypeVar("PS", contravariant=True)  # prognostic state
TO = TypeVar("TO", bound="TrainOutputABC")  # train output
BD = TypeVar("BD")  # batch data for training
FD = TypeVar("FD")  # forcing data for inference
SD = TypeVar("SD")  # stepped data from inference


class AggregatorBuilderABC(abc.ABC, Generic[PS, TO, SD]):
    @abc.abstractmethod
    def get_train_aggregator(self) -> AggregatorABC[TO]:
        pass

    @abc.abstractmethod
    def get_validation_aggregator(self) -> AggregatorABC[TO]:
        pass

    @abc.abstractmethod
    def get_inference_aggregator(self) -> InferenceAggregatorABC[PS, SD]:
        pass


class CheckpointPaths:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir

    @property
    def latest_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ckpt.tar")

    @property
    def best_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_ckpt.tar")

    @property
    def best_inference_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "best_inference_ckpt.tar")

    @property
    def ema_checkpoint_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "ema_ckpt.tar")

    def epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ckpt_{epoch:04d}.tar")

    def ema_epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"ema_ckpt_{epoch:04d}.tar")

    def best_inference_epoch_checkpoint_path(self, epoch: int) -> str:
        return os.path.join(self.checkpoint_dir, f"best_inference_ckpt_{epoch:04d}.tar")


def chain_signal_handler(sig, handler):
    prev_handler = signal.getsignal(sig)

    def on_sig(signum, frame):
        handler(signum, frame)
        if callable(prev_handler):
            prev_handler(signum, frame)
        sys.exit(0)

    signal.signal(sig, on_sig)


class Trainer:
    def __init__(
        self,
        train_data: GriddedDataABC[BD],
        validation_data: GriddedDataABC[BD],
        inference_data: InferenceDataABC[PS, FD],
        stepper: TrainStepperABC[PS, BD, FD, SD, TO],
        build_optimization: Callable[[torch.nn.ModuleList], Optimization],
        build_ema: Callable[[torch.nn.ModuleList], EMATracker],
        config: TrainConfigProtocol,
        aggregator_builder: AggregatorBuilderABC[PS, TO, SD],
        end_of_batch_callback: EndOfBatchCallback = lambda: None,
        end_of_epoch_callback: EndOfEpochCallback = null_end_of_epoch_callback,
        do_gc_collect: bool = True,
    ):
        logging.info(f"Current device is {fme.get_device()}")
        dist = Distributed.get_instance()
        if dist.is_root():
            if not os.path.isdir(config.experiment_dir):
                os.makedirs(config.experiment_dir)
            if not os.path.isdir(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
        self.config = config
        self.paths = CheckpointPaths(config.checkpoint_dir)

        self.train_data = train_data
        self.valid_data = validation_data
        for gridded_data, name in zip(
            (self.train_data, self.valid_data), ("train", "valid")
        ):
            gridded_data.log_info(name)

        self.num_batches_seen = 0
        self._start_epoch = 0
        self._epochs_trained = self._start_epoch
        self._current_epoch_num_batches_seen = 0
        self._best_validation_loss = torch.inf
        self._best_inference_error = torch.inf

        self.stepper = stepper
        self.stepper.update_training_history(TrainingJob.from_env())

        self.optimization = build_optimization(stepper.modules)
        self._end_of_batch_callback = end_of_batch_callback
        self._end_of_epoch_callback = end_of_epoch_callback
        self._no_optimization = NullOptimization()
        self._aggregator_builder = aggregator_builder

        resuming = os.path.isfile(self.paths.latest_checkpoint_path)
        if resuming:
            logging.info(f"Resuming training from {self.paths.latest_checkpoint_path}")
            self.restore_checkpoint(
                self.paths.latest_checkpoint_path, self.paths.ema_checkpoint_path
            )

        wandb = WandB.get_instance()
        wandb.watch(self.stepper.modules)

        n_params = count_parameters(self.stepper.modules)
        logging.info(f"Number of trainable model parameters: {n_params}")

        self._inference_data = inference_data
        self._ema = build_ema(stepper.modules)
        self._do_gc_collect = do_gc_collect
        self._in_ema_context = False
        self._started_training = False

        def on_terminate(signum, frame):
            dist = Distributed.get_instance()
            # have all ranks check in to logs
            dist.barrier()
            if self._current_epoch_num_batches_seen > 0 and dist.is_root():
                if self._in_ema_context:
                    logging.info(
                        "In EMA context during interrupt, not saving "
                        "restart checkpoints as it is unsafe to do so"
                    )
                elif not self._started_training:
                    logging.info(
                        "Not saving restart checkpoints as training has not started"
                    )
                else:
                    self._save_restart_checkpoints()
            dist.shutdown()

        chain_signal_handler(signal.SIGTERM, on_terminate)
        chain_signal_handler(signal.SIGINT, on_terminate)

    def set_end_of_epoch_callback(self, end_of_epoch_callback: EndOfEpochCallback):
        self._end_of_epoch_callback = end_of_epoch_callback

    def switch_off_grad(self, model: torch.nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        logging.info("Starting Training Loop...")
        dist = Distributed.get_instance()

        inference_epochs = self.config.get_inference_epochs()
        if self.config.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self._start_epoch + self.config.segment_epochs, self.config.max_epochs
            )

        if self.config.evaluate_before_training and self._epochs_trained == 0:
            logging.info("Starting validation before training")
            valid_logs = self.validate_one_epoch()
            if self._epochs_trained in inference_epochs:
                logging.info("Starting inline inference before training")
                inference_logs = self.inference_one_epoch()
            else:
                inference_logs = {}
            valid_loss = valid_logs["val/mean/loss"]
            logging.info(f"Validation loss before training: {valid_loss}")
            logging.info("Logging to wandb")
            all_logs = valid_logs | inference_logs | {"epoch": self._epochs_trained}
            wandb = WandB.get_instance()
            wandb.log(all_logs, step=self.num_batches_seen)

        while self._epochs_trained < segment_max_epochs:
            if self._do_gc_collect:
                # garbage collect to avoid CUDA error in some contexts
                # https://github.com/pytorch/pytorch/issues/67978#issuecomment-1661986812  # noqa: E501
                gc.collect()
            logging.info(
                f"Beginning epoch after {self._epochs_trained} complete epochs"
            )
            start_time = time.time()
            train_logs = self.train_one_epoch()
            train_end = time.time()
            valid_logs = self.validate_one_epoch()
            valid_end = time.time()
            if self._epochs_trained in inference_epochs:
                inference_logs = self.inference_one_epoch()
                inference_end: float | None = time.time()
            else:
                inference_logs = {}
                inference_end = None

            train_loss = train_logs.get("train/mean/loss")
            valid_loss = valid_logs["val/mean/loss"]
            inference_error = inference_logs.get(
                "inference/time_mean_norm/rmse/channel_mean", None
            )
            # need to get the learning rate before stepping the scheduler
            lr = self.optimization.learning_rate
            self.optimization.step_scheduler(valid_loss=valid_loss, is_iteration=False)

            time_elapsed = time.time() - start_time
            logging.info(
                f"Time taken for epoch {self._epochs_trained} is {time_elapsed} sec"
            )
            if train_loss is not None:
                logging.info(f"Train loss: {train_loss}")
            else:
                logging.info("No train loss available")
            logging.info(f"Valid loss: {valid_loss}")
            if inference_error is not None:
                logging.info(f"Inference error: {inference_error}")

            with self.validation_context():
                additional_logs = self._end_of_epoch_callback(self._epochs_trained)

            logging.info("Logging to wandb")
            all_logs = {
                **train_logs,
                **valid_logs,
                **inference_logs,
                **additional_logs,
                **{
                    "lr": lr,
                    "epoch": self._epochs_trained,
                    "epoch_train_seconds": train_end - start_time,
                    "epoch_validation_seconds": valid_end - train_end,
                    "epoch_total_seconds": time_elapsed,
                    "best_val_loss": min(valid_loss, self._best_validation_loss),
                    "best_inference_error": min(
                        inference_error or torch.inf, self._best_inference_error
                    ),
                },
            }
            if inference_end is not None:
                all_logs["epoch_inference_seconds"] = inference_end - valid_end
            wandb = WandB.get_instance()
            wandb.log(all_logs, step=self.num_batches_seen)

            if dist.is_root():
                if self.config.save_checkpoint:
                    logging.info(f"Saving checkpoints for epoch {self._epochs_trained}")
                    self.save_all_checkpoints(valid_loss, inference_error)

    def _log_first_batch_metrics(self):
        wandb = WandB.get_instance()
        dist = Distributed.get_instance()
        with torch.no_grad(), GlobalTimer():
            batch = next(iter(self.train_data.loader))
            stepped = self.stepper.train_on_batch(
                batch,
                optimization=self._no_optimization,
            )

            if self.config.log_train_every_n_batches > 0:
                with torch.no_grad():
                    metrics = {
                        f"batch_{name}": dist.reduce_mean(metric)
                        for name, metric in sorted(stepped.get_metrics().items())
                    }
                wandb.log(metrics, step=self.num_batches_seen)

    def train_one_epoch(self):
        """Train for one epoch and return logs from TrainAggregator."""
        logging.info(
            f"Starting training step for model trained for {self._epochs_trained} "
            "complete epochs"
        )
        self.train_data.set_epoch(self._epochs_trained + 1)
        wandb = WandB.get_instance()
        dist = Distributed.get_instance()
        names_to_log = ("batch_loss", "training_samples_per_second_on_rank_0", "lr")
        aggregator = self._aggregator_builder.get_train_aggregator()
        n_samples_seen_since_logging = 0
        self.stepper.set_train()
        if self.num_batches_seen == 0:
            # Before training, log the loss on the first batch.
            self._log_first_batch_metrics()
        if self._current_epoch_num_batches_seen > 0:
            logging.info(
                "Subsetting train loader sampler to skip first "
                f"{self._current_epoch_num_batches_seen} batches since these were "
                "already processed for this epoch in a previous training run."
            )
        epoch_data = self.train_data.subset_loader(self._current_epoch_num_batches_seen)
        if self._current_epoch_num_batches_seen > 0:
            logging.info(
                f"Subsetted train loader created, has {len(epoch_data)} batches"
            )
        self._last_saved_num_batches_seen = self.num_batches_seen
        self._started_training = True
        current_time = time.time()
        metrics_aggregator = MetricsAggregator()
        for batch in epoch_data:
            with GlobalTimer():
                stepped = self.stepper.train_on_batch(batch, self.optimization)
            aggregator.record_batch(stepped)
            self._end_of_batch_callback()
            self._ema(model=self.stepper.modules)
            # Step scheduler per-iteration if configured to do so
            self.optimization.step_scheduler(is_iteration=True)
            self.num_batches_seen += 1
            self._current_epoch_num_batches_seen += 1
            n_samples_seen_since_logging += self.train_data.batch_size
            metrics_aggregator.record(stepped.get_metrics())
            if (
                self.config.log_train_every_n_batches > 0
                and self.num_batches_seen % self.config.log_train_every_n_batches == 0
            ):
                metrics = {
                    f"batch_{name}": value
                    for name, value in metrics_aggregator.get_metrics().items()
                }
                metrics_aggregator.clear()
                duration = time.time() - current_time
                current_time = time.time()
                samples_per_second = n_samples_seen_since_logging / duration
                metrics["training_samples_per_second_on_rank_0"] = samples_per_second
                metrics["lr"] = self.optimization.learning_rate
                wandb.log(metrics, step=self.num_batches_seen)
                metrics_to_log = {k: metrics[k] for k in names_to_log if k in metrics}
                logging.info(f"Step {self.num_batches_seen}: {metrics_to_log}")
                n_samples_seen_since_logging = 0
            if (
                dist.is_root()
                and self.config.checkpoint_every_n_batches > 0
                and self.num_batches_seen % self.config.checkpoint_every_n_batches == 0
            ):
                self._save_restart_checkpoints()
                self._last_saved_num_batches_seen = self.num_batches_seen
        if dist.is_root() and self.num_batches_seen > self._last_saved_num_batches_seen:
            self._save_restart_checkpoints()  # before incrementing epoch so we will validate after resuming  # noqa: E501
        # we will save restart checkpoints again after validation/inference
        # are recorded to wandb
        self._epochs_trained += 1
        self._current_epoch_num_batches_seen = 0
        aggregator.flush_diagnostics(subdir=f"epoch_{self._epochs_trained:04d}")
        return aggregator.get_logs(label="train")

    def _save_restart_checkpoints(self):
        logging.info(
            f"Saving latest checkpoint model trained for {self._epochs_trained} "
            f"complete epochs and {self._current_epoch_num_batches_seen} additional "
            f"batches, or {self.num_batches_seen} total batches, with "
            f"best_validation_loss {self._best_validation_loss} and "
            f"best_inference_error {self._best_inference_error}"
        )
        self.save_checkpoint(
            self.paths.latest_checkpoint_path,
            ema_checkpoint_path=self.paths.ema_checkpoint_path,
            include_optimization=True,
        )

    @contextlib.contextmanager
    def validation_context(self):
        """
        The context for running validation.

        In this context, the stepper uses the EMA model if
        `self.config.validate_using_ema` is True.
        """
        if self.config.validate_using_ema:
            with self._ema_context():
                yield
        else:
            yield

    @contextlib.contextmanager
    def _ema_context(self):
        """
        A context where the stepper uses the EMA model.
        """
        self._ema.store(parameters=self.stepper.modules.parameters())
        self._in_ema_context = True
        self._ema.copy_to(model=self.stepper.modules)
        try:
            yield
        finally:
            self._ema.restore(parameters=self.stepper.modules.parameters())
            self._in_ema_context = False

    def validate_one_epoch(self):
        if self._current_epoch_num_batches_seen > 0:
            raise RuntimeError(
                "Validation should only be called after a full epoch has been trained"
            )
        logging.info(
            f"Starting validation step for model trained for "
            f"{self._epochs_trained} epochs"
        )
        self.stepper.set_eval()
        aggregator = self._aggregator_builder.get_validation_aggregator()
        logging.info("Starting loop over validation data")
        with torch.no_grad(), self.validation_context(), GlobalTimer():
            for batch in self.valid_data.loader:
                stepped = self.stepper.train_on_batch(
                    batch,
                    optimization=NullOptimization(),
                    compute_derived_variables=True,
                )
                aggregator.record_batch(
                    batch=stepped,
                )
        logging.info("Starting flush of reduced diagnostics to disk")
        aggregator.flush_diagnostics(subdir=f"epoch_{self._epochs_trained:04d}")
        logging.info("Getting validation aggregator logs")
        return aggregator.get_logs(label="val")

    def inference_one_epoch(
        self,
    ):
        if self._current_epoch_num_batches_seen > 0:
            raise RuntimeError(
                "Inference should only be called after a full epoch has been trained"
            )
        logging.info(
            f"Starting inference step for model trained for "
            f"{self._epochs_trained} epochs"
        )
        logging.info("Starting inline inference run")
        return inference_one_epoch(
            stepper=self.stepper,
            validation_context=self.validation_context,
            dataset=self._inference_data,
            aggregator=self._aggregator_builder.get_inference_aggregator(),
            label="inference",
            epoch=self._epochs_trained,
        )

    def save_checkpoint(
        self,
        checkpoint_path: str,
        ema_checkpoint_path: str | None = None,
        include_optimization: bool = False,
    ):
        if not Distributed.get_instance().is_root():
            raise RuntimeError("Only the root process should save checkpoints")
        # save to a temporary file in case we get pre-empted during save
        temporary_location = os.path.join(
            os.path.dirname(checkpoint_path), f".{uuid.uuid4()}.tmp"
        )
        if ema_checkpoint_path is not None:
            ema_temporary_location: str | None = os.path.join(
                os.path.dirname(ema_checkpoint_path), f".{uuid.uuid4()}.tmp"
            )
        else:
            ema_temporary_location = None
        try:
            data = {
                "num_batches_seen": self.num_batches_seen,
                "current_epoch_num_batches_seen": self._current_epoch_num_batches_seen,
                "epoch": self._epochs_trained,
                "best_validation_loss": self._best_validation_loss,
                "best_inference_error": self._best_inference_error,
                "stepper": self.stepper.get_state(),
                "ema": self._ema.get_state(),
            }
            if include_optimization:
                data["optimization"] = self.optimization.get_state()
            if ema_temporary_location is not None:
                with self._ema_context():
                    ema_data = dict(
                        data,
                        stepper=self.stepper.get_state(),
                        ema=self._ema.get_state(),
                    )
                    # never include optimization in EMA checkpoint
                    if "optimization" in ema_data:
                        ema_data.pop("optimization")
                    torch.save(ema_data, ema_temporary_location)
            torch.save(data, temporary_location)
            if ema_temporary_location is not None and ema_checkpoint_path is not None:
                os.replace(ema_temporary_location, ema_checkpoint_path)
            os.replace(temporary_location, checkpoint_path)
        finally:
            if os.path.exists(temporary_location):
                os.remove(temporary_location)
            if ema_temporary_location is not None and os.path.exists(
                ema_temporary_location
            ):
                os.remove(ema_temporary_location)

    def restore_checkpoint(self, checkpoint_path, ema_checkpoint_path):
        """
        Restore the checkpoint from the given path. This includes the existing state of
        the stepper, optimization, training epoch, and EMA. This is most suitable
        for resuming training from a checkpoint without changing the training schedule,
        i.e., to manage preemption.
        """
        _restore_checkpoint(self, checkpoint_path, ema_checkpoint_path)

    def _epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.config.max_epochs, self.config.checkpoint_save_epochs
        )

    def _ema_epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.config.max_epochs, self.config.ema_checkpoint_save_epochs
        )

    def save_all_checkpoints(self, valid_loss: float, inference_error: float | None):
        if self.config.validate_using_ema:
            best_checkpoint_context = self._ema_context
        else:
            best_checkpoint_context = contextlib.nullcontext  # type: ignore
        with best_checkpoint_context():
            save_best_checkpoint = False
            if valid_loss <= self._best_validation_loss:
                logging.info(
                    "Saving lowest validation loss checkpoint to "
                    f"{self.paths.best_checkpoint_path}"
                )
                self._best_validation_loss = valid_loss
                save_best_checkpoint = True  # wait until inference error is updated
            if inference_error is not None and (
                inference_error <= self._best_inference_error
            ):
                logging.info(
                    f"Epoch inference error ({inference_error}) is lower than "
                    f"previous best inference error ({self._best_inference_error})."
                )
                logging.info(
                    "Saving lowest inference error checkpoint to "
                    f"{self.paths.best_inference_checkpoint_path}"
                )
                self._best_inference_error = inference_error
                self.save_checkpoint(self.paths.best_inference_checkpoint_path)

                # Save epoch-specific best inference checkpoint if configured
                if self.config.save_best_inference_epoch_checkpoints:
                    best_inference_epoch_path = (
                        self.paths.best_inference_epoch_checkpoint_path(
                            self._epochs_trained
                        )
                    )
                    logging.info(
                        "Saving best inference checkpoint for epoch "
                        f"{self._epochs_trained} to {best_inference_epoch_path}"
                    )
                    self.save_checkpoint(best_inference_epoch_path)
            if save_best_checkpoint:
                self.save_checkpoint(self.paths.best_checkpoint_path)

        self._save_restart_checkpoints()

        if self._ema_epoch_checkpoint_enabled(self._epochs_trained):
            ema_epoch_checkpoint_path = self.paths.ema_epoch_checkpoint_path(
                self._epochs_trained
            )
            logging.info(f"Saving EMA epoch checkpoint to {ema_epoch_checkpoint_path}")
            with self._ema_context():
                self.save_checkpoint(ema_epoch_checkpoint_path)
        if self._epoch_checkpoint_enabled(self._epochs_trained):
            epoch_checkpoint_path = self.paths.epoch_checkpoint_path(
                self._epochs_trained
            )
            logging.info(f"Saving epoch checkpoint to {epoch_checkpoint_path}")
            self.save_checkpoint(epoch_checkpoint_path)


def inference_one_epoch(
    stepper: TrainStepperABC[PS, BD, FD, SD, TO],
    validation_context: Callable[[], contextlib.AbstractContextManager],
    dataset: InferenceDataABC[PS, FD],
    aggregator: InferenceAggregatorABC[PS, SD],
    label: str,
    epoch: int,
):
    stepper.set_eval()
    with torch.no_grad(), validation_context(), GlobalTimer():
        run_inference(
            predict=stepper.predict_paired,
            data=dataset,
            aggregator=aggregator,
        )
    logging.info("Starting flush of reduced diagnostics to disk")
    aggregator.flush_diagnostics(subdir=f"epoch_{epoch:04d}")
    logging.info("Getting inline inference aggregator logs")
    logs = aggregator.get_summary_logs()
    return {f"{label}/{k}": v for k, v in logs.items()}


def _restore_checkpoint(trainer: Trainer, checkpoint_path, ema_checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path, map_location=fme.get_device(), weights_only=False
    )
    trainer.stepper.load_state(checkpoint["stepper"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer._current_epoch_num_batches_seen = checkpoint[
        "current_epoch_num_batches_seen"
    ]
    trainer._start_epoch = checkpoint["epoch"]
    trainer._epochs_trained = checkpoint["epoch"]
    trainer._best_validation_loss = checkpoint["best_validation_loss"]
    trainer._best_inference_error = checkpoint["best_inference_error"]
    ema_checkpoint = torch.load(
        ema_checkpoint_path, map_location=fme.get_device(), weights_only=False
    )
    ema_stepper: TrainStepperABC = type(trainer.stepper).from_state(
        ema_checkpoint["stepper"]
    )
    trainer._ema = EMATracker.from_state(checkpoint["ema"], ema_stepper.modules)


def count_parameters(modules: torch.nn.ModuleList) -> int:
    parameters = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters += parameter.numel()
    return parameters


def epoch_checkpoint_enabled(
    epoch: int, max_epochs: int, save_epochs: Slice | None
) -> bool:
    if save_epochs is None:
        return False
    return epoch in range(max_epochs)[save_epochs.slice]
