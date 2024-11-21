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
import dataclasses
import gc
import itertools
import logging
import os
import time
import uuid
from datetime import timedelta
from typing import Callable, Dict, Generic, Mapping, Optional, Sequence, TypeVar

import dacite
import dask
import torch
import xarray as xr
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.inference import run_inference_evaluator
from fme.ace.inference.timing import GlobalTimer
from fme.ace.train.train_config import (
    EndOfBatchCallback,
    TrainBuilders,
    TrainConfig,
    TrainConfigProtocol,
)
from fme.core.aggregator import OneStepAggregator, TrainAggregator
from fme.core.aggregator.inference.main import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
)
from fme.core.data_loading.batch_data import (
    GriddedData,
    GriddedDataABC,
    PairedData,
    PrognosticState,
)
from fme.core.data_loading.config import Slice
from fme.core.data_loading.data_typing import (
    HorizontalCoordinates,
    SigmaCoordinates,
    VariableMetadata,
)
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.generics.aggregator import AggregatorABC, InferenceAggregatorABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.optimization import NullOptimization, Optimization
from fme.core.stepper import (
    SingleModuleStepper,
    StepperABC,
    TrainOutput,
    TrainOutputABC,
)
from fme.core.typing_ import TensorDict, TensorMapping
from fme.core.wandb import WandB

# dask used on individual workers to load batches
dask.config.set(scheduler="synchronous")


def count_parameters(modules: torch.nn.ModuleList) -> int:
    parameters = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters += parameter.numel()
    return parameters


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> "Trainer":
    # note for devs: you don't have to use this function to build a custom
    # trainer, you can build it however you like. This is here for convenience.
    train_data = builder.get_train_data()
    validation_data = builder.get_validation_data()
    inference_data = builder.get_inference_data()

    for batch in train_data.loader:
        shapes = {k: v.shape for k, v in batch.data.items()}
        for value in shapes.values():
            img_shape = value[-2:]
            break
        break
    logging.info("Starting model initialization")
    stepper = builder.get_stepper(
        img_shape=img_shape,
        gridded_operations=train_data.gridded_operations,
        sigma_coordinates=train_data.sigma_coordinates,
        timestep=train_data.timestep,
    )
    optimization = builder.get_optimization(
        itertools.chain(*[m.parameters() for m in stepper.modules])
    )
    ema = builder.get_ema(stepper.modules)
    end_of_batch_ops = builder.get_end_of_batch_ops(stepper.modules)

    for batch in inference_data.loader:
        initial_inference_times = batch.times.isel(time=0)
        break
    aggregator_builder = AggregatorBuilder(
        inference_config=config.inference_aggregator,
        gridded_operations=train_data.gridded_operations,
        sigma_coordinates=train_data.sigma_coordinates,
        horizontal_coordinates=train_data.horizontal_coordinates,
        timestep=train_data.timestep,
        initial_inference_times=initial_inference_times,
        record_step_20=config.inference_n_forward_steps >= 20,
        n_timesteps=config.inference_n_forward_steps + stepper.n_ic_timesteps,
        variable_metadata=train_data.variable_metadata,
        loss_scaling=stepper.effective_loss_scaling,
        channel_mean_names=stepper.out_names,
        normalize=stepper.normalizer.normalize,
    )
    return Trainer(
        train_data=train_data,
        validation_data=validation_data,
        inference_data=inference_data,
        stepper=stepper,
        optimization=optimization,
        ema=ema,
        config=config,
        aggregator_builder=aggregator_builder,
        end_of_batch_callback=end_of_batch_ops,
    )


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


PS = TypeVar("PS")  # prognostic state
TO = TypeVar("TO", bound="TrainOutputABC")  # train output
BD = TypeVar("BD")  # batch data
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


class AggregatorBuilder(AggregatorBuilderABC[PrognosticState, TrainOutput, PairedData]):
    def __init__(
        self,
        inference_config: InferenceEvaluatorAggregatorConfig,
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: timedelta,
        initial_inference_times: xr.DataArray,
        record_step_20: bool,
        n_timesteps: int,
        normalize: Callable[[TensorMapping], TensorDict],
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        loss_scaling: Optional[Dict[str, torch.Tensor]] = None,
        channel_mean_names: Optional[Sequence[str]] = None,
    ):
        self.inference_config = inference_config
        self.gridded_operations = gridded_operations
        self.sigma_coordinates = sigma_coordinates
        self.horizontal_coordinates = horizontal_coordinates
        self.timestep = timestep
        self.initial_inference_times = initial_inference_times
        self.record_step_20 = record_step_20
        self.n_timesteps = n_timesteps
        self.variable_metadata = variable_metadata
        self.loss_scaling = loss_scaling
        self.channel_mean_names = channel_mean_names
        self.normalize = normalize

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator()

    def get_validation_aggregator(self) -> OneStepAggregator:
        return OneStepAggregator(
            gridded_operations=self.gridded_operations,
            sigma_coordinates=self.sigma_coordinates.to(device=fme.get_device()),
            variable_metadata=self.variable_metadata,
            loss_scaling=self.loss_scaling,
        )

    def get_inference_aggregator(
        self,
    ) -> InferenceEvaluatorAggregator:
        return self.inference_config.build(
            sigma_coordinates=self.sigma_coordinates,
            horizontal_coordinates=self.horizontal_coordinates,
            timestep=self.timestep,
            initial_times=self.initial_inference_times,
            record_step_20=self.record_step_20,
            n_timesteps=self.n_timesteps,
            variable_metadata=self.variable_metadata,
            channel_mean_names=self.channel_mean_names,
            normalize=self.normalize,
        )


class Trainer:
    def __init__(
        self,
        train_data: GriddedDataABC[BD],
        validation_data: GriddedDataABC[BD],
        inference_data: GriddedDataABC[BD],
        stepper: StepperABC[BD, TO],
        optimization: Optimization,
        ema: EMATracker,
        config: TrainConfigProtocol,
        # TODO: PS and SD don't constrain anything yet, they will need to when
        # inference is implemented generically.
        aggregator_builder: AggregatorBuilderABC[PS, TO, SD],
        end_of_batch_callback: EndOfBatchCallback = lambda: None,
    ):
        logging.info(f"Current device is {fme.get_device()}")
        self.dist = Distributed.get_instance()
        if self.dist.is_root():
            if not os.path.isdir(config.experiment_dir):
                os.makedirs(config.experiment_dir)
            if not os.path.isdir(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
        self.config = config
        self.paths = CheckpointPaths(config.checkpoint_dir)

        logging.info("rank %d, begin data loader init" % self.dist.rank)
        self.train_data = train_data
        self.valid_data = validation_data
        logging.info("rank %d, data loader initialized" % self.dist.rank)
        for gridded_data, name in zip(
            (self.train_data, self.valid_data), ("train", "valid")
        ):
            gridded_data.log_info(name)

        self.num_batches_seen = 0
        self._start_epoch = 0
        self._model_epoch = self._start_epoch
        self.num_batches_seen = 0
        self._best_validation_loss = torch.inf
        self._best_inference_error = torch.inf

        self.stepper = stepper
        self.optimization = optimization
        self._end_of_batch_ops = end_of_batch_callback
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

        logging.info(
            (
                "Number of trainable model parameters: "
                f"{count_parameters(self.stepper.modules)}"
            )
        )

        self._inference_data = inference_data
        self._ema = ema

    def switch_off_grad(self, model: torch.nn.Module):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        logging.info("Starting Training Loop...")

        self._model_epoch = self._start_epoch
        inference_epochs = self.config.get_inference_epochs()
        if self.config.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self._start_epoch + self.config.segment_epochs, self.config.max_epochs
            )
        # "epoch" describes the loop, self._model_epoch describes model weights
        # needed so we can describe the loop even after weights are updated
        for epoch in range(self._start_epoch, segment_max_epochs):
            # garbage collect to avoid CUDA error in some contexts
            # https://github.com/pytorch/pytorch/issues/67978#issuecomment-1661986812  # noqa: E501
            gc.collect()
            logging.info(f"Epoch: {epoch+1}")
            self.train_data.set_epoch(epoch)

            start_time = time.time()
            logging.info(f"Starting training step on epoch {epoch + 1}")
            train_logs = self.train_one_epoch()
            train_end = time.time()
            logging.info(f"Starting validation step on epoch {epoch + 1}")
            valid_logs = self.validate_one_epoch()
            valid_end = time.time()
            if epoch in inference_epochs:
                logging.info(f"Starting inference step on epoch {epoch + 1}")
                inference_logs = self.inference_one_epoch()
                inference_end: Optional[float] = time.time()
            else:
                inference_logs = {}
                inference_end = None

            train_loss = train_logs["train/mean/loss"]
            valid_loss = valid_logs["val/mean/loss"]
            inference_error = inference_logs.get(
                "inference/time_mean_norm/rmse/channel_mean", None
            )
            # need to get the learning rate before stepping the scheduler
            lr = self.optimization.learning_rate
            self.optimization.step_scheduler(valid_loss)

            if self.dist.is_root():
                if self.config.save_checkpoint:
                    logging.info(f"Saving checkpoints for epoch {epoch + 1}")
                    self.save_all_checkpoints(valid_loss, inference_error)

            time_elapsed = time.time() - start_time
            logging.info(f"Time taken for epoch {epoch + 1} is {time_elapsed} sec")
            logging.info(f"Train loss: {train_loss}. Valid loss: {valid_loss}")
            if inference_error is not None:
                logging.info(f"Inference error: {inference_error}")

            logging.info("Logging to wandb")
            all_logs = {
                **train_logs,
                **valid_logs,
                **inference_logs,
                **{
                    "lr": lr,
                    "epoch": epoch,
                    "epoch_train_seconds": train_end - start_time,
                    "epoch_validation_seconds": valid_end - train_end,
                    "epoch_total_seconds": time_elapsed,
                },
            }
            if inference_end is not None:
                all_logs["epoch_inference_seconds"] = inference_end - valid_end
            wandb = WandB.get_instance()
            wandb.log(all_logs, step=self.num_batches_seen)
        if segment_max_epochs == self.config.max_epochs:
            self.config.clean_wandb(experiment_dir=self.config.experiment_dir)

    def train_one_epoch(self):
        """Train for one epoch and return logs from TrainAggregator."""
        wandb = WandB.get_instance()
        aggregator = self._aggregator_builder.get_train_aggregator()
        if self.num_batches_seen == 0:
            # Before training, log the loss on the first batch.
            with torch.no_grad():
                batch = next(iter(self.train_data.loader))
                stepped = self.stepper.train_on_batch(
                    batch,
                    optimization=self._no_optimization,
                )

                if self.config.log_train_every_n_batches > 0:
                    with torch.no_grad():
                        metrics = {
                            f"batch_{name}": self.dist.reduce_mean(metric)
                            for name, metric in sorted(stepped.get_metrics().items())
                        }
                    wandb.log(metrics, step=self.num_batches_seen)
        current_time = time.time()
        for batch in self.train_data.loader:
            stepped = self.stepper.train_on_batch(
                batch,
                self.optimization,
            )
            aggregator.record_batch(stepped)
            self._end_of_batch_ops()
            self._ema(model=self.stepper.modules)
            self.num_batches_seen += 1
            if (
                self.config.log_train_every_n_batches > 0
                and self.num_batches_seen % self.config.log_train_every_n_batches == 0
            ):
                with torch.no_grad():
                    metrics = {
                        f"batch_{name}": self.dist.reduce_mean(metric)
                        for name, metric in sorted(stepped.get_metrics().items())
                    }
                duration = time.time() - current_time
                current_time = time.time()
                n_samples = (
                    self.train_data.batch_size * self.config.log_train_every_n_batches
                )
                samples_per_second = n_samples / duration
                metrics["training_samples_per_second"] = samples_per_second
                wandb.log(metrics, step=self.num_batches_seen)
        self._model_epoch += 1

        return aggregator.get_logs(label="train")

    @contextlib.contextmanager
    def _validation_context(self):
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
        self._ema.copy_to(model=self.stepper.modules)
        try:
            yield
        finally:
            self._ema.restore(parameters=self.stepper.modules.parameters())

    def validate_one_epoch(self):
        aggregator = self._aggregator_builder.get_validation_aggregator()
        with torch.no_grad(), self._validation_context():
            for batch in self.valid_data.loader:
                stepped = self.stepper.train_on_batch(
                    batch,
                    optimization=NullOptimization(),
                    keep_initial_condition=True,
                )

                stepped = stepped.compute_derived_variables()
                aggregator.record_batch(
                    batch=stepped,
                )
        return aggregator.get_logs(label="val")

    def inference_one_epoch(self):
        aggregator = self._aggregator_builder.get_inference_aggregator()
        with torch.no_grad(), self._validation_context(), GlobalTimer():
            if (
                isinstance(aggregator, InferenceEvaluatorAggregator)
                and isinstance(
                    self.stepper,
                    SingleModuleStepper,
                )
                and isinstance(self._inference_data, GriddedData)
            ):
                # TODO: update inference to be generic, then remove this
                # if-block and this comment
                run_inference_evaluator(
                    aggregator=aggregator,
                    stepper=self.stepper,
                    data=self._inference_data,
                )
        logs = aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def save_checkpoint(self, checkpoint_path):
        # save to a temporary file in case we get pre-empted during save
        temporary_location = os.path.join(
            os.path.dirname(checkpoint_path), f".{uuid.uuid4()}.tmp"
        )
        try:
            torch.save(
                {
                    "num_batches_seen": self.num_batches_seen,
                    "epoch": self._model_epoch,
                    "best_validation_loss": self._best_validation_loss,
                    "best_inference_error": self._best_inference_error,
                    "stepper": self.stepper.get_state(),
                    "optimization": self.optimization.get_state(),
                    "ema": self._ema.get_state(),
                },
                temporary_location,
            )
            os.replace(temporary_location, checkpoint_path)
        finally:
            if os.path.exists(temporary_location):
                os.remove(temporary_location)

    def restore_checkpoint(self, checkpoint_path, ema_checkpoint_path):
        _restore_checkpoint(self, checkpoint_path, ema_checkpoint_path)

    def _epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.config.max_epochs, self.config.checkpoint_save_epochs
        )

    def _ema_epoch_checkpoint_enabled(self, epoch: int) -> bool:
        return epoch_checkpoint_enabled(
            epoch, self.config.max_epochs, self.config.ema_checkpoint_save_epochs
        )

    def save_all_checkpoints(self, valid_loss: float, inference_error: Optional[float]):
        logging.info(f"Saving latest checkpoint to {self.paths.latest_checkpoint_path}")
        self.save_checkpoint(self.paths.latest_checkpoint_path)
        if self._epoch_checkpoint_enabled(self._model_epoch):
            epoch_checkpoint_path = self.paths.epoch_checkpoint_path(self._model_epoch)
            logging.info(f"Saving epoch checkpoint to {epoch_checkpoint_path}")
            self.save_checkpoint(epoch_checkpoint_path)
        if self._ema_epoch_checkpoint_enabled(self._model_epoch):
            ema_epoch_checkpoint_path = self.paths.ema_epoch_checkpoint_path(
                self._model_epoch
            )
            logging.info(f"Saving EMA epoch checkpoint to {ema_epoch_checkpoint_path}")
            with self._ema_context():
                self.save_checkpoint(ema_epoch_checkpoint_path)
        if self.config.validate_using_ema:
            best_checkpoint_context = self._ema_context
        else:
            best_checkpoint_context = contextlib.nullcontext  # type: ignore
        with best_checkpoint_context():
            if valid_loss <= self._best_validation_loss:
                logging.info(
                    "Saving lowest validation loss checkpoint to "
                    f"{self.paths.best_checkpoint_path}"
                )
                self._best_validation_loss = valid_loss
                self.save_checkpoint(self.paths.best_checkpoint_path)
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
        with self._ema_context():
            logging.info(
                f"Saving latest EMA checkpoint to {self.paths.ema_checkpoint_path}"
            )
            self.save_checkpoint(self.paths.ema_checkpoint_path)


def epoch_checkpoint_enabled(
    epoch: int, max_epochs: int, save_epochs: Optional[Slice]
) -> bool:
    if save_epochs is None:
        return False
    return epoch in range(max_epochs)[save_epochs.slice]


def _restore_checkpoint(trainer: Trainer, checkpoint_path, ema_checkpoint_path):
    # separated into a function only to make it easier to mock
    checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
    # restore checkpoint is used for finetuning as well as resuming.
    # If finetuning (i.e., not resuming), restore checkpoint
    # does not load optimizer state, instead uses config specified lr.
    trainer.stepper.load_state(checkpoint["stepper"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer._start_epoch = checkpoint["epoch"]
    trainer._best_validation_loss = checkpoint["best_validation_loss"]
    trainer._best_inference_error = checkpoint["best_inference_error"]
    ema_checkpoint = torch.load(ema_checkpoint_path, map_location=fme.get_device())
    ema_stepper: SingleModuleStepper = SingleModuleStepper.from_state(
        ema_checkpoint["stepper"]
    )
    trainer._ema = EMATracker.from_state(checkpoint["ema"], ema_stepper.modules)


def run_train_from_config(config: TrainConfig):
    run_train(TrainBuilders(config), config)


def run_train(builders: TrainBuilders, config: TrainConfig):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config.logging.configure_logging(config.experiment_dir, log_filename="out.log")
    env_vars = logging_utils.retrieve_env_vars()
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    config_as_dict = to_flat_dict(dataclasses.asdict(config))
    config.logging.configure_wandb(
        config=config_as_dict, env_vars=env_vars, resume=True, notes=beaker_url
    )
    trainer = build_trainer(builders, config)
    trainer.train()
    logging.info("DONE ---- rank %d" % dist.rank)


def main(yaml_config: str):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir, exist_ok=True)
    with open(os.path.join(train_config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    run_train_from_config(train_config)
