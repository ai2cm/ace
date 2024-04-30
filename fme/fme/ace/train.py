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

import argparse
import contextlib
import dataclasses
import logging
import os
import time
from typing import Optional

import dacite
import torch
import yaml

import fme
from fme.ace.inference import run_inference
from fme.ace.inference.derived_variables import compute_stepped_derived_quantities
from fme.ace.train_config import TrainConfig
from fme.ace.utils import logging_utils
from fme.core.aggregator import (
    InferenceAggregatorConfig,
    OneStepAggregator,
    TrainAggregator,
)
from fme.core.data_loading.getters import get_data_loader, get_inference_data
from fme.core.data_loading.utils import BatchData
from fme.core.distributed import Distributed
from fme.core.optimization import NullOptimization
from fme.core.wandb import WandB


def count_parameters(modules: torch.nn.ModuleList) -> int:
    parameters = 0
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad:
                parameters += parameter.numel()
    return parameters


class Trainer:
    def __init__(self, config: TrainConfig):
        self.dist = Distributed.get_instance()
        if self.dist.is_root():
            if not os.path.isdir(config.experiment_dir):
                os.makedirs(config.experiment_dir)
            if not os.path.isdir(config.checkpoint_dir):
                os.makedirs(config.checkpoint_dir)
        self.config = config

        data_requirements = config.stepper.get_data_requirements(
            n_forward_steps=self.config.n_forward_steps
        )
        logging.info("rank %d, begin data loader init" % self.dist.rank)
        self.train_data = get_data_loader(
            config.train_loader,
            requirements=data_requirements,
            train=True,
        )
        self.valid_data = get_data_loader(
            config.validation_loader,
            requirements=data_requirements,
            train=False,
        )
        logging.info("rank %d, data loader initialized" % self.dist.rank)
        for gridded_data, name in zip(
            (self.train_data, self.valid_data), ("train", "valid")
        ):
            n_samples = len(gridded_data.loader.dataset)
            n_batches = len(gridded_data.loader)
            logging.info(f"{name} data: {n_samples} samples, {n_batches} batches")
            first_time = gridded_data.loader.dataset[0][1].values[0]
            last_time = gridded_data.loader.dataset[-1][1].values[0]
            logging.info(f"{name} data: first sample's initial time: {first_time}")
            logging.info(f"{name} data: last sample's initial time: {last_time}")

        self.num_batches_seen = 0
        self.startEpoch = 0
        self._model_epoch = self.startEpoch
        self.num_batches_seen = 0
        self._best_validation_loss = torch.inf
        self._best_inference_error = torch.inf

        for batch in self.train_data.loader:
            shapes = {k: v.shape for k, v in batch.data.items()}
            for value in shapes.values():
                img_shape = value[-2:]
                break
            break
        logging.info("Starting model initialization")
        self.stepper = config.stepper.get_stepper(
            img_shape=img_shape,
            area=self.train_data.area_weights,
            sigma_coordinates=self.train_data.sigma_coordinates,
        )
        self.optimization = config.optimization.build(
            self.stepper.module.parameters(), config.max_epochs
        )
        self._base_weights = self.config.stepper.get_base_weights()
        self._copy_after_batch = config.copy_weights_after_batch
        self._no_optimization = NullOptimization()

        if config.resuming:
            logging.info("Loading checkpoint %s" % config.latest_checkpoint_path)
            self.restore_checkpoint(config.latest_checkpoint_path)

        wandb = WandB.get_instance()
        wandb.watch(self.stepper.modules)

        logging.info(
            (
                "Number of trainable model parameters: "
                f"{count_parameters(self.stepper.modules)}"
            )
        )
        inference_data_requirements = dataclasses.replace(data_requirements)
        inference_data_requirements.n_timesteps = config.inference.n_forward_steps + 1

        self._inference_data = get_inference_data(
            config.inference.loader,
            config.inference.forward_steps_in_memory,
            inference_data_requirements,
        )

        self._ema = self.config.ema.build(self.stepper.modules)

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        logging.info("Starting Training Loop...")

        self._model_epoch = self.startEpoch
        inference_epochs = list(range(0, self.config.max_epochs))[
            self.config.inference.epochs.slice
        ]
        if self.config.segment_epochs is None:
            segment_max_epochs = self.config.max_epochs
        else:
            segment_max_epochs = min(
                self.startEpoch + self.config.segment_epochs, self.config.max_epochs
            )
        # "epoch" describes the loop, self._model_epoch describes model weights
        # needed so we can describe the loop even after weights are updated
        for epoch in range(self.startEpoch, segment_max_epochs):
            logging.info(f"Epoch: {epoch+1}")
            if self.train_data.sampler is not None:
                self.train_data.sampler.set_epoch(epoch)

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
            self.config.clean_wandb()

    def train_one_epoch(self):
        """Train for one epoch and return logs from TrainAggregator."""
        wandb = WandB.get_instance()
        aggregator = TrainAggregator()
        if self.num_batches_seen == 0:
            # Before training, log the loss on the first batch.
            with torch.no_grad():
                batch = next(iter(self.train_data.loader))
                stepped = self.stepper.run_on_batch(
                    batch.data,
                    optimization=self._no_optimization,
                    n_forward_steps=self.config.n_forward_steps,
                )

                if self.config.log_train_every_n_batches > 0:
                    with torch.no_grad():
                        metrics = {
                            f"batch_{name}": self.dist.reduce_mean(metric)
                            for name, metric in sorted(stepped.metrics.items())
                        }
                    wandb.log(metrics, step=self.num_batches_seen)
        batch: BatchData
        current_time = time.time()
        for batch in self.train_data.loader:
            stepped = self.stepper.run_on_batch(
                batch.data,
                self.optimization,
                n_forward_steps=self.config.n_forward_steps,
            )
            aggregator.record_batch(stepped.metrics["loss"])
            if self._base_weights is not None:
                self._copy_after_batch.apply(
                    weights=self._base_weights, modules=self.stepper.modules
                )
            self._ema(model=self.stepper.modules)
            self.num_batches_seen += 1
            if (
                self.config.log_train_every_n_batches > 0
                and self.num_batches_seen % self.config.log_train_every_n_batches == 0
            ):
                with torch.no_grad():
                    metrics = {
                        f"batch_{name}": self.dist.reduce_mean(metric)
                        for name, metric in sorted(stepped.metrics.items())
                    }
                duration = time.time() - current_time
                current_time = time.time()
                n_samples = (
                    self.train_data.loader.batch_size
                    * self.config.log_train_every_n_batches
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
        aggregator = OneStepAggregator(
            self.train_data.area_weights.to(fme.get_device()),
            self.train_data.sigma_coordinates,
            self.train_data.metadata,
        )

        with torch.no_grad(), self._validation_context():
            for batch in self.valid_data.loader:
                stepped = self.stepper.run_on_batch(
                    batch.data,
                    optimization=NullOptimization(),
                    n_forward_steps=self.config.n_forward_steps,
                )
                # Prepend initial condition back to start of windows
                # as it's used to compute differenced quantities
                ic, normed_ic = self.stepper.get_initial_condition(batch.data)
                stepped = stepped.prepend_initial_condition(ic, normed_ic)

                stepped = compute_stepped_derived_quantities(
                    stepped, self.valid_data.sigma_coordinates
                )
                aggregator.record_batch(
                    loss=stepped.metrics["loss"],
                    target_data=stepped.target_data,
                    gen_data=stepped.gen_data,
                    target_data_norm=stepped.target_data_norm,
                    gen_data_norm=stepped.gen_data_norm,
                )
        return aggregator.get_logs(label="val")

    def inference_one_epoch(self):
        record_step_20 = self.config.inference.n_forward_steps >= 20
        aggregator_config: InferenceAggregatorConfig = self.config.inference.aggregator
        aggregator = aggregator_config.build(
            area_weights=self.train_data.area_weights.to(fme.get_device()),
            sigma_coordinates=self.train_data.sigma_coordinates,
            record_step_20=record_step_20,
            n_timesteps=self.config.inference.n_forward_steps + 1,
            metadata=self.train_data.metadata,
        )
        with torch.no_grad(), self._validation_context():
            run_inference(
                aggregator=aggregator,
                stepper=self.stepper,
                data=self._inference_data,
                n_forward_steps=self.config.inference.n_forward_steps,
                forward_steps_in_memory=self.config.inference.forward_steps_in_memory,
            )
        logs = aggregator.get_logs(label="inference")
        if "inference/mean/series" in logs:
            # Tables don't work well when reported every epoch, this is a quick
            # workaround to remove them. Could refactor to avoid returning
            # at all, but it's used when converting the logs to epoch-wise
            # wandb logs in standalone inference.
            logs.pop("inference/mean/series")
        if "inference/mean_norm/series" in logs:
            logs.pop("inference/mean_norm/series")
        return logs

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                "num_batches_seen": self.num_batches_seen,
                "epoch": self._model_epoch,
                "best_validation_loss": self._best_validation_loss,
                "best_inference_error": self._best_inference_error,
                "stepper": self.stepper.get_state(),
                "optimization": self.optimization.get_state(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        _restore_checkpoint(self, checkpoint_path)

    def save_all_checkpoints(self, valid_loss: float, inference_error: Optional[float]):
        logging.info(
            f"Saving latest checkpoint to {self.config.latest_checkpoint_path}"
        )
        self.save_checkpoint(self.config.latest_checkpoint_path)
        if self.config.epoch_checkpoint_enabled(self._model_epoch):
            epoch_checkpoint_path = self.config.epoch_checkpoint_path(self._model_epoch)
            logging.info(f"Saving epoch checkpoint to {epoch_checkpoint_path}")
            self.save_checkpoint(epoch_checkpoint_path)
        if self.config.validate_using_ema:
            best_checkpoint_context = self._ema_context
        else:
            best_checkpoint_context = contextlib.nullcontext  # type: ignore
        with best_checkpoint_context():
            if valid_loss <= self._best_validation_loss:
                logging.info(
                    "Saving lowest validation loss checkpoint to "
                    f"{self.config.best_checkpoint_path}"
                )
                self._best_validation_loss = valid_loss
                self.save_checkpoint(self.config.best_checkpoint_path)
            if inference_error is not None and (
                inference_error <= self._best_inference_error
            ):
                logging.info(
                    f"Epoch inference error ({inference_error}) is lower than "
                    f"previous best inference error ({self._best_inference_error})."
                )
                logging.info(
                    "Saving lowest inference error checkpoint to "
                    f"{self.config.best_inference_checkpoint_path}"
                )
                self._best_inference_error = inference_error
                self.save_checkpoint(self.config.best_inference_checkpoint_path)
        with self._ema_context():
            logging.info(
                f"Saving latest EMA checkpoint to {self.config.ema_checkpoint_path}"
            )
            self.save_checkpoint(self.config.ema_checkpoint_path)


def _restore_checkpoint(trainer: Trainer, checkpoint_path):
    # separated into a function only to make it easier to mock
    checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
    # restore checkpoint is used for finetuning as well as resuming.
    # If finetuning (i.e., not resuming), restore checkpoint
    # does not load optimizer state, instead uses config specified lr.
    trainer.stepper.load_state(checkpoint["stepper"])
    trainer.optimization.load_state(checkpoint["optimization"])
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["epoch"]
    trainer._best_validation_loss = checkpoint["best_validation_loss"]
    trainer._best_inference_error = checkpoint["best_inference_error"]


def main(yaml_config: str):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir)
    with open(os.path.join(train_config.experiment_dir, "config.yaml"), "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    train_config.configure_logging(log_filename="out.log")
    env_vars = logging_utils.retrieve_env_vars()
    logging_utils.log_versions()
    beaker_url = logging_utils.log_beaker_url()
    train_config.configure_wandb(env_vars=env_vars, resume=True, notes=beaker_url)
    trainer = Trainer(train_config)
    trainer.train()
    logging.info("DONE ---- rank %d" % dist.rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", required=True, type=str)

    args = parser.parse_args()
    main(yaml_config=args.yaml_config)
