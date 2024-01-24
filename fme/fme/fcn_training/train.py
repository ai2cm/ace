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
from fme.core.aggregator import InferenceAggregator, OneStepAggregator, TrainAggregator
from fme.core.aggregator.null import NullAggregator
from fme.core.data_loading.get_loader import get_data_loader
from fme.core.distributed import Distributed
from fme.core.optimization import NullOptimization
from fme.core.wandb import WandB
from fme.fcn_training.inference import run_inference
from fme.fcn_training.inference.derived_variables import (
    compute_stepped_derived_quantities,
)
from fme.fcn_training.train_config import TrainConfig
from fme.fcn_training.utils import gcs_utils, logging_utils


class Trainer:
    def count_parameters(self):
        parameters = 0
        for module in self.stepper.modules:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameters += parameter.numel()
        return parameters

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
            config.train_data,
            requirements=data_requirements,
            train=True,
        )
        self.valid_data = get_data_loader(
            config.validation_data,
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

        self.num_batches_seen = 0
        self.startEpoch = 0

        self._model_epoch = self.startEpoch
        self.num_batches_seen = 0

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
        self._no_optimization = NullOptimization()

        if config.resuming:
            logging.info("Loading checkpoint %s" % config.latest_checkpoint_path)
            self.restore_checkpoint(config.latest_checkpoint_path)

        wandb = WandB.get_instance()
        wandb.watch(self.stepper.modules)

        logging.info(f"Number of trainable model parameters: {self.count_parameters()}")
        inference_data_requirements = dataclasses.replace(data_requirements)
        inference_data_requirements.n_timesteps = config.inference.n_forward_steps + 1

        def get_inference_data_loader(window_time_slice: Optional[slice] = None):
            with logging_utils.log_level(logging.WARNING):
                return get_data_loader(
                    config.inference.data,
                    train=False,
                    requirements=inference_data_requirements,
                    window_time_slice=window_time_slice,
                )

        self._inference_data_loader_factory = get_inference_data_loader
        self._ema = self.config.ema.build(self.stepper.modules)

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        logging.info("Starting Training Loop...")

        best_valid_loss = torch.inf
        best_inference_error = torch.inf
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
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.config.latest_checkpoint_path)
                    if self.config.epoch_checkpoint_enabled(self._model_epoch):
                        self.save_checkpoint(
                            self.config.epoch_checkpoint_path(self._model_epoch)
                        )
                    if self.config.validate_using_ema:
                        best_checkpoint_context = self._ema_context
                    else:
                        best_checkpoint_context = contextlib.nullcontext
                    with best_checkpoint_context():
                        if valid_loss <= best_valid_loss:
                            self.save_checkpoint(self.config.best_checkpoint_path)
                            best_valid_loss = valid_loss
                        if inference_error is not None and (
                            inference_error <= best_inference_error
                        ):
                            self.save_checkpoint(
                                self.config.best_inference_checkpoint_path
                            )
                            best_inference_error = inference_error
                    with self._ema_context():
                        self.save_checkpoint(self.config.ema_checkpoint_path)

            time_elapsed = time.time() - start_time
            logging.info(f"Time taken for epoch {epoch + 1} is {time_elapsed} sec")
            logging.info(f"Train loss: {train_loss}. Valid loss: {valid_loss}")

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
        for batch in self.train_data.loader:
            stepped = self.stepper.run_on_batch(
                batch.data,
                self.optimization,
                n_forward_steps=self.config.n_forward_steps,
                aggregator=aggregator,
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
                    aggregator=NullAggregator(),
                )
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
        aggregator = InferenceAggregator(
            self.train_data.area_weights.to(fme.get_device()),
            self.train_data.sigma_coordinates,
            record_step_20=record_step_20,
            log_video=False,
            log_zonal_mean_images=True,
            n_timesteps=self.config.inference.n_forward_steps + 1,
            enable_extended_videos=False,
            metadata=self.train_data.metadata,
        )
        with torch.no_grad(), self._validation_context():
            run_inference(
                aggregator=aggregator,
                stepper=self.stepper,
                data_loader_factory=self._inference_data_loader_factory,
                n_forward_steps=self.config.inference.n_forward_steps,
                forward_steps_in_memory=self.config.inference.forward_steps_in_memory,
            )
        logs = aggregator.get_logs(label="inference")
        return logs

    def save_checkpoint(self, checkpoint_path):
        torch.save(
            {
                "num_batches_seen": self.num_batches_seen,
                "epoch": self._model_epoch,
                "stepper": self.stepper.get_state(),
                "optimization": self.optimization.get_state(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        _restore_checkpoint(self, checkpoint_path)


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
    gcs_utils.authenticate()
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
