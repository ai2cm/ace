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

import os
import time
from fme.core.aggregator import OneStepAggregator, InferenceAggregator, TrainAggregator
import dacite
from fme.core.distributed import Distributed
import argparse
import torch
import logging
from fme.fcn_training.utils import logging_utils
import yaml

from fme.fcn_training.utils.data_loader_multifiles import (
    get_data_loader,
)
from fme.fcn_training.utils.data_utils import load_series_data
from fme.core.wandb import WandB
from fme.fcn_training.train_config import TrainConfig

import fme

wandb = WandB.get_instance()


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

        self.n_forward_steps = 1
        if self.n_forward_steps != 1:
            raise NotImplementedError(
                "diagnostics code not updated for n_forward_steps != 1"
            )
        data_requirements = config.stepper.get_data_requirements(
            n_forward_steps=self.n_forward_steps
        )
        logging.info("rank %d, begin data loader init" % self.dist.rank)
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            config.train_data,
            requirements=data_requirements,
            train=True,
        )
        self.valid_data_loader, self.valid_dataset = get_data_loader(
            config.validation_data,
            requirements=data_requirements,
            train=False,
        )
        logging.info("rank %d, data loader initialized" % self.dist.rank)

        self.num_batches_seen = 0
        self.startEpoch = 0

        self.epoch = self.startEpoch
        self.num_batches_seen = 0

        for data in self.train_data_loader:
            shapes = {k: v.shape for k, v in data.items()}
            break
        self.stepper = config.stepper.get_stepper(shapes, max_epochs=config.max_epochs)

        if config.resuming:
            logging.info("Loading checkpoint %s" % config.checkpoint_path)
            self.restore_checkpoint(config.checkpoint_path)

        wandb.watch(self.stepper.modules)

        logging.info(
            "Number of trainable model parameters: {}".format(self.count_parameters())
        )
        self.inference_n_forward_steps = config.inference_n_forward_steps
        # TODO: refactor this into its own dataset configuration
        self.inference_data = load_series_data(
            idx=0,
            n_steps=config.inference_n_forward_steps + 1,
            ds=self.valid_dataset.ds,
            names=data_requirements.names,
        )

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        logging.info("Starting Training Loop...")

        best_valid_loss = 1.0e6
        for epoch in range(self.startEpoch, self.config.max_epochs):
            self.epoch = epoch
            logging.info(f"Epoch: {epoch+1}")
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            start_time = time.time()
            train_logs = self.train_one_epoch()
            valid_logs = self.validate_one_epoch()
            inference_logs = self.inference_one_epoch()

            train_loss = train_logs["train/mean/loss"]
            valid_loss = valid_logs["val/mean/loss"]

            self.stepper.step_scheduler(valid_loss)

            if self.dist.is_root():
                if self.config.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.config.checkpoint_path)
                    if valid_loss <= best_valid_loss:
                        self.save_checkpoint(self.config.best_checkpoint_path)
                        best_valid_loss = valid_loss

            time_elapsed = time.time() - start_time
            logging.info(f"Time taken for epoch {epoch + 1} is {time_elapsed} sec")
            logging.info(f"Train loss: {train_loss}. Valid loss: {valid_loss}")

            logging.info("Logging to wandb")
            for pg in self.stepper.optimization.optimizer.param_groups:
                lr = pg["lr"]
            all_logs = {
                **train_logs,
                **valid_logs,
                **inference_logs,
                **{
                    "lr": lr,
                    "epoch": self.epoch,
                },
            }
            wandb.log(all_logs, step=self.num_batches_seen)

    def train_one_epoch(self):
        """Train for one epoch and log batch losses to wandb. Returns the final
        batch loss for the current epoch."""
        # TODO: clean up and merge train_one_epoch and validate_one_epoch
        # deduplicate code through helper routines or if conditionals

        aggregator = TrainAggregator()
        if self.num_batches_seen == 0:
            # Before training, log the loss on the first batch.
            with torch.no_grad():
                data = next(iter(self.train_data_loader))
                batch_loss, _, _, _ = self.stepper.run_on_batch(
                    data, train=False, n_forward_steps=self.n_forward_steps
                )

                if self.config.log_train_every_n_batches > 0:
                    wandb.log(
                        {"batch_loss": self.dist.reduce_mean(batch_loss)},
                        step=self.num_batches_seen,
                    )
        for data in self.train_data_loader:
            batch_loss, _, _, _ = self.stepper.run_on_batch(
                data,
                train=True,
                n_forward_steps=self.n_forward_steps,
                aggregator=aggregator,
            )
            self.num_batches_seen += 1
            if (
                self.config.log_train_every_n_batches > 0
                and self.num_batches_seen % self.config.log_train_every_n_batches == 0
            ):
                reduced_batch_loss = self.dist.reduce_mean(batch_loss)
                wandb.log(
                    {"batch_loss": reduced_batch_loss},
                    step=self.num_batches_seen,
                )

        return aggregator.get_logs(label="train")

    def validate_one_epoch(self):
        n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler

        aggregator = OneStepAggregator(
            self.train_dataset.area_weights.to(fme.get_device())
        )

        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                if i >= n_valid_batches:
                    break
                self.stepper.run_on_batch(
                    data,
                    train=False,
                    n_forward_steps=self.n_forward_steps,
                    aggregator=aggregator,
                )
        return aggregator.get_logs(label="val")

    def inference_one_epoch(self):
        # TODO: refactor inference to be more clearly reusable between
        # training, validation, and inference
        logging.info("Starting inference on validation set...")

        valid_data = {
            name: tensor[: self.config.inference_n_forward_steps + 1].unsqueeze(0)
            for name, tensor in self.inference_data.items()
        }
        record_step_20 = self.config.inference_n_forward_steps >= 20
        aggregator = InferenceAggregator(
            self.train_dataset.area_weights.to(fme.get_device()),
            record_step_20=record_step_20,
            log_video=False,
        )
        with torch.no_grad():
            self.stepper.run_on_batch(
                data=valid_data,
                train=False,
                n_forward_steps=self.config.inference_n_forward_steps,
                aggregator=aggregator,
            )
        return aggregator.get_logs(label="inference")

    def save_checkpoint(self, checkpoint_path, model=None):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        if not model:
            model = self.stepper.module

        torch.save(
            {
                "num_batches_seen": self.num_batches_seen,
                "epoch": self.epoch,
                "stepper": self.stepper.get_state(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""
        _restore_checkpoint(self, checkpoint_path)


def _restore_checkpoint(trainer: Trainer, checkpoint_path):
    # separated into a function only to make it easier to mock
    checkpoint = torch.load(checkpoint_path, map_location=fme.get_device())
    # restore checkpoint is used for finetuning as well as resuming.
    # If finetuning (i.e., not resuming), restore checkpoint
    # does not load optimizer state, instead uses config specified lr.
    load_optimizer = trainer.config.resuming
    trainer.stepper.load_state(checkpoint["stepper"], load_optimizer=load_optimizer)
    trainer.num_batches_seen = checkpoint["num_batches_seen"]
    trainer.startEpoch = checkpoint["epoch"]


def main(
    yaml_config: str,
):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
        with open(os.path.join(data["experiment_dir"], "config.yaml"), "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )

    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir)
    train_config.configure_logging(log_filename="out.log")
    train_config.configure_wandb(resume=True)
    logging_utils.log_versions()
    logging_utils.log_beaker_url()
    trainer = Trainer(train_config)
    trainer.train()
    logging.info("DONE ---- rank %d" % dist.rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)

    args = parser.parse_args()
    main(
        yaml_config=args.yaml_config,
    )
