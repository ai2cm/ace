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
import dacite
from fme.core.distributed import Distributed
from fme.fcn_training.utils.data_loader_fv3gfs import load_series_data
import argparse
import torch
import torch.nn as nn
import logging
from fme.fcn_training.utils import logging_utils
import yaml

from fme.fcn_training.utils.data_loader_multifiles import (
    get_data_loader,
)
from fme.core.wandb import WandB
from fme.fcn_training.train_config import TrainConfig

from fme.fcn_training.inference import inference
import fme
import netCDF4

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

        self.iters = 0
        self.startEpoch = 0

        self.epoch = self.startEpoch

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

        # TODO: refactor this into its own dataset configuration
        inference_ds = netCDF4.MFDataset(
            os.path.join(config.validation_data.data_path, "*.nc")
        )
        self.inference_data = load_series_data(
            idx=0,
            n_steps=config.prediction_length + 1,
            ds=inference_ds,
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

            train_loss = train_logs["loss"]
            valid_loss = valid_logs["valid_loss"]

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
            all_logs = {**train_logs, **valid_logs, **inference_logs, **{"lr": lr}}
            wandb.log(all_logs, step=self.epoch)

    def train_one_epoch(self):
        # TODO: clean up and merge train_one_epoch and validate_one_epoch
        # deduplicate code through helper routines or if conditionals

        for data in self.train_data_loader:
            loss, _, _, _ = self.stepper.run_on_batch(
                data, train=True, n_forward_steps=self.n_forward_steps
            )

        logs = {"loss": loss}

        for key in sorted(logs.keys()):
            logs[key] = self.dist.reduce_mean(logs[key].detach())

        return logs

    def validate_one_epoch(self):
        n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler

        area_weights = fme.spherical_area_weights(
            self.valid_dataset.img_shape_x,
            self.valid_dataset.img_shape_y,
            device=fme.get_device(),
        )
        valid_buff = torch.zeros((3), dtype=torch.float32, device=fme.get_device())
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)

        # It's not ideal to reach so deeply into the stepper config for the out names,
        # once we use aggregators we can instead lazily initialize these buffers
        # when we have an example of the output data and its names
        valid_weighted_rmse = {
            k: torch.zeros((1), dtype=torch.float32, device=fme.get_device())
            for k in self.config.stepper.out_names
        }
        valid_gradient_magnitude_diff = {
            k: torch.zeros((1), dtype=torch.float32, device=fme.get_device())
            for k in self.config.stepper.out_names
        }

        with torch.no_grad():
            image_logs = {}
            for i, data in enumerate(self.valid_data_loader, 0):
                if i >= n_valid_batches:
                    break
                valid_steps += 1.0
                (
                    batch_loss,
                    gen,
                    gen_norm,
                    data_norm,
                ) = self.stepper.run_on_batch(
                    data, train=False, n_forward_steps=self.n_forward_steps
                )
                valid_loss += batch_loss
                for name in gen_norm:
                    time_dim = 1
                    input_time = 0
                    target_time = 1
                    valid_l1 += nn.functional.l1_loss(
                        gen_norm[name].select(dim=time_dim, index=target_time),
                        data_norm[name].select(dim=time_dim, index=target_time),
                    ) / len(gen_norm[name].select(dim=time_dim, index=target_time))

                    # direct prediction weighted rmse
                    assert gen_norm[name].shape[time_dim] == 2, (
                        "if this assert fails, need to update diagnostics to "
                        "aggregate across multiple time dimension outputs",
                    )
                    gen_for_rmse = gen[name].select(dim=time_dim, index=target_time)
                    tar_for_rmse = (
                        data[name].select(dim=time_dim, index=target_time)
                    ).to(fme.get_device())
                    valid_weighted_rmse[name] += fme.root_mean_squared_error(
                        tar_for_rmse, gen_for_rmse, weights=area_weights, dim=[-2, -1]
                    ).mean(dim=0)
                    grad_percent_diff = fme.gradient_magnitude_percent_diff(
                        tar_for_rmse, gen_for_rmse, weights=area_weights, dim=[-2, -1]
                    ).mean(dim=0)
                    valid_gradient_magnitude_diff[name] += grad_percent_diff

                if i == 0:
                    for name in gen_norm:
                        gap = torch.zeros((self.valid_dataset.img_shape_x, 4)).to(
                            fme.get_device(), dtype=torch.float
                        )
                        gen_for_image = gen_norm[name].select(
                            dim=time_dim, index=target_time
                        )[
                            0
                        ]  # first sample in batch
                        target_for_image = data_norm[name].select(
                            dim=time_dim, index=target_time
                        )[0]
                        input_for_image = data_norm[name].select(
                            dim=time_dim, index=input_time
                        )[0]
                        image_error = gen_for_image - target_for_image
                        image_full_field = torch.cat(
                            (gen_for_image, gap, target_for_image), axis=1
                        )
                        image_residual = torch.cat(
                            (
                                gen_for_image - input_for_image,
                                gap,
                                target_for_image - input_for_image,
                            ),
                            axis=1,
                        )
                        caption = (
                            f"{name} one step full field for "
                            f"sample {i}; (left) generated and (right) target."
                        )
                        wandb_image = wandb.Image(image_full_field, caption=caption)
                        image_logs[f"image-full-field/sample{i}/{name}"] = wandb_image
                        caption = (
                            f"{name} one step residual for "
                            f"sample {i}; (left) generated and (right) target."
                        )
                        wandb_image = wandb.Image(image_residual, caption=caption)
                        image_logs[f"image-residual/sample{i}/{name}"] = wandb_image
                        caption = (
                            f"{name} one step error "
                            f"(generated - target) for sample {i}."
                        )
                        wandb_image = wandb.Image(image_error, caption=caption)
                        image_logs[f"image-error/sample{i}/{name}"] = wandb_image

        valid_buff = self.dist.reduce_sum(valid_buff)
        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        logs = {
            "valid_l1": valid_buff_cpu[1],
            "valid_loss": valid_buff_cpu[0],
        }
        for name in valid_weighted_rmse:
            valid_weighted_rmse[name] = self.dist.reduce_sum(valid_weighted_rmse[name])

            valid_weighted_rmse[name] = valid_weighted_rmse[name] / valid_buff[2]
            valid_gradient_magnitude_diff[name] = (
                valid_gradient_magnitude_diff[name] / valid_buff[2]
            )

            # download buffers
            valid_weighted_rmse_cpu = valid_weighted_rmse[name].detach().cpu().numpy()
            valid_gradient_magnitude_diff_cpu = (
                valid_gradient_magnitude_diff[name].detach().cpu().numpy()
            )
            logs[f"valid_rmse/{name}"] = valid_weighted_rmse_cpu
            logs[
                f"valid_gradient_magnitude_percent_diff/{name}"
            ] = valid_gradient_magnitude_diff_cpu

        validation_logs = {**logs, **image_logs}
        return validation_logs

    def inference_one_epoch(self):
        # TODO: refactor inference to be more clearly reusable between
        # training, validation, and inference
        logging.info("Starting inference on validation set...")

        with torch.no_grad():
            (
                _,
                _,
                inference_logs,
            ) = inference.autoregressive_inference(
                ic=0,
                valid_data_full=self.inference_data,
                stepper=self.stepper,
                log_on_each_unroll_step=False,
                prediction_length=self.config.prediction_length,
            )

        return inference_logs

    def save_checkpoint(self, checkpoint_path, model=None):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        if not model:
            model = self.stepper.module

        torch.save(
            {
                "iters": self.iters,
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
    trainer.iters = checkpoint["iters"]
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
    train_config.configure_logging(log_filename="out.log")
    train_config.configure_wandb()
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
