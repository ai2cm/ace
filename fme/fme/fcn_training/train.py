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
import numpy as np
import argparse
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
import wandb
from utils.weighted_acc_rmse import (
    weighted_rmse_torch,
    weighted_global_mean_gradient_magnitude,
)
from apex import optimizers
from utils.darcy_loss import LpLoss
from collections import OrderedDict

DECORRELATION_TIME = 36  # 9 days
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from registry import NET_REGISTRY
from inference import inference


class Trainer:
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, world_rank):
        self.params = params
        self.world_rank = world_rank
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        )

        if params.log_to_wandb:
            wandb.init(config=params, project=params.project, entity=params.entity)
            logging_utils.log_beaker_url()

        logging.info("rank %d, begin data loader init" % world_rank)
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params, params.train_data_path, dist.is_initialized(), train=True
        )
        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params, params.valid_data_path, dist.is_initialized(), train=False
        )
        self.loss_obj = LpLoss()
        logging.info("rank %d, data loader initialized" % world_rank)

        params.crop_size_x = self.valid_dataset.crop_size_x
        params.crop_size_y = self.valid_dataset.crop_size_y
        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y
        # following two params needed by FourierNeuralOperatorNet
        params.img_crop_shape_x = self.valid_dataset.img_shape_x
        params.img_crop_shape_y = self.valid_dataset.img_shape_y
        params.N_in_channels = self.train_dataset.n_in_channels
        params.N_out_channels = self.train_dataset.n_out_channels
        params.in_names = self.train_dataset.in_names
        params.out_names = self.train_dataset.out_names

        BackboneNet = NET_REGISTRY[params.nettype]

        self.model = BackboneNet(params).to(self.device)

        if self.params.enable_nhwc:
            # NHWC: Convert model to channels_last memory format
            self.model = self.model.to(memory_format=torch.channels_last)

        if params.log_to_wandb:
            wandb.watch(self.model)

        if params.optimizer_type == "FusedAdam":
            self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr=params.lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)

        if params.enable_amp:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[params.local_rank],
                output_device=[params.local_rank],
                find_unused_parameters=True,
            )

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)
        if params.two_step_training:
            if params.pretrained and (not params.resuming):
                logging.info(
                    "Starting from pretrained one-step afno model at %s"
                    % params.pretrained_ckpt_path
                )
                self.restore_checkpoint(params.pretrained_ckpt_path)
                self.iters = 0
                self.startEpoch = 0

        self.epoch = self.startEpoch

        if params.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif params.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=params.max_epochs, last_epoch=self.startEpoch - 1
            )
        else:
            self.scheduler = None

        if params.log_to_screen:
            logging.info(
                "Number of trainable model parameters: {}".format(
                    self.count_parameters()
                )
            )

    def switch_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def train(self):
        if self.params.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.0e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if self.params.log_to_screen:
                logging.info(f"Epoch: {epoch+1}")
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)

            start_time = time.time()
            _, _, train_logs = self.train_one_epoch()
            _, valid_logs = self.validate_one_epoch()
            inference_logs = self.inference_one_epoch()

            train_loss = train_logs["loss"]
            valid_loss = valid_logs["valid_loss"]

            if self.params.scheduler == "ReduceLROnPlateau":
                self.scheduler.step(valid_loss)
            elif self.params.scheduler == "CosineAnnealingLR":
                self.scheduler.step()

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    # checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_loss <= best_valid_loss:
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_loss

            if self.params.log_to_screen:
                time_elapsed = time.time() - start_time
                logging.info(f"Time taken for epoch {epoch + 1} is {time_elapsed} sec")
                logging.info(f"Train loss: {train_loss}. Valid loss: {valid_loss}")

            if self.params.log_to_wandb:
                logging.info("Logging to wandb")
                for pg in self.optimizer.param_groups:
                    lr = pg["lr"]
                all_logs = {**train_logs, **valid_logs, **inference_logs, **{"lr": lr}}
                wandb.log(all_logs, step=self.epoch)

    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        self.model.train()

        for i, data in enumerate(self.train_data_loader, 0):
            self.iters += 1
            data_start = time.time()
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

            if self.params.enable_nhwc:
                inp = inp.to(memory_format=torch.channels_last)
                tar = tar.to(memory_format=torch.channels_last)

            data_time += time.time() - data_start

            tr_start = time.time()

            self.model.zero_grad()
            if self.params.two_step_training:
                with amp.autocast(self.params.enable_amp):
                    gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                    loss_step_one = self.loss_obj(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )
                    gen_step_two = self.model(gen_step_one).to(
                        self.device, dtype=torch.float
                    )
                    loss_step_two = self.loss_obj(
                        gen_step_two,
                        tar[
                            :,
                            self.params.N_out_channels : 2 * self.params.N_out_channels,
                        ],
                    )
                    loss = loss_step_one + loss_step_two
            else:
                with amp.autocast(self.params.enable_amp):
                    gen = self.model(inp).to(self.device, dtype=torch.float)
                    loss = self.loss_obj(gen, tar)

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            tr_time += time.time() - tr_start

        if self.params.two_step_training:
            logs = {
                "loss": loss,
                "loss_step_one": loss_step_one,
                "loss_step_two": loss_step_two,
            }
        else:
            logs = {"loss": loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()
        n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler
        if self.params.normalization == "minmax":
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == "zscore":
            mult = torch.as_tensor(self._load_global_output_stds()).to(self.device)

        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros(
            (self.params.N_out_channels), dtype=torch.float32, device=self.device
        )
        valid_gradient_magnitude_diff = torch.zeros(
            (self.params.N_out_channels), dtype=torch.float32, device=self.device
        )

        valid_start = time.time()

        with torch.no_grad():
            image_logs = {}
            for i, data in enumerate(self.valid_data_loader, 0):
                if i >= n_valid_batches:
                    break
                inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

                if self.params.two_step_training:
                    gen_step_one = self.model(inp).to(self.device, dtype=torch.float)
                    loss_step_one = self.loss_obj(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )

                    gen_step_two = self.model(gen_step_one).to(
                        self.device, dtype=torch.float
                    )

                    loss_step_two = self.loss_obj(
                        gen_step_two,
                        tar[
                            :,
                            self.params.N_out_channels : 2 * self.params.N_out_channels,
                        ],
                    )
                    valid_loss += loss_step_one + loss_step_two
                    valid_l1 += nn.functional.l1_loss(
                        gen_step_one, tar[:, 0 : self.params.N_out_channels]
                    )
                else:
                    gen = self.model(inp).to(self.device, dtype=torch.float)
                    valid_loss += self.loss_obj(gen, tar)
                    valid_l1 += nn.functional.l1_loss(gen, tar)

                valid_steps += 1.0

                # direct prediction weighted rmse
                if self.params.two_step_training:
                    gen_for_rmse = gen_step_one
                    tar_for_rmse = tar[:, 0 : self.params.N_out_channels]
                else:
                    gen_for_rmse = gen
                    tar_for_rmse = tar
                valid_weighted_rmse += weighted_rmse_torch(gen_for_rmse, tar_for_rmse)
                gen_gradient_magnitude = weighted_global_mean_gradient_magnitude(
                    gen_for_rmse
                )
                tar_gradient_magnitude = weighted_global_mean_gradient_magnitude(
                    tar_for_rmse
                )
                valid_gradient_magnitude_diff += (
                    100
                    * (gen_gradient_magnitude - tar_gradient_magnitude)
                    / tar_gradient_magnitude
                )

                if i % 10 == 0:
                    for j in range(gen.shape[1]):
                        name = self.valid_dataset.out_names[j]
                        gap = torch.zeros((self.valid_dataset.img_shape_x, 4)).to(
                            self.device, dtype=torch.float
                        )
                        gen_for_image = (
                            gen_step_one[0, j]
                            if self.params.two_step_training
                            else gen[0, j]
                        )
                        image_error = gen_for_image - tar[0, j]
                        image_full_field = torch.cat(
                            (gen_for_image, gap, tar[0, j]), axis=1
                        )
                        image_residual = torch.cat(
                            (gen_for_image - inp[0, j], gap, tar[0, j] - inp[0, j]),
                            axis=1,
                        )
                        if self.params.log_to_wandb:
                            caption = (
                                f"Channel {j} ({name}) one step full field for "
                                f"sample {i}; (left) generated and (right) target."
                            )
                            wandb_image = wandb.Image(image_full_field, caption=caption)
                            image_logs[
                                f"image-full-field/sample{i}/channel{j}-{name}"
                            ] = wandb_image
                            caption = (
                                f"Channel {j} ({name}) one step residual for "
                                f"sample {i}; (left) generated and (right) target."
                            )
                            wandb_image = wandb.Image(image_residual, caption=caption)
                            image_logs[
                                f"image-residual/sample{i}/channel{j}-{name}"
                            ] = wandb_image
                            caption = (
                                f"Channel {j} ({name}) one step error "
                                f"(generated - target) for sample {i}."
                            )
                            wandb_image = wandb.Image(image_error, caption=caption)
                            image_logs[
                                f"image-error/sample{i}/channel{j}-{name}"
                            ] = wandb_image
                        else:
                            image_path = os.path.join(
                                self.params["experiment_dir"],
                                f"sample{i}",
                                f"channel{j}",
                                f"epoch{self.epoch}.png",
                            )
                            os.makedirs(os.path.dirname(image_path), exist_ok=True)
                            save_image(image_full_field, image_path)

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        valid_gradient_magnitude_diff = valid_gradient_magnitude_diff / valid_buff[2]
        valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()
        valid_gradient_magnitude_diff_cpu = (
            valid_gradient_magnitude_diff.detach().cpu().numpy()
        )

        valid_time = time.time() - valid_start
        valid_weighted_rmse = mult * torch.mean(valid_weighted_rmse, axis=0)
        logs = {
            "valid_l1": valid_buff_cpu[1],
            "valid_loss": valid_buff_cpu[0],
            "valid_rmse_u10": valid_weighted_rmse_cpu[0],
            "valid_rmse_v10": valid_weighted_rmse_cpu[1],
        }
        metric_name_format = "valid_gradient_magnitude_percent_diff/channel{c}-{name}"
        grad_mag_logs = {
            metric_name_format.format(
                c=c, name=name
            ): valid_gradient_magnitude_diff_cpu[c]
            for c, name in enumerate(self.valid_dataset.out_names)
        }

        validation_logs = {**logs, **grad_mag_logs, **image_logs}
        return valid_time, validation_logs

    def inference_one_epoch(self):
        if self.params.log_to_screen:
            logging.info("Starting inference on validation set...")
        import copy

        with torch.no_grad():
            inference_params = copy.copy(self.params)
            inference_params.means = self.valid_dataset.out_means[0]
            inference_params.stds = self.valid_dataset.out_stds[0]
            inference_params.time_means = self.valid_dataset.out_time_means[0]
            inference_params.use_daily_climatology = (
                False  # TODO(gideond) default value?
            )
            inference_params.epoch = self.epoch
            inference_params.iters = self.iters
            inference_params.get_data_loader = False
            inference_params.log_on_each_unroll_step_inference = False
            inference_params.log_to_wandb = True
            inference_params.log_to_screen = False  # reduce noise in logs
            (
                _,
                _,
                _,
                _,
                _,
                inference_logs,
            ) = inference.autoregressive_inference(
                inference_params, 0, self.valid_dataset.data_array, self.model
            )

        return inference_logs

    def _load_global_output_stds(self):
        if self.params.data_type == "ERA5":
            return np.load(self.params.global_stds_path)[
                0, self.params.out_channels, 0, 0
            ]
        elif self.params.data_type == "FV3GFS":
            return self.valid_dataset.out_stds.squeeze()
        else:
            raise NotImplementedError(f"data_type {self.params.data_type} is unknown.")

    def load_model_wind(self, model_path):
        if self.params.log_to_screen:
            logging.info("Loading the wind model weights from {}".format(model_path))
        checkpoint = torch.load(
            model_path, map_location="cuda:{}".format(self.params.local_rank)
        )
        if dist.is_initialized():
            self.model_wind.load_state_dict(checkpoint["model_state"])
        else:
            new_model_state = OrderedDict()
            model_key = "model_state" if "model_state" in checkpoint else "state_dict"
            for key in checkpoint[model_key].keys():
                if "module." in key:  # model was stored using ddp which prepends module
                    name = str(key[7:])
                    new_model_state[name] = checkpoint[model_key][key]
                else:
                    new_model_state[key] = checkpoint[model_key][key]
            self.model_wind.load_state_dict(new_model_state)
            self.model_wind.eval()

    def save_checkpoint(self, checkpoint_path, model=None):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""

        if not model:
            model = self.model

        torch.save(
            {
                "iters": self.iters,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function"""
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda:{}".format(self.params.local_rank)
        )
        try:
            self.model.load_state_dict(checkpoint["model_state"])
        except:  # noqa: E722
            new_state_dict = OrderedDict()
            for key, val in checkpoint["model_state"].items():
                name = key[7:]
                new_state_dict[name] = val
            self.model.load_state_dict(new_state_dict)
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]
        if self.params.resuming:
            # restore checkpoint is used for finetuning as well as resuming.
            # If finetuning (i.e., not resuming), restore checkpoint
            # does not load optimizer state, instead uses config specified lr.
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def main(
    run_num: str, yaml_config: str, config: str, enable_amp: bool, epsilon_factor: float
):
    params = YParams(os.path.abspath(yaml_config), config)
    params["epsilon_factor"] = epsilon_factor

    params["world_size"] = 1
    if "WORLD_SIZE" in os.environ:
        params["world_size"] = int(os.environ["WORLD_SIZE"])

    world_rank = 0
    local_rank = 0
    if params["world_size"] > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        world_rank = dist.get_rank()
        params["global_batch_size"] = params.batch_size
        params["batch_size"] = int(params.batch_size // params["world_size"])

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, config, str(run_num))
    if world_rank == 0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
            os.makedirs(os.path.join(expDir, "training_checkpoints/"))

    params["experiment_dir"] = os.path.abspath(expDir)
    params["checkpoint_path"] = os.path.join(expDir, "training_checkpoints/ckpt.tar")
    params["best_checkpoint_path"] = os.path.join(
        expDir, "training_checkpoints/best_ckpt.tar"
    )

    # Do not comment this line out please:
    resuming = True if os.path.isfile(params.checkpoint_path) else False

    params["resuming"] = resuming
    params["local_rank"] = local_rank
    params["enable_amp"] = enable_amp

    # wandb parameters
    params["project"] = "fourcastnet-era5"
    params["entity"] = "ai2cm"
    if world_rank == 0:
        logging_utils.log_to_file(
            logger_name=None, log_filename=os.path.join(expDir, "out.log")
        )
        logging_utils.log_versions()
        params.log()

    params["log_to_wandb"] = (world_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (world_rank == 0) and params["log_to_screen"]

    if world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(os.path.join(expDir, "hyperparams.yaml"), "w") as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, world_rank)
    trainer.train()
    logging.info("DONE ---- rank %d" % world_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument("--enable_amp", action="store_true")
    parser.add_argument("--epsilon_factor", default=0, type=float)

    args = parser.parse_args()
    main(
        args.run_num,
        args.yaml_config,
        args.config,
        args.enable_amp,
        args.epsilon_factor,
    )
