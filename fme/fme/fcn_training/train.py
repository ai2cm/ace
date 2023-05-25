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
import types
from typing import List, Optional, Literal
from fme.fcn_training.utils.data_loader_fv3gfs import load_series_data
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import logging
from fme.fcn_training.utils import logging_utils

logging_utils.config_logger()
from fme.fcn_training.utils.YParams import YParams
from fme.fcn_training.utils.data_loader_multifiles import (
    get_data_loader,
    DataLoaderParams,
)
import wandb
from fme.fcn_training.utils.darcy_loss import LpLoss

DECORRELATION_TIME = 36  # 9 days
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from networks.geometric_v1.sfnonet import FourierNeuralOperatorBuilder
from fourcastnet.networks.afnonet import AFNONetBuilder
from fme.fcn_training.registry import ModuleBuilder
from fme.fcn_training.inference import inference
import dataclasses
import fme
from fme.fcn_training.stepper import SingleModuleStepperConfig
import netCDF4


@dataclasses.dataclass
class DataParams:
    img_shape_x: int
    img_shape_y: int
    img_crop_shape_x: int
    img_crop_shape_y: int
    N_in_channels: int
    N_out_channels: int
    in_names: List[str]
    out_names: List[str]

    @classmethod
    def from_datasets(cls, train_dataset, valid_dataset) -> "DataParams":
        return cls(
            img_shape_x=valid_dataset.img_shape_x,
            img_shape_y=valid_dataset.img_shape_y,
            img_crop_shape_x=valid_dataset.img_shape_x,
            img_crop_shape_y=valid_dataset.img_shape_y,
            N_in_channels=train_dataset.n_in_channels,
            N_out_channels=train_dataset.n_out_channels,
            in_names=train_dataset.in_names,
            out_names=train_dataset.out_names,
        )


@dataclasses.dataclass
class TrainerParams:
    run_num: str
    # TODO: remove "config" key, require a single config per yaml file
    config: str
    enable_automatic_mixed_precision: bool
    world_size: int
    world_rank: int
    experiment_dir: str
    checkpoint_path: str
    best_checkpoint_path: str
    resuming: bool
    local_rank: int
    project: str
    entity: str
    log_to_wandb: bool
    log_to_screen: bool
    scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"]
    optimizer_type: Literal["Adam", "FusedAdam"]
    nettype: str
    train_data_path: str
    valid_data_path: str
    max_epochs: int
    save_checkpoint: bool
    data_type: str
    num_data_workers: int
    # note: dt is only used for inference
    dt: float
    global_means_path: str
    global_stds_path: str
    time_means_path: str
    lr: float
    in_names: List[str]
    out_names: List[str]
    batch_size: Optional[int] = None
    data_params: Optional[DataParams] = None
    spectral_transform: Literal["sht", "fft"] = "sht"
    filter_type: Literal["linear", "non-linear"] = "non-linear"
    scale_factor: int = 16
    embed_dim: int = 256
    num_layers: int = 12
    num_blocks: int = 16
    hard_thresholding_fraction: float = 1.0
    normalization_layer: Literal["instance_norm", "layer_norm"] = "instance_norm"
    mlp_mode: str = "none"
    big_skip: bool = True
    compression: Optional[str] = None
    rank: int = 128  # not the same as local_rank
    complex_network: bool = True
    complex_activation: str = "real"
    spectral_layers: int = 1
    laplace_weighting: bool = False
    checkpointing: bool = False
    patch_size: int = 16
    pretrained: bool = False
    pretrained_ckpt_path: Optional[str] = None
    # parameters only for inference
    prediction_length: int = 2
    perturb: bool = False

    def __post_init__(self):
        assert len(self.in_names) > 0
        assert len(self.out_names) > 0

    @classmethod
    def new(
        cls,
        run_num: str,
        yaml_config: str,
        config: str,
        enable_automatic_mixed_precision: bool,
    ):
        """
        Create a new TrainerParams instance.

        Args:
            run_num: a unique identifier for this run
            yaml_config: path to a yaml file containing the training configuration
            config: name of the configuration to use from the yaml file
            enable_automatic_mixed_precision: whether to use automatic mixed precision

        Side-effects include:
            - creating the experiment directory and a training_checkpoints subdirectory
            - calling `dist.init_process_group`, if world_size is greater than 1
            - setting the GPU device to the local rank
            - setting global logging configuration
        """

        params = YParams(os.path.abspath(yaml_config), config)

        if "precip" in params:
            raise NotImplementedError("precip training feature has been removed")
        if "orography" in params:
            raise NotImplementedError(
                "feature to add orography to inputs has been removed"
            )

        params["world_size"] = 1
        if "WORLD_SIZE" in os.environ:
            params["world_size"] = int(os.environ["WORLD_SIZE"])

        world_rank = 0
        local_rank = 0
        if params["world_size"] > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
            local_rank = int(os.environ["LOCAL_RANK"])
            world_rank = dist.get_rank()
            params["batch_size"] = int(
                params.batch_size // params["world_size"]  # type: ignore
            )

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            torch.backends.cudnn.benchmark = True

        # Set up directory
        expDir = os.path.join(params.exp_dir, config, str(run_num))  # type: ignore
        if world_rank == 0:
            if not os.path.isdir(expDir):
                os.makedirs(expDir)
            if not os.path.isdir(os.path.join(expDir, "training_checkpoints/")):
                os.makedirs(os.path.join(expDir, "training_checkpoints/"))

        params["experiment_dir"] = os.path.abspath(expDir)
        params["checkpoint_path"] = os.path.join(
            expDir, "training_checkpoints/ckpt.tar"
        )
        params["best_checkpoint_path"] = os.path.join(
            expDir, "training_checkpoints/best_ckpt.tar"
        )

        checkpoint_file_exists = os.path.isfile(params.checkpoint_path)  # type: ignore
        resuming = True if checkpoint_file_exists else False

        params["resuming"] = resuming
        params["local_rank"] = local_rank

        # wandb parameters
        if world_rank == 0:
            logging_utils.log_to_file(
                logger_name=None, log_filename=os.path.join(expDir, "out.log")
            )
            logging_utils.log_versions()
            params.log()

        model_params = {}
        for param_name in [
            "spectral_transform",
            "filter_type",
            "scale_factor",
            "embed_dim",
            "num_layers",
            "num_blocks",
            "hard_thresholding_fraction",
            "normalization_layer",
            "mlp_mode",
            "big_skip",
            "compression",
            "rank",
            "complex_network",
            "complex_activation",
            "spectral_layers",
            "laplace_weighting",
            "checkpointing",
            "patch_size",
        ]:
            if param_name in params:
                model_params[param_name] = params[param_name]

        optional_args = {}
        for optional_param in [
            "pretrained",
            "pretrained_ckpt_path",
            "in_channels",
            "out_channels",
            "prediction_length",
            "perturb",
        ]:
            if optional_param in params:
                optional_args[optional_param] = params[optional_param]

        params_instance = TrainerParams(
            run_num=run_num,
            config=config,
            enable_automatic_mixed_precision=enable_automatic_mixed_precision,
            world_size=params["world_size"],
            world_rank=world_rank,
            batch_size=params["batch_size"],
            experiment_dir=os.path.abspath(expDir),
            checkpoint_path=os.path.join(expDir, "training_checkpoints/ckpt.tar"),
            best_checkpoint_path=os.path.join(
                expDir, "training_checkpoints/best_ckpt.tar"
            ),
            resuming=resuming,
            local_rank=local_rank,
            project="fourcastnet-era5",
            entity="ai2cm",
            log_to_wandb=(world_rank == 0) and params["log_to_wandb"],
            log_to_screen=(world_rank == 0) and params["log_to_screen"],
            scheduler=params["scheduler"],
            optimizer_type=params["optimizer_type"],
            nettype=params["nettype"],
            train_data_path=params["train_data_path"],
            valid_data_path=params["valid_data_path"],
            max_epochs=params["max_epochs"],
            save_checkpoint=params["save_checkpoint"],
            data_type=params["data_type"],
            num_data_workers=params["num_data_workers"],
            dt=params["dt"],
            global_means_path=params["global_means_path"],
            global_stds_path=params["global_stds_path"],
            time_means_path=params["time_means_path"],
            lr=params["lr"],
            in_names=params["in_names"],
            out_names=params["out_names"],
            **model_params,
            **optional_args,
        )

        if world_rank == 0:
            params_instance._log(config, os.path.abspath(yaml_config))
        return params_instance

    def _log(self, config_name, yaml_filename):
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(yaml_filename))
        logging.info("Configuration name: " + str(config_name))
        for key, val in self.__dict__.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")

    @property
    def data_loader_params(self):
        return DataLoaderParams(
            data_type=self.data_type,
            batch_size=self.batch_size,
            num_data_workers=self.num_data_workers,
            dt=self.dt,
            global_means_path=self.global_means_path,
            global_stds_path=self.global_stds_path,
            time_means_path=self.time_means_path,
            in_names=self.in_names,
            out_names=self.out_names,
        )

    @property
    def module_builder(self) -> ModuleBuilder:
        if self.nettype == "FourierNeuralOperatorNet":
            params = FourierNeuralOperatorBuilder(
                spectral_transform=self.spectral_transform,
                filter_type=self.filter_type,
                scale_factor=self.scale_factor,
                embed_dim=self.embed_dim,
                num_layers=self.num_layers,
                num_blocks=self.num_blocks,
                hard_thresholding_fraction=self.hard_thresholding_fraction,
                normalization_layer=self.normalization_layer,
                mlp_mode=self.mlp_mode,
                big_skip=self.big_skip,
                compression=self.compression,
                rank=self.rank,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                laplace_weighting=self.laplace_weighting,
                checkpointing=self.checkpointing,
            )
        elif self.nettype == "afno":
            params = AFNONetBuilder(
                patch_size=self.patch_size,
                embed_dim=self.embed_dim,
                num_blocks=self.num_blocks,
            )
        else:
            raise ValueError("Unknown nettype: " + str(self.nettype))
        return params


class Trainer:
    def count_parameters(self):
        parameters = 0
        for module in self.stepper.modules:
            for parameter in module.parameters():
                if parameter.requires_grad:
                    parameters += parameter.numel()
        return parameters

    def __init__(self, params: TrainerParams, world_rank):
        self.params = params
        self.world_rank = world_rank

        if params.log_to_wandb:
            wandb.init(
                config=params,
                project=params.project,
                entity=params.entity,
                resume=True,
                dir=params.experiment_dir,
            )
            logging_utils.log_beaker_url()

        stepper_config = SingleModuleStepperConfig(
            builder=params.module_builder,
            in_names=params.in_names,
            out_names=params.out_names,
            optimizer_type=params.optimizer_type,
            lr=params.lr,
            scheduler=params.scheduler,
            max_epochs=params.max_epochs,
            loss_obj=LpLoss(),
            enable_automatic_mixed_precision=params.enable_automatic_mixed_precision,
        )

        self.n_forward_steps = 1
        if self.n_forward_steps != 1:
            raise NotImplementedError(
                "diagnostics code not updated for n_forward_steps != 1"
            )
        data_requirements = stepper_config.get_data_requirements(
            n_forward_steps=self.n_forward_steps
        )
        logging.info("rank %d, begin data loader init" % world_rank)
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params.data_loader_params,
            params.train_data_path,
            dist.is_initialized(),
            requirements=data_requirements,
            train=True,
        )
        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params.data_loader_params,
            params.valid_data_path,
            dist.is_initialized(),
            requirements=data_requirements,
            train=False,
        )
        logging.info("rank %d, data loader initialized" % world_rank)

        self.iters = 0
        self.startEpoch = 0

        self.epoch = self.startEpoch

        for data in self.train_data_loader:
            shapes = {k: v.shape for k, v in data.items()}
            break
        normalizer = fme.get_normalizer(
            global_means_path=params.global_means_path,
            global_stds_path=params.global_stds_path,
            names=data_requirements.names,
        )
        self.stepper = stepper_config.get_stepper(shapes, normalizer)

        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        if params.log_to_wandb:
            wandb.watch(self.stepper.modules)

        if params.log_to_screen:
            logging.info(
                "Number of trainable model parameters: {}".format(
                    self.count_parameters()
                )
            )

        # TODO: refactor this into its own dataset configuration
        inference_ds = netCDF4.MFDataset(os.path.join(params.valid_data_path, "*.nc"))
        self.inference_data = load_series_data(
            idx=0,
            n_steps=params.prediction_length + 1,
            ds=inference_ds,
            names=list(set(params.in_names).union(params.out_names)),
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
            train_logs = self.train_one_epoch()
            valid_logs = self.validate_one_epoch()
            inference_logs = self.inference_one_epoch()

            train_loss = train_logs["loss"]
            valid_loss = valid_logs["valid_loss"]

            self.stepper.step_scheduler(valid_loss)

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
                for pg in self.stepper.optimization.optimizer.param_groups:
                    lr = pg["lr"]
                all_logs = {**train_logs, **valid_logs, **inference_logs, **{"lr": lr}}
                wandb.log(all_logs, step=self.epoch)

    def train_one_epoch(self):
        # TODO: clean up and merge train_one_epoch and validate_one_epoch
        # deduplicate code through helper routines or if conditionals
        self.epoch += 1

        for data in self.train_data_loader:
            loss, _, _, _ = self.stepper.run_on_batch(
                data, train=True, n_forward_steps=self.n_forward_steps
            )

        logs = {"loss": loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

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
        valid_weighted_rmse = {
            k: torch.zeros((1), dtype=torch.float32, device=fme.get_device())
            for k in self.params.out_names
        }
        valid_gradient_magnitude_diff = {
            k: torch.zeros((1), dtype=torch.float32, device=fme.get_device())
            for k in self.params.out_names
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
                        if self.params.log_to_wandb:
                            caption = (
                                f"{name} one step full field for "
                                f"sample {i}; (left) generated and (right) target."
                            )
                            wandb_image = wandb.Image(image_full_field, caption=caption)
                            image_logs[
                                f"image-full-field/sample{i}/{name}"
                            ] = wandb_image
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

        if dist.is_initialized():
            dist.all_reduce(valid_buff)
        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        logs = {
            "valid_l1": valid_buff_cpu[1],
            "valid_loss": valid_buff_cpu[0],
        }
        for name in self.params.out_names:
            if dist.is_initialized():
                dist.all_reduce(valid_weighted_rmse[name])

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
        if self.params.log_to_screen:
            logging.info("Starting inference on validation set...")

        with torch.no_grad():
            inference_params = types.SimpleNamespace(**self.params.__dict__)
            inference_params.log_on_each_unroll_step_inference = False
            inference_params.log_to_wandb = True
            inference_params.log_to_screen = False  # reduce noise in logs
            (
                _,
                _,
                inference_logs,
            ) = inference.autoregressive_inference(
                inference_params, 0, self.inference_data, self.stepper
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
    load_optimizer = trainer.params.resuming
    trainer.stepper.load_state(checkpoint["stepper"], load_optimizer=load_optimizer)
    trainer.iters = checkpoint["iters"]
    trainer.startEpoch = checkpoint["epoch"]


def main(
    run_num: str,
    yaml_config: str,
    config: str,
    enable_automatic_mixed_precision: bool,
):
    params = TrainerParams.new(
        run_num=run_num,
        yaml_config=yaml_config,
        config=config,
        enable_automatic_mixed_precision=enable_automatic_mixed_precision,
    )

    if params.world_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.__dict__.items():
            hparams[str(key)] = str(value)
        with open(
            os.path.join(params.experiment_dir, "hyperparams.yaml"), "w"
        ) as hpfile:
            yaml.dump(hparams, hpfile)

    trainer = Trainer(params, params.world_rank)
    trainer.train()
    logging.info("DONE ---- rank %d" % params.world_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="default", type=str)
    parser.add_argument("--enable_amp", action="store_true")

    args = parser.parse_args()
    main(
        run_num=args.run_num,
        yaml_config=args.yaml_config,
        config=args.config,
        enable_automatic_mixed_precision=args.enable_amp,
    )
