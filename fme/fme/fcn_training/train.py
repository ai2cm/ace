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
import numpy as np
import argparse
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from fme.fcn_training.utils import logging_utils

logging_utils.config_logger()
from fme.fcn_training.utils.YParams import YParams
from fme.fcn_training.utils.data_loader_multifiles import (
    get_data_loader,
    DataLoaderParams,
)
import wandb
from fme.fcn_training.utils.weighted_acc_rmse import (
    weighted_rmse_torch,
    weighted_global_mean_gradient_magnitude,
)
from apex import optimizers
from fme.fcn_training.utils.darcy_loss import LpLoss

DECORRELATION_TIME = 36  # 9 days
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from networks.geometric_v1.sfnonet import FourierNeuralOperatorParams
from fourcastnet.networks.afnonet import AFNONetParams
from fme.fcn_training.registry import NET_REGISTRY
from fme.fcn_training.inference import inference
import dataclasses
import fme


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


class SingleModuleStepper:
    """
    Stepper class for a single pytorch module.
    """

    def __init__(
        self,
        module: nn.Module,
        optimizer_type: Literal["Adam", "FusedAdam"],
        lr: float,
        enable_automatic_mixed_precision: bool,
        scheduler: Literal["ReduceLROnPlateau", "CosineAnnealingLR"],
        max_epochs: int,
        start_epoch: int,
        loss_obj: nn.Module,
    ):
        """
        Args:
            module: The module to train.
            optimizer_type: The optimizer type. Currently supports "Adam"
                and "FusedAdam".
            lr: The learning rate.
            enable_automatic_mixed_precision: Whether to use automatic mixed precision.
            scheduler: The scheduler type. Currently supports
                "ReduceLROnPlateau" and "CosineAnnealingLR".
            max_epochs: The maximum number of epochs to train for.
            start_epoch: The epoch to start training from.
            loss_obj: The loss object to use.
        """
        self.module = module
        if optimizer_type == "FusedAdam":
            self.optimizer = optimizers.FusedAdam(self.module.parameters(), lr=lr)
        else:
            self.optimizer = torch.optim.Adam(self.module.parameters(), lr=lr)

        if enable_automatic_mixed_precision:
            self.gscaler: Optional[amp.GradScaler] = amp.GradScaler()
        else:
            self.gscaler = None

        if dist.is_initialized():
            self.module = DistributedDataParallel(
                self.module,
                device_ids=[int(os.environ["LOCAL_RANK"])],
                output_device=[int(os.environ["LOCAL_RANK"])],
                find_unused_parameters=True,
            )

        if scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, factor=0.2, patience=5, mode="min"
            )
        elif scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=max_epochs, last_epoch=start_epoch - 1
            )
        else:
            self.scheduler = None

        self.loss_obj = loss_obj

    @property
    def modules(self) -> List[nn.Module]:
        """
        Returns:
            A list of modules being trained.
        """
        return [self.module]

    def step_scheduler(self, valid_loss: float):
        """
        Step the scheduler.

        Args:
            valid_loss: The validation loss. Used in schedulers which change the
                learning rate based on whether the validation loss is decreasing.
        """
        if self.scheduler is not None:
            try:
                self.scheduler.step(metrics=valid_loss)
            except TypeError:
                self.scheduler.step()

    def run_on_batch(self, input_data, target_data, train: bool):
        """
        Run the model on a batch of data.

        Args:
            input_data: The input data.
            target_data: The target data.
            train: Whether to train the model.
        """
        if train:
            self.module.train()
            self.module.zero_grad()
        else:
            self.module.eval()
        automatic_mixed_precision_enabled = self.gscaler is not None
        with amp.autocast(automatic_mixed_precision_enabled):
            gen = self.module(input_data).to(fme.get_device(), dtype=torch.float)
            loss = self.loss_obj(gen, target_data)

        if train:
            if self.gscaler is not None:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.gscaler is not None:
                self.gscaler.update()
        return loss, gen

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

    def load_state(self, state, load_optimizer: bool = True):
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
            load_optimizer: Whether to load the optimizer state.
        """
        if "module" in state:
            module_state = state["module"]
            for key in module_state:
                if key.startswith("module."):
                    name = key[7:]
                    module_state[name] = module_state.pop(key)
            self.module.load_state_dict(module_state)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if load_optimizer and "scheduler" in state:
            self.scheduler.load_state_dict(state["scheduler"])


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
    # TODO: is there a way to hard-code nhwc, which controls channels first/last?
    enable_nhwc: bool
    nettype: str
    train_data_path: str
    valid_data_path: str
    max_epochs: int
    save_checkpoint: bool
    normalization: str
    data_type: str
    num_data_workers: int
    # note: dt is only used for inference
    dt: float
    n_history: int
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
    normalize: bool = True
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
            "normalize",
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
            enable_nhwc=params["enable_nhwc"],
            nettype=params["nettype"],
            train_data_path=params["train_data_path"],
            valid_data_path=params["valid_data_path"],
            max_epochs=params["max_epochs"],
            save_checkpoint=params["save_checkpoint"],
            normalization=params["normalization"],
            data_type=params["data_type"],
            num_data_workers=params["num_data_workers"],
            dt=params["dt"],
            n_history=params["n_history"],
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

    def register_data_params(self, data_params: DataParams):
        self.data_params = data_params

    @property
    def img_shape_x(self) -> int:
        if self.data_params is not None:
            return self.data_params.img_shape_x
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def img_shape_y(self) -> int:
        if self.data_params is not None:
            return self.data_params.img_shape_y
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def img_crop_shape_x(self) -> int:
        if self.data_params is not None:
            return self.data_params.img_crop_shape_x
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def img_crop_shape_y(self) -> int:
        if self.data_params is not None:
            return self.data_params.img_crop_shape_y
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def N_in_channels(self) -> int:
        if self.data_params is not None:
            return self.data_params.N_in_channels
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def N_out_channels(self) -> int:
        if self.data_params is not None:
            return self.data_params.N_out_channels
        else:
            raise RuntimeError(
                "data_params must be registered before accessing this attribute"
            )

    @property
    def data_loader_params(self):
        return DataLoaderParams(
            data_type=self.data_type,
            batch_size=self.batch_size,
            num_data_workers=self.num_data_workers,
            dt=self.dt,
            n_history=self.n_history,
            global_means_path=self.global_means_path,
            global_stds_path=self.global_stds_path,
            time_means_path=self.time_means_path,
            normalize=self.normalize,
            in_names=self.in_names,
            out_names=self.out_names,
            normalization=self.normalization,
        )

    @property
    def model_params(self):
        if self.nettype == "FourierNeuralOperatorNet":
            params = FourierNeuralOperatorParams(
                spectral_transform=self.spectral_transform,
                filter_type=self.filter_type,
                img_crop_shape_x=self.img_crop_shape_x,
                img_crop_shape_y=self.img_crop_shape_y,
                scale_factor=self.scale_factor,
                N_in_channels=self.N_in_channels,
                N_out_channels=self.N_out_channels,
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
            params = AFNONetParams(
                img_shape_x=self.img_shape_x,
                img_shape_y=self.img_shape_y,
                patch_size=self.patch_size,
                N_in_channels=self.N_in_channels,
                N_out_channels=self.N_out_channels,
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

        logging.info("rank %d, begin data loader init" % world_rank)
        (
            self.train_data_loader,
            self.train_dataset,
            self.train_sampler,
        ) = get_data_loader(
            params.data_loader_params,
            params.train_data_path,
            dist.is_initialized(),
            train=True,
        )
        self.valid_data_loader, self.valid_dataset = get_data_loader(
            params.data_loader_params,
            params.valid_data_path,
            dist.is_initialized(),
            train=False,
        )
        logging.info("rank %d, data loader initialized" % world_rank)

        data_params = DataParams.from_datasets(self.train_dataset, self.valid_dataset)
        self.params.register_data_params(data_params)

        BackboneNet, ParamsClass = NET_REGISTRY[params.nettype]
        assert isinstance(params.model_params, ParamsClass)

        self.iters = 0
        self.startEpoch = 0
        if params.resuming:
            logging.info("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.startEpoch

        model = BackboneNet(params.model_params).to(fme.get_device())
        if self.params.enable_nhwc:
            # NHWC: Convert model to channels_last memory format
            model = model.to(memory_format=torch.channels_last)
        self.stepper = SingleModuleStepper(
            model,
            optimizer_type=params.optimizer_type,
            lr=params.lr,
            scheduler=params.scheduler,
            max_epochs=params.max_epochs,
            start_epoch=self.startEpoch,
            loss_obj=LpLoss(),
            enable_automatic_mixed_precision=params.enable_automatic_mixed_precision,
        )

        if params.log_to_wandb:
            wandb.watch(self.stepper.modules)

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
                for pg in self.stepper.optimizer.param_groups:
                    lr = pg["lr"]
                all_logs = {**train_logs, **valid_logs, **inference_logs, **{"lr": lr}}
                wandb.log(all_logs, step=self.epoch)

    def train_one_epoch(self):
        # TODO: clean up and merge train_one_epoch and validate_one_epoch
        # deduplicate code through helper routines or if conditionals
        self.epoch += 1

        for data in self.train_data_loader:
            self.iters += 1
            input_data, target_data = map(
                lambda x: x.to(fme.get_device(), dtype=torch.float), data
            )

            if self.params.enable_nhwc:
                input_data = input_data.to(memory_format=torch.channels_last)
                target_data = target_data.to(memory_format=torch.channels_last)
            loss, _ = self.stepper.run_on_batch(input_data, target_data, train=True)

        logs = {"loss": loss}

        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(logs[key].detach())
                logs[key] = float(logs[key] / dist.get_world_size())

        return logs

    def validate_one_epoch(self):
        n_valid_batches = 20  # do validation on first 20 images, just for LR scheduler
        if self.params.normalization == "minmax":
            raise Exception("minmax normalization not supported")
        elif self.params.normalization == "zscore":
            # TODO: unify this normalization logic with what's in the data loading
            mult = torch.as_tensor(self._load_global_output_stds()).to(fme.get_device())

        valid_buff = torch.zeros((3), dtype=torch.float32, device=fme.get_device())
        valid_loss = valid_buff[0].view(-1)
        valid_l1 = valid_buff[1].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros(
            (self.params.N_out_channels), dtype=torch.float32, device=fme.get_device()
        )
        valid_gradient_magnitude_diff = torch.zeros(
            (self.params.N_out_channels), dtype=torch.float32, device=fme.get_device()
        )

        with torch.no_grad():
            image_logs = {}
            for i, data in enumerate(self.valid_data_loader, 0):
                if i >= n_valid_batches:
                    break
                input, target = map(
                    lambda x: x.to(fme.get_device(), dtype=torch.float), data
                )
                batch_loss, gen = self.stepper.run_on_batch(input, target, train=False)
                valid_loss += batch_loss
                valid_l1 += nn.functional.l1_loss(gen, target)

                valid_steps += 1.0

                # direct prediction weighted rmse
                gen_for_rmse = gen
                tar_for_rmse = target
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
                            fme.get_device(), dtype=torch.float
                        )
                        gen_for_image = gen[0, j]
                        image_error = gen_for_image - target[0, j]
                        image_full_field = torch.cat(
                            (gen_for_image, gap, target[0, j]), axis=1
                        )
                        image_residual = torch.cat(
                            (
                                gen_for_image - input[0, j],
                                gap,
                                target[0, j] - input[0, j],
                            ),
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
                                self.params.experiment_dir,
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
        return validation_logs

    def inference_one_epoch(self):
        # TODO: refactor inference to be more clearly reusable between
        # training, validation, and inference
        if self.params.log_to_screen:
            logging.info("Starting inference on validation set...")

        with torch.no_grad():
            inference_params = types.SimpleNamespace(
                **dict(self.params.__dict__, **self.params.data_params.__dict__)
            )
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
                inference_params, 0, self.valid_dataset.data_array, self.stepper.module
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
        checkpoint = torch.load(
            checkpoint_path, map_location="cuda:{}".format(self.params.local_rank)
        )
        # restore checkpoint is used for finetuning as well as resuming.
        # If finetuning (i.e., not resuming), restore checkpoint
        # does not load optimizer state, instead uses config specified lr.
        load_optimizer = self.params.resuming
        self.stepper.load_state(checkpoint["stepper"], load_optimizer=load_optimizer)
        self.iters = checkpoint["iters"]
        self.startEpoch = checkpoint["epoch"]


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
