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

import dataclasses
import logging
import os
from datetime import timedelta
from typing import Callable, Dict, Mapping, Optional, Sequence

import dacite
import dask
import torch
import xarray as xr
import yaml

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator import OneStepAggregator, TrainAggregator
from fme.ace.aggregator.inference.main import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
)
from fme.ace.data_loading.batch_data import PairedData, PrognosticState
from fme.ace.stepper import TrainOutput
from fme.ace.train.train_config import TrainBuilders, TrainConfig
from fme.core.coordinates import HorizontalCoordinates, HybridSigmaPressureCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.generics.trainer import AggregatorBuilderABC, TrainConfigProtocol, Trainer
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping

# dask used on individual workers to load batches
dask.config.set(scheduler="synchronous")


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> "Trainer":
    # note for devs: you don't have to use this function to build a custom
    # trainer, you can build it however you like. This is here for convenience.
    train_data = builder.get_train_data()
    validation_data = builder.get_validation_data()
    inference_data = builder.get_evaluation_inference_data()

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
        vertical_coordinate=train_data.vertical_coordinate,
        timestep=train_data.timestep,
    )
    end_of_batch_ops = builder.get_end_of_batch_ops(stepper.modules)

    for batch in inference_data.loader:
        initial_inference_times = batch.time.isel(time=0)
        break
    aggregator_builder = AggregatorBuilder(
        inference_config=config.inference_aggregator,
        gridded_operations=train_data.gridded_operations,
        vertical_coordinate=train_data.vertical_coordinate,
        horizontal_coordinates=train_data.horizontal_coordinates,
        timestep=train_data.timestep,
        initial_inference_time=initial_inference_times,
        record_step_20=config.inference_n_forward_steps >= 20,
        n_timesteps=config.inference_n_forward_steps + stepper.n_ic_timesteps,
        variable_metadata=train_data.variable_metadata,
        loss_scaling=stepper.effective_loss_scaling,
        channel_mean_names=stepper.out_names,
        normalize=stepper.normalizer.normalize,
    )
    do_gc_collect = fme.get_device() != torch.device("cpu")
    trainer_config: TrainConfigProtocol = config  # documenting trainer input type
    return Trainer(
        train_data=train_data,
        validation_data=validation_data,
        inference_data=inference_data,
        stepper=stepper,
        build_optimization=builder.get_optimization,
        build_ema=builder.get_ema,
        config=trainer_config,
        aggregator_builder=aggregator_builder,
        end_of_batch_callback=end_of_batch_ops,
        do_gc_collect=do_gc_collect,
    )


class AggregatorBuilder(
    AggregatorBuilderABC[PrognosticState, TrainOutput, PairedData],
):
    def __init__(
        self,
        inference_config: InferenceEvaluatorAggregatorConfig,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        timestep: timedelta,
        initial_inference_time: xr.DataArray,
        record_step_20: bool,
        n_timesteps: int,
        normalize: Callable[[TensorMapping], TensorDict],
        variable_metadata: Optional[Mapping[str, VariableMetadata]] = None,
        loss_scaling: Optional[Dict[str, torch.Tensor]] = None,
        channel_mean_names: Optional[Sequence[str]] = None,
    ):
        self.inference_config = inference_config
        self.gridded_operations = gridded_operations
        self.vertical_coordinate = vertical_coordinate
        self.horizontal_coordinates = horizontal_coordinates
        self.timestep = timestep
        self.initial_inference_time = initial_inference_time
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
            variable_metadata=self.variable_metadata,
            loss_scaling=self.loss_scaling,
        )

    def get_inference_aggregator(
        self,
    ) -> InferenceEvaluatorAggregator:
        return self.inference_config.build(
            vertical_coordinate=self.vertical_coordinate,
            horizontal_coordinates=self.horizontal_coordinates,
            timestep=self.timestep,
            initial_time=self.initial_inference_time,
            record_step_20=self.record_step_20,
            n_timesteps=self.n_timesteps,
            variable_metadata=self.variable_metadata,
            channel_mean_names=self.channel_mean_names,
            normalize=self.normalize,
        )


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
