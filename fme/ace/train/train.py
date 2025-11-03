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
from collections.abc import Callable, Sequence

import dacite
import torch
import xarray as xr

import fme
import fme.core.logging_utils as logging_utils
from fme.ace.aggregator import (
    OneStepAggregator,
    OneStepAggregatorConfig,
    TrainAggregator,
)
from fme.ace.aggregator.inference.main import (
    InferenceEvaluatorAggregator,
    InferenceEvaluatorAggregatorConfig,
)
from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.stepper import TrainOutput
from fme.ace.train.train_config import TrainBuilders, TrainConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.dataset_info import DatasetInfo
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.generics.data import InferenceDataABC
from fme.core.generics.trainer import (
    AggregatorBuilderABC,
    TrainConfigProtocol,
    Trainer,
    inference_one_epoch,
)
from fme.core.typing_ import TensorDict, TensorMapping


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> "Trainer":
    # note for devs: you don't have to use this function to build a custom
    # trainer, you can build it however you like. This is here for convenience.
    logging.info("Initializing training data loader")
    train_data = builder.get_train_data()
    logging.info("Initializing validation data loader")
    validation_data = builder.get_validation_data()
    if config.inference is None:
        logging.info("Skipping inline inference")
    else:
        logging.info("Initializing inline inference data loader")
    inference_data = builder.get_evaluation_inference_data()

    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata

    dataset_info = train_data.dataset_info
    logging.info("Starting model initialization")
    stepper = builder.get_stepper(
        dataset_info=dataset_info,
    )
    end_of_batch_ops = builder.get_end_of_batch_ops(
        modules=stepper.modules, base_weights=stepper.get_base_weights()
    )

    if config.inference is None:
        initial_inference_times = None
    else:
        initial_inference_times = inference_data.initial_time
    inference_n_forward_steps = config.inference_n_forward_steps
    record_step_20 = inference_n_forward_steps >= 20

    aggregator_builder = AggregatorBuilder(
        inference_config=config.inference_aggregator,
        dataset_info=dataset_info.update_variable_metadata(variable_metadata),
        output_dir=config.output_dir,
        initial_inference_time=initial_inference_times,
        record_step_20=record_step_20,
        n_timesteps=inference_n_forward_steps + stepper.n_ic_timesteps,
        loss_scaling=stepper.effective_loss_scaling,
        channel_mean_names=stepper.loss_names,
        normalize=stepper.normalizer.normalize,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
        validation_config=config.validation_aggregator,
    )
    do_gc_collect = fme.get_device() != torch.device("cpu")
    trainer_config: TrainConfigProtocol = config  # documenting trainer input type
    trainer = Trainer(
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

    def inference(
        data: InferenceDataABC[PrognosticState, BatchData],
        aggregator: InferenceEvaluatorAggregator,
        label: str,
        epoch: int,
    ):
        logging.info("Starting weather evaluation inference run")
        return inference_one_epoch(
            stepper=stepper,
            validation_context=trainer.validation_context,
            dataset=data,
            aggregator=aggregator,
            label=label,
            epoch=epoch,
        )

    end_of_epoch_ops = builder.get_end_of_epoch_callback(
        inference,
        normalize=stepper.normalizer.normalize,
        channel_mean_names=stepper.loss_names,
        output_dir=config.output_dir,
        variable_metadata=variable_metadata,
        save_diagnostics=config.save_per_epoch_diagnostics,
        n_ic_timesteps=stepper.n_ic_timesteps,
    )
    trainer.set_end_of_epoch_callback(end_of_epoch_ops)
    return trainer


class AggregatorBuilder(
    AggregatorBuilderABC[PrognosticState, TrainOutput, PairedData],
):
    def __init__(
        self,
        inference_config: InferenceEvaluatorAggregatorConfig | None,
        dataset_info: DatasetInfo,
        initial_inference_time: xr.DataArray | None,
        record_step_20: bool,
        n_timesteps: int,
        output_dir: str,
        normalize: Callable[[TensorMapping], TensorDict],
        loss_scaling: dict[str, torch.Tensor] | None = None,
        channel_mean_names: Sequence[str] | None = None,
        save_per_epoch_diagnostics: bool = False,
        validation_config: OneStepAggregatorConfig = dataclasses.field(
            default_factory=lambda: OneStepAggregatorConfig(),
        ),
    ):
        self.inference_config = inference_config
        self.dataset_info = dataset_info
        self.initial_inference_time = initial_inference_time
        self.record_step_20 = record_step_20
        self.n_timesteps = n_timesteps
        self.loss_scaling = loss_scaling
        self.channel_mean_names = channel_mean_names
        self.normalize = normalize
        self.output_dir = output_dir
        self.save_per_epoch_diagnostics = save_per_epoch_diagnostics
        self.validation_config = validation_config

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator()

    def get_validation_aggregator(self) -> OneStepAggregator:
        return self.validation_config.build(
            dataset_info=self.dataset_info,
            loss_scaling=self.loss_scaling,
            save_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=os.path.join(self.output_dir, "val"),
            channel_mean_names=self.channel_mean_names,
        )

    def get_inference_aggregator(
        self,
    ) -> InferenceEvaluatorAggregator:
        if isinstance(self.inference_config, InferenceEvaluatorAggregatorConfig):
            return self.inference_config.build(
                dataset_info=self.dataset_info,
                initial_time=self.initial_inference_time,
                record_step_20=self.record_step_20,
                n_timesteps=self.n_timesteps,
                channel_mean_names=self.channel_mean_names,
                normalize=self.normalize,
                save_diagnostics=self.save_per_epoch_diagnostics,
                output_dir=os.path.join(self.output_dir, "inference"),
            )
        else:
            raise ValueError(
                "Trying to build an inference aggregator, but inference config not set."
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
        config=config_as_dict,
        env_vars=env_vars,
        notes=beaker_url,
    )
    if config.resume_results is not None:
        logging.info(
            f"Resuming training from results in {config.resume_results.existing_dir}"
        )
    trainer = build_trainer(builders, config)
    try:
        trainer.train()
        logging.info(f"DONE ---- rank {dist.rank}")
    finally:
        dist.shutdown()


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None, h_parallel_size=1, w_parallel_size=1):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()
    config.resume_results = prepare_directory(
        config.experiment_dir, config_data, config.resume_results
    )
    dist = Distributed()
    if (h_parallel_size>1) or (w_parallel_size >1):
      dist._init_distributed(h_parallel_size =  h_parallel_size, w_parallel_size=w_parallel_size)
    run_train_from_config(config)
