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

import contextlib
import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Any

import dacite
import torch

import fme
from fme.ace.aggregator import (
    OneStepAggregator,
    OneStepAggregatorConfig,
    TrainAggregator,
)
from fme.ace.aggregator.train import TrainAggregatorConfig
from fme.ace.stepper import TrainOutput
from fme.ace.train.train_config import TrainBuilders, TrainConfig
from fme.core.cli import prepare_config, prepare_directory
from fme.core.dataset_info import DatasetInfo
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.distributed import Distributed
from fme.core.generics.trainer import (
    AggregatorBuilderABC,
    TrainConfigProtocol,
    Trainer,
    inference_one_epoch,
)
from fme.core.generics.validation import run_validation


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> "Trainer":
    # note for devs: you don't have to use this function to build a custom
    # trainer, you can build it however you like. This is here for convenience.
    logging.info("Initializing training data loader")
    train_data = builder.get_train_data()

    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata

    if config.validation_list:
        logging.info("Initializing validation data loaders")
    else:
        logging.info("Skipping validation")
    validation_entries = builder.get_validation_data()

    for data, name in zip(
        [train_data] + [data for _, data, _ in validation_entries],
        ["train"] + [name for _, _, name in validation_entries],
    ):
        data.log_info(name)

    if config.inference_list:
        logging.info("Initializing inline inference data loaders")
    else:
        logging.info("Skipping inline inference")
    inference_entries = builder.get_inference_data(variable_metadata)
    inference_epochs = config.get_inference_epochs()
    inference_epoch_sets = config.get_inference_epoch_sets()

    dataset_info = train_data.dataset_info
    logging.info("Starting model initialization")
    stepper = builder.get_stepper(
        dataset_info=dataset_info,
    )
    end_of_batch_ops = builder.get_end_of_batch_ops(
        modules=stepper.modules, base_weights=stepper.get_base_weights()
    )

    primary_validation_config = (
        config.validation_list[0].aggregator if config.validation_list else None
    )
    aggregator_builder = AggregatorBuilder(
        train_config=config.train_aggregator,
        dataset_info=dataset_info.update_variable_metadata(variable_metadata),
        output_dir=config.output_dir,
        loss_scaling=stepper.effective_loss_scaling,
        channel_mean_names=stepper.loss_names,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
        validation_config=primary_validation_config,
    )

    def validation_callback(epoch: int) -> tuple[dict[str, Any], float]:
        all_logs: dict[str, Any] = {}
        weighted_loss = 0.0
        for entry_config, data, name in validation_entries:
            data.set_epoch(epoch)
            aggregator = entry_config.aggregator.build(
                dataset_info=dataset_info.update_variable_metadata(variable_metadata),
                loss_scaling=stepper.effective_loss_scaling,
                save_diagnostics=config.save_per_epoch_diagnostics,
                output_dir=os.path.join(config.output_dir, name),
                channel_mean_names=stepper.loss_names,
            )
            logs = run_validation(
                train_stepper=stepper,
                validation_data=data,
                aggregator=aggregator,
                label=name,
                diagnostics_subdir=f"epoch_{epoch:04d}",
                record_logs=lambda logs: None,
            )
            all_logs.update(logs)
            if entry_config.weight > 0:
                metric_key = f"{name}/mean/loss"
                loss = logs.get(metric_key)
                if loss is None:
                    raise RuntimeError(
                        f"Validation entry {name!r} with "
                        f"weight={entry_config.weight} did not produce "
                        f"expected metric key {metric_key!r}."
                    )
                weighted_loss += entry_config.weight * loss
        return all_logs, weighted_loss

    def inference_callback(epoch: int) -> tuple[dict[str, Any], float | None]:
        if epoch not in inference_epochs:
            return {}, None
        all_logs: dict[str, Any] = {}
        weighted_error: float | None = None
        for i, (entry_config, data, entry_dataset_info, name) in enumerate(
            inference_entries
        ):
            if epoch not in inference_epoch_sets[i]:
                continue
            aggregator = entry_config.aggregator.build(
                dataset_info=entry_dataset_info,
                n_ic_steps=stepper.n_ic_timesteps,
                n_forward_steps=entry_config.n_forward_steps,
                initial_time=data.initial_time,
                normalize=stepper.normalizer.normalize,
                output_dir=os.path.join(config.output_dir, name),
                channel_mean_names=stepper.loss_names,
                save_diagnostics=config.save_per_epoch_diagnostics,
                n_ensemble_per_ic=entry_config.n_ensemble_per_ic,
                enable_time_series=False,
            )
            logs = inference_one_epoch(
                stepper=stepper,
                validation_context=contextlib.nullcontext,
                dataset=data,
                aggregator=aggregator,
                label=name,
                epoch=epoch,
            )
            all_logs.update(logs)
            if entry_config.weight > 0:
                metric_key = f"{name}/time_mean_norm/rmse/channel_mean"
                error = logs.get(metric_key)
                if error is None:
                    raise RuntimeError(
                        f"Inference entry {name!r} with weight={entry_config.weight} "
                        f"did not produce expected metric key {metric_key!r}. "
                        f"Entries contributing to checkpoint selection must produce "
                        f"this metric."
                    )
                if weighted_error is None:
                    weighted_error = 0.0
                weighted_error += entry_config.weight * error

        return all_logs, weighted_error

    primary_validation_data = validation_entries[0][1]
    do_gc_collect = fme.get_device() != torch.device("cpu")
    trainer_config: TrainConfigProtocol = config  # documenting trainer input type
    return Trainer(
        train_data=train_data,
        validation_data=primary_validation_data,
        stepper=stepper,
        build_optimization=builder.get_optimization,
        build_ema=builder.get_ema,
        config=trainer_config,
        aggregator_builder=aggregator_builder,
        validation_callback=validation_callback,
        end_of_batch_callback=end_of_batch_ops,
        inference_callback=inference_callback,
        do_gc_collect=do_gc_collect,
    )


class AggregatorBuilder(
    AggregatorBuilderABC[TrainOutput],
):
    def __init__(
        self,
        train_config: TrainAggregatorConfig,
        dataset_info: DatasetInfo,
        output_dir: str,
        loss_scaling: dict[str, torch.Tensor] | None = None,
        channel_mean_names: Sequence[str] | None = None,
        save_per_epoch_diagnostics: bool = False,
        validation_config: OneStepAggregatorConfig | None = None,
    ):
        self.train_config = train_config
        self.dataset_info = dataset_info
        self.loss_scaling = loss_scaling
        self.channel_mean_names = channel_mean_names
        self.output_dir = output_dir
        self.save_per_epoch_diagnostics = save_per_epoch_diagnostics
        self.validation_config = validation_config or OneStepAggregatorConfig()

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator(
            config=self.train_config,
            operations=self.dataset_info.gridded_operations,
        )

    def get_validation_aggregator(self) -> OneStepAggregator:
        return self.validation_config.build(
            dataset_info=self.dataset_info,
            loss_scaling=self.loss_scaling,
            save_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=os.path.join(self.output_dir, "val"),
            channel_mean_names=self.channel_mean_names,
        )


def run_train_from_config(config: TrainConfig):
    run_train(TrainBuilders(config), config)


def run_train(builders: TrainBuilders, config: TrainConfig):
    dist = Distributed.get_instance()
    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True
    if not os.path.isdir(config.experiment_dir):
        os.makedirs(config.experiment_dir, exist_ok=True)
    config_data = dataclasses.asdict(config)
    config.logging.configure_logging(
        config.experiment_dir,
        log_filename="out.log",
        config=config_data,
        resumable=True,
    )
    if config.resume_results is not None:
        logging.info(
            f"Resuming training from results in {config.resume_results.existing_dir}"
        )
    trainer = build_trainer(builders, config)
    trainer.train()
    logging.info(f"DONE ---- rank {dist.rank}")


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=TrainConfig, data=config_data, config=dacite.Config(strict=True)
    )
    config.set_random_seed()
    config.resume_results = prepare_directory(
        config.experiment_dir, config_data, config.resume_results
    )
    run_train_from_config(config)
