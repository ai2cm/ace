import contextlib
import dataclasses
import logging
import os
from collections.abc import Sequence
from typing import Any

import dacite
import torch

import fme
from fme.core.cli import prepare_config, prepare_directory
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.distributed import Distributed
from fme.core.ema import EMATracker
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.generics.trainer import AggregatorBuilderABC, Trainer, inference_one_epoch
from fme.core.generics.validation import run_validation
from fme.coupled.aggregator import OneStepAggregator, TrainAggregator
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.stepper import CoupledTrainOutput
from fme.coupled.train.train_config import TrainBuilders, TrainConfig
from fme.coupled.typing_ import CoupledTensorMapping


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> Trainer:
    logging.info("Initializing training data loader")
    train_data = builder.get_train_data()

    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata
    dataset_info = train_data.dataset_info.update_variable_metadata(variable_metadata)

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
    inference_entries = builder.get_inference_data()
    inference_epochs = config.get_inference_epochs()
    inference_epoch_sets = config.get_inference_epoch_sets()

    logging.info("Starting model initialization")
    stepper = builder.get_stepper(train_data.dataset_info)
    end_of_batch_ops = builder.get_end_of_batch_ops(stepper.modules)

    loss_scaling = stepper.effective_loss_scaling
    aggregator_builder = CoupledAggregatorBuilder(
        dataset_info=dataset_info,
        loss_scaling=loss_scaling,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
        output_dir=config.output_dir,
    )

    def validation_callback(
        epoch: int,
        stepper: TrainStepperABC,
        ema: EMATracker,
    ) -> tuple[dict[str, Any], float]:
        all_logs: dict[str, Any] = {}
        weighted_loss = 0.0
        for entry_config, data, name in validation_entries:
            data.set_epoch(epoch)
            aggregator = OneStepAggregator(
                dataset_info=dataset_info,
                save_diagnostics=config.save_per_epoch_diagnostics,
                output_dir=os.path.join(config.output_dir, name),
                loss_scaling=loss_scaling,
            )
            logs = run_validation(
                train_stepper=stepper,
                validation_data=data,
                aggregator=aggregator,
                label=name,
                diagnostics_subdir=f"epoch_{epoch:04d}",
                record_logs=lambda logs: None,
                ema=ema,
                validate_using_ema=config.validate_using_ema,
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
        weighted_error = 0.0
        has_error = False
        for i, (entry_config, data, name) in enumerate(inference_entries):
            if epoch not in inference_epoch_sets[i]:
                continue
            batch = next(iter(data.loader))
            initial_times = batch.ocean_data.time.isel(time=0)
            n_timesteps_ocean = (
                entry_config.n_coupled_steps + stepper.ocean.n_ic_timesteps
            )
            n_timesteps_atmosphere = (
                entry_config.n_coupled_steps * stepper.n_inner_steps
                + stepper.atmosphere.n_ic_timesteps
            )
            aggregator = entry_config.aggregator.build(
                dataset_info=dataset_info,
                n_timesteps_ocean=n_timesteps_ocean,
                n_timesteps_atmosphere=n_timesteps_atmosphere,
                initial_time=initial_times,
                ocean_normalize=stepper.ocean.normalizer.normalize,
                atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
                save_diagnostics=config.save_per_epoch_diagnostics,
                output_dir=os.path.join(config.output_dir, name),
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
            error = logs.get(f"{name}/time_mean_norm/rmse/channel_mean")
            if error is not None:
                weighted_error += entry_config.weight * error
                has_error = True

        return all_logs, weighted_error if has_error else None

    return Trainer(
        train_data=train_data,
        stepper=stepper,
        build_optimization=builder.get_optimization,
        build_ema=builder.get_ema,
        config=config,
        aggregator_builder=aggregator_builder,
        validation_callback=validation_callback,
        end_of_batch_callback=end_of_batch_ops,
        inference_callback=inference_callback,
    )


class CoupledAggregatorBuilder(
    AggregatorBuilderABC[CoupledTrainOutput],
):
    def __init__(
        self,
        dataset_info: CoupledDatasetInfo,
        output_dir: str,
        loss_scaling: CoupledTensorMapping,
        save_per_epoch_diagnostics: bool = False,
    ):
        self.dataset_info = dataset_info
        self.output_dir = output_dir
        self.loss_scaling = loss_scaling
        self.save_per_epoch_diagnostics = save_per_epoch_diagnostics

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator()

    def get_validation_aggregator(self) -> OneStepAggregator:
        return OneStepAggregator(
            dataset_info=self.dataset_info,
            save_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=os.path.join(self.output_dir, "val"),
            loss_scaling=self.loss_scaling,
        )


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
        config.resume_results.verify_wandb_resumption(config.experiment_dir)
    trainer = build_trainer(builders, config)
    trainer.train()
    logging.info(f"DONE ---- rank {dist.rank}")


def run_train_from_config(config: TrainConfig):
    run_train(TrainBuilders(config), config)


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    data = prepare_config(yaml_config, override=override_dotlist)
    train_config: TrainConfig = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    train_config.set_random_seed()
    train_config.resume_results = prepare_directory(
        train_config.experiment_dir, data, train_config.resume_results
    )
    run_train_from_config(train_config)
