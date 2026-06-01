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
from fme.core.generics.data import GriddedDataABC
from fme.core.generics.lr_tuning import ValidateStepper
from fme.core.generics.train_stepper import TrainStepperABC
from fme.core.generics.trainer import (
    AggregatorBuilderABC,
    InferenceCallback,
    InferenceTask,
    Trainer,
    build_inference_callback,
)
from fme.core.generics.validation import run_validation, run_validation_loop
from fme.coupled.aggregator import OneStepAggregator, TrainAggregator
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.stepper import CoupledTrainOutput, CoupledTrainStepper
from fme.coupled.train.train_config import (
    InlineInferenceConfig,
    InlineValidationConfig,
    TrainBuilders,
    TrainConfig,
)
from fme.coupled.typing_ import CoupledTensorMapping


def get_validation_callback(
    validation_entries: Sequence[tuple[InlineValidationConfig, GriddedDataABC, str]],
    stepper: TrainStepperABC,
    dataset_info: CoupledDatasetInfo,
    loss_scaling: CoupledTensorMapping,
    save_per_epoch_diagnostics: bool,
    output_dir: str,
):
    def validation_callback(epoch: int) -> tuple[dict[str, Any], float]:
        all_logs: dict[str, Any] = {}
        weighted_loss = 0.0
        for entry_config, data, name in validation_entries:
            data.set_epoch(epoch)
            aggregator = OneStepAggregator(
                dataset_info=dataset_info,
                save_diagnostics=save_per_epoch_diagnostics,
                output_dir=os.path.join(output_dir, name),
                loss_scaling=loss_scaling,
            )
            logs = run_validation(
                train_stepper=stepper,
                validation_data=data,
                aggregator=aggregator,
                label=name,
                diagnostics_subdir=f"epoch_{epoch:04d}",
                record_logs=lambda logs: None,
            )
            overlap = all_logs.keys() & logs.keys()
            if overlap:
                raise RuntimeError(
                    f"Validation entry {name!r} produced log keys that "
                    f"overlap with earlier entries: {sorted(overlap)}"
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

    return validation_callback


def get_validate_stepper_callback(
    validation_entries: Sequence[tuple[InlineValidationConfig, GriddedDataABC, str]],
    dataset_info: CoupledDatasetInfo,
    loss_scaling: CoupledTensorMapping,
    validate_using_ema: bool,
) -> ValidateStepper:
    # LR tuning passes trial stepper/EMA instances distinct from the Trainer's
    # own stepper, so this callback manages its own EMA via run_validation_loop
    # rather than relying on the Trainer's validation_context().
    def validate_stepper(stepper: TrainStepperABC, ema: EMATracker) -> float:
        weighted_loss = 0.0
        for entry_config, data, name in validation_entries:
            aggregator = OneStepAggregator(
                dataset_info=dataset_info,
                save_diagnostics=False,
                output_dir="",
                loss_scaling=loss_scaling,
            )
            run_validation_loop(
                stepper=stepper,
                valid_data=data,
                aggregator=aggregator,
                ema=ema,
                validate_using_ema=validate_using_ema,
            )
            logs = aggregator.get_logs(label=name)
            if entry_config.weight > 0:
                metric_key = f"{name}/mean/loss"
                loss = logs.get(metric_key)
                if loss is not None:
                    weighted_loss += entry_config.weight * loss
        return weighted_loss

    return validate_stepper


def get_inference_callback(
    inference_entries: Sequence[
        tuple[InlineInferenceConfig, InferenceGriddedData, str]
    ],
    inference_epochs: Sequence[int],
    inference_epoch_sets: Sequence[set[int]],
    stepper: CoupledTrainStepper,
    dataset_info: CoupledDatasetInfo,
    output_dir: str,
    save_per_epoch_diagnostics: bool,
) -> InferenceCallback:
    tasks: list[InferenceTask] = []
    for i, (entry_config, data, name) in enumerate(inference_entries):
        tasks.append(
            InferenceTask(
                name=name,
                data=data,
                aggregator_factory=_make_coupled_aggregator_factory(
                    entry_config=entry_config,
                    data=data,
                    name=name,
                    stepper=stepper,
                    dataset_info=dataset_info,
                    output_dir=output_dir,
                    save_per_epoch_diagnostics=save_per_epoch_diagnostics,
                ),
                epoch_set=frozenset(inference_epoch_sets[i]),
                weight=entry_config.weight,
            )
        )

    return build_inference_callback(
        tasks=tasks,
        inference_epochs=inference_epochs,
        stepper=stepper,
    )


def _make_coupled_aggregator_factory(
    entry_config: InlineInferenceConfig,
    data: InferenceGriddedData,
    name: str,
    stepper: CoupledTrainStepper,
    dataset_info: CoupledDatasetInfo,
    output_dir: str,
    save_per_epoch_diagnostics: bool,
):
    def factory():
        batch = next(iter(data.loader))
        initial_times = batch.ocean_data.time.isel(time=0)
        n_timesteps_ocean = entry_config.n_coupled_steps + stepper.ocean.n_ic_timesteps
        n_timesteps_atmosphere = (
            entry_config.n_coupled_steps * stepper.n_inner_steps
            + stepper.atmosphere.n_ic_timesteps
        )
        return entry_config.aggregator.build(
            dataset_info=dataset_info,
            n_timesteps_ocean=n_timesteps_ocean,
            n_timesteps_atmosphere=n_timesteps_atmosphere,
            initial_time=initial_times,
            ocean_normalize=stepper.ocean.normalizer.normalize,
            atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
            save_diagnostics=save_per_epoch_diagnostics,
            output_dir=os.path.join(output_dir, name),
        )

    return factory


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> Trainer:
    logging.info("Initializing training data loader")
    train_data = builder.get_train_data()

    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata
    dataset_info = train_data.dataset_info.update_variable_metadata(variable_metadata)

    logging.info("Initializing validation data loaders")
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

    validation_callback = get_validation_callback(
        validation_entries=validation_entries,
        stepper=stepper,
        dataset_info=dataset_info,
        loss_scaling=loss_scaling,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
        output_dir=config.output_dir,
    )

    validate_stepper: ValidateStepper | None = None
    if config.lr_tuning is not None:
        validate_stepper = get_validate_stepper_callback(
            validation_entries=validation_entries,
            dataset_info=dataset_info,
            loss_scaling=loss_scaling,
            validate_using_ema=config.validate_using_ema,
        )

    inference_callback = get_inference_callback(
        inference_entries=inference_entries,
        inference_epochs=inference_epochs,
        inference_epoch_sets=inference_epoch_sets,
        stepper=stepper,
        dataset_info=dataset_info,
        output_dir=config.output_dir,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
    )

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
        validate_stepper=validate_stepper,
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
