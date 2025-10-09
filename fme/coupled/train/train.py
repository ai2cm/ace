import dataclasses
import logging
import os
from collections.abc import Callable, Sequence

import dacite
import torch
import xarray as xr

import fme
import fme.core.logging_utils as logging_utils
from fme.core.cli import prepare_config, prepare_directory
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.dicts import to_flat_dict
from fme.core.distributed import Distributed
from fme.core.generics.trainer import AggregatorBuilderABC, Trainer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.aggregator import (
    InferenceEvaluatorAggregatorConfig,
    OneStepAggregator,
    TrainAggregator,
)
from fme.coupled.data_loading.batch_data import (
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.stepper import CoupledTrainOutput
from fme.coupled.train.train_config import TrainBuilders, TrainConfig


def build_trainer(builder: TrainBuilders, config: TrainConfig) -> Trainer:
    logging.info("Initializing training data loader")
    train_data = builder.get_train_data()
    logging.info("Initializing validation data loader")
    validation_data = builder.get_validation_data()
    logging.info("Initializing inline inference data loader")
    dataset_info = train_data.dataset_info
    inference_data = builder.get_evaluation_inference_data()

    variable_metadata = get_derived_variable_metadata() | train_data.variable_metadata
    logging.info("Starting model initialization")
    stepper = builder.get_stepper(dataset_info)
    end_of_batch_ops = builder.get_end_of_batch_ops(stepper.modules)

    batch = next(iter(inference_data.loader))
    initial_inference_times = batch.ocean_data.time.isel(time=0)
    n_timesteps_ocean = config.inference_n_coupled_steps + stepper.ocean.n_ic_timesteps
    n_timesteps_atmosphere = (
        config.inference_n_coupled_steps * stepper.n_inner_steps
        + stepper.atmosphere.n_ic_timesteps
    )
    aggregator_builder = CoupledAggregatorBuilder(
        inference_config=config.inference_aggregator,
        dataset_info=dataset_info.update_variable_metadata(variable_metadata),
        initial_inference_times=initial_inference_times,
        n_timesteps_ocean=n_timesteps_ocean,
        n_timesteps_atmosphere=n_timesteps_atmosphere,
        ocean_normalize=stepper.ocean.normalizer.normalize,
        atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
        ocean_loss_scaling=stepper.ocean.effective_loss_scaling,
        atmosphere_loss_scaling=stepper.atmosphere.effective_loss_scaling,
        save_per_epoch_diagnostics=config.save_per_epoch_diagnostics,
        output_dir=config.output_dir,
    )
    return Trainer(
        train_data=train_data,
        validation_data=validation_data,
        inference_data=inference_data,
        stepper=stepper,
        build_optimization=builder.get_optimization,
        build_ema=builder.get_ema,
        config=config,
        aggregator_builder=aggregator_builder,
        end_of_batch_callback=end_of_batch_ops,
    )


class CoupledAggregatorBuilder(
    AggregatorBuilderABC[CoupledPrognosticState, CoupledTrainOutput, CoupledPairedData]
):
    def __init__(
        self,
        inference_config: InferenceEvaluatorAggregatorConfig,
        dataset_info: CoupledDatasetInfo,
        initial_inference_times: xr.DataArray,
        n_timesteps_ocean: int,
        n_timesteps_atmosphere: int,
        output_dir: str,
        ocean_normalize: Callable[[TensorMapping], TensorDict],
        atmosphere_normalize: Callable[[TensorMapping], TensorDict],
        ocean_loss_scaling: TensorMapping | None = None,
        atmosphere_loss_scaling: TensorMapping | None = None,
        ocean_channel_mean_names: Sequence[str] | None = None,
        atmosphere_channel_mean_names: Sequence[str] | None = None,
        save_per_epoch_diagnostics: bool = False,
    ):
        self.inference_config = inference_config
        self.dataset_info = dataset_info
        self.initial_inference_times = initial_inference_times
        self.n_timesteps_ocean = n_timesteps_ocean
        self.n_timesteps_atmosphere = n_timesteps_atmosphere
        self.output_dir = output_dir
        self.ocean_normalize = ocean_normalize
        self.atmosphere_normalize = atmosphere_normalize
        self.ocean_loss_scaling = ocean_loss_scaling
        self.atmosphere_loss_scaling = atmosphere_loss_scaling
        self.ocean_channel_mean_names = ocean_channel_mean_names
        self.atmosphere_channel_mean_names = atmosphere_channel_mean_names
        self.save_per_epoch_diagnostics = save_per_epoch_diagnostics

    def get_train_aggregator(self) -> TrainAggregator:
        return TrainAggregator()

    def get_validation_aggregator(self) -> OneStepAggregator:
        return OneStepAggregator(
            dataset_info=self.dataset_info,
            save_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=os.path.join(self.output_dir, "val"),
            ocean_loss_scaling=self.ocean_loss_scaling,
            atmosphere_loss_scaling=self.atmosphere_loss_scaling,
            ocean_channel_mean_names=self.ocean_channel_mean_names,
            atmosphere_channel_mean_names=self.atmosphere_channel_mean_names,
        )

    def get_inference_aggregator(self):
        return self.inference_config.build(
            dataset_info=self.dataset_info,
            n_timesteps_ocean=self.n_timesteps_ocean,
            n_timesteps_atmosphere=self.n_timesteps_atmosphere,
            initial_time=self.initial_inference_times,
            ocean_normalize=self.ocean_normalize,
            atmosphere_normalize=self.atmosphere_normalize,
            save_diagnostics=self.save_per_epoch_diagnostics,
            output_dir=os.path.join(self.output_dir, "inference"),
            ocean_channel_mean_names=self.ocean_channel_mean_names,
            atmosphere_channel_mean_names=self.atmosphere_channel_mean_names,
        )


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
        config=config_as_dict, env_vars=env_vars, notes=beaker_url
    )
    if config.resume_results is not None:
        logging.info(
            f"Resuming training from results in {config.resume_results.existing_dir}"
        )
        config.resume_results.verify_wandb_resumption(config.experiment_dir)
    trainer = build_trainer(builders, config)
    try:
        trainer.train()
        logging.info(f"DONE ---- rank {dist.rank}")
    finally:
        dist.shutdown()


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
