import dataclasses
import logging
import pathlib
from collections.abc import Callable, Sequence

import dacite
import torch

import fme
from fme.ace.stepper import StepperOverrideConfig, apply_stepper_override
from fme.ace.stepper import load_stepper as load_single_stepper
from fme.ace.stepper import (
    load_stepper_config_with_override as load_single_stepper_config,
)
from fme.ace.stepper.single_module import apply_stepper_override_to_stepper_config
from fme.core.cli import prepare_config, prepare_directory
from fme.core.cloud import makedirs
from fme.core.derived_variables import get_derived_variable_metadata
from fme.core.generics.inference import get_record_to_wandb, run_inference
from fme.core.logging_utils import LoggingConfig
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.aggregator import InferenceEvaluatorAggregatorConfig
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.getters import get_inference_data
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
from fme.coupled.data_loading.inference import InferenceDataLoaderConfig
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.inference.data_writer import (
    CoupledDataWriterConfig,
    CoupledPairedDataWriter,
    DatasetMetadata,
)
from fme.coupled.inference.loop import CoupledDeriver, run_coupled_dataset_comparison
from fme.coupled.stepper import (
    ComponentConfig,
    CoupledOceanFractionConfig,
    CoupledStepper,
    CoupledStepperConfig,
    load_coupled_stepper,
)


def _validate_coupled_component_override(
    override: StepperOverrideConfig | None,
) -> None:
    """Restrict coupled inference overrides to ``prescribed_prognostic_names``.

    ``CoupledStepperConfig`` caches cross-component forcing-name sets and
    validates component compatibility at construction. Only
    ``prescribed_prognostic_names`` is recomputed on demand; an ``ocean``,
    ``multi_call`` or ``derived_forcings`` override applied afterward would leave
    those caches stale, so reject them rather than silently use stale values.
    """
    if override is None:
        return
    unsupported = [
        name
        for name, value in (
            ("ocean", override.ocean),
            ("multi_call", override.multi_call),
            ("derived_forcings", override.derived_forcings),
        )
        if value != "keep"
    ]
    if unsupported:
        raise ValueError(
            "Coupled inference overrides only support prescribed_prognostic_names, "
            f"but got unsupported override(s): {sorted(unsupported)}."
        )


def apply_coupled_stepper_config_inference_overrides(
    coupled_config: CoupledStepperConfig,
    ocean_override: StepperOverrideConfig | None,
    atmosphere_override: StepperOverrideConfig | None,
) -> None:
    """Mutate ``coupled_config`` in place for inference overrides.

    Forcing-window names are computed on demand from the (now mutated) component
    step configs, so no explicit refresh is needed.
    """
    _validate_coupled_component_override(ocean_override)
    _validate_coupled_component_override(atmosphere_override)
    if ocean_override is not None:
        apply_stepper_override_to_stepper_config(
            coupled_config.ocean.stepper, ocean_override
        )
    if atmosphere_override is not None:
        apply_stepper_override_to_stepper_config(
            coupled_config.atmosphere.stepper, atmosphere_override
        )


@dataclasses.dataclass
class StandaloneComponentConfig:
    """
    Configuration specifying the path to one of the components (ocean or
    atmosphere) within a CoupledStepper. Intended for inference with separate
    pretrained Stepper training checkpoints.

    """

    timedelta: str
    path: str


@dataclasses.dataclass
class StandaloneComponentCheckpointsConfig:
    """
    Configuration for creating a CoupledStepper from two separate Stepper
    checkpoints, for standalone inference.

    Parameters:
        ocean: The ocean component configuration.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include 'ocean'.
        sst_name: Name of the sea surface temperature field in the ocean data.
        ocean_stepper_override: Optional overrides when loading the ocean Stepper.
        atmosphere_stepper_override: Optional overrides when loading the atmosphere
            Stepper (e.g. prescribed_prognostic_names for inference).

    """

    ocean: StandaloneComponentConfig
    atmosphere: StandaloneComponentConfig
    sst_name: str = "sst"
    ocean_fraction_prediction: CoupledOceanFractionConfig | None = None
    ocean_stepper_override: StepperOverrideConfig | None = None
    atmosphere_stepper_override: StepperOverrideConfig | None = None

    def load_stepper_config(self) -> CoupledStepperConfig:
        return CoupledStepperConfig(
            ocean=ComponentConfig(
                timedelta=self.ocean.timedelta,
                stepper=load_single_stepper_config(
                    self.ocean.path, self.ocean_stepper_override
                ),
            ),
            atmosphere=ComponentConfig(
                timedelta=self.atmosphere.timedelta,
                stepper=load_single_stepper_config(
                    self.atmosphere.path, self.atmosphere_stepper_override
                ),
            ),
            sst_name=self.sst_name,
            ocean_fraction_prediction=self.ocean_fraction_prediction,
        )

    def load_stepper(self) -> CoupledStepper:
        ocean = load_single_stepper(self.ocean.path, self.ocean_stepper_override)
        atmosphere = load_single_stepper(
            self.atmosphere.path, self.atmosphere_stepper_override
        )
        dataset_info = CoupledDatasetInfo(
            ocean=ocean.training_dataset_info,
            atmosphere=atmosphere.training_dataset_info,
        )

        return CoupledStepper(
            config=self.load_stepper_config(),
            ocean=ocean,
            atmosphere=atmosphere,
            dataset_info=dataset_info,
        )


def load_stepper_config(
    checkpoint_path: str | pathlib.Path | StandaloneComponentCheckpointsConfig,
    ocean_stepper_override: StepperOverrideConfig | None = None,
    atmosphere_stepper_override: StepperOverrideConfig | None = None,
) -> CoupledStepperConfig:
    """Load a coupled stepper configuration.

    Args:
        checkpoint_path: The path to the serialized CoupledStepper checkpoint, or a
            StandaloneComponentCheckpointsConfig.
        ocean_stepper_override: When ``checkpoint_path`` is a single coupled checkpoint
            file, optional overrides merged into the ocean ``StepperConfig`` **before**
            computing forcing-window requirements (e.g. ``prescribed_prognostic_names``
            for a checkpoint that was saved without them). Ignored for
            ``StandaloneComponentCheckpointsConfig`` (use overrides on that object).
        atmosphere_stepper_override: Same for the atmosphere component.

    Returns:
        The CoupledStepperConfig from the serialized checkpoint or constructed from the
        standalone ocean and atmosphere checkpoints.
    """
    if isinstance(checkpoint_path, StandaloneComponentCheckpointsConfig):
        logging.info(
            f"Loading ocean model checkpoint from {checkpoint_path.ocean.path}"
        )
        logging.info(
            "Loading atmosphere model checkpoint from "
            f"{checkpoint_path.atmosphere.path}"
        )
        return checkpoint_path.load_stepper_config()

    logging.info(f"Loading trained coupled model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = CoupledStepperConfig.from_state(checkpoint["stepper"]["config"])
    apply_coupled_stepper_config_inference_overrides(
        config,
        ocean_override=ocean_stepper_override,
        atmosphere_override=atmosphere_stepper_override,
    )
    return config


def load_stepper(
    checkpoint_path: str | pathlib.Path | StandaloneComponentCheckpointsConfig,
    ocean_stepper_override: StepperOverrideConfig | None = None,
    atmosphere_stepper_override: StepperOverrideConfig | None = None,
) -> CoupledStepper:
    """Load a coupled stepper.

    Args:
        checkpoint_path: The path to the serialized CoupledStepper checkpoint, or a
            StandaloneComponentCheckpointsConfig.
        ocean_stepper_override: When loading a single coupled checkpoint, optional
            overrides for the ocean Stepper (ignored for
            StandaloneComponentCheckpointsConfig).
        atmosphere_stepper_override: When loading a single coupled checkpoint, optional
            overrides for the atmosphere Stepper (ignored for
            StandaloneComponentCheckpointsConfig).

    Returns:
        The CoupledStepper serialized in the checkpoint or constructed from the
        standalone ocean and atmosphere checkpoints.
    """
    if isinstance(checkpoint_path, StandaloneComponentCheckpointsConfig):
        logging.info(
            f"Loading ocean model checkpoint from {checkpoint_path.ocean.path}"
        )
        logging.info(
            "Loading atmosphere model checkpoint from "
            f"{checkpoint_path.atmosphere.path}"
        )
        return checkpoint_path.load_stepper()

    stepper = load_coupled_stepper(checkpoint_path)
    _validate_coupled_component_override(ocean_stepper_override)
    _validate_coupled_component_override(atmosphere_stepper_override)
    # Overrides mutate each component Stepper's config, which the CoupledStepper
    # aliases as its nested CoupledStepperConfig component config, so forcing
    # windows (computed on demand) reflect the overrides automatically.
    if atmosphere_stepper_override is not None:
        apply_stepper_override(stepper.atmosphere, atmosphere_stepper_override)
    if ocean_stepper_override is not None:
        apply_stepper_override(stepper.ocean, ocean_stepper_override)
    return stepper


def _validate_coupled_steps_config(n_coupled_steps: int, coupled_steps_in_memory: int):
    if n_coupled_steps % coupled_steps_in_memory:
        raise ValueError("n_coupled_steps must be divisible by coupled_steps_in_memory")


def _validate_stepper_overrides(
    checkpoint_path: str | pathlib.Path | StandaloneComponentCheckpointsConfig,
    ocean_stepper_override: StepperOverrideConfig | None,
    atmosphere_stepper_override: StepperOverrideConfig | None,
) -> None:
    """Reject top-level stepper overrides for standalone component checkpoints.

    The top-level ``ocean_stepper_override`` / ``atmosphere_stepper_override`` only
    apply when loading a single coupled checkpoint. A
    ``StandaloneComponentCheckpointsConfig`` carries its own per-component overrides,
    so top-level ones would be silently ignored; raise instead.
    """
    if isinstance(checkpoint_path, StandaloneComponentCheckpointsConfig) and (
        ocean_stepper_override is not None or atmosphere_stepper_override is not None
    ):
        raise ValueError(
            "ocean_stepper_override / atmosphere_stepper_override are only "
            "supported when loading a single coupled checkpoint. Set overrides on "
            "the StandaloneComponentCheckpointsConfig components instead."
        )


@dataclasses.dataclass
class InferenceEvaluatorConfig:
    """
    Configuration for running inference including comparison to reference data.

    Parameters:
        experiment_dir: Directory to save results to.
        n_coupled_steps: Number of steps to run the model forward for.
        checkpoint_path: Path to a CoupledStepper training checkpoint to load, or a
            mapping to two separate Stepper training checkpoints.
        logging: configuration for logging.
        loader: Configuration for data to be used as initial conditions, forcing, and
            target in inference.
        coupled_steps_in_memory: Number of coupled steps to complete in memory
            at a time, will load one more step for initial condition.
        data_writer: Configuration for data writers.
        aggregator: Configuration for inference evaluator aggregator.
        prediction_loader: Configuration for prediction data to evaluate. If given,
            model evaluation will not run, and instead predictions will be evaluated.
            Model checkpoint will still be used to determine inputs and outputs.
        ocean_stepper_override: Optional overrides when loading a **single** coupled
            checkpoint (not ``StandaloneComponentCheckpointsConfig``), applied to the
            ocean ``Stepper`` and to ``CoupledStepperConfig`` used for forcing windows
            (e.g. ``StepperOverrideConfig(prescribed_prognostic_names=[...])``).
        atmosphere_stepper_override: Optional overrides for the atmosphere Stepper
            when loading a single coupled checkpoint.
    """

    experiment_dir: str
    n_coupled_steps: int
    checkpoint_path: str | StandaloneComponentCheckpointsConfig
    logging: LoggingConfig
    loader: InferenceDataLoaderConfig
    coupled_steps_in_memory: int = 1
    data_writer: CoupledDataWriterConfig = dataclasses.field(
        default_factory=lambda: CoupledDataWriterConfig()
    )
    aggregator: InferenceEvaluatorAggregatorConfig = dataclasses.field(
        default_factory=lambda: InferenceEvaluatorAggregatorConfig()
    )
    prediction_loader: InferenceDataLoaderConfig | None = None
    ocean_stepper_override: StepperOverrideConfig | None = None
    atmosphere_stepper_override: StepperOverrideConfig | None = None

    def __post_init__(self):
        _validate_coupled_steps_config(
            self.n_coupled_steps, self.coupled_steps_in_memory
        )
        _validate_stepper_overrides(
            self.checkpoint_path,
            self.ocean_stepper_override,
            self.atmosphere_stepper_override,
        )

    def configure_logging(self, log_filename: str):
        config = dataclasses.asdict(self)
        self.logging.configure_logging(
            self.experiment_dir, log_filename, config=config, resumable=False
        )

    def load_stepper(self) -> CoupledStepper:
        return load_stepper(
            self.checkpoint_path,
            ocean_stepper_override=self.ocean_stepper_override,
            atmosphere_stepper_override=self.atmosphere_stepper_override,
        )

    def load_stepper_config(self) -> CoupledStepperConfig:
        return load_stepper_config(
            self.checkpoint_path,
            ocean_stepper_override=self.ocean_stepper_override,
            atmosphere_stepper_override=self.atmosphere_stepper_override,
        )

    def get_data_writer(
        self,
        data: InferenceGriddedData,
    ) -> CoupledPairedDataWriter:
        if self.data_writer.ocean.time_coarsen is not None:
            try:
                self.data_writer.ocean.time_coarsen.validate(
                    self.coupled_steps_in_memory,
                    self.n_coupled_steps,
                )
            except ValueError as err:
                raise ValueError(
                    f"Ocean time_coarsen config invalid with error: {str(err)}"
                )
        if self.data_writer.atmosphere.time_coarsen is not None:
            try:
                self.data_writer.atmosphere.time_coarsen.validate(
                    self.coupled_steps_in_memory * data.n_inner_steps,
                    self.n_coupled_steps * data.n_inner_steps,
                )
            except ValueError as err:
                raise ValueError(
                    f"Atmosphere time_coarsen config invalid with error: {str(err)}"
                )

        variable_metadata = get_derived_variable_metadata() | data.variable_metadata
        dataset_metadata = DatasetMetadata.from_env()
        coupled_dataset_metadata = {
            "ocean": dataset_metadata,
            "atmosphere": dataset_metadata,
        }
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            initial_condition_times=data.initial_time.to_numpy(),
            n_timesteps_ocean=self.n_coupled_steps,
            n_timesteps_atmosphere=self.n_coupled_steps * data.n_inner_steps,
            ocean_timestep=data.ocean_timestep,
            atmosphere_timestep=data.atmosphere_timestep,
            variable_metadata=variable_metadata,
            coords=data.coords,
            dataset_metadata=coupled_dataset_metadata,
        )


class _Deriver(CoupledDeriver):
    """
    Deriver for coupled dataset comparison: removes initial condition and
    computes derived variables on CoupledBatchData.
    """

    def __init__(
        self,
        n_ic_timesteps_ocean: int,
        n_ic_timesteps_atmosphere: int,
        ocean_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        atmosphere_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
    ):
        self._n_ic_timesteps_ocean = n_ic_timesteps_ocean
        self._n_ic_timesteps_atmosphere = n_ic_timesteps_atmosphere
        self._ocean_derive_func = ocean_derive_func
        self._atmosphere_derive_func = atmosphere_derive_func

    def get_forward_data(
        self,
        data: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> CoupledBatchData:
        if compute_derived_variables:
            timer = GlobalTimer.get_instance()
            with timer.context("compute_derived_variables"):
                data = data.compute_derived_variables(
                    ocean_derive_func=self._ocean_derive_func,
                    atmosphere_derive_func=self._atmosphere_derive_func,
                    forcing_data=data,
                )
        return data.remove_initial_condition(
            n_ic_timesteps_ocean=self._n_ic_timesteps_ocean,
            n_ic_timesteps_atmosphere=self._n_ic_timesteps_atmosphere,
        )


def main(yaml_config: str, override_dotlist: Sequence[str] | None = None):
    config_data = prepare_config(yaml_config, override=override_dotlist)
    config = dacite.from_dict(
        data_class=InferenceEvaluatorConfig,
        data=config_data,
        config=dacite.Config(strict=True),
    )
    prepare_directory(config.experiment_dir, config_data)
    with GlobalTimer(), torch.no_grad():
        return run_evaluator_from_config(config)


def run_evaluator_from_config(config: InferenceEvaluatorConfig):
    timer = GlobalTimer.get_instance()
    timer.start_outer("inference")
    timer.start("initialization")

    makedirs(config.experiment_dir, exist_ok=True)
    config.configure_logging(log_filename="inference_out.log")

    if fme.using_gpu():
        torch.backends.cudnn.benchmark = True

    stepper_config = config.load_stepper_config()
    logging.info("Loading inference data")
    window_requirements = stepper_config.get_evaluation_window_data_requirements(
        n_coupled_steps=config.coupled_steps_in_memory
    )
    initial_condition_requirements = (
        stepper_config.get_prognostic_state_data_requirements()
    )
    stepper = config.load_stepper()
    data = get_inference_data(
        config=config.loader,
        total_coupled_steps=config.n_coupled_steps,
        window_requirements=window_requirements,
        initial_condition=initial_condition_requirements,
        dataset_info=stepper.training_dataset_info,
    )
    stepper.set_eval()

    aggregator_config: InferenceEvaluatorAggregatorConfig = config.aggregator
    batch = next(iter(data.loader))
    initial_time = batch.ocean_data.time.isel(time=0)
    variable_metadata = get_derived_variable_metadata() | data.variable_metadata
    dataset_info = data.dataset_info.update_variable_metadata(variable_metadata)
    n_timesteps_ocean = config.n_coupled_steps + stepper.ocean.n_ic_timesteps
    n_timesteps_atmosphere = (
        config.n_coupled_steps * stepper.n_inner_steps
        + stepper.atmosphere.n_ic_timesteps
    )
    aggregator = aggregator_config.build(
        dataset_info=dataset_info,
        n_timesteps_ocean=n_timesteps_ocean,
        n_timesteps_atmosphere=n_timesteps_atmosphere,
        initial_time=initial_time,
        ocean_normalize=stepper.ocean.normalizer.normalize,
        atmosphere_normalize=stepper.atmosphere.normalizer.normalize,
        output_dir=config.experiment_dir,
    )

    writer = config.get_data_writer(data)

    timer.stop("initialization")
    logging.info("Starting inference")
    logger = get_record_to_wandb(label="inference")
    if config.prediction_loader is not None:
        prediction_data = get_inference_data(
            config=config.prediction_loader,
            total_coupled_steps=config.n_coupled_steps,
            window_requirements=window_requirements,
            initial_condition=initial_condition_requirements,
            dataset_info=stepper.training_dataset_info,
        )
        deriver = _Deriver(
            n_ic_timesteps_ocean=stepper.ocean.n_ic_timesteps,
            n_ic_timesteps_atmosphere=stepper.atmosphere.n_ic_timesteps,
            ocean_derive_func=stepper.ocean.derive_func,
            atmosphere_derive_func=stepper.atmosphere.derive_func,
        )
        run_coupled_dataset_comparison(
            aggregator=aggregator,
            prediction_data=prediction_data,
            target_data=data,
            deriver=deriver,
            writer=writer,
            record_logs=logger.log,
            all_names=stepper_config.all_names,
        )
    else:
        run_inference(
            predict=stepper.predict_paired,
            data=data,
            aggregator=aggregator,
            writer=writer,
            record_logs=logger.log,
        )

    timer.start("final_writer_flush")
    logging.info("Starting final flush of data writer")
    writer.finalize()
    logging.info("Writing reduced metrics to disk in netcdf format.")
    aggregator.flush_diagnostics()
    timer.stop("final_writer_flush")

    timer.stop_outer("inference")
    total_steps = (
        config.n_coupled_steps * stepper.n_inner_steps
    ) * config.loader.n_initial_conditions
    inference_duration = timer.get_duration("inference")
    wandb_logging_duration = timer.get_duration("inference/wandb_logging")
    total_steps_per_second = total_steps / (inference_duration - wandb_logging_duration)
    timer.log_durations()
    logging.info(
        "Total steps per second (ignoring wandb logging): "
        f"{total_steps_per_second:.2f} steps/second"
    )

    summary_logs = {
        "total_steps_per_second": total_steps_per_second,
        **aggregator.get_summary_logs(),
    }
    logger.log_to_current_step(summary_logs)
    logger.log_to_current_step(timer.get_durations(), label="")
