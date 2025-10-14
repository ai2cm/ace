import contextlib
import dataclasses
import datetime
import logging
import pathlib
import warnings
from collections.abc import Callable, Generator, Mapping
from typing import Any, Literal, cast

import dacite
import dacite.exceptions
import torch
import xarray as xr
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.ace.stepper.parameter_init import (
    ParameterInitializationConfig,
    ParameterInitializer,
    StepperWeightsAndHistory,
    Weights,
    WeightsAndHistoryLoader,
    null_weights_and_history,
)
from fme.ace.stepper.time_length_probabilities import TimeLengthProbabilities
from fme.core.coordinates import (
    NullPostProcessFn,
    SerializableVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.utils import encode_timestep
from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.device import get_device
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.loss import StepLoss, StepLossConfig
from fme.core.masking import NullMasking, StaticMaskingConfig
from fme.core.multi_call import MultiCallConfig
from fme.core.normalizer import (
    NetworkAndLossNormalizationConfig,
    NormalizationConfig,
    StandardNormalizer,
)
from fme.core.ocean import OceanConfig
from fme.core.optimization import NullOptimization
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.step.multi_call import MultiCallStepConfig, replace_multi_call
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepABC, StepSelector
from fme.core.tensors import (
    add_ensemble_dim,
    fold_ensemble_dim,
    fold_sized_ensemble_dim,
    repeat_interleave_batch_dim,
    unfold_ensemble_dim,
)
from fme.core.timing import GlobalTimer
from fme.core.training_history import TrainingHistory, TrainingJob
from fme.core.typing_ import EnsembleTensorDict, TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


def load_weights_and_history(path: str | None) -> StepperWeightsAndHistory:
    if path is None:
        return null_weights_and_history()
    stepper = load_stepper(path)
    return_weights: Weights = []
    for module in stepper.modules:
        return_weights.append(module.state_dict())
    return return_weights, stepper.training_history


@dataclasses.dataclass
class SingleModuleStepperConfig:
    """
    Configuration for a single module stepper.

    Parameters:
        builder: The module builder.
        in_names: Names of input variables.
        out_names: Names of output variables.
        normalization: The normalization configuration.
        parameter_init: The parameter initialization configuration.
        ocean: The ocean configuration.
        loss: The loss configuration.
        corrector: The corrector configuration.
        next_step_forcing_names: Names of forcing variables for the next timestep.
        loss_normalization: The normalization configuration for the loss.
        residual_normalization: Optional alternative to configure loss normalization.
            If provided, it will be used for all *prognostic* variables in loss scaling.
        multi_call: The configuration of multi-called diagnostics.
        include_multi_call_in_loss: Whether to include multi-call diagnostics in the
            loss. The same loss configuration as specified in 'loss' is used.
        crps_training: Whether to use CRPS training for stochastic models.
        residual_prediction: Whether to have ML module predict tendencies for
            prognostic variables.
    """

    builder: ModuleSelector
    in_names: list[str]
    out_names: list[str]
    normalization: NormalizationConfig
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )
    ocean: OceanConfig | None = None
    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    corrector: AtmosphereCorrectorConfig | CorrectorSelector = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: list[str] = dataclasses.field(default_factory=list)
    loss_normalization: NormalizationConfig | None = None
    residual_normalization: NormalizationConfig | None = None
    multi_call: MultiCallConfig | None = None
    include_multi_call_in_loss: bool = False
    crps_training: bool = False
    residual_prediction: bool = False

    def __post_init__(self):
        for name in self.next_step_forcing_names:
            if name not in self.in_names:
                raise ValueError(
                    f"next_step_forcing_name '{name}' not in in_names: {self.in_names}"
                )
            if name in self.out_names:
                raise ValueError(
                    f"next_step_forcing_name is an output variable: '{name}'"
                )
        if (
            self.residual_normalization is not None
            and self.loss_normalization is not None
        ):
            raise ValueError(
                "Only one of residual_normalization, loss_normalization can "
                "be provided."
                "If residual_normalization is provided, it will be used for all "
                "*prognostic* variables in loss scalng. "
                "If loss_normalization is provided, it will be used for all variables "
                "in loss scaling."
            )
        if self.multi_call is not None:
            self.multi_call.validate(self.in_names, self.out_names)
        if self.include_multi_call_in_loss:
            if self.multi_call is None:
                raise ValueError(
                    "include_multi_calls_in_loss is True but no multi_call config "
                    "was provided."
                )

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @classmethod
    def remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        _unsupported_key_defaults = {
            "conserve_dry_air": False,
            "optimization": None,
            "conservation_loss": {"dry_air_penalty": None},
        }
        state_copy = state.copy()
        for key, default in _unsupported_key_defaults.items():
            if key in state_copy:
                if state_copy[key] == default or state_copy[key] is None:
                    del state_copy[key]
                else:
                    raise ValueError(
                        f"The stepper config option {key} is deprecated and the setting"
                        f" provided, {state_copy[key]}, is no longer implemented. The "
                        "SingleModuleStepper being loaded from state cannot be run by "
                        "this version of the code."
                    )
        for normalization_key in [
            "normalization",
            "loss_normalization",
            "residual_normalization",
        ]:
            if state_copy.get(normalization_key) is not None:
                if "exclude_names" in state_copy[normalization_key]:
                    if state_copy[normalization_key]["exclude_names"] is not None:
                        raise ValueError(
                            "The exclude_names option in normalization config is no "
                            "longer supported, but excluded names were found in "
                            f"{normalization_key}."
                        )
                    else:
                        del state_copy[normalization_key]["exclude_names"]
        if "prescriber" in state_copy:
            # want to maintain backwards compatibility for this particular feature
            if state_copy["prescriber"] is not None:
                if state_copy.get("ocean") is not None:
                    raise ValueError("Cannot specify both prescriber and ocean.")
                state_copy["ocean"] = {
                    "surface_temperature_name": state_copy["prescriber"][
                        "prescribed_name"
                    ],
                    "ocean_fraction_name": state_copy["prescriber"]["mask_name"],
                    "interpolate": state_copy["prescriber"]["interpolate"],
                }
            del state_copy["prescriber"]
        if "activation_checkpointing" in state_copy:
            del state_copy["activation_checkpointing"]
        return state_copy

    def to_stepper_config(
        self,
        normalizer: StandardNormalizer,
        loss_normalizer: StandardNormalizer,
    ) -> "StepperConfig":
        """
        Convert the current config to a stepper config.

        Overwriting normalization configuration is needed to avoid
        a checkpoint trying to load normalization data from netCDF files
        which are no longer present when running inference.

        Args:
            normalizer: overwrite the normalization config
                with data from this normalizer
            loss_normalizer: overwrite the loss normalization config
                with data from this normalizer

        Returns:
            A stepper config.
        """
        return StepperConfig(
            step=self._to_step_config(normalizer, loss_normalizer),
            loss=self.loss,
            crps_training=self.crps_training,
            parameter_init=self.parameter_init,
        )

    def _to_step_config(
        self,
        normalizer: StandardNormalizer | None = None,
        loss_normalizer: StandardNormalizer | None = None,
    ) -> StepSelector:
        return StepSelector(
            type="multi_call",
            config=dataclasses.asdict(
                MultiCallStepConfig(
                    wrapped_step=StepSelector(
                        type="single_module",
                        config=dataclasses.asdict(
                            self._to_single_module_step_config(
                                normalizer=normalizer,
                                loss_normalizer=loss_normalizer,
                            )
                        ),
                    ),
                    config=self.multi_call,
                    include_multi_call_in_loss=self.include_multi_call_in_loss,
                )
            ),
        )

    def _to_single_module_step_config(
        self,
        normalizer: StandardNormalizer | None = None,
        loss_normalizer: StandardNormalizer | None = None,
    ) -> "SingleModuleStepConfig":
        if normalizer is not None:
            normalization = normalizer.get_normalization_config()
        else:
            normalization = self.normalization
        if loss_normalizer is not None:
            loss_normalization: NormalizationConfig | None = (
                loss_normalizer.get_normalization_config()
            )
            residual_normalization: NormalizationConfig | None = None
        else:
            loss_normalization = self.loss_normalization
            residual_normalization = self.residual_normalization
        return SingleModuleStepConfig(
            builder=self.builder,
            in_names=self.in_names,
            out_names=self.out_names,
            normalization=NetworkAndLossNormalizationConfig(
                network=normalization,
                loss=loss_normalization,
                residual=residual_normalization,
            ),
            ocean=self.ocean,
            corrector=self.corrector,
            next_step_forcing_names=self.next_step_forcing_names,
            crps_training=self.crps_training,
            residual_prediction=self.residual_prediction,
        )


def _prepend_timesteps(
    data: EnsembleTensorDict, timesteps: TensorMapping, time_dim: int = 2
) -> EnsembleTensorDict:
    for v in data.values():
        n_ensemble = v.shape[1]
        break
    else:
        return data  # data is length zero
    timesteps = add_ensemble_dim(timesteps, repeats=n_ensemble)
    return EnsembleTensorDict(
        {k: torch.cat([timesteps[k], v], dim=time_dim) for k, v in data.items()}
    )


def _get_time_dim_size(data: TensorDict) -> int:
    for v in data.values():
        return v.shape[1]
    raise ValueError("data is empty")


def _clip_time_dim(data: TensorDict, time_dim_size: int) -> TensorDict:
    return {k: v[:, :time_dim_size] for k, v in data.items()}


@dataclasses.dataclass
class TrainOutput(TrainOutputABC):
    metrics: TensorDict
    gen_data: EnsembleTensorDict
    target_data: EnsembleTensorDict
    time: xr.DataArray
    normalize: Callable[[TensorDict], TensorDict]
    derive_func: Callable[[TensorMapping, TensorMapping], TensorDict] = (
        lambda x, _: dict(x)
    )

    def __post_init__(self):
        for v in self.target_data.values():
            if v.shape[1] != 1:
                raise ValueError(
                    f"target_data can only have one ensemble member, got {v.shape[1]}"
                )

    def ensemble_derive_func(
        self, data: EnsembleTensorDict, forcing_data: TensorMapping
    ) -> EnsembleTensorDict:
        """
        Compute derived variables for an ensemble of data.

        Args:
            data: The data to compute derived variables for.
            forcing_data: The forcing data to use for the derived variables.
                Time dimension must be at least as long as present in the data,
                if longer it will be clipped.

        Returns:
            The derived variables.
        """
        flattened_data, n_ensemble = fold_ensemble_dim(data)
        if n_ensemble > 1:
            ensemble_forcing_data = add_ensemble_dim(forcing_data, repeats=n_ensemble)
            flattened_forcing_data = fold_sized_ensemble_dim(
                ensemble_forcing_data, n_ensemble
            )
        else:
            flattened_forcing_data = dict(forcing_data)
        flattened_forcing_data = _clip_time_dim(
            flattened_forcing_data, _get_time_dim_size(flattened_data)
        )
        derived_data = self.derive_func(flattened_data, flattened_forcing_data)
        return unfold_ensemble_dim(derived_data, n_ensemble)

    def remove_initial_condition(self, n_ic_timesteps: int) -> "TrainOutput":
        return TrainOutput(
            metrics=self.metrics,
            gen_data=EnsembleTensorDict(
                {k: v[:, :, n_ic_timesteps:] for k, v in self.gen_data.items()}
            ),
            target_data=EnsembleTensorDict(
                {k: v[:, :, n_ic_timesteps:] for k, v in self.target_data.items()}
            ),
            time=self.time[:, n_ic_timesteps:],
            normalize=self.normalize,
            derive_func=self.derive_func,
        )

    def copy(self) -> "TrainOutput":
        """Creates new dictionaries for the data but with the same tensors."""
        return TrainOutput(
            metrics=self.metrics,
            gen_data=EnsembleTensorDict({k: v for k, v in self.gen_data.items()}),
            target_data=EnsembleTensorDict({k: v for k, v in self.target_data.items()}),
            time=self.time,
            normalize=self.normalize,
            derive_func=self.derive_func,
        )

    def prepend_initial_condition(
        self,
        initial_condition: PrognosticState,
    ) -> "TrainOutput":
        """
        Prepends an initial condition to the existing stepped data.
        Assumes data are on the same device.
        For data windows > 0, the target IC is different from the generated IC
            and may be provided for correct calculation of tendencies.

        Args:
            initial_condition: Initial condition data.
        """
        batch_data = initial_condition.as_batch_data()
        return TrainOutput(
            metrics=self.metrics,
            gen_data=_prepend_timesteps(self.gen_data, batch_data.data),
            target_data=_prepend_timesteps(
                self.target_data,
                batch_data.data,
            ),
            time=xr.concat([batch_data.time, self.time], dim="time"),
            normalize=self.normalize,
            derive_func=self.derive_func,
        )

    def compute_derived_variables(
        self,
    ) -> "TrainOutput":
        gen_data = self.ensemble_derive_func(
            self.gen_data, fold_sized_ensemble_dim(self.target_data, 1)
        )
        target_data = self.ensemble_derive_func(
            self.target_data, fold_sized_ensemble_dim(self.target_data, 1)
        )
        return TrainOutput(
            metrics=self.metrics,
            gen_data=gen_data,
            target_data=target_data,
            time=self.time,
            normalize=self.normalize,
            derive_func=self.derive_func,
        )

    def get_metrics(self) -> TensorDict:
        return self.metrics


def stack_list_of_tensor_dicts(
    dict_list: list[TensorDict],
    time_dim: int,
) -> TensorDict:
    keys = next(iter(dict_list)).keys()
    stack_dict = {}
    for k in keys:
        stack_dict[k] = torch.stack([d[k] for d in dict_list], dim=time_dim)
    return stack_dict


def process_ensemble_prediction_generator_list(
    output_list: list[EnsembleTensorDict],
) -> EnsembleTensorDict:
    output_timeseries = stack_list_of_tensor_dicts(
        cast(list[TensorDict], output_list), time_dim=2
    )
    return EnsembleTensorDict(
        {k: v for k, v in output_timeseries.items()},
    )


def process_prediction_generator_list(
    output_list: list[TensorDict],
    time: xr.DataArray,
    labels: list[set[str]],
    horizontal_dims: list[str] | None = None,
) -> BatchData:
    output_timeseries = stack_list_of_tensor_dicts(output_list, time_dim=1)
    return BatchData.new_on_device(
        data=output_timeseries,
        time=time,
        horizontal_dims=horizontal_dims,
        labels=labels,
    )


@dataclasses.dataclass
class StepperConfig:
    """
    Configuration for a stepper.

    Parameters:
        step: The step configuration.
        loss: The loss configuration.
        optimize_last_step_only: Whether to optimize only the last step.
        n_ensemble: The number of ensemble members evaluated for each training
            batch member. Default is 2 if the loss type is EnsembleLoss, otherwise
            the default is 1. Must be 2 for EnsembleLoss to be valid.
        crps_training: Deprecated, kept for backwards compatibility. Use
            n_ensemble=2 with a CRPS loss instead.
        parameter_init: The parameter initialization configuration.
        input_masking: Config for masking step inputs.
        train_n_forward_steps: The number of timesteps to train on and associated
            sampling probabilities. By default, the stepper will train on the full
            number of timesteps present in the training dataset samples. Values must
            be less than or equal to the number of timesteps present
            in the training dataset samples.
    """

    step: StepSelector
    loss: StepLossConfig = dataclasses.field(default_factory=lambda: StepLossConfig())
    optimize_last_step_only: bool = False
    n_ensemble: int = -1  # sentinel value to avoid None typing of attribute
    crps_training: bool = False
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )
    input_masking: StaticMaskingConfig | None = None
    train_n_forward_steps: TimeLengthProbabilities | int | None = None

    @property
    def train_n_forward_steps_sampler(self) -> TimeLengthProbabilities | None:
        if isinstance(self.train_n_forward_steps, int):
            return TimeLengthProbabilities.from_constant(self.train_n_forward_steps)
        return self.train_n_forward_steps

    def __post_init__(self):
        if self.crps_training:
            warnings.warn(
                "crps_training is deprecated, use n_ensemble=2 "
                "with a CRPS loss instead",
                DeprecationWarning,
            )
            self.n_ensemble = 2
            self.loss = StepLossConfig(
                type="EnsembleLoss",
                kwargs={"crps_weight": 1.0},
            )
        if self.n_ensemble == -1:
            if self.loss.type == "EnsembleLoss":
                self.n_ensemble = 2
            else:
                self.n_ensemble = 1

    @property
    def n_ic_timesteps(self) -> int:
        return self.step.n_ic_timesteps

    def get_train_window_data_requirements(
        self,
        default_n_forward_steps: int | None,
    ) -> DataRequirements:
        if self.train_n_forward_steps is None:
            if default_n_forward_steps is None:
                raise ValueError(
                    "default_n_forward_steps is required if "
                    "train_n_forward_steps is not provided"
                )
            n_forward_steps = default_n_forward_steps
        elif isinstance(self.train_n_forward_steps, int):
            n_forward_steps = self.train_n_forward_steps
        else:
            n_forward_steps = self.train_n_forward_steps.max_n_forward_steps
        return DataRequirements(
            names=self.all_names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def get_evaluation_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=self.all_names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def get_prognostic_state_data_requirements(self) -> PrognosticStateDataRequirements:
        return PrognosticStateDataRequirements(
            names=self.prognostic_names,
            n_timesteps=self.n_ic_timesteps,
        )

    @property
    def input_only_names(self) -> list[str]:
        return list(set(self.input_names) - set(self.output_names))

    def get_forcing_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=list(
                set(self.input_only_names).union(self.step.next_step_input_names)
            ),
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def _window_steps_required(self, n_forward_steps: int) -> int:
        return n_forward_steps + self.n_ic_timesteps

    def as_loaded_dict(self):
        self.step.load()
        return dataclasses.asdict(self)

    def get_stepper(
        self,
        dataset_info: DatasetInfo,
        apply_parameter_init: bool = True,
        training_history: TrainingHistory | None = None,
        load_weights_and_history: WeightsAndHistoryLoader = load_weights_and_history,
    ):
        """
        Args:
            dataset_info: Information about the training dataset.
            apply_parameter_init: Whether to apply parameter initialization.
            training_history: History of the stepper's training jobs.
            load_weights_and_history: Function for loading weights and history.
                Default implementation loads a Trainer checkpoint containing
                a Stepper.

        """
        logging.info("Initializing stepper from provided config")
        if apply_parameter_init:
            parameter_initializer = self.get_parameter_initializer(
                load_weights_and_history
            )
        else:
            parameter_initializer = ParameterInitializer()
        step = self.step.get_step(
            dataset_info, init_weights=parameter_initializer.freeze_weights
        )
        derive_func = dataset_info.vertical_coordinate.build_derive_function(
            dataset_info.timestep
        )
        if self.input_masking is None:
            input_masking = NullMasking()
        else:
            input_masking = self.input_masking.build(
                mask=dataset_info.mask_provider,
                means=step.normalizer.means,
            )
        try:
            output_process_func = dataset_info.mask_provider.build_output_masker()
        except MissingDatasetInfo:
            output_process_func = NullPostProcessFn()
        return Stepper(
            config=self,
            step=step,
            dataset_info=dataset_info,
            input_process_func=input_masking,
            output_process_func=output_process_func,
            derive_func=derive_func,
            parameter_initializer=parameter_initializer,
            training_history=training_history,
        )

    @classmethod
    def from_stepper_state(cls, state) -> "StepperConfig":
        """
        Initialize a StepperConfig from a stepper state.

        This is required for backwards compatibility with older steppers,
        whose configuration did not provide normalization constants, but rather
        pointed to files on disk. Newer stepper configurations load these
        constants into the configuration before checkpoints are saved.

        Args:
            state: The state of the stepper.

        Returns:
            The stepper config.
        """
        try:
            legacy_config = SingleModuleStepperConfig.from_state(state["config"])
            normalizer = StandardNormalizer.from_state(
                state.get("normalizer", state.get("normalization"))
            )
            if normalizer is None:
                raise KeyError(
                    "No normalization found in state, available keys: "
                    + ", ".join(state.keys())
                )
            loss_normalizer_config = state.get(
                "loss_normalizer", state.get("loss_normalization")
            )
            if loss_normalizer_config is None:
                loss_normalizer = normalizer
            else:
                loss_normalizer = StandardNormalizer.from_state(loss_normalizer_config)
            return legacy_config.to_stepper_config(
                normalizer=normalizer, loss_normalizer=loss_normalizer
            )
        except (dacite.exceptions.DaciteError, KeyError):
            state = cls.remove_deprecated_keys(state["config"])
            return dacite.from_dict(
                data_class=cls, data=state, config=dacite.Config(strict=True)
            )

    @property
    def loss_names(self):
        """Names of variables to include in loss."""
        return self.step.loss_names

    @property
    def input_names(self) -> list[str]:
        """Names of variables which are required as inputs."""
        return self.step.input_names

    @property
    def all_names(self) -> list[str]:
        """Names of all variables."""
        return list(set(self.input_names + self.output_names))

    @property
    def next_step_forcing_names(self) -> list[str]:
        """
        Names of variables which are given as inputs but taken from the output timestep.

        An example might be solar insolation taken during the output window period.
        """
        return self.step.get_next_step_forcing_names()

    @property
    def prognostic_names(self) -> list[str]:
        """Names of variables which both inputs and outputs."""
        return self.step.prognostic_names

    @property
    def output_names(self) -> list[str]:
        """Names of variables which are outputs only."""
        return self.step.output_names

    @classmethod
    def remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        return state_copy

    def replace_ocean(self, ocean: OceanConfig | None):
        self.step.replace_ocean(ocean)

    def get_ocean(self) -> OceanConfig | None:
        return self.step.get_ocean()

    def replace_multi_call(
        self, multi_call: MultiCallConfig | None, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace the multi-call configuration of self.step and ensure the
        associated state can be loaded as a multi-call step.

        A value of `None` for `multi_call` will remove the multi-call configuration.

        If the selected type supports it, the multi-call configuration will be
        updated in place. Otherwise, it will be wrapped in the multi_call step
        configuration with the given multi_call config or None.

        Note this updates self.step in place, but returns a new state dictionary.

        Args:
            multi_call: MultiCallConfig for the resulting self.step.
            state: state dictionary associated with the loaded step.

        Returns:
            The state dictionary updated to ensure consistency with that of a
            serialized multi-call step.
        """
        self.step, new_state = replace_multi_call(self.step, multi_call, state)
        return new_state

    def get_parameter_initializer(
        self,
        load_weights_and_history: WeightsAndHistoryLoader,
    ) -> ParameterInitializer:
        """
        Get the parameter initializer for this stepper configuration.
        """
        return self.parameter_init.build(
            load_weights_and_history=load_weights_and_history
        )

    @classmethod
    def from_state(cls, state) -> "StepperConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


class Stepper(
    TrainStepperABC[
        PrognosticState,
        BatchData,
        BatchData,
        PairedData,
        TrainOutput,
    ]
):
    """
    Stepper class for selectable step configurations.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: StepperConfig,
        step: StepABC,
        dataset_info: DatasetInfo,
        input_process_func: Callable[[TensorMapping], TensorDict],
        output_process_func: Callable[[TensorMapping], TensorDict],
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        parameter_initializer: ParameterInitializer,
        training_history: TrainingHistory | None = None,
    ):
        """
        Args:
            config: The configuration.
            step: The step object.
            dataset_info: Information about dataset used for training.
            output_process_func: Function to post-process the output of the step
                function.
            derive_func: Function to compute derived variables.
            input_process_func: Optional function for processing inputs and next-step
                inputs before passing them to the step object, e.g., by masking
                specific regions.
            parameter_initializer: The parameter initializer to use for loading weights
                from an external source.
            training_history: History of the stepper's training jobs.
        """
        self._config = config
        self._step_obj = step
        self._dataset_info = dataset_info
        self._derive_func = derive_func
        self._output_process_func = output_process_func
        self._input_process_func = input_process_func
        self._no_optimization = NullOptimization()
        self._parameter_initializer = parameter_initializer
        self._train_n_forward_steps_sampler = config.train_n_forward_steps_sampler

        def get_loss_obj() -> StepLoss:
            loss_normalizer = step.get_loss_normalizer()
            if config.loss is None:
                raise ValueError("Loss is not configured")
            return config.loss.build(
                dataset_info.gridded_operations,
                out_names=config.loss_names,
                channel_dim=self.CHANNEL_DIM,
                normalizer=loss_normalizer,
            )

        self._loss_normalizer: StandardNormalizer | None = None

        self._get_loss_obj = get_loss_obj
        self._loss_obj: StepLoss | None = None

        self._parameter_initializer.apply_weights(
            step.modules,
        )

        self._l2_sp_tuning_regularizer = (
            self._parameter_initializer.get_l2_sp_tuning_regularizer(
                step.modules,
            )
        )

        self._training_history = (
            training_history if training_history is not None else TrainingHistory()
        )
        self._append_training_history_from(
            base_training_history=self._parameter_initializer.training_history
        )

        _1: PredictFunction[  # for type checking
            PrognosticState,
            BatchData,
            BatchData,
        ] = self.predict

        _2: PredictFunction[  # for type checking
            PrognosticState,
            BatchData,
            PairedData,
        ] = self.predict_paired

        self._dataset_info = dataset_info

    @property
    def _loaded_loss_normalizer(self) -> StandardNormalizer:
        if self._loss_normalizer is None:
            loss_normalizer = self._step_obj.get_loss_normalizer()
            self._loss_normalizer = loss_normalizer
        return self._loss_normalizer

    @property
    def loss_obj(self) -> StepLoss:
        if self._loss_obj is None:
            self._loss_obj = self._get_loss_obj()
        return self._loss_obj

    @property
    def config(self) -> StepperConfig:
        return self._config

    @property
    def derive_func(self) -> Callable[[TensorMapping, TensorMapping], TensorDict]:
        return self._derive_func

    @property
    def surface_temperature_name(self) -> str | None:
        return self._step_obj.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> str | None:
        return self._step_obj.ocean_fraction_name

    def prescribe_sst(
        self,
        mask_data: TensorMapping,
        gen_data: TensorMapping,
        target_data: TensorMapping,
    ) -> TensorDict:
        """
        Prescribe sea surface temperature onto the generated surface temperature field.

        Args:
            mask_data: Source for the prescriber mask field.
            gen_data: Contains the generated surface temperature field.
            target_data: Contains the target surface temperature that will
                be prescribed onto the generated one according to the mask.
        """
        return self._step_obj.prescribe_sst(mask_data, gen_data, target_data)

    @property
    def training_dataset_info(self) -> DatasetInfo:
        return self._dataset_info

    @property
    def training_variable_metadata(self) -> Mapping[str, VariableMetadata]:
        return self._dataset_info.variable_metadata

    @property
    def training_history(self) -> TrainingHistory:
        return self._training_history

    def _append_training_history_from(
        self, base_training_history: TrainingHistory | None
    ):
        """
        When the stepper receives weights from a base stepper via parameter
        initialization, this helper is used to extend its training history to include
        the training history of the base stepper.

        Args:
            base_training_history: The training history from a base stepper to append.
        """
        if base_training_history is not None:
            self._training_history.extend(base_training_history)

    @property
    def effective_loss_scaling(self) -> TensorDict:
        """
        Effective loss scalings used to normalize outputs before computing loss.
        y_loss_normalized_i = (y_i - y_mean_i) / loss_scaling_i
        where loss_scaling_i = loss_normalizer_std_i / weight_i.
        """
        return self.loss_obj.effective_loss_scaling

    def replace_multi_call(self, multi_call: MultiCallConfig | None):
        """
        Replace the MultiCall object with a new one. Note this is only
        meant to be used at inference time and may result in the loss
        function being unusable.

        Args:
            multi_call: The new multi_call configuration or None.
        """
        state = self._step_obj.get_state()
        new_state = self._config.replace_multi_call(multi_call, state)
        new_stepper: Stepper = self._config.get_stepper(
            dataset_info=self._dataset_info, apply_parameter_init=False
        )
        new_stepper._step_obj.load_state(new_state)
        self._step_obj = new_stepper._step_obj

    def replace_ocean(self, ocean: OceanConfig | None):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self._config.replace_ocean(ocean)
        new_stepper: Stepper = self._config.get_stepper(
            dataset_info=self._dataset_info,
            apply_parameter_init=False,
        )
        new_stepper._step_obj.load_state(self._step_obj.get_state())
        self._step_obj = new_stepper._step_obj

    def get_base_weights(self) -> Weights | None:
        """
        Get the base weights of the stepper.

        Returns:
            A list of weight dictionaries for each module in the stepper.
        """
        return self._parameter_initializer.base_weights

    @property
    def prognostic_names(self) -> list[str]:
        return self._step_obj.prognostic_names

    @property
    def out_names(self) -> list[str]:
        return self._step_obj.output_names

    @property
    def loss_names(self) -> list[str]:
        return self._step_obj.loss_names

    @property
    def n_ic_timesteps(self) -> int:
        return self._step_obj.n_ic_timesteps

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return self._step_obj.modules

    @property
    def normalizer(self) -> StandardNormalizer:
        return self._step_obj.normalizer

    def step(
        self,
        input: TensorMapping,
        next_step_input_data: TensorMapping,
        wrapper: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon] containing denormalized data from the
                initial timestep.
            next_step_input_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon] containing denormalized data from
                the output timestep.
            wrapper: Wrapper to apply over each nn.Module before calling.

        Returns:
            The denormalized output data at the next time step.
        """
        input = self._input_process_func(input)
        next_step_input_data = self._input_process_func(next_step_input_data)
        output = self._step_obj.step(input, next_step_input_data, wrapper=wrapper)
        return self._output_process_func(output)

    def get_prediction_generator(
        self,
        initial_condition: PrognosticState,
        forcing_data: BatchData,
        n_forward_steps: int,
        optimizer: OptimizationABC,
    ) -> Generator[TensorDict, None, None]:
        """
        Predict multiple steps forward given initial condition and forcing data.

        Uses low-level inputs and does not compute derived variables, to separate
        concerns from the `predict` method.

        Args:
            initial_condition: The initial condition, containing tensors of shape
                [n_batch, self.n_ic_timesteps, <horizontal_dims>].
            forcing_data: The forcing data, containing tensors of shape
                [n_batch, n_forward_steps + self.n_ic_timesteps, <horizontal_dims>].
            n_forward_steps: The number of forward steps to predict, corresponding
                to the data shapes of forcing_data.
            optimizer: The optimizer to use for updating the module.

        Returns:
            Generator yielding the output data at each timestep.
        """
        ic_dict = initial_condition.as_batch_data().data
        forcing_dict = forcing_data.data
        return self._predict_generator(
            ic_dict, forcing_dict, n_forward_steps, optimizer
        )

    @property
    def _input_only_names(self) -> list[str]:
        return list(
            set(self._step_obj.input_names).difference(set(self._step_obj.output_names))
        )

    def _predict_generator(
        self,
        ic_dict: TensorMapping,
        forcing_dict: TensorMapping,
        n_forward_steps: int,
        optimizer: OptimizationABC,
    ) -> Generator[TensorDict, None, None]:
        state = {k: ic_dict[k].squeeze(self.TIME_DIM) for k in ic_dict}
        for step in range(n_forward_steps):
            input_forcing = {
                k: (
                    forcing_dict[k][:, step]
                    if k not in self._step_obj.next_step_forcing_names
                    else forcing_dict[k][:, step + 1]
                )
                for k in self._input_only_names
            }
            next_step_input_dict = {
                k: forcing_dict[k][:, step + 1]
                for k in self._step_obj.next_step_input_names
            }
            input_data = {**state, **input_forcing}

            def checkpoint(module):
                return optimizer.checkpoint(module, step=step)

            state = self.step(
                input_data,
                next_step_input_dict,
                wrapper=checkpoint,
            )
            yield state
            state = optimizer.detach_if_using_gradient_accumulation(state)

    def predict(
        self,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[BatchData, PrognosticState]:
        """
        Predict multiple steps forward given initial condition and reference data.

        Args:
            initial_condition: Prognostic state data with tensors of shape
                [n_batch, self.n_ic_timesteps, <horizontal_dims>]. This data is assumed
                to contain all prognostic variables and be denormalized.
            forcing: Contains tensors of shape
                [n_batch, self.n_ic_timesteps + n_forward_steps, n_lat, n_lon]. This
                contains the forcing and ocean data for the initial condition and all
                subsequent timesteps.
            compute_derived_variables: Whether to compute derived variables for the
                prediction.

        Returns:
            A batch data containing the prediction and the prediction's final state
            which can be used as a new initial condition.
        """
        timer = GlobalTimer.get_instance()
        forcing_names = set(self._input_only_names).union(
            self._step_obj.next_step_input_names
        )
        with timer.context("forward_prediction"):
            ic_batch_data = initial_condition.as_batch_data()
            if ic_batch_data.labels != forcing.labels:
                raise ValueError(
                    "Initial condition and forcing data must have the same labels, "
                    f"got {ic_batch_data.labels} and {forcing.labels}."
                )
            forcing_data = forcing.subset_names(forcing_names)
            if ic_batch_data.n_timesteps != self.n_ic_timesteps:
                raise ValueError(
                    f"Initial condition must have {self.n_ic_timesteps} timesteps, got "
                    f"{ic_batch_data.n_timesteps}."
                )
            n_forward_steps = forcing_data.n_timesteps - self.n_ic_timesteps
            output_list = list(
                self.get_prediction_generator(
                    initial_condition,
                    forcing_data,
                    n_forward_steps,
                    NullOptimization(),
                )
            )
        data = process_prediction_generator_list(
            output_list,
            time=forcing_data.time[:, self.n_ic_timesteps :],
            horizontal_dims=forcing_data.horizontal_dims,
            labels=forcing.labels,
        )
        if compute_derived_variables:
            with timer.context("compute_derived_variables"):
                data = (
                    data.prepend(initial_condition)
                    .compute_derived_variables(
                        derive_func=self.derive_func,
                        forcing_data=forcing_data,
                    )
                    .remove_initial_condition(self.n_ic_timesteps)
                )
        prognostic_state = data.get_end(self.prognostic_names, self.n_ic_timesteps)
        data = BatchData.new_on_device(
            data=data.data,
            time=data.time,
            horizontal_dims=data.horizontal_dims,
            labels=data.labels,
        )
        return data, prognostic_state

    def predict_paired(
        self,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[PairedData, PrognosticState]:
        """
        Predict multiple steps forward given initial condition and reference data.

        Args:
            initial_condition: Prognostic state data with tensors of shape
                [n_batch, self.n_ic_timesteps, <horizontal_dims>]. This data is assumed
                to contain all prognostic variables and be denormalized.
            forcing: Contains tensors of shape
                [n_batch, self.n_ic_timesteps + n_forward_steps, n_lat, n_lon]. This
                contains the forcing and ocean data for the initial condition and all
                subsequent timesteps.
            compute_derived_variables: Whether to compute derived variables for the
                prediction.

        Returns:
            A tuple of 1) a paired data object, containing the prediction paired with
            all target/forcing data at the same timesteps, and 2) the prediction's
            final state, which can be used as a new initial condition.
        """
        prediction, new_initial_condition = self.predict(
            initial_condition, forcing, compute_derived_variables
        )
        forward_data = self.get_forward_data(
            forcing, compute_derived_variables=compute_derived_variables
        )
        return (
            PairedData.from_batch_data(
                prediction=prediction,
                reference=BatchData.new_on_device(
                    data=forward_data.data,
                    time=forward_data.time,
                    horizontal_dims=forward_data.horizontal_dims,
                    labels=forward_data.labels,
                ),
            ),
            new_initial_condition,
        )

    def get_forward_data(
        self, data: BatchData, compute_derived_variables: bool = False
    ) -> BatchData:
        if compute_derived_variables:
            timer = GlobalTimer.get_instance()
            with timer.context("compute_derived_variables"):
                data = data.compute_derived_variables(
                    derive_func=self.derive_func,
                    forcing_data=data,
                )
        return data.remove_initial_condition(self.n_ic_timesteps)

    def _get_regularizer_loss(self) -> torch.Tensor:
        return self._l2_sp_tuning_regularizer() + self._step_obj.get_regularizer_loss()

    def train_on_batch(
        self,
        data: BatchData,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TrainOutput:
        """
        Train the model on a batch of data with one or more forward steps.

        If gradient accumulation is used by the optimization, the computational graph is
        detached between steps to reduce memory consumption. This means the model learns
        how to deal with inputs on step N but does not try to improve the behavior at
        step N by modifying the behavior for step N-1.

        Args:
            data: The batch data where each tensor in data.data has shape
                [n_sample, n_forward_steps + self.n_ic_timesteps, <horizontal_dims>].
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            compute_derived_variables: Whether to compute derived variables for the
                prediction and target data.

        Returns:
            The loss metrics, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        metrics: dict[str, float] = {}
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        target_data = self.get_forward_data(data, compute_derived_variables=False)

        optimization.set_mode(self._step_obj.modules)
        output_list = self._accumulate_loss(
            input_data,
            data,
            target_data,
            optimization,
            metrics,
        )

        regularizer_loss = self._get_regularizer_loss()
        if torch.any(regularizer_loss > 0):
            optimization.accumulate_loss(regularizer_loss)
        metrics["loss"] = optimization.get_accumulated_loss().detach()
        optimization.step_weights()

        gen_data = process_ensemble_prediction_generator_list(output_list)

        stepped = TrainOutput(
            metrics=metrics,
            gen_data=gen_data,
            target_data=add_ensemble_dim(target_data.data),
            time=target_data.time,
            normalize=self.normalizer.normalize,
            derive_func=self.derive_func,
        )
        ic = data.get_start(
            set(data.data.keys()), self.n_ic_timesteps
        )  # full data and not just prognostic get prepended
        stepped = stepped.prepend_initial_condition(ic)
        if compute_derived_variables:
            stepped = stepped.compute_derived_variables()
        # apply post-processing and return
        return stepped

    def _accumulate_loss(
        self,
        input_data: PrognosticState,
        data: BatchData,
        target_data: BatchData,
        optimization: OptimizationABC,
        metrics: dict[str, float],
    ) -> list[EnsembleTensorDict]:
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        # output from self.predict_paired does not include initial condition
        n_forward_steps = data.time.shape[1] - self.n_ic_timesteps
        n_ensemble = self._config.n_ensemble
        input_ensemble_data: TensorMapping = repeat_interleave_batch_dim(
            input_data.as_batch_data().data, repeats=n_ensemble
        )
        forcing_ensemble_data: TensorMapping = repeat_interleave_batch_dim(
            data.data, repeats=n_ensemble
        )
        output_generator = self._predict_generator(
            input_ensemble_data,
            forcing_ensemble_data,
            n_forward_steps,
            optimization,
        )
        output_list: list[EnsembleTensorDict] = []
        output_iterator = iter(output_generator)
        if self._train_n_forward_steps_sampler is not None:
            stochastic_n_forward_steps = self._train_n_forward_steps_sampler.sample()
            if stochastic_n_forward_steps > n_forward_steps:
                raise RuntimeError(
                    "The number of forward steps to train on "
                    f"({stochastic_n_forward_steps}) is greater than the number of "
                    f"forward steps in the data ({n_forward_steps}), "
                    "This is supposed to be ensured by the StepperConfig when train "
                    "data requirements are retrieved, so this is a bug."
                )
            n_forward_steps = stochastic_n_forward_steps
        for step in range(n_forward_steps):
            optimize_step = (
                step == n_forward_steps - 1 or not self._config.optimize_last_step_only
            )
            if optimize_step:
                context = contextlib.nullcontext()
            else:
                context = torch.no_grad()
            with context:
                gen_step = next(output_iterator)
                gen_step = unfold_ensemble_dim(gen_step, n_ensemble=n_ensemble)
                output_list.append(gen_step)
                # Note: here we examine the loss for a single timestep,
                # not a single model call (which may contain multiple timesteps).
                target_step = add_ensemble_dim(
                    {
                        k: v.select(self.TIME_DIM, step)
                        for k, v in target_data.data.items()
                    }
                )
                step_loss = self.loss_obj(gen_step, target_step, step=step)
                metrics[f"loss_step_{step}"] = step_loss.detach()
            if optimize_step:
                optimization.accumulate_loss(step_loss)
        return output_list

    def update_training_history(self, training_job: TrainingJob) -> None:
        """
        Update the stepper's history of training jobs.

        Args:
            training_job: The training job to add to the history.
        """
        self._training_history.append(training_job)

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "config": self._config.as_loaded_dict(),
            "dataset_info": self._dataset_info.to_state(),
            "step": self._step_obj.get_state(),
            "training_history": self._training_history.get_state(),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        self._step_obj.load_state(state["step"])

    @classmethod
    def from_state(cls, state) -> "Stepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        try:
            legacy_config = SingleModuleStepperConfig.from_state(state["config"])
            dataset_state = {}
            dataset_state["timestep"] = state.get(
                "encoded_timestep", DEFAULT_ENCODED_TIMESTEP
            )
            if "sigma_coordinates" in state:
                # for backwards compatibility with old checkpoints
                dataset_state["vertical_coordinate"] = state["sigma_coordinates"]
            else:
                dataset_state["vertical_coordinate"] = state["vertical_coordinate"]

            if "area" in state:
                # backwards-compatibility, these older checkpoints are always lat-lon
                dataset_state["gridded_operations"] = {
                    "type": "LatLonOperations",
                    "state": {"area_weights": state["area"]},
                }
            else:
                dataset_state["gridded_operations"] = state["gridded_operations"]

            if "img_shape" in state:
                dataset_state["img_shape"] = state["img_shape"]
            elif "data_shapes" in state:
                for _, shape in state["data_shapes"].items():
                    if len(shape) == 4:
                        dataset_state["img_shape"] = shape[-2:]
                        break

            normalizer = StandardNormalizer.from_state(
                state.get("normalizer", state.get("normalization"))
            )
            if normalizer is None:
                raise ValueError(
                    f"No normalizer state found, keys include {state.keys()}"
                )
            if "loss_normalizer" in state or "loss_normalization" in state:
                loss_normalizer = StandardNormalizer.from_state(
                    state.get("loss_normalizer", state.get("loss_normalization"))
                )
            else:
                loss_normalizer = normalizer
            config = legacy_config.to_stepper_config(
                normalizer=normalizer, loss_normalizer=loss_normalizer
            )
            dataset_info = DatasetInfo.from_state(dataset_state)
            state["step"] = {
                # SingleModuleStep inside MultiCallStep
                "wrapped_step": {"module": state["module"]}
            }
        except dacite.exceptions.DaciteError:
            config = StepperConfig.from_stepper_state(state)
            dataset_info = DatasetInfo.from_state(state["dataset_info"])
        training_history = TrainingHistory.from_state(state.get("training_history", []))
        stepper = config.get_stepper(
            dataset_info=dataset_info,
            training_history=training_history,
            # don't need to initialize weights, we're about to load_state
            apply_parameter_init=False,
        )
        stepper.load_state(state)
        return stepper


def get_serialized_stepper_vertical_coordinate(
    state: dict[str, Any],
) -> VerticalCoordinate:
    if "vertical_coordinate" in state:
        return dacite.from_dict(
            data_class=SerializableVerticalCoordinate,
            data={"vertical_coordinate": state["vertical_coordinate"]},
            config=dacite.Config(strict=True),
        ).vertical_coordinate
    elif "sigma_coordinates" in state:
        return dacite.from_dict(
            data_class=SerializableVerticalCoordinate,
            data={"vertical_coordinate": state["sigma_coordinates"]},
            config=dacite.Config(strict=True),
        ).vertical_coordinate
    else:
        dataset_info = DatasetInfo.from_state(state["dataset_info"])
        return dataset_info.vertical_coordinate


@dataclasses.dataclass
class StepperOverrideConfig:
    """
    Configuration for overriding stepper configuration options.

    The default value for each parameter is ``"keep"``, which denotes that the
    serialized stepper's configuration will not be modified when loaded. Passing
    other values will override the configuration of the loaded stepper.

    Parameters:
        ocean: Ocean configuration to override that used in producing a serialized
            stepper.
        multi_call: MultiCall configuration to override that used in producing a
            serialized stepper.
    """

    ocean: Literal["keep"] | OceanConfig | None = "keep"
    multi_call: Literal["keep"] | MultiCallConfig | None = "keep"


def load_stepper_config(
    checkpoint_path: str | pathlib.Path,
    override_config: StepperOverrideConfig | None = None,
) -> StepperConfig:
    """Load a stepper configuration, optionally overriding certain aspects.

    Args:
        checkpoint_path: The path to the serialized checkpoint.
        override_config: Configuration options to override (optional).

    Returns:
        The configuration of the stepper serialized in the checkpoint, with
        appropriate options overridden.
    """
    stepper = load_stepper(checkpoint_path, override_config)
    return stepper._config


def load_stepper(
    checkpoint_path: str | pathlib.Path,
    override_config: StepperOverrideConfig | None = None,
) -> Stepper:
    """Load a stepper, optionally overriding certain aspects.

    Args:
        checkpoint_path: The path to the serialized checkpoint.
        override_config: Configuration options to override (optional).

    Returns:
        The stepper serialized in the checkpoint, with appropriate options
        overridden.
    """
    if override_config is None:
        override_config = StepperOverrideConfig()

    checkpoint = torch.load(
        checkpoint_path, map_location=get_device(), weights_only=False
    )
    stepper = Stepper.from_state(checkpoint["stepper"])

    if override_config.ocean != "keep":
        logging.info(
            "Overriding training ocean configuration with a new ocean configuration."
        )
        stepper.replace_ocean(override_config.ocean)

    if override_config.multi_call != "keep":
        logging.info(
            "Overriding training multi_call configuration with a new "
            "multi_call configuration."
        )
        stepper.replace_multi_call(override_config.multi_call)
    return stepper
