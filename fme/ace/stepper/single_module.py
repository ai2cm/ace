import dataclasses
import datetime
import logging
import pathlib
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import dacite
import dacite.exceptions
import torch
import xarray as xr
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.ace.stepper.parameter_init import ParameterInitializationConfig
from fme.core.coordinates import SerializableVerticalCoordinate, VerticalCoordinate
from fme.core.corrector.atmosphere import AtmosphereCorrectorConfig
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import WeightedMappingLoss, WeightedMappingLossConfig
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
from fme.core.step.step import InferenceDataProtocol, StepABC, StepSelector
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


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
    in_names: List[str]
    out_names: List[str]
    normalization: NormalizationConfig
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )
    ocean: Optional[OceanConfig] = None
    loss: WeightedMappingLossConfig = dataclasses.field(
        default_factory=lambda: WeightedMappingLossConfig()
    )
    corrector: Union[AtmosphereCorrectorConfig, CorrectorSelector] = dataclasses.field(
        default_factory=lambda: AtmosphereCorrectorConfig()
    )
    next_step_forcing_names: List[str] = dataclasses.field(default_factory=list)
    loss_normalization: Optional[NormalizationConfig] = None
    residual_normalization: Optional[NormalizationConfig] = None
    multi_call: Optional[MultiCallConfig] = None
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

    def load(self):
        self.normalization.load()
        if self.loss_normalization is not None:
            self.loss_normalization.load()
        if self.residual_normalization is not None:
            self.residual_normalization.load()

    @property
    def n_ic_timesteps(self) -> int:
        return 1

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

    def get_forcing_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=self.input_only_names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def _window_steps_required(self, n_forward_steps: int) -> int:
        return n_forward_steps + self.n_ic_timesteps

    def get_state(self):
        self.load()
        return dataclasses.asdict(self)

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        """
        If the model is being initialized from another model's weights for fine-tuning,
        returns those weights. Otherwise, returns None.

        The list mirrors the order of `modules` in the `Stepper` class.
        """
        return self.parameter_init.get_base_weights(_load_weights)

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
        init_weights: bool = True,
    ) -> "Stepper":
        """
        Args:
            img_shape: Shape of domain as (n_lat, n_lon).
            gridded_operations: Gridded operations to use.
            vertical_coordinate: Vertical coordinate to use.
            timestep: Timestep of the model.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        logging.info("Initializing stepper from provided legacy config")
        normalizer = self.normalization.build(self.normalize_names)
        combined_normalization_config = NetworkAndLossNormalizationConfig(
            network=self.normalization,
            loss=self.loss_normalization,
            residual=self.residual_normalization,
        )
        loss_normalizer = combined_normalization_config.get_loss_normalizer(
            self.normalize_names, residual_scaled_names=self.prognostic_names
        )
        new_config = self.to_stepper_config(
            normalizer=normalizer, loss_normalizer=loss_normalizer
        )
        return new_config.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
            init_weights=init_weights,
        )

    def get_ocean(self) -> Optional[OceanConfig]:
        return self.ocean

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @property
    def input_names(self) -> List[str]:
        return self.in_names

    @property
    def output_names(self) -> List[str]:
        return self.out_names

    @property
    def all_names(self):
        """Names of all variables required, including auxiliary ones."""
        extra_names = []
        if self.ocean is not None:
            extra_names.extend(self.ocean.forcing_names)
        if self.multi_call is not None:
            extra_names.extend(self.multi_call.names)
        all_names = list(set(self.in_names).union(self.out_names).union(extra_names))
        return all_names

    @property
    def normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        extra_names = []
        if self.multi_call is not None:
            extra_names.extend(self.multi_call.names)
        return list(set(self.in_names).union(self.out_names).union(extra_names))

    @property
    def input_only_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        return list(set(self.all_names) - set(self.out_names))

    @property
    def prognostic_names(self) -> List[str]:
        """Names of variables which both inputs and outputs."""
        return list(set(self.out_names).intersection(self.in_names))

    @property
    def loss_names(self) -> List[str]:
        extra_names = []
        if self.multi_call is not None:
            extra_names.extend(self.multi_call.names)
        return list(set(self.out_names).union(extra_names))

    @property
    def diagnostic_names(self) -> List[str]:
        """Names of variables which are outputs only."""
        extra_names = []
        if self.multi_call is not None:
            extra_names = self.multi_call.names
        out_names = list(set(self.out_names).union(extra_names))
        return list(set(out_names).difference(self.in_names))

    @classmethod
    def remove_deprecated_keys(cls, state: Dict[str, Any]) -> Dict[str, Any]:
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
        normalizer: Optional[StandardNormalizer] = None,
        loss_normalizer: Optional[StandardNormalizer] = None,
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
        normalizer: Optional[StandardNormalizer] = None,
        loss_normalizer: Optional[StandardNormalizer] = None,
    ) -> "SingleModuleStepConfig":
        if normalizer is not None:
            normalization = normalizer.get_normalization_config()
        else:
            normalization = self.normalization
        if loss_normalizer is not None:
            loss_normalization: Optional[NormalizationConfig] = (
                loss_normalizer.get_normalization_config()
            )
            residual_normalization: Optional[NormalizationConfig] = None
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

    def replace_multi_call(self, multi_call: Optional[MultiCallConfig]):
        self.multi_call = multi_call

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        self.ocean = ocean


def _load_weights(path: str) -> List[Mapping[str, Any]]:
    stepper = load_stepper(path)
    return_weights: List[Mapping[str, Any]] = []
    for module in stepper.modules:
        return_weights.append(module.state_dict())
    return return_weights


@dataclasses.dataclass
class ExistingStepperConfig:
    """
    Configuration for an existing stepper. This is only designed to point to
    a serialized stepper checkpoint for loading, e.g., in the case of training
    resumption.

    Parameters:
        checkpoint_path: The path to the serialized checkpoint.
    """

    checkpoint_path: str

    def __post_init__(self):
        self._stepper_config = StepperConfig.from_stepper_state(
            self._load_checkpoint()["stepper"]
        )

    def _load_checkpoint(self) -> Mapping[str, Any]:
        return torch.load(
            self.checkpoint_path, map_location=get_device(), weights_only=False
        )

    def get_evaluation_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return self._stepper_config.get_evaluation_window_data_requirements(
            n_forward_steps
        )

    def get_prognostic_state_data_requirements(self) -> PrognosticStateDataRequirements:
        return self._stepper_config.get_prognostic_state_data_requirements()

    def get_forcing_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return self._stepper_config.get_forcing_window_data_requirements(
            n_forward_steps
        )

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        return self._stepper_config.get_base_weights()

    def get_stepper(
        self,
        img_shape,
        gridded_operations,
        vertical_coordinate,
        timestep,
    ):
        logging.info(f"Initializing stepper from {self.checkpoint_path}")
        return Stepper.from_state(self._load_checkpoint()["stepper"])


def _prepend_timesteps(
    data: TensorMapping, timesteps: TensorMapping, time_dim: int = 1
) -> TensorDict:
    return {k: torch.cat([timesteps[k], v], dim=time_dim) for k, v in data.items()}


@dataclasses.dataclass
class TrainOutput(TrainOutputABC):
    metrics: TensorDict
    gen_data: TensorDict
    target_data: TensorDict
    time: xr.DataArray
    normalize: Callable[[TensorDict], TensorDict]
    derive_func: Callable[[TensorMapping, TensorMapping], TensorDict] = (
        lambda x, _: dict(x)
    )

    def remove_initial_condition(self, n_ic_timesteps: int) -> "TrainOutput":
        return TrainOutput(
            metrics=self.metrics,
            gen_data={k: v[:, n_ic_timesteps:] for k, v in self.gen_data.items()},
            target_data={k: v[:, n_ic_timesteps:] for k, v in self.target_data.items()},
            time=self.time[:, n_ic_timesteps:],
            normalize=self.normalize,
            derive_func=self.derive_func,
        )

    def copy(self) -> "TrainOutput":
        """Creates new dictionaries for the data but with the same tensors."""
        return TrainOutput(
            metrics=self.metrics,
            gen_data={k: v for k, v in self.gen_data.items()},
            target_data={k: v for k, v in self.target_data.items()},
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
        gen_data = self.derive_func(self.gen_data, self.target_data)
        target_data = self.derive_func(self.target_data, self.target_data)
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


def crps_loss(
    gen_norm_step: TensorDict,
    target_norm_step: TensorDict,
    names: List[str],
) -> torch.Tensor:
    """
    Compute the CRPS loss for a single timestep.

    Args:
        gen_norm_step: The generated normalized step with each variable having
            shape [n_batch, 2, ...] where the 2 represents the two samples.
        target_norm_step: The target normalized step with each variable having
            shape [n_batch, ...].
        names: The names of the variables to compute the loss for.

    Returns:
        The CRPS loss for the given variables.
    """
    total = torch.tensor(0.0, device=get_device())
    for name in names:
        total += _crps_loss_single(
            gen_norm_step[name],
            target_norm_step[name],
        )
    return total / len(names)


def _crps_loss_single(
    gen_norm_step: torch.Tensor,
    target_norm_step: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the CRPS loss for a single variable at a single timestep.

    Args:
        gen_norm_step: The generated normalized step, of shape
            [n_batch, 2, ...] where the 2 represents the two samples.
        target_norm_step: The target normalized step, of shape
            [n_batch, ...].

    Returns:
        The CRPS loss.
    """
    if gen_norm_step.shape[1] != 2:
        raise NotImplementedError(
            "CRPS loss is written here specifically for 2 samples, "
            f"got {gen_norm_step.shape[1]} samples"
        )
    # CRPS is `E[|X - y|] - 1/2 E[|X - X'|]`
    # below we compute the first term as the average of two samples
    # meaning the 0.5 factor can be pulled out
    # we are using "almost fair" CRPS from https://arxiv.org/html/2412.15832v1
    # with a value of alpha = 0.95 as used in the paper
    alpha = 0.95
    epsilon = (1 - alpha) / 2
    target_term = torch.abs(gen_norm_step - target_norm_step[:, None, ...]).mean(axis=1)
    internal_term = -0.5 * torch.abs(
        gen_norm_step[:, 0, ...] - gen_norm_step[:, 1, ...]
    )
    return (target_term + (1 - epsilon) * internal_term).mean()


def repeat_interleave_batch_dim(data: TensorMapping, repeats: int) -> TensorDict:
    return {k: v.repeat_interleave(repeats, dim=0) for k, v in data.items()}


def reshape_with_sample_dim(data: TensorMapping, repeats: int) -> TensorDict:
    return {
        k: v.reshape(v.shape[0] // repeats, repeats, *v.shape[1:])
        for k, v in data.items()
    }


def stack_list_of_tensor_dicts(
    dict_list: List[TensorDict],
    time_dim: int,
) -> TensorDict:
    keys = next(iter(dict_list)).keys()
    stack_dict = {}
    for k in keys:
        stack_dict[k] = torch.stack([d[k] for d in dict_list], dim=time_dim)
    return stack_dict


def process_prediction_generator_list(
    output_list: List[TensorDict],
    time_dim: int,
    time: xr.DataArray,
    horizontal_dims: Optional[List[str]] = None,
) -> BatchData:
    output_timeseries = stack_list_of_tensor_dicts(output_list, time_dim)
    return BatchData.new_on_device(
        data=output_timeseries,
        time=time,
        horizontal_dims=horizontal_dims,
    )


@dataclasses.dataclass
class StepperConfig:
    """
    Configuration for a stepper.

    Parameters:
        step: The step configuration.
        loss: The loss configuration.
        crps_training: Whether to use CRPS training for stochastic models.
        parameter_init: The parameter initialization configuration.
    """

    step: StepSelector
    loss: WeightedMappingLossConfig = dataclasses.field(
        default_factory=lambda: WeightedMappingLossConfig()
    )
    crps_training: bool = False
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )

    @property
    def n_ic_timesteps(self) -> int:
        return self.step.n_ic_timesteps

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
    def input_only_names(self) -> List[str]:
        return list(set(self.input_names) - set(self.output_names))

    def get_forcing_window_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=self.input_only_names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def _window_steps_required(self, n_forward_steps: int) -> int:
        return n_forward_steps + self.n_ic_timesteps

    def as_loaded_dict(self):
        self.step.load()
        return dataclasses.asdict(self)

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
        init_weights: bool = True,
    ):
        """
        Args:
            img_shape: Shape of domain as (n_lat, n_lon).
            gridded_operations: Gridded operations to use.
            vertical_coordinate: Vertical coordinate to use.
            timestep: Timestep of the model.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        dataset_info = DatasetInfo(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        )
        logging.info("Initializing stepper from provided config")
        step = self.step.get_step(dataset_info)
        derive_func = dataset_info.vertical_coordinate.build_derive_function(
            dataset_info.timestep
        )
        post_process_func = (
            dataset_info.vertical_coordinate.build_post_process_function()
        )
        return Stepper(
            config=self,
            step=step,
            dataset_info=dataset_info,
            post_process_func=post_process_func,
            derive_func=derive_func,
            init_weights=init_weights,
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
    def input_names(self) -> List[str]:
        """Names of variables which are required as inputs."""
        return self.step.input_names

    @property
    def all_names(self) -> List[str]:
        """Names of all variables."""
        return list(set(self.input_names + self.output_names))

    @property
    def next_step_forcing_names(self) -> List[str]:
        """
        Names of variables which are given as inputs but taken from the output timestep.

        An example might be solar insolation taken during the output window period.
        """
        return self.step.get_next_step_forcing_names()

    @property
    def prognostic_names(self) -> List[str]:
        """Names of variables which both inputs and outputs."""
        return self.step.prognostic_names

    @property
    def output_names(self) -> List[str]:
        """Names of variables which are outputs only."""
        return self.step.output_names

    @classmethod
    def remove_deprecated_keys(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        state_copy = state.copy()
        return state_copy

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        self.step.replace_ocean(ocean)

    def get_ocean(self) -> Optional[OceanConfig]:
        return self.step.get_ocean()

    def replace_multi_call(self, multi_call: Optional[MultiCallConfig]):
        self.step = replace_multi_call(self.step, multi_call)

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        """
        If the model is being initialized from another model's weights for fine-tuning,
        returns those weights. Otherwise, returns None.

        The list mirrors the order of `modules` in the `Stepper` class.
        """
        return self.parameter_init.get_base_weights(_load_weights)


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
        post_process_func: Callable[[TensorMapping], TensorDict],
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        init_weights: bool = True,
    ):
        """
        Args:
            config: The configuration.
            step: The step object.
            dataset_info: Information about dataset used for training.
            post_process_func: Function to post-process the output of the step function.
            derive_func: Function to compute derived variables.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        self._config = config
        self._step_obj = step
        self._dataset_info = dataset_info
        self._derive_func = derive_func
        self._post_process_func = post_process_func
        self._no_optimization = NullOptimization()

        def get_loss_obj():
            loss_normalizer = step.get_loss_normalizer()
            return config.loss.build(
                dataset_info.gridded_operations,
                out_names=config.loss_names,
                channel_dim=self.CHANNEL_DIM,
                normalizer=loss_normalizer,
            )

        self._loss_normalizer: Optional[StandardNormalizer] = None

        self._get_loss_obj = get_loss_obj
        self._loss_obj: Optional[WeightedMappingLoss] = None

        self._l2_sp_tuning_regularizer = config.parameter_init.apply(
            step.modules, init_weights=init_weights, load_weights=_load_weights
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
    def loss_obj(self) -> WeightedMappingLoss:
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
    def surface_temperature_name(self) -> Optional[str]:
        return self._step_obj.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        return self._step_obj.ocean_fraction_name

    def validate_inference_data(self, data: InferenceDataProtocol):
        self._step_obj.validate_inference_data(data)

    @property
    def effective_loss_scaling(self) -> TensorDict:
        """
        Effective loss scalings used to normalize outputs before computing loss.
        y_loss_normalized_i = (y_i - y_mean_i) / loss_scaling_i
        where loss_scaling_i = loss_normalizer_std_i / weight_i.
        """
        return self.loss_obj.effective_loss_scaling

    def replace_multi_call(self, multi_call: Optional[MultiCallConfig]):
        """
        Replace the MultiCall object with a new one. Note this is only
        meant to be used at inference time and may result in the loss
        function being unusable.

        Args:
            multi_call: The new multi_call configuration or None.
        """
        self._config.replace_multi_call(multi_call)
        new_stepper: "Stepper" = self._config.get_stepper(
            img_shape=self._dataset_info.img_shape,
            gridded_operations=self._dataset_info.gridded_operations,
            vertical_coordinate=self._dataset_info.vertical_coordinate,
            timestep=self._dataset_info.timestep,
            init_weights=False,
        )
        new_stepper._step_obj.load_state(self._step_obj.get_state())
        self._step_obj = new_stepper._step_obj

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self._config.replace_ocean(ocean)
        new_stepper: "Stepper" = self._config.get_stepper(
            img_shape=self._dataset_info.img_shape,
            gridded_operations=self._dataset_info.gridded_operations,
            vertical_coordinate=self._dataset_info.vertical_coordinate,
            timestep=self._dataset_info.timestep,
            init_weights=False,
        )
        new_stepper._step_obj.load_state(self._step_obj.get_state())
        self._step_obj = new_stepper._step_obj

    @property
    def prognostic_names(self) -> List[str]:
        return self._step_obj.prognostic_names

    @property
    def out_names(self) -> List[str]:
        return self._step_obj.output_names

    @property
    def loss_names(self) -> List[str]:
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
        return self._step_obj.step(input, next_step_input_data, wrapper=wrapper)

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
        concerns from the public `predict` method.

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
    def _input_only_names(self) -> List[str]:
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
    ) -> Tuple[BatchData, PrognosticState]:
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
            forcing_data = forcing.subset_names(forcing_names)
            if initial_condition.as_batch_data().n_timesteps != self.n_ic_timesteps:
                raise ValueError(
                    f"Initial condition must have {self.n_ic_timesteps} timesteps, got "
                    f"{initial_condition.as_batch_data().n_timesteps}."
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
            time_dim=self.TIME_DIM,
            time=forcing_data.time[:, self.n_ic_timesteps :],
            horizontal_dims=forcing_data.horizontal_dims,
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
            data=self._post_process_func(data.data),
            time=data.time,
            horizontal_dims=data.horizontal_dims,
        )
        return data, prognostic_state

    def predict_paired(
        self,
        initial_condition: PrognosticState,
        forcing: BatchData,
        compute_derived_variables: bool = False,
    ) -> Tuple[PairedData, PrognosticState]:
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
                    data=self._post_process_func(forward_data.data),
                    time=forward_data.time,
                    horizontal_dims=forward_data.horizontal_dims,
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
        metrics: Dict[str, float] = {}
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        target_data = self.get_forward_data(data, compute_derived_variables=False)

        optimization.set_mode(self._step_obj.modules)
        output_list = self._accumulate_loss(
            input_data,
            data,
            target_data,
            optimization,
            metrics,
            use_crps=self._config.crps_training,
        )

        regularizer_loss = self._get_regularizer_loss()
        if torch.any(regularizer_loss > 0):
            optimization.accumulate_loss(regularizer_loss)
        metrics["loss"] = optimization.get_accumulated_loss().detach()
        optimization.step_weights()
        gen_data = process_prediction_generator_list(
            output_list,
            time_dim=self.TIME_DIM,
            time=data.time[:, self.n_ic_timesteps :],
            horizontal_dims=data.horizontal_dims,
        ).data

        stepped = TrainOutput(
            metrics=metrics,
            gen_data=self._post_process_func(gen_data),
            target_data=self._post_process_func(target_data.data),
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
        return stepped

    def _accumulate_loss(
        self,
        input_data: PrognosticState,
        data: BatchData,
        target_data: BatchData,
        optimization: OptimizationABC,
        metrics: Dict[str, float],
        use_crps: bool = False,
    ):
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)
        # output from self.predict_paired does not include initial condition
        n_forward_steps = data.time.shape[1] - self.n_ic_timesteps
        if use_crps:
            n_samples = 2  # must be 2 for CRPS loss calculation
            input_sample_data: TensorMapping = repeat_interleave_batch_dim(
                input_data.as_batch_data().data, repeats=n_samples
            )
            forcing_sample_data: TensorMapping = repeat_interleave_batch_dim(
                data.data, repeats=n_samples
            )
        else:
            input_sample_data = input_data.as_batch_data().data
            forcing_sample_data = data.data
        output_generator = self._predict_generator(
            input_sample_data,
            forcing_sample_data,
            n_forward_steps,
            optimization,
        )
        output_list = []
        for step, gen_step in enumerate(output_generator):
            if use_crps:
                gen_step = reshape_with_sample_dim(gen_step, repeats=n_samples)
                output_list.append(
                    {k: v.select(1, index=0) for k, v in gen_step.items()}
                )
            else:
                output_list.append(gen_step)
            # Note: here we examine the loss for a single timestep,
            # not a single model call (which may contain multiple timesteps).
            target_step = {
                k: v.select(self.TIME_DIM, step) for k, v in target_data.data.items()
            }

            if use_crps:
                step_loss: torch.Tensor = crps_loss(
                    self._loaded_loss_normalizer.normalize(gen_step),
                    self._loaded_loss_normalizer.normalize(target_step),
                    names=self.out_names,
                )
            else:
                step_loss = self.loss_obj(gen_step, target_step)
            metrics[f"loss_step_{step}"] = step_loss.detach()
            optimization.accumulate_loss(step_loss)
        return output_list

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "config": self._config.as_loaded_dict(),
            "dataset_info": self._dataset_info.to_state(),
            "step": self._step_obj.get_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
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
            encoded_timestep = state.get("encoded_timestep", DEFAULT_ENCODED_TIMESTEP)
            timestep = decode_timestep(encoded_timestep)
            if "sigma_coordinates" in state:
                # for backwards compatibility with old checkpoints
                state["vertical_coordinate"] = state["sigma_coordinates"]
            vertical_coordinate = dacite.from_dict(
                data_class=SerializableVerticalCoordinate,
                data={"vertical_coordinate": state["vertical_coordinate"]},
                config=dacite.Config(strict=True),
            ).vertical_coordinate

            if "area" in state:
                # backwards-compatibility, these older checkpoints are always lat-lon
                gridded_operations: GriddedOperations = LatLonOperations(state["area"])
            else:
                gridded_operations = GriddedOperations.from_state(
                    state["gridded_operations"]
                )
            normalizer = StandardNormalizer.from_state(
                state.get("normalizer", state.get("normalization"))
            )
            if normalizer is None:
                raise ValueError(
                    f"No normalizer state found, keys include {state.keys()}"
                )
            loss_normalizer = StandardNormalizer.from_state(
                state.get("loss_normalizer", state.get("loss_normalization"))
            )
            if loss_normalizer is None:
                loss_normalizer = normalizer
            config = legacy_config.to_stepper_config(
                normalizer=normalizer, loss_normalizer=loss_normalizer
            )
            dataset_info = DatasetInfo(
                img_shape=state["img_shape"],
                timestep=timestep,
                vertical_coordinate=vertical_coordinate,
                gridded_operations=gridded_operations,
            )
            state["step"] = {
                # SingleModuleStep inside MultiCallStep
                "wrapped_step": {"module": state["module"]}
            }
        except dacite.exceptions.DaciteError:
            config = StepperConfig.from_stepper_state(state)
            dataset_info = DatasetInfo.from_state(state["dataset_info"])
        stepper = config.get_stepper(
            img_shape=dataset_info.img_shape,
            gridded_operations=dataset_info.gridded_operations,
            vertical_coordinate=dataset_info.vertical_coordinate,
            timestep=dataset_info.timestep,
            init_weights=False,
        )
        stepper.load_state(state)
        return stepper


def get_serialized_stepper_vertical_coordinate(
    state: Dict[str, Any],
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
    override_config: Optional[StepperOverrideConfig] = None,
) -> StepperConfig:
    """Load a stepper configuration, optionally overriding certain aspects.

    Args:
        checkpoint_path: The path to the serialized checkpoint.
        override_config: Configuration options to override (optional).

    Returns:
        The configuration of the stepper serialized in the checkpoint, with
        appropriate options overridden.
    """
    if override_config is None:
        override_config = StepperOverrideConfig()

    checkpoint = torch.load(
        checkpoint_path, map_location=get_device(), weights_only=False
    )

    config = StepperConfig.from_stepper_state(checkpoint["stepper"])

    if override_config.ocean != "keep":
        logging.info(
            "Overriding training ocean configuration with a new ocean configuration."
        )
        config.replace_ocean(override_config.ocean)
    if override_config.multi_call != "keep":
        logging.info(
            "Overriding training multi_call configuration with a new "
            "multi_call configuration."
        )
        config.replace_multi_call(override_config.multi_call)
    return config


def load_stepper(
    checkpoint_path: str | pathlib.Path,
    override_config: Optional[StepperOverrideConfig] = None,
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
