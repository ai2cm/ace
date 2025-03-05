import dataclasses
import datetime
import logging
import pathlib
from copy import copy
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
import torch
import xarray as xr
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.multi_call import MultiCallConfig
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.core.coordinates import (
    HybridSigmaPressureCoordinate,
    SerializableVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.corrector.corrector import CorrectorConfig
from fme.core.corrector.registry import CorrectorABC
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import NormalizationConfig, StandardNormalizer
from fme.core.ocean import Ocean, OceanConfig
from fme.core.optimization import ActivationCheckpointingConfig, NullOptimization
from fme.core.packer import Packer
from fme.core.parameter_init import ParameterInitializationConfig
from fme.core.registry import CorrectorSelector, ModuleSelector
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
        activation_checkpointing: Configuration for activation checkpointing to trade
            increased computation for lowered memory during training.
        crps_training: Whether to use CRPS training for stochastic models.
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
    corrector: Union[CorrectorConfig, CorrectorSelector] = dataclasses.field(
        default_factory=lambda: CorrectorConfig()
    )
    next_step_forcing_names: List[str] = dataclasses.field(default_factory=list)
    loss_normalization: Optional[NormalizationConfig] = None
    residual_normalization: Optional[NormalizationConfig] = None
    multi_call: Optional[MultiCallConfig] = None
    include_multi_call_in_loss: bool = False
    activation_checkpointing: ActivationCheckpointingConfig = dataclasses.field(
        default_factory=lambda: ActivationCheckpointingConfig()
    )
    crps_training: bool = False

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
        if self.ocean is None:
            names = self.forcing_names
        else:
            names = list(set(self.forcing_names).union(self.ocean.forcing_names))

        return DataRequirements(
            names=names,
            n_timesteps=self._window_steps_required(n_forward_steps),
        )

    def _window_steps_required(self, n_forward_steps: int) -> int:
        return n_forward_steps + self.n_ic_timesteps

    def get_state(self):
        return dataclasses.asdict(self)

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        """
        If the model is being initialized from another model's weights for fine-tuning,
        returns those weights. Otherwise, returns None.

        The list mirrors the order of `modules` in the `SingleModuleStepper` class.
        """
        base_weights = self.parameter_init.get_base_weights()
        if base_weights is not None:
            return [base_weights]
        else:
            return None

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        timestep: datetime.timedelta,
    ):
        logging.info("Initializing stepper from provided config")
        vertical_coordinate = vertical_coordinate.to(get_device())
        derive_func = vertical_coordinate.build_derive_function(timestep)
        corrector = vertical_coordinate.build_corrector(
            config=self.corrector,
            gridded_operations=gridded_operations,
            timestep=timestep,
        )
        normalizer = self.normalization.build(self.normalize_names)
        step = SingleModuleStep(
            config=self,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            corrector=corrector,
            normalizer=normalizer,
            timestep=timestep,
        )
        return SingleModuleStepper(
            config=self,
            step=step,
            derive_func=derive_func,
        )

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

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
    def forcing_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        return list(set(self.in_names) - set(self.out_names))

    @property
    def prognostic_names(self) -> List[str]:
        """Names of variables which both inputs and outputs."""
        return list(set(self.out_names).intersection(self.in_names))

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
        return state_copy


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
        self._stepper_config = SingleModuleStepperConfig.from_state(
            self._load_checkpoint()["stepper"]["config"]
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
        del img_shape  # unused
        logging.info(f"Initializing stepper from {self.checkpoint_path}")
        return SingleModuleStepper.from_state(self._load_checkpoint()["stepper"])


def _combine_normalizers(
    residual_normalizer: StandardNormalizer,
    model_normalizer: StandardNormalizer,
) -> StandardNormalizer:
    # Combine residual and model normalizers by overwriting the model normalizer
    # values that are present in residual normalizer. The residual normalizer
    # is assumed to have a subset of prognostic keys only.
    means, stds = copy(model_normalizer.means), copy(model_normalizer.stds)
    means.update(residual_normalizer.means)
    stds.update(residual_normalizer.stds)
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=model_normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=model_normalizer.fill_nans_on_denormalize,
    )


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


class SingleModuleStep:
    """
    Step class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepperConfig,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: VerticalCoordinate,
        corrector: CorrectorABC,
        normalizer: StandardNormalizer,
        timestep: datetime.timedelta,
        init_weights: bool = True,
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            gridded_operations: The gridded operations, e.g. for area weighting.
            vertical_coordinate: The vertical coordinate.
            corrector: The corrector to use at the end of each step.
            normalizer: The normalizer to use.
            timestep: Timestep of the model.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        self._gridded_operations = gridded_operations  # stored for serializing
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.normalizer = normalizer
        if config.ocean is not None:
            self.ocean: Optional[Ocean] = config.ocean.build(
                config.in_names, config.out_names, timestep
            )
        else:
            self.ocean = None
        self.module = config.builder.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )
        module, self._l2_sp_tuning_regularizer = config.parameter_init.apply(
            self.module, init_weights=init_weights
        )
        self.module = module.to(get_device())
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self.module = dist.wrap_module(self.module)

        self._vertical_coordinate = vertical_coordinate.to(get_device())
        self._timestep = timestep

        self._corrector = corrector
        self.in_names = config.in_names
        self.out_names = config.out_names

        self._activation_checkpointing = config.activation_checkpointing

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        return self._vertical_coordinate

    @property
    def gridded_operations(self) -> GriddedOperations:
        return self._gridded_operations

    @property
    def timestep(self) -> datetime.timedelta:
        return self._timestep

    @property
    def surface_temperature_name(self) -> Optional[str]:
        if self._config.ocean is not None:
            return self._config.ocean.surface_temperature_name
        return None

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        if self._config.ocean is not None:
            return self._config.ocean.ocean_fraction_name
        return None

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self._config.ocean = ocean
        if ocean is None:
            self.ocean = ocean
        else:
            self.ocean = ocean.build(self.in_names, self.out_names, self.timestep)

    @property
    def forcing_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        if self.ocean is None:
            return self._config.forcing_names
        return list(set(self._config.forcing_names).union(self.ocean.forcing_names))

    @property
    def prognostic_names(self) -> List[str]:
        return sorted(self._config.prognostic_names)

    @property
    def diagnostic_names(self) -> List[str]:
        return sorted(self._config.diagnostic_names)

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    @property
    def modules(self) -> nn.ModuleList:
        """
        Returns:
            A list of modules being trained.
        """
        return nn.ModuleList([self.module])

    def step(
        self,
        input: TensorMapping,
        next_step_forcing_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for `self.module`
                and is assumed to contain all input variables and be denormalized.
            next_step_forcing_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary forcing
                data at the output timestep for the ocean model and corrector.
            use_activation_checkpointing: If True, wrap the module call with
                torch.utils.checkpoint.checkpoint, reducing memory consumption
                in exchange for increased computation. This is only relevant during
                training and otherwise has no effect.

        Returns:
            The denormalized output data at the next time step.
        """
        input_norm = self.normalizer.normalize(input)
        input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        if use_activation_checkpointing:
            output_tensor = torch.utils.checkpoint.checkpoint(
                self.module,
                input_tensor,
                use_reentrant=False,
                **self._activation_checkpointing.kwargs,
            )
        else:
            output_tensor = self.module(input_tensor)
        output_norm = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
        output = self.normalizer.denormalize(output_norm)
        if self._corrector is not None:
            output = self._corrector(input, output, next_step_forcing_data)
        if self.ocean is not None:
            output = self.ocean(input, output, next_step_forcing_data)
        return output

    def get_regularizer_loss(self):
        return self._l2_sp_tuning_regularizer()

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "module": self.module.state_dict(),
            "normalizer": self.normalizer.get_state(),
            "img_shape": self._img_shape,
            "config": self._config.get_state(),
            "gridded_operations": self._gridded_operations.to_state(),
            "vertical_coordinate": self._vertical_coordinate.as_dict(),
            "encoded_timestep": encode_timestep(self.timestep),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        if "module" in state:
            module = state["module"]
            if "module.device_buffer" in module:
                # for backwards compatibility with old checkpoints
                del module["module.device_buffer"]
            self.module.load_state_dict(module)

    @classmethod
    def from_state(cls, state) -> "SingleModuleStep":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        config_data = {**state["config"]}  # make a copy to avoid mutating input
        config_data["normalization"] = state["normalizer"]

        if "area" in state:
            # backwards-compatibility, these older checkpoints are always lat-lon
            gridded_operations: GriddedOperations = LatLonOperations(state["area"])
        else:
            gridded_operations = GriddedOperations.from_state(
                state["gridded_operations"]
            )

        if "sigma_coordinates" in state:
            # for backwards compatibility with old checkpoints
            state["vertical_coordinate"] = state["sigma_coordinates"]

        vertical_coordinate = dacite.from_dict(
            data_class=SerializableVerticalCoordinate,
            data={"vertical_coordinate": state["vertical_coordinate"]},
            config=dacite.Config(strict=True),
        ).vertical_coordinate

        # for backwards compatibility with original ACE checkpoint which
        # serialized vertical coordinates as float64
        if isinstance(vertical_coordinate, HybridSigmaPressureCoordinate):
            if vertical_coordinate.ak.dtype == torch.float64:
                vertical_coordinate.ak = vertical_coordinate.ak.to(dtype=torch.float32)
            if vertical_coordinate.bk.dtype == torch.float64:
                vertical_coordinate.bk = vertical_coordinate.bk.to(dtype=torch.float32)
        encoded_timestep = state.get("encoded_timestep", DEFAULT_ENCODED_TIMESTEP)
        timestep = decode_timestep(encoded_timestep)
        if "img_shape" in state:
            img_shape = state["img_shape"]
        else:
            # this is for backwards compatibility with old checkpoints
            for v in state["data_shapes"].values():
                img_shape = v[-2:]
                break
        config = SingleModuleStepperConfig.from_state(config_data)
        corrector = vertical_coordinate.build_corrector(
            config=config.corrector,
            gridded_operations=gridded_operations,
            timestep=timestep,
        )
        step = cls(
            config=config,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            corrector=corrector,
            timestep=timestep,
            normalizer=config.normalization.build(config.normalize_names),
            # don't need to initialize weights, we're about to load_state
            init_weights=False,
        )
        step.load_state(state)
        return step


class SingleModuleStepper(
    TrainStepperABC[
        PrognosticState,
        BatchData,
        BatchData,
        PairedData,
        TrainOutput,
    ],
):
    """
    Stepper class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepperConfig,
        step: SingleModuleStep,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
    ):
        """
        Args:
            config: The configuration.
            step: The step object.
            derive_func: Function to compute derived variables.
        """
        self._step_obj = step
        self._derive_func = derive_func
        self._post_process_func = (
            self._step_obj.vertical_coordinate.build_post_process_function()
        )
        self._config = config
        self._no_optimization = NullOptimization()

        self.loss_obj = config.loss.build(
            step.gridded_operations.area_weighted_mean,
            config.out_names,
            self.CHANNEL_DIM,
        )

        if config.loss_normalization is not None:
            self.loss_normalizer = config.loss_normalization.build(
                names=config.normalize_names
            )
        elif config.residual_normalization is not None:
            # Use residual norm for prognostic variables and input/output
            # normalizer for diagnostic variables in loss
            self.loss_normalizer = _combine_normalizers(
                residual_normalizer=config.residual_normalization.build(
                    config.prognostic_names
                ),
                model_normalizer=config.normalization.build(config.normalize_names),
            )
        else:
            self.loss_normalizer = config.normalization.build(config.normalize_names)

        if config.multi_call is not None:
            self._multi_call = config.multi_call.build(self.step)
            if config.include_multi_call_in_loss:
                self._multi_call_loss = config.loss.build(
                    step.gridded_operations.area_weighted_mean,
                    self._multi_call.names,
                    self.CHANNEL_DIM,
                )
            else:
                zero_loss = torch.tensor(0.0, device=get_device())
                self._multi_call_loss = lambda x, y: zero_loss
        else:
            self._multi_call = None

        self._activation_checkpointing = config.activation_checkpointing

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

    @property
    def derive_func(self) -> Callable[[TensorMapping, TensorMapping], TensorDict]:
        return self._derive_func

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        return self._step_obj.vertical_coordinate

    @property
    def surface_temperature_name(self) -> Optional[str]:
        return self._step_obj.surface_temperature_name

    @property
    def ocean_fraction_name(self) -> Optional[str]:
        return self._step_obj.ocean_fraction_name

    @property
    def timestep(self) -> datetime.timedelta:
        return self._step_obj.timestep

    @property
    def effective_loss_scaling(self) -> TensorDict:
        """
        Effective loss scalings used to normalize outputs before computing loss.
        y_loss_normalized_i = (y_i - y_mean_i) / loss_scaling_i
        where loss_scaling_i = loss_normalizer_std_i / weight_i.
        """
        custom_weights = self._config.loss.weights
        loss_normalizer_stds = self.loss_normalizer.stds
        return {
            k: loss_normalizer_stds[k] / custom_weights.get(k, 1.0)
            for k in self._config.out_names
        }

    def replace_ocean(self, ocean: Optional[OceanConfig]):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model configuration or None.
        """
        self._step_obj.replace_ocean(ocean)

    def replace_multi_call(self, multi_call: Optional[MultiCallConfig]):
        """
        Replace the MultiCall object with a new one. Note this is only
        meant to be used at inference time and does not affect the loss
        function.

        Args:
            multi_call: The new multi_call configuration or None.
        """
        self._config.multi_call = multi_call
        if multi_call is None:
            self._multi_call = None
        else:
            multi_call.validate(self.in_names, self.out_names)
            self._multi_call = multi_call.build(self.step)

    @property
    def forcing_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        return self._step_obj.forcing_names

    @property
    def prognostic_names(self) -> List[str]:
        return self._step_obj.prognostic_names

    @property
    def diagnostic_names(self) -> List[str]:
        return self._step_obj.diagnostic_names

    @property
    def in_names(self) -> List[str]:
        return self._step_obj.in_names

    @property
    def out_names(self) -> List[str]:
        return self._step_obj.out_names

    @property
    def n_ic_timesteps(self) -> int:
        return self._step_obj.n_ic_timesteps

    @property
    def module(self) -> nn.Module:
        return self._step_obj.module

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
        next_step_forcing_data: TensorMapping,
        use_activation_checkpointing: bool = False,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for `self.module`
                and is assumed to contain all input variables and be denormalized.
            next_step_forcing_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary forcing
                data at the output timestep for the ocean model and corrector.
            use_activation_checkpointing: If True, wrap the module call with
                torch.utils.checkpoint.checkpoint, reducing memory consumption
                in exchange for increased computation. This is only relevant during
                training and otherwise has no effect.

        Returns:
            The denormalized output data at the next time step.
        """
        return self._step_obj.step(
            input, next_step_forcing_data, use_activation_checkpointing
        )

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

    def _predict_generator(
        self,
        ic_dict: TensorMapping,
        forcing_dict: TensorMapping,
        n_forward_steps: int,
        optimizer: OptimizationABC,
    ) -> Generator[TensorDict, None, None]:
        state = {k: ic_dict[k].squeeze(self.TIME_DIM) for k in ic_dict}
        ml_forcing_names = self._config.forcing_names
        for step in range(n_forward_steps):
            ml_input_forcing = {
                k: (
                    forcing_dict[k][:, step]
                    if k not in self._config.next_step_forcing_names
                    else forcing_dict[k][:, step + 1]
                )
                for k in ml_forcing_names
            }
            next_step_forcing_dict = {
                k: forcing_dict[k][:, step + 1] for k in self._forcing_names()
            }
            input_data = {**state, **ml_input_forcing}
            use_activation_checkpointing = (
                step >= self._activation_checkpointing.after_n_forward_steps
            )
            state = self.step(
                input_data,
                next_step_forcing_dict,
                use_activation_checkpointing,
            )
            if self._multi_call is not None:
                multi_called_outputs = self._multi_call.step(
                    input_data, next_step_forcing_dict, use_activation_checkpointing
                )
                state = {**multi_called_outputs, **state}
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
        with timer.context("forward_prediction"):
            forcing_data = forcing.subset_names(self._forcing_names())
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

    def _forcing_names(self) -> List[str]:
        return self._step_obj.forcing_names

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

        regularizer_loss = self._step_obj.get_regularizer_loss()
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
            gen_norm_step = self.loss_normalizer.normalize(gen_step)
            target_norm_step = self.loss_normalizer.normalize(target_step)

            if use_crps:
                step_loss: torch.Tensor = crps_loss(
                    gen_norm_step, target_norm_step, names=self.out_names
                )
            else:
                step_loss = self.loss_obj(gen_norm_step, target_norm_step)
            metrics[f"loss_step_{step}"] = step_loss.detach()

            if self._multi_call is not None:
                if use_crps:
                    mc_loss = crps_loss(
                        gen_norm_step, target_norm_step, names=self._multi_call.names
                    )
                else:
                    mc_loss = self._multi_call_loss(gen_norm_step, target_norm_step)
                step_loss = step_loss + mc_loss
                metrics[f"loss_multi_call_step_{step}"] = mc_loss.detach()

            optimization.accumulate_loss(step_loss)
        return output_list

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        step_config = self._step_obj.get_state()
        step_config.pop("config")
        assert "loss_normalizer" not in step_config
        return {
            "config": self._config.get_state(),
            "loss_normalizer": self.loss_normalizer.get_state(),
            **step_config,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load the state of the stepper.

        Args:
            state: The state to load.
        """
        self._step_obj.load_state(state)

    @classmethod
    def from_state(cls, state) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        step = SingleModuleStep.from_state(state)
        config_data = {**state["config"]}  # make a copy to avoid mutating input

        # for backwards compatibility with previous steppers created w/o
        # loss_normalization or residual_normalization
        loss_normalizer_state = state.get(
            "loss_normalizer", state.get("normalizer", None)
        )
        if loss_normalizer_state is None:
            raise ValueError(
                f"No loss normalizer state found, keys include {state.keys()}"
            )
        config_data["loss_normalization"] = loss_normalizer_state

        # Overwrite the residual_normalization key if it exists, since the combined
        # loss scalings are saved in initial training as the loss_normalization
        config_data["residual_normalization"] = None
        config = SingleModuleStepperConfig.from_state(config_data)

        encoded_timestep = state.get("encoded_timestep", DEFAULT_ENCODED_TIMESTEP)
        timestep = decode_timestep(encoded_timestep)
        derive_func = step.vertical_coordinate.build_derive_function(timestep)
        stepper = cls(
            config=config,
            step=step,
            derive_func=derive_func,
        )
        stepper.load_state(state)
        return stepper


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
) -> SingleModuleStepperConfig:
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
    config = SingleModuleStepperConfig.from_state(checkpoint["stepper"]["config"])

    if override_config.ocean != "keep":
        logging.info(
            "Overriding training ocean configuration with a new ocean configuration."
        )
        config.ocean = override_config.ocean
    if override_config.multi_call != "keep":
        logging.info(
            "Overriding training multi_call configuration with a new "
            "multi_call configuration."
        )
        config.multi_call = override_config.multi_call
    return config


def load_stepper(
    checkpoint_path: str | pathlib.Path,
    override_config: Optional[StepperOverrideConfig] = None,
) -> SingleModuleStepper:
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
    stepper = SingleModuleStepper.from_state(checkpoint["stepper"])

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
