import dataclasses
import datetime
import logging
from copy import copy
from typing import Any, Dict, List, Mapping, Optional, Protocol, Tuple, Union

import dacite
import torch
from torch import nn

from fme.core.corrector import CorrectorConfig
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.requirements import DataRequirements
from fme.core.data_loading.utils import BatchData, decode_timestep, encode_timestep
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations, LatLonOperations
from fme.core.loss import WeightedMappingLossConfig
from fme.core.normalizer import (
    FromStateNormalizer,
    NormalizationConfig,
    StandardNormalizer,
)
from fme.core.ocean import Ocean, OceanConfig
from fme.core.packer import Packer
from fme.core.registry import ModuleSelector

from .optimization import NullOptimization, Optimization
from .parameter_init import ParameterInitializationConfig
from .typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


@dataclasses.dataclass
class SingleModuleStepperConfig:
    """
    Configuration for a single module stepper.

    Attributes:
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
    """

    builder: ModuleSelector
    in_names: List[str]
    out_names: List[str]
    normalization: Union[NormalizationConfig, FromStateNormalizer]
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )
    ocean: Optional[OceanConfig] = None
    loss: WeightedMappingLossConfig = dataclasses.field(
        default_factory=lambda: WeightedMappingLossConfig()
    )
    corrector: CorrectorConfig = dataclasses.field(
        default_factory=lambda: CorrectorConfig()
    )
    next_step_forcing_names: List[str] = dataclasses.field(default_factory=list)
    loss_normalization: Optional[Union[NormalizationConfig, FromStateNormalizer]] = None
    residual_normalization: Optional[
        Union[NormalizationConfig, FromStateNormalizer]
    ] = None

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

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self.all_names,
            n_timesteps=n_forward_steps + 1,
        )

    def get_forcing_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        if self.ocean is None:
            names = self.forcing_names
        else:
            names = list(set(self.forcing_names).union(self.ocean.forcing_names))

        return DataRequirements(names=names, n_timesteps=n_forward_steps + 1)

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
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ):
        logging.info("Initializing stepper from provided config")
        return SingleModuleStepper(
            config=self,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
            timestep=timestep,
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
        all_names = list(set(self.in_names).union(self.out_names).union(extra_names))
        return all_names

    @property
    def normalize_names(self):
        """Names of variables which require normalization. I.e. inputs/outputs."""
        return list(set(self.in_names).union(self.out_names))

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
        """Names of variables which both inputs and outputs."""
        return list(set(self.out_names).difference(self.in_names))

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

    Attributes:
        checkpoint_path: The path to the serialized checkpoint.
    """

    checkpoint_path: str

    def _load_checkpoint(self) -> Mapping[str, Any]:
        return torch.load(self.checkpoint_path, map_location=get_device())

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return SingleModuleStepperConfig.from_state(
            self._load_checkpoint()["stepper"]["config"]
        ).get_data_requirements(n_forward_steps)

    def get_base_weights(self) -> Optional[List[Mapping[str, Any]]]:
        return SingleModuleStepperConfig.from_state(
            self._load_checkpoint()["stepper"]["config"]
        ).get_base_weights()

    def get_stepper(self, img_shape, gridded_operations, sigma_coordinates, timestep):
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
    return StandardNormalizer(means=means, stds=stds)


def _prepend_timesteps(
    data: TensorDict, timesteps: TensorDict, time_dim: int = 1
) -> TensorDict:
    return {k: torch.cat([timesteps[k], v], dim=time_dim) for k, v in data.items()}


@dataclasses.dataclass
class SteppedData:
    metrics: TensorDict
    gen_data: TensorDict
    target_data: TensorDict
    gen_data_norm: TensorDict
    target_data_norm: TensorDict

    def remove_initial_condition(self, n_ic_timesteps: int) -> "SteppedData":
        return SteppedData(
            metrics=self.metrics,
            gen_data={k: v[:, n_ic_timesteps:] for k, v in self.gen_data.items()},
            target_data={k: v[:, n_ic_timesteps:] for k, v in self.target_data.items()},
            gen_data_norm={
                k: v[:, n_ic_timesteps:] for k, v in self.gen_data_norm.items()
            },
            target_data_norm={
                k: v[:, n_ic_timesteps:] for k, v in self.target_data_norm.items()
            },
        )

    def copy(self) -> "SteppedData":
        """Creates new dictionaries for the data but with the same tensors."""
        return SteppedData(
            metrics=self.metrics,
            gen_data={k: v for k, v in self.gen_data.items()},
            target_data={k: v for k, v in self.target_data.items()},
            gen_data_norm={k: v for k, v in self.gen_data_norm.items()},
            target_data_norm={k: v for k, v in self.target_data_norm.items()},
        )

    def prepend_initial_condition(
        self,
        initial_condition: TensorDict,
        normalized_initial_condition: TensorDict,
        target_initial_condition: Optional[TensorDict] = None,
        normalized_target_initial_condition: Optional[TensorDict] = None,
    ) -> "SteppedData":
        """
        Prepends an initial condition to the existing stepped data.
        Assumes data are on the same device.
        For data windows > 0, the target IC is different from the generated IC
            and may be provided for correct calculation of tendencies.
        """
        return SteppedData(
            metrics=self.metrics,
            gen_data=_prepend_timesteps(self.gen_data, initial_condition),
            target_data=_prepend_timesteps(
                self.target_data, target_initial_condition or initial_condition
            ),
            gen_data_norm=_prepend_timesteps(
                self.gen_data_norm, normalized_initial_condition
            ),
            target_data_norm=_prepend_timesteps(
                self.target_data_norm,
                normalized_target_initial_condition or normalized_initial_condition,
            ),
        )


class HasDeviceData(Protocol):
    @property
    def device_data(self) -> TensorMapping:
        ...


class SingleModuleStepper:
    """
    Stepper class for a single pytorch module.
    """

    TIME_DIM = 1
    CHANNEL_DIM = -3

    def __init__(
        self,
        config: SingleModuleStepperConfig,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
        init_weights: bool = True,
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            area: (n_lat, n_lon) array containing relative gridcell area,
                in any units including unitless.
            sigma_coordinates: The sigma coordinates.
            timestep: Timestep of the model.
            init_weights: Whether to initialize the weights. Should pass False if
                the weights are about to be overwritten by a checkpoint.
        """
        self._gridded_operations = gridded_operations  # stored for serializing
        n_in_channels = len(config.in_names)
        n_out_channels = len(config.out_names)
        self.in_packer = Packer(config.in_names)
        self.out_packer = Packer(config.out_names)
        self.normalizer = config.normalization.build(config.normalize_names)
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
        self._is_distributed = dist.is_distributed()
        self.module = dist.wrap_module(self.module)

        self.sigma_coordinates = sigma_coordinates.to(get_device())
        self.timestep = timestep

        self.loss_obj = config.loss.build(
            gridded_operations.area_weighted_mean, config.out_names, self.CHANNEL_DIM
        )

        self._corrector = config.corrector.build(
            gridded_operations=gridded_operations,
            sigma_coordinates=self.sigma_coordinates,
            timestep=timestep,
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
                model_normalizer=self.normalizer,
            )
        else:
            self.loss_normalizer = self.normalizer
        self.in_names = config.in_names
        self.out_names = config.out_names

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

    def get_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return self._config.get_data_requirements(n_forward_steps)

    @property
    def effective_loss_scaling(self) -> TensorMapping:
        """
        Effective loss scalings used to normalize outputs before computing loss.
        y_loss_normalized_i = (y_i - y_mean_i) / loss_scaling_i
        where loss_scaling_i = loss_normalizer_std_i / weight_i
        """
        custom_weights = self._config.loss.weights
        loss_normalizer_stds = self.loss_normalizer.stds
        return {
            k: loss_normalizer_stds[k] / custom_weights.get(k, 1.0)
            for k in self._config.out_names
        }

    def replace_ocean(self, ocean: Ocean):
        """
        Replace the ocean model with a new one.
        Args:
            ocean: The new ocean model.
        """
        self.ocean = ocean

    @property
    def prognostic_names(self) -> List[str]:
        return sorted(
            list(set(self.out_packer.names).intersection(self.in_packer.names))
        )

    @property
    def diagnostic_names(self) -> List[str]:
        return sorted(list(set(self.out_packer.names).difference(self.in_packer.names)))

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    @property
    def n_output_timesteps(self) -> int:
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
        ocean_data: TensorMapping,
    ) -> TensorDict:
        """
        Step the model forward one timestep given input data.

        Args:
            input: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This data is used as input for `self.module`
                and is assumed to contain all input variables and be denormalized.
            ocean_data: Mapping from variable name to tensor of shape
                [n_batch, n_lat, n_lon]. This must contain the necessary data at the
                output timestep for the ocean model (e.g. surface temperature,
                mixed-layer depth etc.).

        Returns:
            The denormalized output data at the next time step.
        """
        input_norm = self.normalizer.normalize(input)
        input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        output_tensor = self.module(input_tensor)
        output_norm = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
        output = self.normalizer.denormalize(output_norm)
        if self._corrector is not None:
            output = self._corrector(input, output)
        if self.ocean is not None:
            output = self.ocean(ocean_data, input, output)
        return output

    def predict(
        self,
        initial_condition: TensorMapping,
        forcing_data: TensorMapping,
        n_forward_steps: int,
    ) -> TensorDict:
        """
        Predict multiple steps forward given initial condition and forcing data.

        Args:
            initial_condition: Mapping from variable name to tensors of shape
                [n_batch, self.n_ic_steps, <horizontal_dims>]. This data is assumed
                to contain all prognostic variables and be denormalized.
            forcing_data: Mapping from variable name to tensors of shape
                [n_batch, n_forward_steps + 1, n_lat, n_lon]. This contains the forcing
                and ocean data for the initial condition and all subsequent timesteps.
            n_forward_steps: The number of timesteps to run the model forward for.

        Returns:
            The denormalized output data for all the forward timesteps. Shape of
            each tensor will be [n_batch, n_forward_steps, n_lat, n_lon].
        """
        output_list = []
        state = {
            k: initial_condition[k].squeeze(self.TIME_DIM) for k in initial_condition
        }
        forcing_names = self._config.forcing_names
        ocean_forcing_names = self.ocean.forcing_names if self.ocean is not None else []
        for step in range(n_forward_steps):
            current_step_forcing = {
                k: (
                    forcing_data[k][:, step]
                    if k not in self._config.next_step_forcing_names
                    else forcing_data[k][:, step + 1]
                )
                for k in forcing_names
            }
            next_step_ocean_data = {
                k: forcing_data[k][:, step + 1] for k in ocean_forcing_names
            }
            input_data = {**state, **current_step_forcing}
            state = self.step(input_data, next_step_ocean_data)
            output_list.append(state)
        output_timeseries = {}
        for name in state:
            output_timeseries[name] = torch.stack(
                [x[name] for x in output_list], dim=self.TIME_DIM
            )
        return output_timeseries

    def get_initial_condition(self, data: BatchData) -> Tuple[TensorDict, TensorDict]:
        if self.TIME_DIM != 1:
            raise NotImplementedError(
                "get_initial_condition hard-codes time dimension at index 1"
            )
        ic = {k: v[:, : self.n_ic_timesteps, ...] for k, v in data.device_data.items()}
        return ic, self.normalizer.normalize(ic)

    def run_on_batch(
        self,
        data: HasDeviceData,
        optimization: Union[Optimization, NullOptimization],
        n_forward_steps: int = 1,
    ) -> SteppedData:
        """
        Step the model forward multiple steps on a batch of data.

        Args:
            data: The batch data where each tensor in data.device_data has shape
                [n_sample, n_forward_steps + self.n_ic_timesteps, <horizontal_dims>].
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            n_forward_steps: The number of timesteps to run the model for.

        Returns:
            The loss metrics, the generated data, the normalized generated data,
                and the normalized batch data.
        """
        data_ = data.device_data
        time_dim = self.TIME_DIM
        n_ic_timesteps = self.n_ic_timesteps
        if self.ocean is None:
            forcing_names = self._config.forcing_names
        else:
            forcing_names = self._config.forcing_names + self.ocean.forcing_names
        forcing_data = {k: data_[k] for k in forcing_names}

        loss = torch.tensor(0.0, device=get_device())
        metrics = {}

        input_data = {
            k: data_[k][:, :n_ic_timesteps] for k in self._config.prognostic_names
        }
        # Remove the initial condition from target data
        data_ = {k: data_[k][:, n_ic_timesteps:] for k in data_}

        optimization.set_mode(self.module)
        with optimization.autocast():
            # output from self.predict does not include initial condition
            gen_data = self.predict(input_data, forcing_data, n_forward_steps)

            # compute loss for each timestep
            for step in range(n_forward_steps):
                # Note: here we examine the loss for a single timestep,
                # not a single model call (which may contain multiple timesteps).
                gen_step = {k: v.select(time_dim, step) for k, v in gen_data.items()}
                target_step = {k: v.select(time_dim, step) for k, v in data_.items()}
                gen_norm_step = self.loss_normalizer.normalize(gen_step)
                target_norm_step = self.loss_normalizer.normalize(target_step)

                step_loss = self.loss_obj(gen_norm_step, target_norm_step)
                loss += step_loss
                metrics[f"loss_step_{step}"] = step_loss.detach()

        loss += self._l2_sp_tuning_regularizer()

        metrics["loss"] = loss.detach()
        optimization.step_weights(loss)

        gen_data_norm = self.normalizer.normalize(gen_data)
        full_data_norm = self.normalizer.normalize(data_)

        return SteppedData(
            metrics=metrics,
            gen_data=gen_data,
            target_data=data_,
            gen_data_norm=gen_data_norm,
            target_data_norm=full_data_norm,
        )

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
            "sigma_coordinates": self.sigma_coordinates.as_dict(),
            "encoded_timestep": encode_timestep(self.timestep),
            "loss_normalizer": self.loss_normalizer.get_state(),
        }

    def load_state(self, state):
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
    def from_state(cls, state) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        config = {**state["config"]}  # make a copy to avoid mutating input
        config["normalization"] = FromStateNormalizer(state["normalizer"])

        # for backwards compatibility with previous steppers created w/o
        # loss_normalization or residual_normalization
        loss_normalizer_state = state.get("loss_normalizer", state["normalizer"])
        config["loss_normalization"] = FromStateNormalizer(loss_normalizer_state)
        # Overwrite the residual_normalization key if it exists, since the combined
        # loss scalings are saved in initial training as the loss_normalization
        config["residual_normalization"] = None

        if "area" in state:
            # backwards-compatibility, these older checkpoints are always lat-lon
            gridded_operations: GriddedOperations = LatLonOperations(state["area"])
        else:
            gridded_operations = GriddedOperations.from_state(
                state["gridded_operations"]
            )

        sigma_coordinates = dacite.from_dict(
            data_class=SigmaCoordinates,
            data=state["sigma_coordinates"],
            config=dacite.Config(strict=True),
        )
        # for backwards compatibility with original ACE checkpoint which
        # serialized vertical coordinates as float64
        if sigma_coordinates.ak.dtype == torch.float64:
            sigma_coordinates.ak = sigma_coordinates.ak.to(dtype=torch.float32)
        if sigma_coordinates.bk.dtype == torch.float64:
            sigma_coordinates.bk = sigma_coordinates.bk.to(dtype=torch.float32)
        encoded_timestep = state.get("encoded_timestep", DEFAULT_ENCODED_TIMESTEP)
        timestep = decode_timestep(encoded_timestep)
        if "img_shape" in state:
            img_shape = state["img_shape"]
        else:
            # this is for backwards compatibility with old checkpoints
            for v in state["data_shapes"].values():
                img_shape = v[-2:]
                break
        stepper = cls(
            config=SingleModuleStepperConfig.from_state(config),
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
            timestep=timestep,
            # don't need to initialize weights, we're about to load_state
            init_weights=False,
        )
        stepper.load_state(state)
        return stepper
