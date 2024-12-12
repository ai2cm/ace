import dataclasses
import datetime
import logging
from copy import copy
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import dacite
import torch
import xarray as xr
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.inference.derived_variables import compute_derived_quantities
from fme.ace.requirements import PrognosticStateDataRequirements
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.corrector.corrector import CorrectorConfig
from fme.core.dataset.requirements import DataRequirements
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
from fme.core.optimization import NullOptimization
from fme.core.packer import Packer
from fme.core.parameter_init import ParameterInitializationConfig
from fme.core.registry import CorrectorSelector, ModuleSelector
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping

DEFAULT_TIMESTEP = datetime.timedelta(hours=6)
DEFAULT_ENCODED_TIMESTEP = encode_timestep(DEFAULT_TIMESTEP)


class AtmosphericDeriveFn:
    def __init__(
        self,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        timestep: datetime.timedelta,
    ):
        self.vertical_coordinate = vertical_coordinate.to(
            "cpu"
        )  # must be on cpu for multiprocessing fork context
        self.timestep = timestep

    def __call__(self, data: TensorMapping, forcing_data: TensorMapping) -> TensorDict:
        return compute_derived_quantities(
            dict(data),
            vertical_coordinate=self.vertical_coordinate.to(get_device()),
            timestep=self.timestep,
            forcing_data=dict(forcing_data),
        )


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
        vertical_coordinate: HybridSigmaPressureCoordinate,
        timestep: datetime.timedelta,
    ):
        logging.info("Initializing stepper from provided config")
        derive_func = AtmosphericDeriveFn(vertical_coordinate, timestep)
        return SingleModuleStepper(
            config=self,
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
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
        return torch.load(self.checkpoint_path, map_location=get_device())

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

    def get_stepper(self, img_shape, gridded_operations, vertical_coordinate, timestep):
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
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        timestep: datetime.timedelta,
        init_weights: bool = True,
    ):
        """
        Args:
            config: The configuration.
            img_shape: Shape of domain as (n_lat, n_lon).
            gridded_operations: The gridded operations, e.g. for area weighting.
            vertical_coordinate: The vertical coordinate.
            derive_func: Function to compute derived variables.
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
        self.derive_func = derive_func
        self._img_shape = img_shape
        self._config = config
        self._no_optimization = NullOptimization()

        dist = Distributed.get_instance()
        self._is_distributed = dist.is_distributed()
        self.module = dist.wrap_module(self.module)

        self._vertical_coordinates = vertical_coordinate.to(get_device())
        self._timestep = timestep

        self.loss_obj = config.loss.build(
            gridded_operations.area_weighted_mean, config.out_names, self.CHANNEL_DIM
        )

        self._corrector = config.corrector.build(
            gridded_operations=gridded_operations,
            vertical_coordinate=self.vertical_coordinate,
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
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
        return self._vertical_coordinates

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

    def replace_ocean(self, ocean: Ocean):
        """
        Replace the ocean model with a new one.

        Args:
            ocean: The new ocean model.
        """
        self.ocean = ocean

    @property
    def forcing_names(self) -> List[str]:
        """Names of variables which are inputs only."""
        return self._config.forcing_names

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

        Returns:
            The denormalized output data at the next time step.
        """
        input_norm = self.normalizer.normalize(input)
        input_tensor = self.in_packer.pack(input_norm, axis=self.CHANNEL_DIM)
        output_tensor = self.module(input_tensor)
        output_norm = self.out_packer.unpack(output_tensor, axis=self.CHANNEL_DIM)
        output = self.normalizer.denormalize(output_norm)
        if self._corrector is not None:
            output = self._corrector(input, output, next_step_forcing_data)
        if self.ocean is not None:
            output = self.ocean(input, output, next_step_forcing_data)
        return output

    def _predict(
        self,
        initial_condition: TensorMapping,
        forcing_data: TensorMapping,
        n_forward_steps: int,
    ) -> TensorDict:
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

        Returns:
            The output data at each timestep.
        """
        state = {
            k: initial_condition[k].squeeze(self.TIME_DIM) for k in initial_condition
        }
        ml_forcing_names = self._config.forcing_names
        output_list = []
        for step in range(n_forward_steps):
            ml_input_forcing = {
                k: (
                    forcing_data[k][:, step]
                    if k not in self._config.next_step_forcing_names
                    else forcing_data[k][:, step + 1]
                )
                for k in ml_forcing_names
            }
            next_step_forcing_data = {
                k: forcing_data[k][:, step + 1] for k in self._forcing_names()
            }
            input_data = {**state, **ml_input_forcing}
            state = self.step(input_data, next_step_forcing_data)
            output_list.append(state)
        output_timeseries = {}
        for name in state:
            output_timeseries[name] = torch.stack(
                [x[name] for x in output_list], dim=self.TIME_DIM
            )
        return output_timeseries

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
            initial_condition_state = initial_condition.as_batch_data()
            if initial_condition_state.time.shape[1] != self.n_ic_timesteps:
                raise ValueError(
                    f"Initial condition must have {self.n_ic_timesteps} timesteps, got "
                    f"{initial_condition_state.time.shape[1]}."
                )
            n_forward_steps = forcing_data.time.shape[1] - self.n_ic_timesteps
            output_timeseries = self._predict(
                initial_condition_state.data, forcing_data.data, n_forward_steps
            )
            data = BatchData.new_on_device(
                output_timeseries,
                forcing_data.time[:, self.n_ic_timesteps :],
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
        return data, data.get_end(self.prognostic_names, self.n_ic_timesteps)

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
            A paired data containing the prediction paired with all forcing data at the
            same timesteps and the prediction's final state which can be used as a
            new initial condition.
        """
        prediction, new_initial_condition = self.predict(
            initial_condition, forcing, compute_derived_variables
        )
        return (
            PairedData.from_batch_data(
                prediction=prediction,
                target=self.get_forward_data(
                    forcing, compute_derived_variables=compute_derived_variables
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
        if self.ocean is None:
            return self._config.forcing_names
        return list(set(self._config.forcing_names).union(self.ocean.forcing_names))

    def train_on_batch(
        self,
        data: BatchData,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> TrainOutput:
        """
        Step the model forward multiple steps on a batch of data.

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
        time_dim = self.TIME_DIM

        loss = torch.tensor(0.0, device=get_device())
        metrics: Dict[str, float] = {}
        input_data = data.get_start(self.prognostic_names, self.n_ic_timesteps)

        optimization.set_mode(self.module)
        with optimization.autocast():
            # output from self.predict does not include initial condition
            output, _ = self.predict_paired(
                input_data,
                forcing=data,
            )
            gen_data = output.prediction
            target_data = output.target
            n_forward_steps = output.time.shape[1]

            # compute loss for each timestep
            for step in range(n_forward_steps):
                # Note: here we examine the loss for a single timestep,
                # not a single model call (which may contain multiple timesteps).
                gen_step = {k: v.select(time_dim, step) for k, v in gen_data.items()}
                target_step = {
                    k: v.select(time_dim, step) for k, v in target_data.items()
                }
                gen_norm_step = self.loss_normalizer.normalize(gen_step)
                target_norm_step = self.loss_normalizer.normalize(target_step)

                step_loss = self.loss_obj(gen_norm_step, target_norm_step)
                loss += step_loss
                metrics[f"loss_step_{step}"] = step_loss.detach()

        loss += self._l2_sp_tuning_regularizer()

        metrics["loss"] = loss.detach()
        optimization.step_weights(loss)

        stepped = TrainOutput(
            metrics=metrics,
            gen_data=dict(gen_data),
            target_data=dict(target_data),
            time=output.time,
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
            "vertical_coordinate": self.vertical_coordinate.as_dict(),
            "encoded_timestep": encode_timestep(self.timestep),
            "loss_normalizer": self.loss_normalizer.get_state(),
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
    def from_state(cls, state) -> "SingleModuleStepper":
        """
        Load the state of the stepper.

        Args:
            state: The state to load.

        Returns:
            The stepper.
        """
        config = {**state["config"]}  # make a copy to avoid mutating input
        config["normalization"] = state["normalizer"]

        # for backwards compatibility with previous steppers created w/o
        # loss_normalization or residual_normalization
        loss_normalizer_state = state.get("loss_normalizer", state["normalizer"])
        config["loss_normalization"] = loss_normalizer_state

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

        if "sigma_coordinates" in state:
            # for backwards compatibility with old checkpoints
            state["vertical_coordinate"] = state["sigma_coordinates"]

        vertical_coordinate = dacite.from_dict(
            data_class=HybridSigmaPressureCoordinate,
            data=state["vertical_coordinate"],
            config=dacite.Config(strict=True),
        )
        # for backwards compatibility with original ACE checkpoint which
        # serialized vertical coordinates as float64
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
        derive_func = AtmosphericDeriveFn(vertical_coordinate, timestep)
        stepper = cls(
            config=SingleModuleStepperConfig.from_state(config),
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
            derive_func=derive_func,
            # don't need to initialize weights, we're about to load_state
            init_weights=False,
        )
        stepper.load_state(state)
        return stepper
