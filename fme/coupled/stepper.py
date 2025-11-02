import dataclasses
import datetime
import logging
import pathlib
from collections.abc import Callable, Generator, Iterable
from typing import Any, Literal

import dacite
import numpy as np
import pandas as pd
import torch
from torch import nn

import fme
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.requirements import DataRequirements
from fme.ace.stepper import (
    Stepper,
    TrainOutput,
    process_prediction_generator_list,
    stack_list_of_tensor_dicts,
)
from fme.ace.stepper.parameter_init import (
    StepperWeightsAndHistory,
    Weights,
    WeightsAndHistoryLoader,
)
from fme.ace.stepper.single_module import StepperConfig
from fme.ace.stepper.single_module import (
    load_weights_and_history as load_uncoupled_weights_and_history,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.ocean import OceanConfig
from fme.core.ocean_data import OceanData
from fme.core.optimization import NullOptimization
from fme.core.tensors import add_ensemble_dim
from fme.core.timing import GlobalTimer
from fme.core.training_history import TrainingHistory, TrainingJob
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.loss import LossContributionsConfig, StepLossABC, StepPredictionABC
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledPrognosticStateDataRequirements,
)


@dataclasses.dataclass
class ComponentConfig:
    """
    Configuration for one of the components (ocean or atmosphere) within a
    CoupledStepper.

    Parameters:
        timedelta: An ISO 8601 Duration string specifying the size of this component's
            stepper step.
        stepper: The single module stepper configuration for this component.
        loss_contributions: The loss contributions configuration for this component.
    """

    timedelta: str
    stepper: StepperConfig
    loss_contributions: LossContributionsConfig = dataclasses.field(
        default_factory=lambda: LossContributionsConfig()
    )


@dataclasses.dataclass
class CoupledOceanFractionConfig:
    """
    Configuration for computing ocean fraction from the ocean-predicted sea ice
    fraction.

    Parameters:
        sea_ice_fraction_name: Name of the sea ice fraction field in the ocean
            data. Must be an ocean prognostic variable. If the atmosphere uses
            the same name as an ML forcing then the generated sea ice fraction
            is also passed as an input to the atmosphere.
        land_fraction_name: Name of the land fraction field in the atmosphere
            data. If needed, will be passed to the ocean stepper as a forcing.
        sea_ice_fraction_name_in_atmosphere: Name of the sea ice fraction field in
            the atmosphere data, if required and different from sea_ice_fraction_name.

    """

    sea_ice_fraction_name: str
    land_fraction_name: str
    sea_ice_fraction_name_in_atmosphere: str | None = None

    def validate_ocean_prognostic_names(self, prognostic_names: Iterable[str]):
        if self.sea_ice_fraction_name not in prognostic_names:
            raise ValueError(
                f"CoupledOceanFractionConfig expected {self.sea_ice_fraction_name} "
                "to be a prognostic variable of the ocean model, but it is not."
            )

    def validate_atmosphere_forcing_names(self, forcing_names: Iterable[str]):
        if self.land_fraction_name not in forcing_names:
            raise ValueError(
                f"CoupledOceanFractionConfig expected {self.land_fraction_name} "
                "to be an ML forcing of the atmosphere model, but it is not."
            )

    def filter_atmosphere_forcing_names(
        self,
        unfiltered_names: Iterable[str],
        ocean_fraction_name: str,
    ) -> list[str]:
        """Remove ocean fraction and sea ice fraction from atmosphere forcing names.

        When ocean fraction is predicted from ocean model outputs, these
        variables should not be loaded from atmosphere data since they will be
        computed at runtime.

        Args:
            unfiltered_names: The full list of atmosphere forcing names to filter.
            ocean_fraction_name: Name of the ocean fraction field in atmosphere data.

        Returns:
            Filtered list of atmosphere forcing names.

        """
        filtered = [name for name in unfiltered_names if name != ocean_fraction_name]

        sea_ice_fraction_name = (
            self.sea_ice_fraction_name_in_atmosphere or self.sea_ice_fraction_name
        )
        filtered = [name for name in filtered if name != sea_ice_fraction_name]

        return filtered

    def build_ocean_data(
        self, forcings_from_ocean: TensorMapping, atmos_forcing_data: TensorMapping
    ) -> OceanData:
        # compute ocean frac from land frac and ocean-predicted sea ice frac
        land_frac_name = self.land_fraction_name
        sea_ice_frac_name = self.sea_ice_fraction_name
        # fill nans with 0s
        sea_ice_frac = torch.nan_to_num(forcings_from_ocean[sea_ice_frac_name])
        land_frac = atmos_forcing_data[land_frac_name]
        return OceanData({land_frac_name: land_frac, sea_ice_frac_name: sea_ice_frac})


def _load_stepper_weights_and_history_factory(
    stepper: Stepper,
) -> WeightsAndHistoryLoader:
    def load_stepper_weights_and_history(*_) -> StepperWeightsAndHistory:
        return_weights: Weights = []
        for module in stepper.modules:
            return_weights.append(module.state_dict())
        return return_weights, stepper.training_history

    return load_stepper_weights_and_history


@dataclasses.dataclass
class CoupledParameterInitConfig:
    """
    Enables component parameter initialization via a coupled stepper checkpoint.

    The default behavior when checkpoint_path is None is to rely on the
    component steppers' parameter initialization config to load uncoupled
    component checkpoints, if provided. When checkpoint_path is non-None, the
    component steppers' ParameterInitializationConfig.weights_path must be None.

    Parameters:
        checkpoint_path: Path to a coupled stepper checkpoint.
    """

    checkpoint_path: str | None = None

    def build_weights_and_history_loaders(self) -> "CoupledWeightsAndHistoryLoaders":
        if self.checkpoint_path is None:
            return CoupledWeightsAndHistoryLoaders(
                ocean=load_uncoupled_weights_and_history,
                atmosphere=load_uncoupled_weights_and_history,
            )
        coupled_stepper = load_coupled_stepper(self.checkpoint_path)
        return CoupledWeightsAndHistoryLoaders(
            ocean=_load_stepper_weights_and_history_factory(coupled_stepper.ocean),
            atmosphere=_load_stepper_weights_and_history_factory(
                coupled_stepper.atmosphere
            ),
        )


@dataclasses.dataclass
class CoupledWeightsAndHistoryLoaders:
    ocean: WeightsAndHistoryLoader
    atmosphere: WeightsAndHistoryLoader
    training_history: TrainingHistory | None = None


@dataclasses.dataclass
class CoupledStepperConfig:
    """
    Configuration for a coupled atmosphere-ocean stepper. From a common initial
    condition time the atmosphere steps first and takes as many steps as fit in
    a single ocean step, while being forced by the ocean's initial condition
    SST. The ocean then steps forward once, receiving required forcings from the
    atmosphere-generated output as averages over its step window. This completes
    a single "coupled step". For subsequent coupled steps, the generated SST
    from the ocean forces the atmosphere's steps.

    For example, with an atmosphere:ocean step size ratio of 2:1, the following
    sequence results in 2 coupled steps (4 atmosphere steps and 2 ocean steps):

    1. IC -> atmos_step_1 -> atmos_step_2
    2. (IC, atmos_step_1_and_2_avg) -> ocean_step_1
    3. (atmos_step_2, ocean_step_1) -> atmos_step_3 -> atmos_step_4
    4. (ocean_step_1, atmos_step_3_and_4_avg) -> ocean_step_2

    Parameters:
        ocean: The ocean component configuration. Output variable names must be distinct
            from the atmosphere's output names. Outputs that are input names in the
            atmosphere must be prognostic variables in the ocean.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include OceanConfig so that ocean-generated SSTs
            can be written on the atmosphere's surface temperature field. Output
            variable names must be distinct from the ocean's output names.
        sst_name: Name of the liquid sea surface temperature field in the ocean data.
            Must be present in the ocean's output names.
        ocean_fraction_prediction: (Optional) Configuration for ocean-generated
            ocean fraction to replace the ocean fraction variable specified in the
            atmosphere's OceanConfig. If the atmosphere uses the ocean fraction as
            an ML forcing, the generated ocean fraction is also passed as an input.
        parameter_init: The parameter initialization configuration.

    """

    ocean: ComponentConfig
    atmosphere: ComponentConfig
    sst_name: str = "sst"
    ocean_fraction_prediction: CoupledOceanFractionConfig | None = None
    parameter_init: CoupledParameterInitConfig = dataclasses.field(
        default_factory=lambda: CoupledParameterInitConfig()
    )

    def __post_init__(self):
        self._validate_component_configs()

        atmosphere_ocean_config = self.atmosphere.stepper.get_ocean()
        # this was already checked in _validate_component_configs, so an
        # assertion will do fine here to appease mypy
        assert atmosphere_ocean_config is not None
        self._atmosphere_ocean_config = atmosphere_ocean_config

        # set timesteps
        self._ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
        self._atmosphere_timestep = pd.Timedelta(
            self.atmosphere.timedelta
        ).to_pytimedelta()

        # calculate forcing sets
        self._ocean_forcing_exogenous_names = list(
            set(self.ocean.stepper.input_only_names).difference(
                self.atmosphere.stepper.output_names
            )
        )
        unfiltered_atmosphere_forcing_names = list(
            set(self.atmosphere.stepper.input_only_names).difference(
                self.ocean.stepper.output_names
            )
        )
        if self.ocean_fraction_prediction is not None:
            self._atmosphere_forcing_exogenous_names = (
                self.ocean_fraction_prediction.filter_atmosphere_forcing_names(
                    unfiltered_atmosphere_forcing_names,
                    self._atmosphere_ocean_config.ocean_fraction_name,
                )
            )
        else:
            self._atmosphere_forcing_exogenous_names = (
                unfiltered_atmosphere_forcing_names
            )
        self._shared_forcing_exogenous_names = list(
            set(self._ocean_forcing_exogenous_names).intersection(
                self._atmosphere_forcing_exogenous_names
            )
        )
        self._atmosphere_to_ocean_forcing_names = list(
            set(self.ocean.stepper.input_only_names).intersection(
                self.atmosphere.stepper.output_names
            )
        )
        extra_forcings_names = [self.sst_name]
        if self.ocean_fraction_prediction is not None:
            # NOTE: this is only necessary for the special case where the
            # atmosphere doesn't use the sea ice fraction as an ML forcing
            extra_forcings_names.append(
                self.ocean_fraction_prediction.sea_ice_fraction_name
            )

        self._ocean_to_atmosphere_forcing_names = list(
            set(self.atmosphere.stepper.input_only_names)
            .intersection(self.ocean.stepper.output_names)
            .union(extra_forcings_names)
        )

        # calculate names for each component's data requirements
        unfiltered_all_atmosphere_names = list(
            set(self.atmosphere.stepper.all_names).difference(
                self.ocean.stepper.output_names
            )
        )
        if self.ocean_fraction_prediction is not None:
            self._all_atmosphere_names = (
                self.ocean_fraction_prediction.filter_atmosphere_forcing_names(
                    unfiltered_all_atmosphere_names,
                    self.ocean_fraction_name,
                )
            )
        else:
            self._all_atmosphere_names = unfiltered_all_atmosphere_names
        # NOTE: this removes "shared" forcings from the ocean data requirements
        self._all_ocean_names = list(
            set(self.ocean.stepper.all_names).difference(self._all_atmosphere_names)
        )
        if self.ocean_fraction_prediction is not None:
            # NOTE: land_fraciton is necessary to derive sea_ice_fraction from
            # ocean_sea_ice_fraction
            self._all_ocean_names.append(
                self.ocean_fraction_prediction.land_fraction_name
            )

    @property
    def timestep(self) -> datetime.timedelta:
        # the "coupled timestep" is the same as the ocean's
        return self._ocean_timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        return self._ocean_timestep

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        return self._atmosphere_timestep

    @property
    def n_inner_steps(self) -> int:
        return self.ocean_timestep // self.atmosphere_timestep

    @property
    def atmosphere_ocean_config(self) -> OceanConfig:
        """The OceanConfig defined in the atmosphere StepperConfig."""
        return self._atmosphere_ocean_config

    @property
    def ocean_fraction_name(self) -> str:
        """Name of the ocean fraction field in the atmosphere data."""
        return self.atmosphere_ocean_config.ocean_fraction_name

    @property
    def surface_temperature_name(self) -> str:
        """Name of the surface temperature field in the atmosphere data."""
        return self.atmosphere_ocean_config.surface_temperature_name

    @property
    def ocean_next_step_forcing_names(self) -> list[str]:
        """Ocean next-step forcings."""
        return self.ocean.stepper.next_step_forcing_names

    @property
    def ocean_forcing_exogenous_names(self) -> list[str]:
        """Ocean forcing variables that are not outputs of the atmosphere."""
        return self._ocean_forcing_exogenous_names

    @property
    def atmosphere_forcing_exogenous_names(self) -> list[str]:
        """Atmosphere forcing variables that are not outputs of the ocean."""
        return self._atmosphere_forcing_exogenous_names

    @property
    def shared_forcing_exogenous_names(self) -> list[str]:
        """Exogenous forcing variables shared by both components. Must be
        present in the atmosphere data on disk. If time-varying, the ocean
        receives the atmosphere data forcings averaged over its step window.

        """
        return self._shared_forcing_exogenous_names

    @property
    def atmosphere_to_ocean_forcing_names(self) -> list[str]:
        """Ocean forcing variables that are outputs of the atmosphere."""
        return self._atmosphere_to_ocean_forcing_names

    @property
    def ocean_to_atmosphere_forcing_names(self) -> list[str]:
        """Atmosphere forcing variables that are outputs of the ocean."""
        return self._ocean_to_atmosphere_forcing_names

    def _validate_component_configs(self):
        # validate parameter_init
        if self.parameter_init.checkpoint_path is not None:
            if (
                self.atmosphere.stepper.parameter_init.weights_path is not None
                or self.ocean.stepper.parameter_init.weights_path is not None
            ):
                raise ValueError(
                    "Please specify CoupledParameterInitConfig.checkpoint_path or the "
                    "component Steppers' ParameterInitializationConfig.weights_path, "
                    "but not both."
                )
        # validate atmosphere's OceanConfig
        atmosphere_ocean_config = self.atmosphere.stepper.get_ocean()
        if atmosphere_ocean_config is None:
            raise ValueError(
                "The atmosphere stepper 'ocean' config is missing but must be set for "
                "coupled emulation."
            )
        if atmosphere_ocean_config.slab is not None:
            raise ValueError(
                "The atmosphere stepper 'ocean' config cannot use 'slab' for "
                "coupled emulation."
            )
        # validate compatibility of ocean and atmosphere timestep sizes
        ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
        atmosphere_timestep = pd.Timedelta(self.atmosphere.timedelta).to_pytimedelta()
        if atmosphere_timestep > ocean_timestep:
            raise ValueError("Atmosphere timedelta must not be larger than ocean's.")
        n_inner_steps = ocean_timestep / atmosphere_timestep
        if n_inner_steps != int(n_inner_steps):
            raise ValueError("Ocean timedelta must be a multiple of the atmosphere's.")

        # check for overlapping output names
        duplicate_outputs = set(self.ocean.stepper.output_names).intersection(
            self.atmosphere.stepper.output_names
        )
        if len(duplicate_outputs) > 0:
            raise ValueError(
                "Output variable names of CoupledStepper components cannot "
                f"overlap. Found the following duplicated names: {duplicate_outputs}"
            )

        # ocean diagnostics cannot be used as atmosphere inputs
        ocean_diags_as_atmos_forcings = list(
            set(self.atmosphere.stepper.input_only_names)
            .intersection(self.ocean.stepper.output_names)
            .difference(self.ocean.stepper.input_names)
        )
        if len(ocean_diags_as_atmos_forcings) > 0:
            raise ValueError(
                "CoupledStepper only supports ocean prognostic variables as atmosphere "
                "forcings, but the following ocean diagnostic variables are inputs to "
                f"the atmosphere: {ocean_diags_as_atmos_forcings}."
            )

        # all ocean inputs that are atmosphere outputs must be "next step"
        # forcings according to the ocean stepper config
        atmosphere_to_ocean_forcing_names = list(
            set(self.ocean.stepper.input_only_names).intersection(
                self.atmosphere.stepper.output_names
            )
        )
        missing_next_step_forcings = list(
            set(atmosphere_to_ocean_forcing_names).difference(
                self.ocean.stepper.next_step_forcing_names
            )
        )
        if len(missing_next_step_forcings) > 0:
            raise ValueError(
                "The following variables which are atmosphere component outputs "
                "and ocean component inputs were not found among the ocean's "
                f"next_step_forcing_names: {missing_next_step_forcings}."
            )

        # sst_name must be present in the ocean's output names
        if self.sst_name not in self.ocean.stepper.output_names:
            raise ValueError(
                f"The variable {self.sst_name} is not in the ocean's output "
                "names but is required for coupling with the atmosphere."
            )

        # validate ocean_fraction_prediction
        if self.ocean_fraction_prediction is not None:
            self.ocean_fraction_prediction.validate_ocean_prognostic_names(
                self.ocean.stepper.prognostic_names,
            )
            self.ocean_fraction_prediction.validate_atmosphere_forcing_names(
                self.atmosphere.stepper.input_only_names
            )

    def _get_ocean_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self._all_ocean_names, n_timesteps=n_forward_steps + 1
        )

    def _get_atmosphere_data_requirements(
        self, n_forward_steps: int
    ) -> DataRequirements:
        return DataRequirements(
            names=self._all_atmosphere_names, n_timesteps=n_forward_steps + 1
        )

    def get_evaluation_window_data_requirements(
        self, n_coupled_steps: int
    ) -> CoupledDataRequirements:
        """Get the DataRequirements for the ocean and atmosphere. For every step
        of the CoupledStepper, the atmosphere takes n_inner_steps (determined by
        the number of atmosphere timesteps that fit in a single ocean timestep)
        steps and the ocean takes a single step. Therefore, we need
        n_coupled_steps number of ocean forward steps and n_coupled_steps *
        n_inner_steps number of atmosphere forward steps.

        n_coupled_steps: The number of CoupledStepper forward steps. During
            training, these steps are included when computing gradients.

        """
        return CoupledDataRequirements(
            ocean_timestep=self.ocean_timestep,
            ocean_requirements=self._get_ocean_data_requirements(n_coupled_steps),
            atmosphere_timestep=self.atmosphere_timestep,
            atmosphere_requirements=self._get_atmosphere_data_requirements(
                n_coupled_steps * self.n_inner_steps
            ),
        )

    def get_prognostic_state_data_requirements(
        self,
    ) -> CoupledPrognosticStateDataRequirements:
        """Get the PrognosticStateDataRequirements for the ocean and atmosphere."""
        return CoupledPrognosticStateDataRequirements(
            ocean=self.ocean.stepper.get_prognostic_state_data_requirements(),
            atmosphere=self.atmosphere.stepper.get_prognostic_state_data_requirements(),
        )

    def get_forcing_window_data_requirements(
        self, n_coupled_steps: int
    ) -> CoupledDataRequirements:
        ocean_forcing_names = list(
            set(self.ocean_forcing_exogenous_names).difference(
                self.shared_forcing_exogenous_names
            )
        )
        return CoupledDataRequirements(
            ocean_timestep=self.ocean_timestep,
            ocean_requirements=DataRequirements(
                ocean_forcing_names, n_timesteps=n_coupled_steps + 1
            ),
            atmosphere_timestep=self.atmosphere_timestep,
            atmosphere_requirements=DataRequirements(
                names=self.atmosphere_forcing_exogenous_names,
                n_timesteps=n_coupled_steps * self.n_inner_steps + 1,
            ),
        )

    def _get_ocean_stepper(
        self,
        dataset_info: DatasetInfo,
        load_weights_and_history: WeightsAndHistoryLoader,
    ) -> Stepper:
        if dataset_info.timestep != self.ocean_timestep:
            raise ValueError(
                "Ocean timestep must match the dataset timestep. "
                f"Got {self.ocean_timestep} and {dataset_info.timestep}, respectively."
            )
        return self.ocean.stepper.get_stepper(
            dataset_info=dataset_info,
            apply_parameter_init=True,
            load_weights_and_history=load_weights_and_history,
        )

    def _get_atmosphere_stepper(
        self,
        dataset_info: DatasetInfo,
        load_weights_and_history: WeightsAndHistoryLoader,
    ) -> Stepper:
        if dataset_info.timestep != self.atmosphere_timestep:
            raise ValueError(
                "Atmosphere timestep must match the dataset timestep. "
                f"Got {self.atmosphere_timestep} and {dataset_info.timestep}, "
                "respectively."
            )
        return self.atmosphere.stepper.get_stepper(
            dataset_info=dataset_info,
            apply_parameter_init=True,
            load_weights_and_history=load_weights_and_history,
        )

    def get_stepper(
        self,
        dataset_info: CoupledDatasetInfo,
    ):
        logging.info("Initializing coupler")
        loaders = self.parameter_init.build_weights_and_history_loaders()
        return CoupledStepper(
            config=self,
            ocean=self._get_ocean_stepper(
                dataset_info=dataset_info.ocean,
                load_weights_and_history=loaders.ocean,
            ),
            atmosphere=self._get_atmosphere_stepper(
                dataset_info=dataset_info.atmosphere,
                load_weights_and_history=loaders.atmosphere,
            ),
            dataset_info=dataset_info,
        )

    def get_ocean_loss(
        self,
        loss_obj: Callable[[TensorMapping, TensorMapping, int], torch.Tensor],
        time_dim: int,
    ) -> StepLossABC:
        return self.ocean.loss_contributions.build(loss_obj, time_dim)

    def get_atmosphere_loss(
        self,
        loss_obj: Callable[[TensorMapping, TensorMapping, int], torch.Tensor],
        time_dim: int,
    ) -> StepLossABC:
        return self.atmosphere.loss_contributions.build(loss_obj, time_dim)

    def get_loss(
        self, ocean_loss: StepLossABC, atmosphere_loss: StepLossABC
    ) -> "CoupledStepperTrainLoss":
        return CoupledStepperTrainLoss(ocean_loss, atmosphere_loss)

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "CoupledStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @classmethod
    def remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        if "sst_mask_name" in state_copy:
            del state_copy["sst_mask_name"]
        return state_copy


class ComponentStepMetrics:
    def __init__(self):
        self._ocean: TensorDict = {}
        self._atmos: TensorDict = {}

    def add_metric(self, key, value, realm: Literal["ocean", "atmosphere"]) -> None:
        if realm == "ocean":
            self._ocean[key] = value
        elif realm == "atmosphere":
            self._atmos[key] = value

    def get_ocean_metrics(self) -> TensorDict:
        if not self._ocean:
            return {"loss/ocean": torch.tensor(0.0, device=fme.get_device())}
        loss = sum(self._ocean.values())
        return {
            "loss/ocean": loss,
            **self._ocean,
        }

    def get_atmosphere_metrics(self) -> TensorDict:
        if not self._atmos:
            return {"loss/atmosphere": torch.tensor(0.0, device=fme.get_device())}
        loss = sum(self._atmos.values())
        return {
            "loss/atmosphere": loss,
            **self._atmos,
        }


@dataclasses.dataclass
class CoupledTrainOutput(TrainOutputABC):
    total_metrics: TensorDict
    ocean: TrainOutput
    atmosphere: TrainOutput

    def remove_initial_condition(self, n_ic_timesteps: int) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=self.ocean.remove_initial_condition(n_ic_timesteps),
            atmosphere=self.atmosphere.remove_initial_condition(n_ic_timesteps),
        )

    def copy(self) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            total_metrics=self.total_metrics.copy(),
            ocean=self.ocean.copy(),
            atmosphere=self.atmosphere.copy(),
        )

    def prepend_initial_condition(
        self,
        initial_condition: CoupledPrognosticState,
    ) -> "CoupledTrainOutput":
        """
        Prepends an initial condition to the existing stepped data.
        Assumes data are on the same device.

        Args:
            initial_condition: Initial condition data.
        """
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=self.ocean.prepend_initial_condition(
                initial_condition.ocean_data,
            ),
            atmosphere=self.atmosphere.prepend_initial_condition(
                initial_condition.atmosphere_data,
            ),
        )

    def compute_derived_variables(self) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=self.ocean.compute_derived_variables(),
            atmosphere=self.atmosphere.compute_derived_variables(),
        )

    def get_metrics(self) -> TensorDict:
        ocean_keys = set(self.ocean.metrics.keys())
        atmos_keys = set(self.atmosphere.metrics.keys())
        overlap = ocean_keys.intersection(atmos_keys)
        if len(overlap) > 0:
            raise ValueError(
                "The following metrics have the same name in the atmosphere and ocean: "
                f"{overlap}."
            )
        overlap = ocean_keys.union(atmos_keys).intersection(self.total_metrics.keys())
        if len(overlap) > 0:
            raise ValueError(
                "The following total metric names conflict with ocean or atmosphere "
                f"metric names: {overlap}."
            )
        return {
            **self.total_metrics,
            **self.ocean.metrics,
            **self.atmosphere.metrics,
        }


class ComponentStepPrediction(StepPredictionABC):
    def __init__(
        self,
        realm: Literal["ocean", "atmosphere"],
        data: TensorDict,
        step: int,
    ):
        self._realm: Literal["ocean", "atmosphere"] = realm
        self._data = data
        self._step = step

    @property
    def realm(self) -> Literal["ocean", "atmosphere"]:
        return self._realm

    @property
    def data(self) -> TensorDict:
        return self._data

    @property
    def step(self) -> int:
        return self._step

    def detach(self, optimizer: OptimizationABC) -> "ComponentStepPrediction":
        """Detach the data tensor map from the computation graph."""
        return ComponentStepPrediction(
            realm=self.realm,
            data=optimizer.detach_if_using_gradient_accumulation(self.data),
            step=self.step,
        )


class CoupledStepperTrainLoss:
    def __init__(
        self,
        ocean_loss: StepLossABC,
        atmosphere_loss: StepLossABC,
    ):
        self._loss_objs = {
            "ocean": ocean_loss,
            "atmosphere": atmosphere_loss,
        }

    def __call__(
        self,
        prediction: ComponentStepPrediction,
        target_data: TensorMapping,
    ) -> torch.Tensor | None:
        loss_obj = self._loss_objs[prediction.realm]
        if loss_obj.step_is_optimized(prediction.step):
            return loss_obj(prediction, target_data)
        return None


class CoupledStepper(
    TrainStepperABC[
        CoupledPrognosticState,
        CoupledBatchData,
        CoupledBatchData,
        CoupledPairedData,
        CoupledTrainOutput,
    ],
):
    TIME_DIM = 1

    def __init__(
        self,
        config: CoupledStepperConfig,
        ocean: Stepper,
        atmosphere: Stepper,
        dataset_info: CoupledDatasetInfo,
    ):
        """
        Args:
            config: The configuration.
            ocean: The ocean stepper.
            atmosphere: The atmosphere stepper.
            dataset_info: The CoupledDatasetInfo.
        """
        if ocean.n_ic_timesteps != 1 or atmosphere.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported.")

        self.ocean = ocean
        self.atmosphere = atmosphere
        self._config = config
        self._dataset_info = dataset_info
        self._ocean_mask_provider = dataset_info.ocean_mask_provider

        ocean_loss = self._config.get_ocean_loss(
            self.ocean.loss_obj,
            ocean.TIME_DIM,
        )
        atmos_loss = self._config.get_atmosphere_loss(
            self.atmosphere.loss_obj,
            atmosphere.TIME_DIM,
        )
        self._loss = self._config.get_loss(ocean_loss, atmos_loss)

        _: PredictFunction[  # for type checking
            CoupledPrognosticState,
            CoupledBatchData,
            CoupledPairedData,
        ] = self.predict_paired

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([*self.atmosphere.modules, *self.ocean.modules])

    def set_train(self):
        self.atmosphere.set_train()
        self.ocean.set_train()

    def set_eval(self):
        self.atmosphere.set_eval()
        self.ocean.set_eval()

    def get_state(self):
        """
        Returns:
            The state of the coupled stepper.
        """
        return {
            "config": self._config.get_state(),
            "atmosphere_state": self.atmosphere.get_state(),
            "ocean_state": self.ocean.get_state(),
            "dataset_info": self._dataset_info.to_state(),
        }

    def load_state(self, state: dict[str, Any]):
        self.atmosphere.load_state(state["atmosphere_state"])
        self.ocean.load_state(state["ocean_state"])

    @property
    def training_dataset_info(self) -> CoupledDatasetInfo:
        return self._dataset_info

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    @property
    def n_inner_steps(self) -> int:
        """Number of atmosphere steps per ocean step."""
        return self._config.n_inner_steps

    @property
    def _ocean_forcing_exogenous_names(self) -> list[str]:
        return self._config.ocean_forcing_exogenous_names

    @property
    def _atmosphere_forcing_exogenous_names(self) -> list[str]:
        return self._config.atmosphere_forcing_exogenous_names

    @property
    def _shared_forcing_exogenous_names(self) -> list[str]:
        return self._config.shared_forcing_exogenous_names

    @property
    def _atmosphere_to_ocean_forcing_names(self) -> list[str]:
        return self._config.atmosphere_to_ocean_forcing_names

    @property
    def _ocean_to_atmosphere_forcing_names(self) -> list[str]:
        return self._config.ocean_to_atmosphere_forcing_names

    def _prescribe_ic_sst(
        self,
        atmos_ic_state: PrognosticState,
        forcing_ic_batch: BatchData,
    ) -> PrognosticState:
        """Prescribe the initial condition SST state on the surface_temperature
        initial condition field.

        atmos_ic_state: The atmosphere prognostic state to be updated.
        forcing_ic_state: The corresponding forcing state at the same time,
            which should be output from _get_atmosphere_forcings.

        """
        ts_name = self.atmosphere.surface_temperature_name
        assert np.all(atmos_ic_state.as_batch_data().time == forcing_ic_batch.time)
        atmos_ic_data = atmos_ic_state.as_batch_data().data
        forcing_ic_data = forcing_ic_batch.data
        assert ts_name in atmos_ic_data
        assert ts_name in forcing_ic_data
        assert self.atmosphere.ocean_fraction_name in forcing_ic_data
        atmos_ic_data = self.atmosphere.prescribe_sst(
            mask_data=forcing_ic_data,
            gen_data=atmos_ic_data,
            target_data=forcing_ic_data,
        )
        return PrognosticState(
            BatchData(
                data=atmos_ic_data,
                time=forcing_ic_batch.time,
                labels=forcing_ic_batch.labels,
            )
        )

    def _forcings_from_ocean_with_ocean_fraction(
        self,
        forcings_from_ocean: TensorMapping,
        atmos_forcing_data: TensorMapping,
    ) -> TensorDict:
        """Get the ocean fraction field and return it with the other fields in
        forcings_from_ocean.

        Returns:
            forcings_from_ocean: A copy of the forcings_from_ocean input,
            including the ocean fraction.

        """
        ocean_frac_name = self._config.ocean_fraction_name
        forcings_from_ocean = dict(forcings_from_ocean)
        if self._config.ocean_fraction_prediction is None:
            # for convenience, move the atmos's ocean fraction to the
            # forcings_from_ocean dict
            forcings_from_ocean[ocean_frac_name] = atmos_forcing_data[ocean_frac_name]
        else:
            # compute ocean frac from land frac and ocean-predicted sea ice frac
            ofrac_config = self._config.ocean_fraction_prediction
            ocean_data = ofrac_config.build_ocean_data(
                forcings_from_ocean, atmos_forcing_data
            )
            sea_ice_frac_name = (
                ofrac_config.sea_ice_fraction_name_in_atmosphere
                or ofrac_config.sea_ice_fraction_name
            )
            forcings_from_ocean[sea_ice_frac_name] = ocean_data.sea_ice_fraction
            forcings_from_ocean[ocean_frac_name] = torch.clip(
                ocean_data.ocean_fraction, min=0
            )
        for name, tensor in forcings_from_ocean.items():
            # set ocean invalid points to 0 based on the ocean masking
            mask = self._ocean_mask_provider.get_mask_tensor_for(name)
            if mask is not None:
                mask = mask.expand(tensor.shape)
                forcings_from_ocean[name] = tensor.where(mask != 0, 0)
        return forcings_from_ocean

    def _get_atmosphere_forcings(
        self,
        atmos_data: TensorMapping,
        ocean_ic: TensorMapping,
    ) -> TensorDict:
        """
        Get the forcings for the atmosphere component.

        Args:
            atmos_data: Atmosphere batch data, including initial condition and forward
                steps.
            ocean_ic: Ocean initial condition state, including SST.
            ocean_forcings: Ocean forcing data, including the SST mask.
        """
        time_dim = self.atmosphere.TIME_DIM
        sizes = [-1] * len(next(iter(atmos_data.values())).shape)
        sizes[time_dim] = self.n_inner_steps + 1
        # exogenous forcings are used as is
        forcing_data = {
            k: atmos_data[k] for k in self._atmosphere_forcing_exogenous_names
        }
        # forcings from ocean are constant during the fast atmosphere steps
        # NOTE: only n_ic_timesteps = 1 is currently supported
        assert next(iter(ocean_ic.values())).shape[self.ocean.TIME_DIM] == 1
        forcings_from_ocean = {
            k: ocean_ic[k].expand(*sizes)
            for k in self._ocean_to_atmosphere_forcing_names
        }
        # rename the ocean surface temperature variable using the corresponding
        # name in the atmosphere
        forcings_from_ocean[self._config.surface_temperature_name] = (
            forcings_from_ocean.pop(self._config.sst_name)
        )
        # get the SST mask (0 if land, 1 if sea surface)
        forcings_from_ocean = self._forcings_from_ocean_with_ocean_fraction(
            forcings_from_ocean, forcing_data
        )
        # update atmosphere forcings
        forcing_data.update(forcings_from_ocean)
        return forcing_data

    def _get_ocean_forcings(
        self,
        ocean_data: TensorMapping,
        atmos_gen: TensorMapping,
        atmos_forcings: TensorMapping,
    ) -> TensorDict:
        """
        Get the forcings for the ocean component.

        Args:
            ocean_data: Ocean data, including initial condition and forward
                steps.
            atmos_gen: Generated atmosphere data covering the ocean forward steps.
            atmos_forcings: Atmosphere forcing data covering the ocean forward steps.
        """
        time_dim = self.ocean.TIME_DIM
        # NOTE: only n_ic_timesteps = 1 is currently supported
        assert (
            next(iter(ocean_data.values())).shape[time_dim] == self.n_ic_timesteps + 1
        )
        # get n_ic_timesteps of ocean exogenous forcings
        forcing_data = {
            k: ocean_data[k]
            for k in set(self._ocean_forcing_exogenous_names).difference(
                self._shared_forcing_exogenous_names
            )
        }
        # get time-averaged forcings from atmosphere
        forcings_from_atmosphere = {
            **{
                k: atmos_gen[k].mean(time_dim, keepdim=True)
                for k in self._atmosphere_to_ocean_forcing_names
            },
            **{
                k: atmos_forcings[k].mean(time_dim, keepdim=True)
                for k in self._shared_forcing_exogenous_names
            },
        }
        # append or prepend nans depending on whether or not the forcing is a
        # "next step" forcing
        forcings_from_atmosphere = {
            k: (
                torch.cat([torch.full_like(v, fill_value=np.nan), v], dim=time_dim)
                if k in self._config.ocean_next_step_forcing_names
                else torch.cat([v, torch.full_like(v, fill_value=np.nan)], dim=time_dim)
            )
            for k, v in forcings_from_atmosphere.items()
        }
        forcing_data.update(forcings_from_atmosphere)
        return forcing_data

    def get_prediction_generator(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        if (
            initial_condition.atmosphere_data.as_batch_data().n_timesteps
            != self.atmosphere.n_ic_timesteps
        ):
            raise ValueError(
                "Atmosphere initial condition must have "
                f"{self.atmosphere.n_ic_timesteps} timesteps, got "
                f"{initial_condition.atmosphere_data.as_batch_data().n_timesteps}."
            )

        if (
            initial_condition.ocean_data.as_batch_data().n_timesteps
            != self.n_ic_timesteps
        ):
            raise ValueError(
                "Ocean initial condition must have "
                f"{self.n_ic_timesteps} timesteps, got "
                f"{initial_condition.ocean_data.as_batch_data().n_timesteps}."
            )
        atmos_ic_state = initial_condition.atmosphere_data
        ocean_ic_state = initial_condition.ocean_data

        n_outer_steps = forcing_data.ocean_data.n_timesteps - self.n_ic_timesteps

        for i_outer in range(n_outer_steps):
            # get the atmosphere window for the initial coupled step
            atmos_window = forcing_data.atmosphere_data.select_time_slice(
                slice(
                    i_outer * self.n_inner_steps,
                    (i_outer + 1) * self.n_inner_steps + self.atmosphere.n_ic_timesteps,
                )
            )
            atmos_forcings = BatchData(
                data=self._get_atmosphere_forcings(
                    atmos_window.data,
                    ocean_ic_state.as_batch_data().data,
                ),
                time=atmos_window.time,
                labels=atmos_window.labels,
            )
            # prescribe the initial condition SST state
            atmos_ic_state = self._prescribe_ic_sst(
                atmos_ic_state,
                atmos_forcings.select_time_slice(
                    slice(None, self.atmosphere.n_ic_timesteps)
                ),
            )
            atmos_generator = self.atmosphere.get_prediction_generator(
                atmos_ic_state,
                atmos_forcings,
                self.n_inner_steps,
                optimizer,
            )
            atmos_steps = []

            # predict and yield atmosphere steps
            for i_inner, atmos_step in enumerate(atmos_generator):
                yield ComponentStepPrediction(
                    realm="atmosphere",
                    data=atmos_step,
                    step=(i_outer * self.n_inner_steps + i_inner),
                )
                atmos_step = optimizer.detach_if_using_gradient_accumulation(atmos_step)
                atmos_steps.append(atmos_step)

            ocean_window = forcing_data.ocean_data.select_time_slice(
                slice(i_outer, i_outer + self.n_ic_timesteps + 1)
            )
            atmos_gen = stack_list_of_tensor_dicts(
                atmos_steps, self.atmosphere.TIME_DIM
            )

            atmos_data_forcings = atmos_window.select_time_slice(
                time_slice=slice(
                    self.atmosphere.n_ic_timesteps,
                    self.n_inner_steps + self.atmosphere.n_ic_timesteps,
                )
            )
            ocean_forcings = BatchData(
                data=self._get_ocean_forcings(
                    ocean_window.data, atmos_gen, atmos_data_forcings.data
                ),
                time=ocean_window.time,
                labels=ocean_window.labels,
            )
            # predict and yield a single ocean step
            ocean_step = next(
                iter(
                    self.ocean.get_prediction_generator(
                        ocean_ic_state,
                        ocean_forcings,
                        n_forward_steps=1,
                        optimizer=optimizer,
                    )
                )
            )
            yield ComponentStepPrediction(
                realm="ocean",
                data=ocean_step,
                step=i_outer,
            )

            # prepare ic states for next coupled step
            atmos_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(
                        {
                            k: v.unsqueeze(self.atmosphere.TIME_DIM)
                            for k, v in atmos_steps[-1].items()
                        }
                    ),
                    time=atmos_window.time.isel(
                        time=slice(-self.atmosphere.n_ic_timesteps, None)
                    ),
                    labels=atmos_window.labels,
                )
            )
            ocean_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(
                        {
                            k: v.unsqueeze(self.ocean.TIME_DIM)
                            for k, v in ocean_step.items()
                        }
                    ),
                    time=ocean_window.time.isel(time=slice(-self.n_ic_timesteps, None)),
                    labels=ocean_window.labels,
                )
            )

    def _process_prediction_generator_list(
        self,
        output_list: list[ComponentStepPrediction],
        forcing_data: CoupledBatchData,
    ) -> CoupledBatchData:
        atmos_data = process_prediction_generator_list(
            [x.data for x in output_list if x.realm == "atmosphere"],
            time=forcing_data.atmosphere_data.time[:, self.atmosphere.n_ic_timesteps :],
            horizontal_dims=forcing_data.atmosphere_data.horizontal_dims,
            labels=forcing_data.atmosphere_data.labels,
        )
        ocean_data = process_prediction_generator_list(
            [x.data for x in output_list if x.realm == "ocean"],
            time=forcing_data.ocean_data.time[:, self.ocean.n_ic_timesteps :],
            horizontal_dims=forcing_data.ocean_data.horizontal_dims,
            labels=forcing_data.ocean_data.labels,
        )
        return CoupledBatchData(ocean_data=ocean_data, atmosphere_data=atmos_data)

    def _predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ):
        timer = GlobalTimer.get_instance()
        output_list = list(
            self.get_prediction_generator(
                initial_condition, forcing, NullOptimization()
            )
        )
        gen_data = self._process_prediction_generator_list(output_list, forcing)
        if compute_derived_variables:
            with timer.context("compute_derived_variables"):
                gen_data = (
                    gen_data.prepend(initial_condition)
                    .compute_derived_variables(
                        ocean_derive_func=self.ocean.derive_func,
                        atmosphere_derive_func=self.atmosphere.derive_func,
                        forcing_data=forcing,
                    )
                    .remove_initial_condition(
                        n_ic_timesteps_ocean=self.ocean.n_ic_timesteps,
                        n_ic_timesteps_atmosphere=self.atmosphere.n_ic_timesteps,
                    )
                )
        return gen_data

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledPairedData, CoupledPrognosticState]:
        """
        Predict multiple steps forward given initial condition and reference data.
        """
        gen_data = self._predict(initial_condition, forcing, compute_derived_variables)
        atmos_forward_data = self.atmosphere.get_forward_data(
            forcing.atmosphere_data, compute_derived_variables=compute_derived_variables
        )
        ocean_forward_data = self.ocean.get_forward_data(
            forcing.ocean_data, compute_derived_variables=compute_derived_variables
        )
        return (
            CoupledPairedData.from_coupled_batch_data(
                prediction=gen_data,
                reference=CoupledBatchData(
                    ocean_data=ocean_forward_data,
                    atmosphere_data=atmos_forward_data,
                ),
            ),
            CoupledPrognosticState(
                ocean_data=gen_data.ocean_data.get_end(
                    self.ocean.prognostic_names,
                    self.n_ic_timesteps,
                ),
                atmosphere_data=gen_data.atmosphere_data.get_end(
                    self.atmosphere.prognostic_names,
                    self.atmosphere.n_ic_timesteps,
                ),
            ),
        )

    def predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        gen_data = self._predict(initial_condition, forcing, compute_derived_variables)
        return (
            CoupledBatchData(
                ocean_data=gen_data.ocean_data, atmosphere_data=gen_data.atmosphere_data
            ),
            CoupledPrognosticState(
                ocean_data=gen_data.ocean_data.get_end(
                    self.ocean.prognostic_names,
                    self.n_ic_timesteps,
                ),
                atmosphere_data=gen_data.atmosphere_data.get_end(
                    self.atmosphere.prognostic_names,
                    self.atmosphere.n_ic_timesteps,
                ),
            ),
        )

    def train_on_batch(
        self,
        data: CoupledBatchData,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ):
        """
        Args:
            data: The coupled batch data, consisting of separate batches for ocean and
                atmosphere with the same initial condition time.
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            compute_derived_variables: Whether to compute derived variables for the
                prediction and target atmosphere data.

        """
        # get initial condition prognostic variables
        input_data = CoupledPrognosticState(
            atmosphere_data=data.atmosphere_data.get_start(
                self.atmosphere.prognostic_names, self.n_ic_timesteps
            ),
            ocean_data=data.ocean_data.get_start(
                self.ocean.prognostic_names, self.n_ic_timesteps
            ),
        )

        atmos_forward_data = self.atmosphere.get_forward_data(
            data.atmosphere_data,
            compute_derived_variables=False,
        )
        ocean_forward_data = self.ocean.get_forward_data(
            data.ocean_data,
            compute_derived_variables=False,
        )

        metrics = ComponentStepMetrics()
        optimization.set_mode(self.modules)
        with optimization.autocast():
            output_generator = self.get_prediction_generator(
                input_data,
                data,
                optimization,
            )
            output_list = []
            for gen_step in output_generator:
                if gen_step.realm == "ocean":
                    # compute ocean step metrics
                    target_step = {
                        k: v.select(self.ocean.TIME_DIM, gen_step.step)
                        for k, v in ocean_forward_data.data.items()
                    }
                else:
                    assert gen_step.realm == "atmosphere"
                    target_step = {
                        k: v.select(self.atmosphere.TIME_DIM, gen_step.step)
                        for k, v in atmos_forward_data.data.items()
                    }
                step_loss = self._loss(
                    gen_step,
                    target_step,
                )
                if step_loss is not None:
                    label = f"loss/{gen_step.realm}_step_{gen_step.step}"
                    metrics.add_metric(label, step_loss.detach(), gen_step.realm)
                    optimization.accumulate_loss(step_loss)
                gen_step = gen_step.detach(optimization)
                output_list.append(gen_step)

        loss = optimization.get_accumulated_loss().detach()
        optimization.step_weights()

        gen_data = self._process_prediction_generator_list(output_list, data)
        ocean_stepped = TrainOutput(
            metrics=metrics.get_ocean_metrics(),
            gen_data=add_ensemble_dim(dict(gen_data.ocean_data.data)),
            target_data=add_ensemble_dim(dict(ocean_forward_data.data)),
            time=gen_data.ocean_data.time,
            normalize=self.ocean.normalizer.normalize,
            derive_func=self.ocean.derive_func,
        )
        atmos_stepped = TrainOutput(
            metrics=metrics.get_atmosphere_metrics(),
            gen_data=add_ensemble_dim(dict(gen_data.atmosphere_data.data)),
            target_data=add_ensemble_dim(dict(atmos_forward_data.data)),
            time=gen_data.atmosphere_data.time,
            normalize=self.atmosphere.normalizer.normalize,
            derive_func=self.atmosphere.derive_func,
        )

        stepped = CoupledTrainOutput(
            total_metrics={"loss": loss},
            ocean=ocean_stepped,
            atmosphere=atmos_stepped,
        )

        # prepend initial conditions
        ocean_data = data.ocean_data
        atmos_data = data.atmosphere_data
        ocean_ic = ocean_data.get_start(
            set(ocean_data.data.keys()), self.n_ic_timesteps
        )
        # TODO: different n_ic_timesteps for atmosphere?
        atmos_ic = atmos_data.get_start(
            set(atmos_data.data.keys()), self.n_ic_timesteps
        )
        ic = CoupledPrognosticState(ocean_data=ocean_ic, atmosphere_data=atmos_ic)
        stepped = stepped.prepend_initial_condition(ic)

        if compute_derived_variables:
            stepped = stepped.compute_derived_variables()

        return stepped

    def update_training_history(self, training_job: TrainingJob) -> None:
        """
        Update the stepper's history of training jobs.

        Args:
            training_job: The training job to add to the history.
        """
        self.ocean.update_training_history(training_job)
        self.atmosphere.update_training_history(training_job)

    @classmethod
    def from_state(cls, state) -> "CoupledStepper":
        ocean = Stepper.from_state(state["ocean_state"])
        atmosphere = Stepper.from_state(state["atmosphere_state"])
        config = CoupledStepperConfig.from_state(state["config"])
        if "dataset_info" in state:
            dataset_info = CoupledDatasetInfo.from_state(state["dataset_info"])
        else:
            # NOTE: this is included for backwards compatibility
            dataset_info = CoupledDatasetInfo(
                ocean=ocean.training_dataset_info,
                atmosphere=atmosphere.training_dataset_info,
            )
        return cls(
            config=config,
            ocean=ocean,
            atmosphere=atmosphere,
            dataset_info=dataset_info,
        )


def load_coupled_stepper(checkpoint_path: str | pathlib.Path) -> CoupledStepper:
    logging.info(f"Loading trained coupled model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=fme.get_device(), weights_only=False
    )
    stepper = CoupledStepper.from_state(checkpoint["stepper"])

    return stepper
