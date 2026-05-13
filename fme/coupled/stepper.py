import contextlib
import dataclasses
import datetime
import logging
import pathlib
from collections.abc import Generator, Iterable
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
    ParameterInitializationConfig,
    ParameterInitializer,
    StepperWeightsAndHistory,
    Weights,
    WeightsAndHistoryLoader,
)
from fme.ace.stepper.single_module import (
    StepperConfig,
    process_ensemble_prediction_generator_list,
)
from fme.ace.stepper.single_module import (
    load_weights_and_history as load_uncoupled_weights_and_history,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.ice import IceConfig
from fme.core.ice_data import IceData
from fme.core.loss import StepLossConfig
from fme.core.ocean import OceanConfig
from fme.core.ocean_data import OceanData
from fme.core.optimization import NullOptimization
from fme.core.tensors import add_ensemble_dim, unfold_ensemble_dim
from fme.core.timing import GlobalTimer
from fme.core.training_history import TrainingHistory, TrainingJob
from fme.core.typing_ import EnsembleTensorDict, TensorDict, TensorMapping
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
from fme.coupled.typing_ import CoupledNames, CoupledOptionalInt, CoupledTensorMapping


@dataclasses.dataclass
class ComponentConfig:
    """
    Configuration for one of the components (ocean, sea ice, or atmosphere) within a
    CoupledStepper.

    Parameters:
        timedelta: An ISO 8601 Duration string specifying the size of this component's
            stepper step.
        stepper: The single module stepper configuration for this component.
    """

    timedelta: str
    stepper: StepperConfig


@dataclasses.dataclass
class CoupledOceanFractionConfig:
    """
    Configuration for computing ocean fraction from the ocean-predicted or
    ice-predicted sea ice fraction.

    Parameters:
        sea_ice_fraction_name: Name of the sea ice fraction field in the ocean
            or sea ice data. Must be an ocean or ice prognostic variable. If the
            atmosphere uses the same name as an ML forcing then the generated sea
            ice fraction is also passed as an input to the atmosphere.
        land_fraction_name: Name of the land fraction field in the atmosphere
            data. If needed, will be passed to the ocean or ice stepper as a forcing.
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
        
    def validate_ice_prognostic_names(self, prognostic_names: Iterable[str]):
        if self.sea_ice_fraction_name not in prognostic_names:
            raise ValueError(
                f"CoupledOceanFractionConfig expected {self.sea_ice_fraction_name} "
                "to be a prognostic variable of the ice model, but it is not."
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

        When ocean fraction is predicted from ocean or sea ice model outputs, these
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
        self, forcings_from_ocean: TensorMapping, external_forcing_data: TensorMapping
    ) -> OceanData:
        # compute ocean frac from land frac and ocean-predicted sea ice frac
        land_frac_name = self.land_fraction_name
        sea_ice_frac_name = self.sea_ice_fraction_name
        # fill nans with 0s
        sea_ice_frac = torch.nan_to_num(forcings_from_ocean[sea_ice_frac_name])
        land_frac = external_forcing_data[land_frac_name]
        return OceanData({land_frac_name: land_frac, sea_ice_frac_name: sea_ice_frac})
    
    def build_ice_data(
        self, forcings_from_ice: TensorMapping, external_forcing_data: TensorMapping
    ) -> IceData:
        # compute ocean frac from land frac and ice-predicted sea ice frac
        land_frac_name = self.land_fraction_name
        sea_ice_frac_name = self.sea_ice_fraction_name
        # fill nans with 0s
        sea_ice_frac = torch.clamp(torch.nan_to_num(
                                forcings_from_ice[sea_ice_frac_name]),
                                min=0.0,max=1.0)
        land_frac = external_forcing_data[land_frac_name]
        return IceData({land_frac_name: land_frac, sea_ice_frac_name: sea_ice_frac})


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
                ice=load_uncoupled_weights_and_history,
                atmosphere=load_uncoupled_weights_and_history,
            )
        coupled_stepper = load_coupled_stepper(self.checkpoint_path)
        
        ocean_loader = None
        if coupled_stepper.ocean is not None:
            ocean_loader = _load_stepper_weights_and_history_factory(coupled_stepper.ocean)
            
        ice_loader = None
        if coupled_stepper.ice is not None:
            ice_loader = _load_stepper_weights_and_history_factory(coupled_stepper.ice)
            
        atmosphere_loader = None
        if coupled_stepper.atmosphere is not None:
            atmosphere_loader = _load_stepper_weights_and_history_factory(
                coupled_stepper.atmosphere
            )
            
        return CoupledWeightsAndHistoryLoaders(
            ocean=ocean_loader,
            ice=ice_loader,
            atmosphere=atmosphere_loader,
        )


@dataclasses.dataclass
class CoupledWeightsAndHistoryLoaders:
    ocean: WeightsAndHistoryLoader | None = None
    ice: WeightsAndHistoryLoader | None = None
    atmosphere: WeightsAndHistoryLoader | None = None
    training_history: TrainingHistory | None = None


@dataclasses.dataclass
class CoupledStepperConfig:
    """
    Configuration for a coupled atmosphere-ice-ocean stepper. From a common initial
    condition time, the atmosphere and ice step first and take as many steps as fit in
    a single ocean step, while being forced by the ocean's initial condition. The 
    ocean then steps forward once, receiving required forcings from the atmosphere- 
    and ice-generated output as averages over its step window. This completes
    a single "coupled step". For subsequent coupled steps, the generated output
    from the ocean forces the atmosphere and ice steps.

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
        ice: The ice component configuration. Output variable names must be distinct
            from the atmosphere and ocean output names. Outputs that are input names in
            the atmosphere and ocean must be prognostic variables in the ice.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include OceanConfig so that ocean-generated SSTs
            can be written on the atmosphere's surface temperature field. Output
            variable names must be distinct from the ocean's output names.
        sst_name: Name of the liquid sea surface temperature field in the ocean data.
            Must be present in the ocean's output names.
        ts_name: Name of the frozen ice (or snow) surface skin temperature field in the
            ice data. Must be present in the ice's output names.
        ocean_fraction_prediction: (Optional) Configuration for ocean-generated
            ocean fraction to replace the ocean fraction variable specified in the
            atmosphere's OceanConfig. If the atmosphere uses the ocean fraction as
            an ML forcing, the generated ocean fraction is also passed as an input.

    """

    ocean: ComponentConfig | None = None
    atmosphere: ComponentConfig | None = None
    ice: ComponentConfig | None = None
    sst_name: str = "sst"
    ts_name: str = "TS"
    ocean_fraction_prediction: CoupledOceanFractionConfig | None = None

    def __post_init__(self):
        self._validate_component_configs()
        
        if self.atmosphere is None:  # ice-ocean, with prescribed atmos
            # set timesteps
            self._ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
            self._ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()

            # calculate forcing sets
            self._ocean_forcing_exogenous_names = list(
                set(self.ocean.stepper.input_only_names).difference(
                    self.ice.stepper.output_names
                )
            )
            self._ice_forcing_exogenous_names = list(
                set(self.ice.stepper.input_only_names).difference(
                    self.ocean.stepper.output_names
                )
            )
            self._shared_forcing_exogenous_names = list(
                set(self._ocean_forcing_exogenous_names).intersection(
                    self._ice_forcing_exogenous_names
                )
            )
            self._ice_to_ocean_forcing_names = list(
                set(self.ocean.stepper.input_only_names).intersection(
                    self.ice.stepper.output_names
                )
            )
            self._ocean_to_ice_forcing_names = list(
                set(self.ice.stepper.input_only_names).intersection(
                    self.ocean.stepper.output_names
                )
            )
            # calculate names for each component's data requirements
            self._all_ice_names = list(
                set(self.ice.stepper.all_names).difference(
                    self.ocean.stepper.output_names
                )
            )
            # NOTE: this removes "shared" forcings from the ocean data requirements
            self._all_ocean_names = list(
                set(self.ocean.stepper.all_names).difference(self._all_ice_names)
            )

        elif self.ice is None:  # atmosphere-ocean (ice may be part of ocean)
            atmosphere_ocean_config = self.atmosphere.stepper.get_ocean()
            if atmosphere_ocean_config is None:
                raise RuntimeError(
                    "atmosphere ocean config is None after validation; "
                    "this should not happen"
                )
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
                
        elif self.ocean is None:  # atmosphere-ice, with prescribed ocean
            atmosphere_ice_config = self.atmosphere.stepper.get_ice()
            if atmosphere_ice_config is None:
                raise RuntimeError(
                    "atmosphere ice config is None after validation; "
                    "this should not happen"
                )
            self._atmosphere_ice_config = atmosphere_ice_config

            # set timesteps
            self._ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()
            self._atmosphere_timestep = pd.Timedelta(
                self.atmosphere.timedelta
            ).to_pytimedelta()

            # calculate forcing sets
            self._ice_forcing_exogenous_names = list(
                set(self.ice.stepper.input_only_names).difference(
                    self.atmosphere.stepper.output_names
                )
            )
            unfiltered_atmosphere_forcing_names = list(
                set(self.atmosphere.stepper.input_only_names).difference(
                    self.ice.stepper.output_names
                )
            )
            if self.ocean_fraction_prediction is not None:
                self._atmosphere_forcing_exogenous_names = (
                    self.ocean_fraction_prediction.filter_atmosphere_forcing_names(
                        unfiltered_atmosphere_forcing_names,
                        self._atmosphere_ice_config.ocean_fraction_name,
                    )
                )
            else:
                self._atmosphere_forcing_exogenous_names = (
                    unfiltered_atmosphere_forcing_names
                )
            self._shared_forcing_exogenous_names = list(
                set(self._ice_forcing_exogenous_names).intersection(
                    self._atmosphere_forcing_exogenous_names
                )
            )
            self._atmosphere_to_ice_forcing_names = list(
                set(self.ice.stepper.input_only_names).intersection(
                    self.atmosphere.stepper.output_names
                )
            )
            extra_forcings_names = [self.ts_name]
            if self.ocean_fraction_prediction is not None:
                # NOTE: this is only necessary for the special case where the
                # atmosphere doesn't use the sea ice fraction as an ML forcing
                extra_forcings_names.append(
                    self.ocean_fraction_prediction.sea_ice_fraction_name
                )

            self._ice_to_atmosphere_forcing_names = list(
                set(self.atmosphere.stepper.input_only_names)
                .intersection(self.ice.stepper.output_names)
                .union(extra_forcings_names)
            )

            # calculate names for each component's data requirements
            unfiltered_all_atmosphere_names = list(
                set(self.atmosphere.stepper.all_names).difference(
                    self.ice.stepper.output_names
                )
            )
            if self.ocean_fraction_prediction is not None:
                self._all_atmosphere_names = (
                    self.ocean_fraction_prediction.filter_atmosphere_forcing_names(
                        unfiltered_all_atmosphere_names,
                        self.sea_ice_fraction_name,
                    )
                )
            else:
                self._all_atmosphere_names = unfiltered_all_atmosphere_names
            # NOTE: this removes "shared" forcings from the ice data requirements
            self._all_ice_names = list(
                set(self.ice.stepper.all_names).difference(self._all_atmosphere_names)
            )
            if self.ocean_fraction_prediction is not None:
                # NOTE: land_fraciton is necessary to derive sea_ice_fraction from
                # ocean_sea_ice_fraction
                self._all_ice_names.append(
                    self.ocean_fraction_prediction.land_fraction_name
                )
        
        else:  # fully-coupled
            atmosphere_ocean_config = self.atmosphere.stepper.get_ocean()
            atmosphere_ice_config = self.atmosphere.stepper.get_ice()
            if atmosphere_ocean_config is None:
                raise RuntimeError(
                    "atmosphere ocean config is None after validation; "
                    "this should not happen"
                )
            if atmosphere_ice_config is None:
                raise RuntimeError(
                    "atmosphere ice config is None after validation; "
                    "this should not happen"
                )
            self._atmosphere_ocean_config = atmosphere_ocean_config
            self._atmosphere_ice_config = atmosphere_ice_config

            # set timesteps
            self._ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
            self._ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()
            self._atmosphere_timestep = pd.Timedelta(
                self.atmosphere.timedelta
            ).to_pytimedelta()

            # calculate forcing sets
            self._ocean_forcing_exogenous_names = list(
                set(self.ocean.stepper.input_only_names).difference(
                    set(self.atmosphere.stepper.output_names).union(
                        self.ice.stepper.output_names
                    )
                )
            )
            self._ice_forcing_exogenous_names = list(
                set(self.ice.stepper.input_only_names).difference(
                    set(self.atmosphere.stepper.output_names).union(
                        self.ocean.stepper.output_names
                    )
                )
            )
            unfiltered_atmosphere_forcing_names = list(
                set(self.atmosphere.stepper.input_only_names).difference(
                    set(self.ice.stepper.output_names).union(
                        self.ocean.stepper.output_names
                    )
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
                    set(self._atmosphere_forcing_exogenous_names).intersection(
                        self._ice_forcing_exogenous_names
                    )
                )
            )
            self._atmosphere_to_ocean_forcing_names = list(
                set(self.ocean.stepper.input_only_names).intersection(
                    self.atmosphere.stepper.output_names
                )
            )
            self._atmosphere_to_ice_forcing_names = list(
                set(self.ice.stepper.input_only_names).intersection(
                    self.atmosphere.stepper.output_names
                )
            )
            self._ice_to_atmosphere_forcing_names = list(
                set(self.atmosphere.stepper.input_only_names)
                .intersection(self.ice.stepper.output_names)
                .union([self.ts_name])
            )
            self._ice_to_ocean_forcing_names = list(
                set(self.ocean.stepper.input_only_names).intersection(
                    self.ice.stepper.output_names
                )
            )
            self._ocean_to_atmosphere_forcing_names = list(
                set(self.atmosphere.stepper.input_only_names)
                .intersection(self.ocean.stepper.output_names)
                .union([self.sst_name])
            )

            self._ocean_to_ice_forcing_names = list(
                set(self.ice.stepper.input_only_names).intersection(
                    set(self.ocean.stepper.output_names)
                )
            )
            # calculate names for each component's data requirements
            unfiltered_all_atmosphere_names = list(
                set(self.atmosphere.stepper.all_names).difference(
                    set(self.ocean.stepper.output_names).union(
                        self.ice.stepper.output_names
                        )
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
            self._all_ice_names = list(
                set(self.ice.stepper.all_names).difference(
                    set(self._all_atmosphere_names).union(
                        self._all_ocean_names
                    )
                )
            )
            if self.ocean_fraction_prediction is not None:
                # NOTE: land_fraciton is necessary to derive sea_ice_fraction from
                # ocean_sea_ice_fraction
                self._all_ice_names.append(
                    self.ocean_fraction_prediction.land_fraction_name
                )


    @property
    def timestep(self) -> datetime.timedelta:
        # the "coupled timestep" is the same as the ocean's
        # or sea ice if doing atmosphere-ice coupling
        if self.ocean is not None:
            return self._ocean_timestep
        elif (self.ocean is None) & (self.ice is not None):
            return self._ice_timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        if self.ocean is not None:
            return self._ocean_timestep
        else:
            raise AttributeError("Ocean component is None")
        
    @property
    def ice_timestep(self) -> datetime.timedelta:
        if self.ice is not None:
            return self._ice_timestep
        else:
            raise AttributeError("Ice component is None")

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        if self.atmosphere is not None:
            return self._atmosphere_timestep
        else:
            raise AttributeError("Atmosphere component is None")

    @property
    def n_inner_steps(self) -> int:
        if self.atmosphere is None:  # ice-ocean coupling
            return self.ocean_timestep // self.ice_timestep
        elif self.ocean is None:  # atmosphere-ice coupling
            return self.ice_timestep // self.atmosphere_timestep
        else:  # atmosphere-ocean or fully-coupled
            return self.ocean_timestep // self.atmosphere_timestep

    @property
    def atmosphere_ocean_config(self) -> OceanConfig:
        """The OceanConfig defined in the atmosphere StepperConfig."""
        if hasattr(self, '_atmosphere_ocean_config'):
            return self._atmosphere_ocean_config
        else:
            raise AttributeError("Atmosphere-ocean coupling not configured")
    
    @property
    def atmosphere_ice_config(self) -> IceConfig:
        """The IceConfig defined in the atmosphere StepperConfig."""
        if hasattr(self, '_atmosphere_ice_config'):
            return self._atmosphere_ice_config
        else:
            raise AttributeError("Atmosphere-ice coupling not configured")

    @property
    def ocean_fraction_name(self) -> str:
        """Name of the ocean fraction field in the atmosphere or ocean data."""
        if self.ocean is not None:
            return self._atmosphere_ocean_config.ocean_fraction_name
        else:
            raise AttributeError("Ocean fraction name not available")
        
    @property
    def sea_ice_fraction_name(self) -> str:
        """Name of the sea ice fraction field in the atmosphere or ice data."""
        if self.ice is not None:
            return self._atmosphere_ice_config.sea_ice_fraction_name
        else:
            raise AttributeError("Sea ice fraction name not available")

    @property
    def surface_temperature_name(self) -> str:
        """Name of the surface temperature field in the atmosphere data."""
        if hasattr(self, '_atmosphere_ocean_config'):
            return self._atmosphere_ocean_config.surface_temperature_name
        elif hasattr(self, '_atmosphere_ice_config'):
            return self._atmosphere_ice_config.ice_surface_temperature_name
        else:
            raise AttributeError("Surface temperature name not available")

    @property
    def ocean_next_step_forcing_names(self) -> list[str]:
        """Ocean next-step forcings."""
        if self.ocean is not None:
            return self.ocean.stepper.next_step_forcing_names
        else:
            raise AttributeError("No ocean component")

    @property
    def ocean_forcing_exogenous_names(self) -> list[str]:
        """Ocean forcing variables that are not outputs of the atmosphere or ice."""
        if self.ocean is not None:
            return self._ocean_forcing_exogenous_names
        else:
            raise AttributeError("No ocean component")
    
    @property
    def ice_forcing_exogenous_names(self) -> list[str]:
        """Ice forcing variables that are not outputs of the atmosphere or ocean."""
        if self.ice is not None:
            return self._ice_forcing_exogenous_names
        else:
            raise AttributeError("No ice component")

    @property
    def atmosphere_forcing_exogenous_names(self) -> list[str]:
        """Atmosphere forcing variables that are not outputs of the ocean or ice."""
        if self.atmosphere is not None:
            return self._atmosphere_forcing_exogenous_names
        else:
            raise AttributeError("No atmosphere component")

    @property
    def shared_forcing_exogenous_names(self) -> list[str]:
        """Exogenous forcing variables shared by both components. Must be
        present in the atmosphere data on disk. If time-varying, the ocean
        receives the atmosphere data forcings averaged over its step window.

        """
        return self._shared_forcing_exogenous_names

    @property
    def all_names(self) -> CoupledNames:
        """All variable names to log (outputs plus input-only forcings)."""
        atmosphere_names = []
        if self.atmosphere is not None:
            atmosphere_names = list(
                set(
                    self.atmosphere.stepper.output_names
                    + self._atmosphere_forcing_exogenous_names
                )
            )
        ocean_names = []
        if self.ocean is not None:
            ocean_names = list(
                set(self.ocean.stepper.output_names + self._ocean_forcing_exogenous_names)
            )
        ice_names = []
        if self.ice is not None:
            ice_names = list(
                set(self.ice.stepper.output_names + self._ice_forcing_exogenous_names)
            )
            
        return CoupledNames(ocean=ocean_names, atmosphere=atmosphere_names, ice=ice_names)

    @property
    def atmosphere_to_ocean_forcing_names(self) -> list[str]:
        """Ocean forcing variables that are outputs of the atmosphere."""
        return getattr(self, '_atmosphere_to_ocean_forcing_names', [])
    
    @property
    def atmosphere_to_ice_forcing_names(self) -> list[str]:
        """Ice forcing variables that are outputs of the atmosphere."""
        return getattr(self, '_atmosphere_to_ice_forcing_names', [])

    @property
    def ocean_to_atmosphere_forcing_names(self) -> list[str]:
        """Atmosphere forcing variables that are outputs of the ocean."""
        return getattr(self, '_ocean_to_atmosphere_forcing_names', [])
    
    @property
    def ocean_to_ice_forcing_names(self) -> list[str]:
        """Ice forcing variables that are outputs of the ocean."""
        return getattr(self, '_ocean_to_ice_forcing_names', [])
    
    @property
    def ice_to_atmosphere_forcing_names(self) -> list[str]:
        """Atmosphere forcing variables that are outputs of the Ice."""
        return getattr(self, '_ice_to_atmosphere_forcing_names', [])
    
    @property
    def ice_to_ocean_forcing_names(self) -> list[str]:
        """Ocean forcing variables that are outputs of the Ice."""
        return getattr(self, '_ice_to_ocean_forcing_names', [])

    def _validate_component_configs(self):
        # Ensure at least two components are present
        components_present = sum([
            self.ocean is not None,
            self.atmosphere is not None, 
            self.ice is not None
        ])
        if components_present < 2:
            raise ValueError("At least two components must be configured for coupling.")
        
        if self.atmosphere is None: #ice-ocean coupling
            ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
            ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()
            if ice_timestep > ocean_timestep:
                raise ValueError("Ice timedelta must not be larger than ocean's.")
            n_inner_steps = ocean_timestep / ice_timestep
            if n_inner_steps != int(n_inner_steps):
                raise ValueError("Ocean timedelta must be a multiple of the ice's.")
            
            # check for overlapping output names
            duplicate_outputs = set(self.ocean.stepper.output_names).intersection(
            self.ice.stepper.output_names
            )
            if len(duplicate_outputs) > 0:
                raise ValueError(
                    "Output variable names of CoupledStepper components cannot "
                    f"overlap. Found the following duplicated names: {duplicate_outputs}"
                )

            # ocean diagnostics cannot be used as atmosphere inputs
            ocean_diags_as_ice_forcings = list(
                set(self.ice.stepper.input_only_names)
                .intersection(self.ocean.stepper.output_names)
                .difference(self.ocean.stepper.input_names)
            )
            if len(ocean_diags_as_ice_forcings) > 0:
                raise ValueError(
                    "CoupledStepper only supports ocean prognostic variables as ice "
                    "forcings, but the following ocean diagnostic variables are inputs to "
                    f"the ice: {ocean_diags_as_ice_forcings}."
                )

            # all ocean inputs that are ice outputs must be "next step"
            # forcings according to the ocean stepper config
            ice_to_ocean_forcing_names = list(
                set(self.ocean.stepper.input_only_names).intersection(
                    self.ice.stepper.output_names
                )
            )
            missing_next_step_forcings = list(
                set(ice_to_ocean_forcing_names).difference(
                    self.ocean.stepper.next_step_forcing_names
                )
            )
            if len(missing_next_step_forcings) > 0:
                raise ValueError(
                    "The following variables which are ice component outputs "
                    "and ocean component inputs were not found among the ocean's "
                    f"next_step_forcing_names: {missing_next_step_forcings}."
                )
            
        elif self.ice is None: #atmosphere-ocean coupling
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
            
        elif self.ocean is None: #atmosphere-ice coupling    
            atmosphere_ice_config = self.atmosphere.stepper.get_ice()
            if atmosphere_ice_config is None:
                raise ValueError(
                    "The atmosphere stepper 'ice' config is missing but must be set for "
                    "coupled emulation."
                )
            # validate compatibility of ice and atmosphere timestep sizes
            ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()
            atmosphere_timestep = pd.Timedelta(self.atmosphere.timedelta).to_pytimedelta()
            if atmosphere_timestep > ice_timestep:
                raise ValueError("Atmosphere timedelta must not be larger than ice's.")
            n_inner_steps = ice_timestep / atmosphere_timestep
            if n_inner_steps != int(n_inner_steps):
                raise ValueError("Ice timedelta must be a multiple of the atmosphere's.")

            # check for overlapping output names
            duplicate_outputs = set(self.ice.stepper.output_names).intersection(
                self.atmosphere.stepper.output_names
            )
            if len(duplicate_outputs) > 0:
                raise ValueError(
                    "Output variable names of CoupledStepper components cannot "
                    f"overlap. Found the following duplicated names: {duplicate_outputs}"
                )

            # ice diagnostics cannot be used as atmosphere inputs
            ice_diags_as_atmos_forcings = list(
                set(self.atmosphere.stepper.input_only_names)
                .intersection(self.ice.stepper.output_names)
                .difference(self.ice.stepper.input_names)
            )
            if len(ice_diags_as_atmos_forcings) > 0:
                raise ValueError(
                    "CoupledStepper only supports ice prognostic variables as atmosphere "
                    "forcings, but the following ice diagnostic variables are inputs to "
                    f"the atmosphere: {ice_diags_as_atmos_forcings}."
                )

            # all ice inputs that are atmosphere outputs must be "next step"
            # forcings according to the ice stepper config
            atmosphere_to_ice_forcing_names = list(
                set(self.ice.stepper.input_only_names).intersection(
                    self.atmosphere.stepper.output_names
                )
            )
            missing_next_step_forcings = list(
                set(atmosphere_to_ice_forcing_names).difference(
                    self.ice.stepper.next_step_forcing_names
                )
            )
            if len(missing_next_step_forcings) > 0:
                raise ValueError(
                    "The following variables which are atmosphere component outputs "
                    "and ice component inputs were not found among the ice's "
                    f"next_step_forcing_names: {missing_next_step_forcings}."
                )
            
            # ts_name must be present in the ice's output names
            if self.ts_name not in self.ice.stepper.output_names:
                raise ValueError(
                    f"The variable {self.ts_name} is not in the ice's output "
                    "names but is required for coupling with the atmosphere."
                )
            
            # validate ocean_fraction_prediction
            if self.ocean_fraction_prediction is not None:
                self.ocean_fraction_prediction.validate_ice_prognostic_names(
                    self.ice.stepper.prognostic_names,
                )
                self.ocean_fraction_prediction.validate_atmosphere_forcing_names(
                    self.atmosphere.stepper.input_only_names
                )

        else: #fully-coupled
            atmosphere_ocean_config = self.atmosphere.stepper.get_ocean()
            atmosphere_ice_config = self.atmosphere.stepper.get_ice()
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
            if atmosphere_ice_config is None:
                raise ValueError(
                    "The atmosphere stepper 'ice' config is missing but must be set for "
                    "coupled emulation."
                )
            # validate compatibility of ocean and atmosphere timestep sizes
            ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
            ice_timestep = pd.Timedelta(self.ice.timedelta).to_pytimedelta()
            atmosphere_timestep = pd.Timedelta(self.atmosphere.timedelta).to_pytimedelta()
            if atmosphere_timestep > ocean_timestep:
                raise ValueError("Atmosphere timedelta must not be larger than ocean's.")
            if atmosphere_timestep > ice_timestep:
                raise ValueError("Atmosphere timedelta must not be larger than ice's.")
            if ice_timestep > ocean_timestep:
                raise ValueError("Ice timedelta must not be larger than ocean's.")
            n_inner_steps = ocean_timestep / atmosphere_timestep
            if n_inner_steps != int(n_inner_steps):
                raise ValueError("Ocean timedelta must be a multiple of the atmosphere's.")

            # check for overlapping output names
            duplicate_outputs = set(self.ocean.stepper.output_names).intersection(
                set(self.atmosphere.stepper.output_names).union(
                    set(self.ice.stepper.output_names)
                )
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
            
            # ocean diagnostics cannot be used as ice inputs
            ocean_diags_as_ice_forcings = list(
                set(self.ice.stepper.input_only_names)
                .intersection(self.ocean.stepper.output_names)
                .difference(self.ocean.stepper.input_names)
            )
            if len(ocean_diags_as_ice_forcings) > 0:
                raise ValueError(
                    "CoupledStepper only supports ocean prognostic variables as ice "
                    "forcings, but the following ocean diagnostic variables are inputs to "
                    f"the ice: {ocean_diags_as_ice_forcings}."
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
            
            # all ocean inputs that are ice outputs must be "next step"
            # forcings according to the ocean stepper config
            ice_to_ocean_forcing_names = list(
                set(self.ocean.stepper.input_only_names).intersection(
                    self.ice.stepper.output_names
                )
            )
            missing_next_step_forcings = list(
                set(ice_to_ocean_forcing_names).difference(
                    self.ocean.stepper.next_step_forcing_names
                )
            )
            if len(missing_next_step_forcings) > 0:
                raise ValueError(
                    "The following variables which are ice component outputs "
                    "and ocean component inputs were not found among the ocean's "
                    f"next_step_forcing_names: {missing_next_step_forcings}."
                )

            # sst_name must be present in the ocean's output names
            if self.sst_name not in self.ocean.stepper.output_names:
                raise ValueError(
                    f"The variable {self.sst_name} is not in the ocean's output "
                    "names but is required for coupling with the atmosphere."
                )
            # ts_name must be present in the ice's output names
            if self.ts_name not in self.ice.stepper.output_names:
                raise ValueError(
                    f"The variable {self.ts_name} is not in the ice's output "
                    "names but is required for coupling with the atmosphere."
                )
            
            # validate ocean_fraction_prediction
            if self.ocean_fraction_prediction is not None:
                self.ocean_fraction_prediction.validate_ice_prognostic_names(
                    self.ice.stepper.prognostic_names,
                )
                self.ocean_fraction_prediction.validate_atmosphere_forcing_names(
                    self.atmosphere.stepper.input_only_names
                )

    def _get_ocean_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self._all_ocean_names, n_timesteps=n_forward_steps + 1
        )
    
    def _get_ice_data_requirements(self, n_forward_steps: int) -> DataRequirements:
        return DataRequirements(
            names=self._all_ice_names, n_timesteps=n_forward_steps + 1
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
        """Get the DataRequirements for the ocean, ice, and atmosphere. For every step
        of the CoupledStepper, the atmosphere and ice take n_inner_steps (determined by
        the number of atmosphere (ice) timesteps that fit in a single ocean timestep)
        steps and the ocean takes a single step. Therefore, we need
        n_coupled_steps number of ocean forward steps and n_coupled_steps *
        n_inner_steps number of atmosphere (ice) forward steps.

        n_coupled_steps: The number of CoupledStepper forward steps. During
            training, these steps are included when computing gradients.

        """
        ocean_requirements = None
        ice_requirements = None
        atmosphere_requirements = None
        
        if self.ocean is not None:
            ocean_requirements = self._get_ocean_data_requirements(n_coupled_steps)
        
        if self.ice is not None:
            if self.ocean is not None:
                n_ice_steps = n_coupled_steps * self.n_inner_steps
            else:
                n_ice_steps = n_coupled_steps
            ice_requirements = self._get_ice_data_requirements(n_ice_steps)

        if self.atmosphere is not None:
            if self.ocean is not None:
                n_atmosphere_steps = n_coupled_steps * self.n_inner_steps
            else:
                n_atmosphere_steps = n_coupled_steps
            atmosphere_requirements = self._get_atmosphere_data_requirements(n_atmosphere_steps)
        
        return CoupledDataRequirements(
            ocean_timestep=getattr(self, '_ocean_timestep', None),
            ocean_requirements=ocean_requirements,
            ice_timestep=getattr(self, '_ice_timestep', None),
            ice_requirements=ice_requirements,
            atmosphere_timestep=getattr(self, '_atmosphere_timestep', None),
            atmosphere_requirements=atmosphere_requirements,
        )

    def get_prognostic_state_data_requirements(
        self,
    ) -> CoupledPrognosticStateDataRequirements:
        """Get the PrognosticStateDataRequirements for the ocean and atmosphere."""
        ocean_requirements = None
        atmosphere_requirements = None
        ice_requirements = None
        
        if self.ocean is not None:
            ocean_requirements = self.ocean.stepper.get_prognostic_state_data_requirements()
        
        if self.atmosphere is not None:
            atmosphere_requirements = self.atmosphere.stepper.get_prognostic_state_data_requirements()
        
        if self.ice is not None:
            ice_requirements = self.ice.stepper.get_prognostic_state_data_requirements()
        
        return CoupledPrognosticStateDataRequirements(
            ocean=ocean_requirements,
            atmosphere=atmosphere_requirements,
            ice=ice_requirements,
        )

    def get_forcing_window_data_requirements(
        self, n_coupled_steps: int
    ) -> CoupledDataRequirements:
        ocean_requirements = None
        ice_requirements = None
        atmosphere_requirements = None
        
        if self.ocean is not None:
            ocean_forcing_names = list(
                set(self.ocean_forcing_exogenous_names).difference(
                    self.shared_forcing_exogenous_names
                )
            )
            ocean_requirements = DataRequirements(
                ocean_forcing_names, n_timesteps=n_coupled_steps + 1
            )

        if self.ice is not None:
            if self.ocean is not None:
                n_ice_steps = n_coupled_steps * self.n_inner_steps
            else:
                n_ice_steps = n_coupled_steps
            ice_requirements = DataRequirements(
                names=self.ice_forcing_exogenous_names,
                n_timesteps=n_ice_steps + 1,
            )
        
        if self.atmosphere is not None:
            if self.ocean is not None:
                n_atmosphere_steps = n_coupled_steps * self.n_inner_steps
            else:
                n_atmosphere_steps = n_coupled_steps
            atmosphere_requirements = DataRequirements(
                names=self.atmosphere_forcing_exogenous_names,
                n_timesteps=n_atmosphere_steps + 1,
            )

        return CoupledDataRequirements(
            ocean_timestep=getattr(self, '_ocean_timestep', None),
            ocean_requirements=ocean_requirements,
            ice_timestep=getattr(self, '_ice_timestep', None),
            ice_requirements=ice_requirements,
            atmosphere_timestep=getattr(self, '_atmosphere_timestep', None),
            atmosphere_requirements=atmosphere_requirements,
        )

    def _get_ocean_stepper(
        self,
        dataset_info: DatasetInfo,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> Stepper:
        if dataset_info.timestep != self.ocean_timestep:
            raise ValueError(
                "Ocean timestep must match the dataset timestep. "
                f"Got {self.ocean_timestep} and {dataset_info.timestep}, respectively."
            )
        return self.ocean.stepper.get_stepper(
            dataset_info=dataset_info,
            parameter_initializer=parameter_initializer,
        )
    
    def _get_ice_stepper(
        self,
        dataset_info: DatasetInfo,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> Stepper:
        if dataset_info.timestep != self.ice_timestep:
            raise ValueError(
                "Ice timestep must match the dataset timestep. "
                f"Got {self.ice_timestep} and {dataset_info.timestep}, "
                "respectively."
            )
        return self.ice.stepper.get_stepper(
            dataset_info=dataset_info,
            parameter_initializer=parameter_initializer,
        )

    def _get_atmosphere_stepper(
        self,
        dataset_info: DatasetInfo,
        parameter_initializer: ParameterInitializer | None = None,
    ) -> Stepper:
        if dataset_info.timestep != self.atmosphere_timestep:
            raise ValueError(
                "Atmosphere timestep must match the dataset timestep. "
                f"Got {self.atmosphere_timestep} and {dataset_info.timestep}, "
                "respectively."
            )
        return self.atmosphere.stepper.get_stepper(
            dataset_info=dataset_info,
            parameter_initializer=parameter_initializer,
        )

    def get_stepper(
        self,
        dataset_info: CoupledDatasetInfo,
        ocean_parameter_initializer: ParameterInitializer | None = None,
        ice_parameter_initializer: ParameterInitializer | None = None,
        atmosphere_parameter_initializer: ParameterInitializer | None = None,
    ):
        logging.info("Initializing coupler")
        
        ocean_stepper = None
        if self.ocean is not None:
            ocean_stepper = self._get_ocean_stepper(
                dataset_info=dataset_info.ocean,
                parameter_initializer=ocean_parameter_initializer,
            )
            
        ice_stepper = None
        if self.ice is not None:
            ice_stepper = self._get_ice_stepper(
                dataset_info=dataset_info.ice,
                parameter_initializer=ice_parameter_initializer,
            )
            
        atmosphere_stepper = None
        if self.atmosphere is not None:
            atmosphere_stepper = self._get_atmosphere_stepper(
                dataset_info=dataset_info.atmosphere,
                parameter_initializer=atmosphere_parameter_initializer,
            )
            
        return CoupledStepper(
            config=self,
            ocean=ocean_stepper,
            ice=ice_stepper,
            atmosphere=atmosphere_stepper,
            dataset_info=dataset_info,
        )

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "CoupledStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls,
            data=state,
            config=dacite.Config(
                strict=True, type_hooks={StepperConfig: StepperConfig.from_state}
            ),
        )

    @classmethod
    def remove_deprecated_keys(cls, state: dict[str, Any]) -> dict[str, Any]:
        state_copy = state.copy()
        if "sst_mask_name" in state_copy:
            del state_copy["sst_mask_name"]
        if "parameter_init" in state_copy:
            del state_copy["parameter_init"]
        components = []
        if state_copy["ocean"] is not None:
            components.append("ocean")
        if state_copy["ice"] is not None:
            components.append("ice")
        if state_copy["atmosphere"] is not None:
            components.append("atmosphere")
        for component_key in components:
            if "loss_contributions" in state_copy[component_key]:
                del state_copy[component_key]["loss_contributions"]
        return state_copy


class ComponentStepMetrics:
    def __init__(self):
        self._ocean: TensorDict = {}
        self._ice: TensorDict = {}
        self._atmos: TensorDict = {}

    def add_metric(self, key, value, realm: Literal["ocean", "ice", "atmosphere"]) -> None:
        if realm == "ocean":
            self._ocean[key] = value
        elif realm == "ice":
            self._ice[key] = value
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

    def get_ice_metrics(self) -> TensorDict:
        if not self._ice:
            return {"loss/ice": torch.tensor(0.0, device=fme.get_device())}
        loss = sum(self._ice.values())
        return {
            "loss/ice": loss,
            **self._ice,
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
    ocean: TrainOutput | None = None
    ice: TrainOutput | None = None
    atmosphere: TrainOutput | None = None

    def remove_initial_condition(self, n_ic_timesteps: int) -> "CoupledTrainOutput":
        if self.ocean is not None:
            ocean_info = self.ocean.remove_initial_condition(n_ic_timesteps)
        else:
            ocean_info = None
        if self.ice is not None:
            ice_info = self.ice.remove_initial_condition(n_ic_timesteps)
        else:
            ice_info = None
        if self.atmosphere is not None:
            atmosphere_info = self.atmosphere.remove_initial_condition(n_ic_timesteps)
        else:
            atmosphere_info = None
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=ocean_info,
            ice=ice_info,
            atmosphere=atmosphere_info
        )

    def copy(self) -> "CoupledTrainOutput":
        if self.ocean is not None:
            ocean_info = self.ocean.copy()
        else:
            ocean_info = None
        if self.ice is not None:
            ice_info = self.ice.copy()
        else:
            ice_info = None
        if self.atmosphere is not None:
            atmosphere_info = self.atmosphere.copy()
        else:
            atmosphere_info = None
        return CoupledTrainOutput(
            total_metrics=self.total_metrics.copy(),
            ocean=ocean_info,
            ice=ice_info,
            atmosphere=atmosphere_info
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
        if self.ocean is not None:
            ocean_info = self.ocean.prepend_initial_condition(
                initial_condition.ocean_data,
            )
        else:
            ocean_info = None
        if self.ice is not None:
            ice_info = self.ice.prepend_initial_condition(
                initial_condition.ice_data,
            )
        else:
            ice_info = None
        if self.atmosphere is not None:
            atmosphere_info = self.atmosphere.prepend_initial_condition(
                initial_condition.atmosphere_data,
            )
        else:
            atmosphere_info = None
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=ocean_info,
            ice=ice_info,
            atmosphere=atmosphere_info
        )

    def compute_derived_variables(self) -> "CoupledTrainOutput":
        ocean_info = None
        if self.ocean is not None:
            ocean_info = self.ocean.compute_derived_variables()
        ice_info = None
        if self.ice is not None:
            ice_info = self.ice.compute_derived_variables()
        atmosphere_info = None
        if self.atmosphere is not None:
            atmosphere_info = self.atmosphere.compute_derived_variables()
        return CoupledTrainOutput(
            total_metrics=self.total_metrics,
            ocean=ocean_info,
            ice=ice_info,
            atmosphere=atmosphere_info
        )

    def get_metrics(self) -> TensorDict:
        all_metrics = dict(self.total_metrics)
        
        if self.ocean is not None:
            ocean_keys = set(self.ocean.metrics.keys())
            # Check for conflicts with existing metrics
            overlap = ocean_keys.intersection(all_metrics.keys())
            if len(overlap) > 0:
                raise ValueError(
                    "The following ocean metric names conflict with existing metric names: "
                    f"{overlap}."
                )
            all_metrics.update(self.ocean.metrics)
            
        if self.ice is not None:
            ice_keys = set(self.ice.metrics.keys())
            # Check for conflicts with existing metrics
            overlap = ice_keys.intersection(all_metrics.keys())
            if len(overlap) > 0:
                raise ValueError(
                    "The following ice metric names conflict with existing metric names: "
                    f"{overlap}."
                )
            all_metrics.update(self.ice.metrics)
            
        if self.atmosphere is not None:
            atmos_keys = set(self.atmosphere.metrics.keys())
            # Check for conflicts with existing metrics
            overlap = atmos_keys.intersection(all_metrics.keys())
            if len(overlap) > 0:
                raise ValueError(
                    "The following atmosphere metric names conflict with existing metric names: "
                    f"{overlap}."
                )
            all_metrics.update(self.atmosphere.metrics)
            
        return all_metrics


class ComponentStepPrediction(StepPredictionABC):
    def __init__(
        self,
        realm: Literal["ocean", "atmosphere", "ice"],
        data: TensorDict,
        step: int,
    ):
        self._realm: Literal["ocean", "atmosphere", "ice"] = realm
        self._data = data
        self._step = step

    @property
    def realm(self) -> Literal["ocean", "atmosphere", "ice"]:
        return self._realm

    @property
    def data(self) -> TensorDict:
        return self._data

    @property
    def step(self) -> int:
        return self._step
    

class CoupledStepper:
    TIME_DIM = 1

    def __init__(
        self,
        config: CoupledStepperConfig,
        ocean: Stepper | None,
        ice: Stepper | None,
        atmosphere: Stepper | None,
        dataset_info: CoupledDatasetInfo,
    ):
        """
        Args:
            config: The configuration.
            ocean: The ocean stepper.
            ice: The ice stepper.
            atmosphere: The atmosphere stepper.
            dataset_info: The CoupledDatasetInfo.
        """
        # Check n_ic_timesteps for non-None steppers
        if ocean is not None and ocean.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported for ocean.")
        if ice is not None and ice.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported for ice.")
        if atmosphere is not None and atmosphere.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported for atmosphere.")

        self.ocean = ocean
        self.ice = ice
        self.atmosphere = atmosphere
        self._config = config
        self._dataset_info = dataset_info
        self._ocean_mask_provider = dataset_info.ocean_mask_provider
        self._ice_mask_provider = dataset_info.ice_mask_provider
        _: PredictFunction[  # for type checking
            CoupledPrognosticState,
            CoupledBatchData,
            CoupledPairedData,
        ] = self.predict_paired

    @property
    def modules(self) -> nn.ModuleList:
        all_modules = []
        if self.atmosphere is not None:
            all_modules.extend(self.atmosphere.modules)
        if self.ocean is not None:
            all_modules.extend(self.ocean.modules)
        if self.ice is not None:
            all_modules.extend(self.ice.modules)
        return nn.ModuleList(all_modules)

    def set_train(self):
        if self.atmosphere is not None:
            self.atmosphere.set_train()
        if self.ocean is not None:
            self.ocean.set_train()
        if self.ice is not None:
            self.ice.set_train()

    def set_eval(self):
        if self.atmosphere is not None:
            self.atmosphere.set_eval()
        if self.ocean is not None:
            self.ocean.set_eval()
        if self.ice is not None:
            self.ice.set_eval()

    def get_state(self):
        """
        Returns:
            The state of the coupled stepper.
        """
        state = {
            "config": self._config.get_state(),
            "dataset_info": self._dataset_info.get_state(),
        }
        if self.atmosphere is not None:
            state["atmosphere_state"] = self.atmosphere.get_state()
        if self.ocean is not None:
            state["ocean_state"] = self.ocean.get_state()
        if self.ice is not None:
            state["ice_state"] = self.ice.get_state()
        return state

    def load_state(self, state: dict[str, Any]):
        if self.atmosphere is not None and "atmosphere_state" in state:
            self.atmosphere.load_state(state["atmosphere_state"])
        if self.ocean is not None and "ocean_state" in state:
            self.ocean.load_state(state["ocean_state"])
        if self.ice is not None and "ice_state" in state:
            self.ice.load_state(state["ice_state"])

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
        return getattr(self._config, 'ocean_forcing_exogenous_names', [])
    
    @property
    def _ice_forcing_exogenous_names(self) -> list[str]:
        return getattr(self._config, 'ice_forcing_exogenous_names', [])

    @property
    def _atmosphere_forcing_exogenous_names(self) -> list[str]:
        return getattr(self._config, 'atmosphere_forcing_exogenous_names', [])

    @property
    def _shared_forcing_exogenous_names(self) -> list[str]:
        return getattr(self._config, 'shared_forcing_exogenous_names', [])

    @property
    def _atmosphere_to_ocean_forcing_names(self) -> list[str]:
        return getattr(self._config, 'atmosphere_to_ocean_forcing_names', [])

    @property
    def _atmosphere_to_ice_forcing_names(self) -> list[str]:
        return getattr(self._config, 'atmosphere_to_ice_forcing_names', [])

    @property
    def _ocean_to_atmosphere_forcing_names(self) -> list[str]:
        return getattr(self._config, 'ocean_to_atmosphere_forcing_names', [])

    @property
    def _ocean_to_ice_forcing_names(self) -> list[str]:
        return getattr(self._config, 'ocean_to_ice_forcing_names', [])

    @property
    def _ice_to_atmosphere_forcing_names(self) -> list[str]:
        return getattr(self._config, 'ice_to_atmosphere_forcing_names', [])

    @property
    def _ice_to_ocean_forcing_names(self) -> list[str]:
        return getattr(self._config, 'ice_to_ocean_forcing_names', [])
    
    def _prescribe_ic_ts(
        self,
        atmos_ic_state: PrognosticState,
        forcing_ic_batch: BatchData,
    ) -> PrognosticState:
        """Prescribe the initial condition TS state on the surface_temperature
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
        assert self.atmosphere.sea_ice_fraction_name in forcing_ic_data
        atmos_ic_data = self.atmosphere.prescribe_ice_ts(
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
        external_forcing_data: TensorMapping,
    ) -> TensorDict:
        """Get the ocean fraction field and return it with the other fields in
        forcings_from_ocean.

        Returns:
            forcings_from_ocean: A copy of the forcings_from_ocean input,
            including the ocean fraction.

        """
        forcings_from_ocean = dict(forcings_from_ocean)
        if self.ice is None:
            ocean_frac_name = self._config.ocean_fraction_name
            if self._config.ocean_fraction_prediction is None:
                # for convenience, move the atmos's ocean fraction to the
                # forcings_from_ocean dict
                forcings_from_ocean[ocean_frac_name] = external_forcing_data[ocean_frac_name]
            else:
                # compute ocean frac from land frac and ocean-predicted sea ice frac
                ofrac_config = self._config.ocean_fraction_prediction
                ocean_data = ofrac_config.build_ocean_data(
                    forcings_from_ocean, external_forcing_data
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
    
    def _forcings_from_ice_with_ocean_fraction(
        self,
        forcings_from_ice: TensorMapping,
        external_forcing_data: TensorMapping,
    ) -> TensorDict:
        """Get the ocean fraction field and return it with the other fields in
        forcings_from_ice.

        Returns:
            forcings_from_ice: A copy of the forcings_from_ice input,
            including the ocean fraction.

        """
        forcings_from_ice = dict(forcings_from_ice)
        if self.atmosphere is not None:
            ice_frac_name = self._config.sea_ice_fraction_name
            if self._config.ocean_fraction_prediction is None:
                # for convenience, move the atmos's ocean fraction to the
                # forcings_from_ice dict
                forcings_from_ice[ice_frac_name] = external_forcing_data[ice_frac_name]
            else:
                ifrac_config = self._config.ice_fraction_prediction
                ice_data = ifrac_config.build_ice_data(
                    forcings_from_ice, external_forcing_data
                )
                sea_ice_frac_name = (
                    ifrac_config.sea_ice_fraction_name_in_atmosphere
                    or ifrac_config.sea_ice_fraction_name
                )
                forcings_from_ice[sea_ice_frac_name] = ice_data.sea_ice_fraction
                forcings_from_ice[ice_frac_name] = torch.clip(
                    ice_data.ice_fraction, min=0
                )
        for name, tensor in forcings_from_ice.items():
            # set ice invalid points to 0 based on the ice masking
            mask = self._ice_mask_provider.get_mask_tensor_for(name)
            if mask is not None:
                mask = mask.expand(tensor.shape)
                forcings_from_ice[name] = tensor.where(mask != 0, 0)
        return forcings_from_ice

    def _get_atmosphere_forcings(
        self,
        atmos_data: TensorMapping,
        ocean_ic: TensorMapping | None = None,
        ice_ic: TensorMapping | None = None,
    ) -> TensorDict:
        """
        Get the forcings for the atmosphere component.

        Args:
            atmos_data: Atmosphere batch data, including initial condition and forward
                steps.
            ocean_ic: Ocean initial condition state, including SST.
            ice_ic: Ice initial condition state, including TS.
        """
        time_dim = self.atmosphere.TIME_DIM
        sizes = [-1] * len(next(iter(atmos_data.values())).shape)
        sizes[time_dim] = self.n_inner_steps + 1
        # exogenous forcings are used as is
        forcing_data = {
            k: atmos_data[k] for k in self._atmosphere_forcing_exogenous_names
        }
        if ocean_ic is not None:
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
        if ice_ic is not None:
            # NOTE: only n_ic_timesteps = 1 is currently supported
            assert next(iter(ice_ic.values())).shape[self.ice.TIME_DIM] == 1
            forcings_from_ice = {
                k: ice_ic[k].expand(*sizes)
                for k in self._ice_to_atmosphere_forcing_names
            }
            # rename the ice surface temperature variable using the corresponding
            # name in the atmosphere
            forcings_from_ice[self._config.surface_temperature_name] = (
                forcings_from_ice.pop(self._config.ts_name)
            )
            # get the TS mask (0 if land, 1 if sea surface)
            forcings_from_ice = self._forcings_from_ice_with_ocean_fraction(
                forcings_from_ice, forcing_data
            )
            # update atmosphere forcings
            forcing_data.update(forcings_from_ice)
        return forcing_data

    def _get_ocean_forcings(
        self,
        ocean_data: TensorMapping,
        atmos_gen: TensorMapping | None = None,
        atmos_forcings: TensorMapping | None = None,
        ice_gen: TensorMapping | None = None,
        ice_forcings: TensorMapping | None = None
    ) -> TensorDict:
        """
        Get the forcings for the ocean component.

        Args:
            ocean_data: Ocean data, including initial condition and forward
                steps.
            atmos_gen: Generated atmosphere data covering the ocean forward steps.
            atmos_forcings: Atmosphere forcing data covering the ocean forward steps.
            ice_gen: Generated ice data covering the ocean forward steps.
            ice_forcings: Ice forcing data covering the ocean forward steps.
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
        if atmos_gen is not None:
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
        if ice_gen is not None:
            # get time-averaged forcings from ice
            forcings_from_ice = {
                **{
                    k: ice_gen[k].mean(time_dim, keepdim=True)
                    for k in self._ice_to_ocean_forcing_names
                },
                **{
                    k: ice_forcings[k].mean(time_dim, keepdim=True)
                    for k in self._shared_forcing_exogenous_names
                },
            }
            # append or prepend nans depending on whether or not the forcing is a
            # "next step" forcing
            forcings_from_ice = {
                k: (
                    torch.cat([torch.full_like(v, fill_value=np.nan), v], dim=time_dim)
                    if k in self._config.ocean_next_step_forcing_names
                    else torch.cat([v, torch.full_like(v, fill_value=np.nan)], dim=time_dim)
                )
                for k, v in forcings_from_ice.items()
            }
            forcings_from_ice = self._forcings_from_ice_with_ocean_fraction(
                forcings_from_ice, forcing_data
            )
            forcing_data.update(forcings_from_ice)
        return forcing_data
    
    def _get_ice_forcings(
        self,
        ice_data: TensorMapping,
        ocean_ic: TensorMapping | None = None,
        atmos_ic: TensorMapping | None = None,
    ) -> TensorDict:
        """
        Get the forcings for the ice component.

        Args:
            ice_data: Ice batch data, including initial condition and forward
                steps.
            ocean_ic: Ocean initial condition state, including SST.
            atmos_ic: Atmosphere initial condition state, including surface temperature.
        """
        time_dim = self.ice.TIME_DIM
        sizes = [-1] * len(next(iter(ice_data.values())).shape)
        sizes[time_dim] = self.n_inner_steps + 1
        # exogenous forcings are used as is
        forcing_data = {
            k: ice_data[k] for k in self._ice_forcing_exogenous_names
        }
        if ocean_ic is not None:
                # forcings from ocean are constant during the fast ice steps
            # NOTE: only n_ic_timesteps = 1 is currently supported
            assert next(iter(ocean_ic.values())).shape[self.ocean.TIME_DIM] == 1
            forcings_from_ocean = {
                k: ocean_ic[k].expand(*sizes)
                for k in self._ocean_to_ice_forcing_names
            }
            # get the SST mask (0 if land, 1 if sea surface)
            forcings_from_ocean = self._forcings_from_ocean_with_ocean_fraction(
                forcings_from_ocean, forcing_data
            )
            # update ice forcings
            forcing_data.update(forcings_from_ocean)
        if atmos_ic is not None:
            forcings_from_atmosphere = {
                k: (
                    torch.cat([torch.full_like(v, fill_value=np.nan), v], dim=time_dim)
                    if k in self._config.ice_next_step_forcing_names
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
        """Generate predictions for all coupling scenarios."""
        
        # Route to appropriate coupling implementation
        if self._config.atmosphere is None:  # ice-ocean coupling
            yield from self._ice_ocean_predictions(
                initial_condition, forcing_data, optimizer
            )
        elif self._config.ice is None:  # atmosphere-ocean coupling  
            yield from self._atmosphere_ocean_predictions(
                initial_condition, forcing_data, optimizer
            )
        elif self._config.ocean is None:  # atmosphere-ice coupling
            yield from self._atmosphere_ice_predictions(
                initial_condition, forcing_data, optimizer
            )
        else:  # fully coupled
            yield from self._fully_coupled_predictions(
                initial_condition, forcing_data, optimizer
            )

    def _process_prediction_generator_list(
        self,
        output_list: list[ComponentStepPrediction],
        forcing_data: CoupledBatchData,
    ) -> CoupledBatchData:
        """Process prediction generator output for all coupling scenarios."""
        
        # Process atmosphere data if present
        atmos_data = None
        if self.atmosphere is not None:
            atmos_data = process_prediction_generator_list(
                [x.data for x in output_list if x.realm == "atmosphere"],
                time=forcing_data.atmosphere_data.time[:, self.atmosphere.n_ic_timesteps :],
                horizontal_dims=forcing_data.atmosphere_data.horizontal_dims,
                labels=forcing_data.atmosphere_data.labels,
            )
        
        # Process ocean data if present  
        ocean_data = None
        if self.ocean is not None:
            ocean_data = process_prediction_generator_list(
                [x.data for x in output_list if x.realm == "ocean"],
                time=forcing_data.ocean_data.time[:, self.ocean.n_ic_timesteps :],
                horizontal_dims=forcing_data.ocean_data.horizontal_dims,
                labels=forcing_data.ocean_data.labels,
            )
        
        # Process ice data if present
        ice_data = None
        if self.ice is not None:
            ice_data = process_prediction_generator_list(
                [x.data for x in output_list if x.realm == "ice"],
                time=forcing_data.ice_data.time[:, self.ice.n_ic_timesteps :],
                horizontal_dims=forcing_data.ice_data.horizontal_dims,
                labels=forcing_data.ice_data.labels,
            )
                
        return CoupledBatchData(ocean_data=ocean_data,
                                atmosphere_data=atmos_data,
                                ice_data=ice_data)
    
    def _atmosphere_ocean_predictions(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        """Original SamudrACE atmosphere-ocean coupling implementation."""
        # Validate inputs
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
                    atmos_data=atmos_window.data,
                    ocean_ic=ocean_ic_state.as_batch_data().data,
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
            for i_inner in range(self.n_inner_steps):
                atmos_step_num = i_outer * self.n_inner_steps + i_inner
                atmos_step = next(atmos_generator)
                yield ComponentStepPrediction(
                    realm="atmosphere",
                    data=atmos_step,
                    step=atmos_step_num,
                )
                atmos_step = optimizer.detach_if_using_gradient_accumulation(atmos_step)
                atmos_steps.append(atmos_step)

            atmos_gen = stack_list_of_tensor_dicts(
                atmos_steps, self.atmosphere.TIME_DIM
            )
            atmos_data_forcings = atmos_window.select_time_slice(
                time_slice=slice(
                    self.atmosphere.n_ic_timesteps,
                    self.n_inner_steps + self.atmosphere.n_ic_timesteps,
                )
            )
            ocean_window = forcing_data.ocean_data.select_time_slice(
                slice(i_outer, i_outer + self.n_ic_timesteps + 1)
            )
            ocean_forcings = BatchData(
                data=self._get_ocean_forcings(
                    ocean_data=ocean_window.data,
                    atmos_gen=atmos_gen,
                    atmos_forcings=atmos_data_forcings.data,
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
            atmos_ic_data = {
                k: v.unsqueeze(self.atmosphere.TIME_DIM)
                for k, v in atmos_steps[-1].items()
            }
            atmos_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(atmos_ic_data),
                    time=atmos_window.time.isel(
                        time=slice(-self.atmosphere.n_ic_timesteps, None)
                    ),
                    labels=atmos_window.labels,
                )
            )
            ocean_ic_data = {
                k: v.unsqueeze(self.ocean.TIME_DIM) for k, v in ocean_step.items()
            }
            ocean_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(ocean_ic_data),
                    time=ocean_window.time.isel(time=slice(-self.n_ic_timesteps, None)),
                    labels=ocean_window.labels,
                )
            )
            
    def _ice_ocean_predictions(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        """Ice-ocean coupling implementation."""
        if (
            initial_condition.ice_data.as_batch_data().n_timesteps
            != self.ice.n_ic_timesteps
        ):
            raise ValueError(
                "Ice initial condition must have "
                f"{self.ice.n_ic_timesteps} timesteps, got "
                f"{initial_condition.ice_data.as_batch_data().n_timesteps}."
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
        ice_ic_state = initial_condition.ice_data
        ocean_ic_state = initial_condition.ocean_data

        n_outer_steps = forcing_data.ocean_data.n_timesteps - self.n_ic_timesteps

        for i_outer in range(n_outer_steps):
            # get the ice window for the initial coupled step
            ice_window = forcing_data.ice_data.select_time_slice(
                slice(
                    i_outer * self.n_inner_steps,
                    (i_outer + 1) * self.n_inner_steps + self.ice.n_ic_timesteps,
                )
            )
            ice_forcings = BatchData(
                data=self._get_ice_forcings(
                    ice_data=ice_window.data,
                    ocean_ic=ocean_ic_state.as_batch_data().data,
                ),
                time=ice_window.time,
                labels=ice_window.labels,
            )
            
            ice_generator = self.ice.get_prediction_generator(
                ice_ic_state,
                ice_forcings,
                self.n_inner_steps,
                optimizer,
            )
            ice_steps = []

            # predict and yield ice steps
            for i_inner, ice_step in enumerate(ice_generator):
                yield ComponentStepPrediction(
                    realm="ice",
                    data=ice_step,
                    step=(i_outer * self.n_inner_steps + i_inner),
                )
                ice_step = optimizer.detach_if_using_gradient_accumulation(ice_step)
                ice_steps.append(ice_step)

            ocean_window = forcing_data.ocean_data.select_time_slice(
                slice(i_outer, i_outer + self.n_ic_timesteps + 1)
            )
            ice_gen = stack_list_of_tensor_dicts(
                ice_steps, self.ice.TIME_DIM
            )

            ice_data_forcings = ice_window.select_time_slice(
                time_slice=slice(
                    self.ice.n_ic_timesteps,
                    self.n_inner_steps + self.ice.n_ic_timesteps,
                )
            )
            ocean_forcings = BatchData(
                data=self._get_ocean_forcings(
                    ocean_data=ocean_window.data,
                    ice_gen=ice_gen,
                    ice_forcings=ice_data_forcings.data
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
            ice_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(
                        {
                            k: v.unsqueeze(self.ice.TIME_DIM)
                            for k, v in ice_steps[-1].items()
                        }
                    ),
                    time=ice_window.time.isel(
                        time=slice(-self.ice.n_ic_timesteps, None)
                    ),
                    labels=ice_window.labels,
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
        
    def _atmosphere_ice_predictions(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        """Atmosphere-ice coupling implementation."""
        # Validate inputs
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
            initial_condition.ice_data.as_batch_data().n_timesteps
            != self.n_ic_timesteps
        ):
            raise ValueError(
                "Ice initial condition must have "
                f"{self.n_ic_timesteps} timesteps, got "
                f"{initial_condition.ice_data.as_batch_data().n_timesteps}."
            )
        atmos_ic_state = initial_condition.atmosphere_data
        ice_ic_state = initial_condition.ice_data

        n_outer_steps = forcing_data.ice_data.n_timesteps - self.n_ic_timesteps

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
                    atmos_data=atmos_window.data,
                    ice_ic=ice_ic_state.as_batch_data().data,
                ),
                time=atmos_window.time,
                labels=atmos_window.labels,
            )
            # prescribe the initial condition TS state  
            atmos_ic_state = self._prescribe_ic_ts(
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

            ice_window = forcing_data.ice_data.select_time_slice(
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
            ice_forcings = BatchData(
                data=self._get_ice_forcings(
                    ice_data=ice_window.data,
                    atmos_gen=atmos_gen,
                    atmos_forcings=atmos_data_forcings.data
                ),
                time=ice_window.time,
                labels=ice_window.labels,
            )
            # predict and yield a single ocean step
            ice_step = next(
                iter(
                    self.ice.get_prediction_generator(
                        ice_ic_state,
                        ice_forcings,
                        n_forward_steps=1,
                        optimizer=optimizer,
                    )
                )
            )
            yield ComponentStepPrediction(
                realm="ice",
                data=ice_step,
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
            ice_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(
                        {
                            k: v.unsqueeze(self.ice.TIME_DIM)
                            for k, v in ice_step.items()
                        }
                    ),
                    time=ice_window.time.isel(time=slice(-self.n_ic_timesteps, None)),
                    labels=ice_window.labels,
                )
            )
        
    def _fully_coupled_predictions(
        self,
        initial_condition: CoupledPrognosticState,
        forcing_data: CoupledBatchData,
        optimizer: OptimizationABC,
    ) -> Generator[ComponentStepPrediction, None, None]:
        """Fully coupled (atmosphere-ice-ocean) implementation."""
        # Validate inputs
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
            initial_condition.ice_data.as_batch_data().n_timesteps
            != self.ice.n_ic_timesteps
        ):
            raise ValueError(
                "Ice initial condition must have "
                f"{self.ice.n_ic_timesteps} timesteps, got "
                f"{initial_condition.ice_data.as_batch_data().n_timesteps}."
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
        ice_ic_state = initial_condition.ice_data
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
                    atmos_data=atmos_window.data,
                    ocean_ic=ocean_ic_state.as_batch_data().data,
                    ice_ic=ice_ic_state.as_batch_data().data,
                ),
                time=atmos_window.time,
                labels=atmos_window.labels,
            )
            # prescribe the initial condition surface temperatures (SST and TS)
            # First prescribe SST from ocean
            atmos_ic_state = self._prescribe_ic_sst(
                atmos_ic_state,
                atmos_forcings.select_time_slice(
                    slice(None, self.atmosphere.n_ic_timesteps)
                ),
            )
            # Then prescribe TS from ice
            atmos_ic_state = self._prescribe_ic_ts(
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

            # get the ice window for the initial coupled step
            ice_window = forcing_data.ice_data.select_time_slice(
                slice(
                    i_outer * self.n_inner_steps,
                    (i_outer + 1) * self.n_inner_steps + self.ice.n_ic_timesteps,
                )
            )
            ice_forcings = BatchData(
                data=self._get_ice_forcings(
                    ice_data=ice_window.data,
                    ocean_ic=ocean_ic_state.as_batch_data().data,
                ),
                time=ice_window.time,
                labels=ice_window.labels,
            )
            # prescribe the initial condition SST state
            ice_ic_state = self._prescribe_ic_sst(
                ice_ic_state,
                ice_forcings.select_time_slice(
                    slice(None, self.ice.n_ic_timesteps)
                ),
            )
            ice_generator = self.ice.get_prediction_generator(
                ice_ic_state,
                ice_forcings,
                self.n_inner_steps,
                optimizer,
            )
            ice_steps = []

            # predict and yield atmosphere steps
            for i_inner, atmos_step in enumerate(atmos_generator):
                yield ComponentStepPrediction(
                    realm="atmosphere",
                    data=atmos_step,
                    step=(i_outer * self.n_inner_steps + i_inner),
                )
                atmos_step = optimizer.detach_if_using_gradient_accumulation(atmos_step)
                atmos_steps.append(atmos_step)

                yield ComponentStepPrediction(
                    realm="ice",
                    data=ice_generator[i_inner],
                    step=(i_outer * self.n_inner_steps + i_inner),
                )
                ice_step = optimizer.detach_if_using_gradient_accumulation(ice_generator[i_inner])
                ice_steps.append(ice_step)

            ocean_window = forcing_data.ocean_data.select_time_slice(
                slice(i_outer, i_outer + self.n_ic_timesteps + 1)
            )
            atmos_gen = stack_list_of_tensor_dicts(
                atmos_steps, self.atmosphere.TIME_DIM
            )
            ice_gen = stack_list_of_tensor_dicts(
                ice_steps, self.ice.TIME_DIM
            )

            atmos_data_forcings = atmos_window.select_time_slice(
                time_slice=slice(
                    self.atmosphere.n_ic_timesteps,
                    self.n_inner_steps + self.atmosphere.n_ic_timesteps,
                )
            )
            ice_data_forcings = ice_window.select_time_slice(
                time_slice=slice(
                    self.ice.n_ic_timesteps,
                    self.n_inner_steps + self.ice.n_ic_timesteps,
                )
            )
            ocean_forcings = BatchData(
                data=self._get_ocean_forcings(
                    ocean_data=ocean_window.data,
                    atmos_gen=atmos_gen,
                    atmos_forcings=atmos_data_forcings.data,
                    ice_gen=ice_gen,
                    ice_forcings=ice_data_forcings.data,
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
            ice_ic_state = PrognosticState(
                BatchData(
                    data=optimizer.detach_if_using_gradient_accumulation(
                        {
                            k: v.unsqueeze(self.ice.TIME_DIM)
                            for k, v in ice_steps[-1].items()
                        }
                    ),
                    time=ice_window.time.isel(
                        time=slice(-self.ice.n_ic_timesteps, None)
                    ),
                    labels=ice_window.labels,
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
                ocean_func = None
                ocean_timesteps = None
                if self.ocean is not None:
                    ocean_func = self.ocean.derive_func
                    ocean_timesteps = self.ocean.n_ic_timesteps
                ice_func = None
                ice_timesteps = None
                if self.ice is not None:
                    ice_func = self.ice.derive_func
                    ice_timesteps = self.ice.n_ic_timesteps
                atmos_func = None
                atmos_timesteps = None
                if self.atmosphere is not None:
                    atmos_func = self.atmosphere.derive_func
                    atmos_timesteps = self.atmosphere.n_ic_timesteps
                gen_data = (
                    gen_data.prepend(initial_condition)
                    .compute_derived_variables(
                        ocean_derive_func=ocean_func,
                        ice_derive_func=ice_func,
                        atmosphere_derive_func=atmos_func,
                        forcing_data=forcing,
                    )
                    .remove_initial_condition(
                        n_ic_timesteps_ocean=ocean_timesteps,
                        n_ic_timesteps_ice=ice_timesteps,
                        n_ic_timesteps_atmosphere=atmos_timesteps
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
        
        # Get forward data for components that exist
        atmos_forward_data = None
        if self.atmosphere is not None:
            atmos_forward_data = self.atmosphere.get_forward_data(
                forcing.atmosphere_data, compute_derived_variables=compute_derived_variables
            )
            
        ocean_forward_data = None
        if self.ocean is not None:
            ocean_forward_data = self.ocean.get_forward_data(
                forcing.ocean_data, compute_derived_variables=compute_derived_variables
            )
            
        ice_forward_data = None
        if self.ice is not None:
            ice_forward_data = self.ice.get_forward_data(
                forcing.ice_data, compute_derived_variables=compute_derived_variables
            )
            
        # Create prognostic state for next step
        ocean_end_data = None
        if self.ocean is not None:
            ocean_end_data = gen_data.ocean_data.get_end(
                self.ocean.prognostic_names,
                self.n_ic_timesteps,
            )
            
        atmosphere_end_data = None
        if self.atmosphere is not None:
            atmosphere_end_data = gen_data.atmosphere_data.get_end(
                self.atmosphere.prognostic_names,
                self.atmosphere.n_ic_timesteps,
            )
            
        ice_end_data = None
        if self.ice is not None:
            ice_end_data = gen_data.ice_data.get_end(
                self.ice.prognostic_names,
                self.ice.n_ic_timesteps,
            )
            
        return (
            CoupledPairedData.from_coupled_batch_data(
                prediction=gen_data,
                reference=CoupledBatchData(
                    ocean_data=ocean_forward_data,
                    atmosphere_data=atmos_forward_data,
                    ice_data=ice_forward_data,
                ),
            ),
            CoupledPrognosticState(
                ocean_data=ocean_end_data,
                atmosphere_data=atmosphere_end_data,
                ice_data=ice_end_data,
            ),
        )

    def predict(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledBatchData, CoupledPrognosticState]:
        gen_data = self._predict(initial_condition, forcing, compute_derived_variables)
        
        # Create prognostic state for next step
        ocean_end_data = None
        ocean_data = None
        if self.ocean is not None:
            ocean_end_data = gen_data.ocean_data.get_end(
                self.ocean.prognostic_names,
                self.n_ic_timesteps,
            )
            ocean_data = gen_data.ocean_data
            
        atmosphere_end_data = None
        atmosphere_data = None
        if self.atmosphere is not None:
            atmosphere_end_data = gen_data.atmosphere_data.get_end(
                self.atmosphere.prognostic_names,
                self.atmosphere.n_ic_timesteps,
            )
            atmosphere_data = gen_data.atmosphere_data
            
        ice_end_data = None
        ice_data = None
        if self.ice is not None:
            ice_end_data = gen_data.ice_data.get_end(
                self.ice.prognostic_names,
                self.ice.n_ic_timesteps,
            )
            ice_data = gen_data.ice_data
        
        return (
            CoupledBatchData(
                ocean_data=ocean_data,
                atmosphere_data=atmosphere_data,
                ice_data=ice_data
            ),
            CoupledPrognosticState(
                ocean_data=ocean_end_data,
                atmosphere_data=atmosphere_end_data,  
                ice_data=ice_end_data,
            ),
        )

    def update_training_history(self, training_job: TrainingJob) -> None:
        """
        Update the stepper's history of training jobs.

        Args:
            training_job: The training job to add to the history.
        """
        if self.ocean is not None:
            self.ocean.update_training_history(training_job)
        if self.atmosphere is not None:
            self.atmosphere.update_training_history(training_job)
        if self.ice is not None:
            self.ice.update_training_history(training_job)

    @classmethod
    def from_state(cls, state) -> "CoupledStepper":
        ocean = None
        ice = None
        atmosphere = None
        
        if "ocean_state" in state:
            ocean = Stepper.from_state(state["ocean_state"])
        if "ice_state" in state:
            ice = Stepper.from_state(state["ice_state"])
        if "atmosphere_state" in state:
            atmosphere = Stepper.from_state(state["atmosphere_state"])
            
        config = CoupledStepperConfig.from_state(state["config"])
        if "dataset_info" in state:
            dataset_info = CoupledDatasetInfo.from_state(state["dataset_info"])
        else:
            # NOTE: this is included for backwards compatibility
            if ocean is not None:
                ocean_info = ocean.training_dataset_info
            else:
                ocean_info = None
            if ice is not None:
                ice_info = ice.training_dataset_info
            else:
                ice_info = None
            if atmosphere is not None:
                atmosphere_info = atmosphere.training_dataset_info
            else:
                atmosphere_info = None
            dataset_info = CoupledDatasetInfo(
                ocean=ocean_info,
                ice=ice_info,
                atmosphere=atmosphere_info,
            )
        return cls(
            config=config,
            ocean=ocean,
            ice=ice,
            atmosphere=atmosphere,
            dataset_info=dataset_info,
        )


class ComponentEnsembleStepPrediction(StepPredictionABC):
    """Like ComponentStepPrediction but with an explicit ensemble dimension."""

    def __init__(
        self,
        realm: Literal["ocean", "atmosphere"],
        data: EnsembleTensorDict,
        step: int,
    ):
        self._realm: Literal["ocean", "atmosphere"] = realm
        self._data = data
        self._step = step

    @property
    def realm(self) -> Literal["ocean", "atmosphere"]:
        return self._realm

    @property
    def data(self) -> EnsembleTensorDict:
        return self._data

    @property
    def step(self) -> int:
        return self._step

    def detach_if_using_gradient_accumulation(
        self, optimizer: OptimizationABC
    ) -> "ComponentEnsembleStepPrediction":
        """Eagerly detach the data tensor map from the computational graph
        if already consumed in backprop, i.e. when using gradient accumulation.

        """
        return ComponentEnsembleStepPrediction(
            realm=self.realm,
            data=EnsembleTensorDict(
                optimizer.detach_if_using_gradient_accumulation(self.data)
            ),
            step=self.step,
        )

    def detach(self) -> "ComponentEnsembleStepPrediction":
        """Detach the data tensor map from the computation graph. Should only be
        called after backprop has finished.

        """
        return ComponentEnsembleStepPrediction(
            realm=self.realm,
            data=EnsembleTensorDict({k: v.detach() for k, v in self.data.items()}),
            step=self.step,
        )


def _process_ensemble_output_list(
    output_list: list[ComponentEnsembleStepPrediction],
) -> tuple[EnsembleTensorDict, EnsembleTensorDict]:
    """Separate and detach generated component outputs. Should be called only
    after OptimizationABC.step_weights().

    Returns:
        A tuple of (ocean_gen_data, atmos_gen_data, ice_gen_data) with explicit
        ensemble dimension.

    """
    atmos_gen_data = [x.detach().data for x in output_list if x.realm == "atmosphere"]
    ocean_gen_data = [x.detach().data for x in output_list if x.realm == "ocean"]
    ice_gen_data = [x.detach().data for x in output_list if x.realm == "ice"]
    if len(atmos_gen_data) > 0:
        atmos_gen_data = process_ensemble_prediction_generator_list(atmos_gen_data)
    if len(ocean_gen_data) > 0:
        ocean_gen_data = process_ensemble_prediction_generator_list(ocean_gen_data)
    if len(ice_gen_data) > 0:
        ice_gen_data = process_ensemble_prediction_generator_list(ice_gen_data)
    
    return ocean_gen_data, atmos_gen_data, ice_gen_data


class CoupledStepperTrainLoss:
    def __init__(
        self,
        ocean_loss: StepLossABC | None = None,
        ice_loss: StepLossABC | None = None,
        atmosphere_loss: StepLossABC | None = None,
    ):
        self._loss_objs = {}
        if ocean_loss is not None:
            self._loss_objs["ocean"] = ocean_loss
        if ice_loss is not None:
            self._loss_objs["ice"] = ice_loss
        if atmosphere_loss is not None:
            self._loss_objs["atmosphere"] = atmosphere_loss

    @property
    def effective_loss_scaling(self) -> CoupledTensorMapping:
        ocean_scaling = None
        if "ocean" in self._loss_objs:
            ocean_scaling = self._loss_objs["ocean"].effective_loss_scaling
        ice_scaling = None
        if "ice" in self._loss_objs:
            ice_scaling = self._loss_objs["ice"].effective_loss_scaling
        atmosphere_scaling = None
        if "atmosphere" in self._loss_objs:
            atmosphere_scaling = self._loss_objs["atmosphere"].effective_loss_scaling

        return CoupledTensorMapping(
            ocean=ocean_scaling,
            ice=ice_scaling,
            atmosphere=atmosphere_scaling,
        )

    def sample_n_steps(self) -> None:
        for loss_obj in self._loss_objs.values():
            loss_obj.sample_n_steps()

    def n_required_outer_steps(self, n_inner_steps: int) -> int:
        """Minimum number of outer (ocean) steps needed so that every
        component step contributing to the current batch's loss is computed.

        Callers must invoke ``sample_n_steps()`` beforehand for stochastic
        configs so the value reflects the current batch.
        """
        if "atmosphere" not in self._loss_objs:
            ice_required = self._loss_objs["ice"].n_required_forward_steps()
            ocean_required = self._loss_objs["ocean"].n_required_forward_steps()
            ice_outer = -(-ice_required // n_inner_steps)  # ceil division
            return max(ocean_required, ice_outer)
        elif "ice" not in self._loss_objs:
            ocean_required = self._loss_objs["ocean"].n_required_forward_steps()
            atmos_required = self._loss_objs["atmosphere"].n_required_forward_steps()
            atmos_outer = -(-atmos_required // n_inner_steps)  # ceil division
            return max(ocean_required, atmos_outer)
        elif "ocean" not in self._loss_objs:
            ice_required = self._loss_objs["ice"].n_required_forward_steps()
            atmos_required = self._loss_objs["atmosphere"].n_required_forward_steps()
            atmos_outer = -(-atmos_required // n_inner_steps)  # ceil division
            return max(ice_required, atmos_outer)
        else:
            ocean_required = self._loss_objs["ocean"].n_required_forward_steps()
            ice_required = self._loss_objs["ice"].n_required_forward_steps()
            atmos_required = self._loss_objs["atmosphere"].n_required_forward_steps()
            ice_outer = -(-ice_required // n_inner_steps)  # ceil division
            atmos_outer = -(-atmos_required // n_inner_steps)  # ceil division
            return max(ocean_required, ice_outer, atmos_outer)

    def step_is_optimized(
        self,
        realm: Literal["ocean", "atmosphere"],
        step: int,
    ) -> bool:
        return self._loss_objs[realm].step_is_optimized(step)

    def __call__(
        self,
        prediction: ComponentEnsembleStepPrediction,
        target_data: TensorMapping,
    ) -> torch.Tensor | None:
        loss_obj = self._loss_objs[prediction.realm]
        if loss_obj.step_is_optimized(prediction.step):
            return loss_obj(prediction, target_data)
        return None


@dataclasses.dataclass
class ComponentTrainingConfig:
    loss: StepLossConfig
    loss_contributions: LossContributionsConfig = dataclasses.field(
        default_factory=lambda: LossContributionsConfig()
    )
    parameter_init: ParameterInitializationConfig = dataclasses.field(
        default_factory=lambda: ParameterInitializationConfig()
    )


@dataclasses.dataclass
class CoupledTrainStepperConfig:
    """Configuration for training-specific aspects of a coupled stepper.

    Parameters:
        n_coupled_steps: Number of forward coupled steps in the optimization.
        ocean: The configuration for the ocean component.
        ice: The configuration for the ice component.
        atmosphere: The configuration for the atmosphere component.
        n_ensemble: The number of ensemble members evaluated for each training
            batch member. Default is 2 if ocean or atmopshere loss type is
            EnsembleLoss, otherwise the default is 1. Must be 2 for EnsembleLoss
            to be valid.
        parameter_init: The coupled parameter initialization configuration for
            fine-tuning a previously-trained coupled stepper.
    """

    n_coupled_steps: int
    ocean: ComponentTrainingConfig | None = None
    ice: ComponentTrainingConfig | None = None
    atmosphere: ComponentTrainingConfig | None = None
    n_ensemble: int = -1  # sentinel value to avoid None typing of attribute
    parameter_init: CoupledParameterInitConfig = dataclasses.field(
        default_factory=lambda: CoupledParameterInitConfig()
    )

    def __post_init__(self):
        """Validate that parameter_init is not specified in conflicting ways.

        Raises ValueError if CoupledParameterInitConfig.checkpoint_path is set
        alongside component-level weights_path values.
        """
        if self.parameter_init.checkpoint_path is not None:
            atmos_weights = None
            ocn_weights = None
            ice_weights = None
            if self.atmosphere is not None:
                atmos_weights = self.atmosphere.parameter_init.weights_path
            if self.ocean is not None:
                ocn_weights = self.ocean.parameter_init.weights_path
            if self.ice is not None:
                ice_weights = self.ice.parameter_init.weights_path
            if (
                atmos_weights is not None
                or ocn_weights is not None
                or ice_weights is not None
            ):
                raise ValueError(
                    "Please specify CoupledParameterInitConfig.checkpoint_path "
                    "or the component training configs' "
                    "ParameterInitializationConfig.weights_path, but not both."
                )
        if (
            self.ocean.loss_contributions.is_null
            and self.atmosphere.loss_contributions.is_null
        ):
            raise ValueError(
                "At least one of ocean or atmosphere loss_contributions must be "
                "non-null (non-zero weight and non-zero n_steps)."
            )
        if self.n_ensemble == -1:
            if self.atmosphere is None:
                use_ensemble_loss = "EnsembleLoss" in (
                    self.ocean.loss.type,
                    self.ice.loss.type,
                )
            elif self.ice is None:
                use_ensemble_loss = "EnsembleLoss" in (
                    self.ocean.loss.type,
                    self.atmosphere.loss.type,
                )
            elif self.ocean is None:
                use_ensemble_loss = "EnsembleLoss" in (
                    self.ice.loss.type,
                    self.atmosphere.loss.type,
                )
            else:
                use_ensemble_loss = "EnsembleLoss" in (
                    self.ocean.loss.type,
                    self.ice.loss.type,
                    self.atmosphere.loss.type,
                )
            if use_ensemble_loss:
                self.n_ensemble = 2
            else:
                self.n_ensemble = 1

    @property
    def component_n_steps_max(self) -> CoupledOptionalInt:
        """Per-component upper bound on optimized loss steps, or ``None`` if
        unbounded. Used by ``TrainConfig`` to validate compatibility with
        ``CoupledStepperConfig.n_inner_steps`` and ``self.n_coupled_steps``.
        """
        atmos_nsteps = None
        ocean_nsteps = None
        ice_nsteps = None
        if self.atmosphere is not None:
            atmos_nsteps = self.atmosphere.loss_contributions.n_steps_max
        if self.ocean is not None:
            ocean_nsteps = self.ocean.loss_contributions.n_steps_max
        if self.ice is not None:
            ice_nsteps = self.ice.loss_contributions.n_steps_max
        return CoupledOptionalInt(
            ocean=ocean_nsteps,
            atmosphere=atmos_nsteps,
            ice=ice_nsteps,
        )

    def _build_loss(
        self, stepper: CoupledStepper, n_coupled_steps: int
    ) -> CoupledStepperTrainLoss:
        ocean_loss = None
        if stepper.ocean is not None:
            max_n_steps = n_coupled_steps
            ocean_step_loss = stepper.ocean.build_loss(self.ocean.loss)
            ocean_loss = self.ocean.loss_contributions.build(
                ocean_step_loss, stepper.ocean.TIME_DIM, n_steps_limit=max_n_steps
            )
        ice_loss = None
        if stepper.ice is not None:
            max_n_steps = n_coupled_steps * stepper.n_inner_steps
            ice_step_loss = stepper.ice.build_loss(self.ice.loss)
            ice_loss = self.ice.loss_contributions.build(
                ice_step_loss, stepper.ice.TIME_DIM, n_steps_limit=max_n_steps
            )
        atmos_loss = None
        if stepper.atmosphere is not None:
            max_n_steps = n_coupled_steps * stepper.n_inner_steps
            atmos_step_loss = stepper.atmosphere.build_loss(self.atmosphere.loss)
            atmos_loss = self.atmosphere.loss_contributions.build(
                atmos_step_loss, stepper.atmosphere.TIME_DIM, n_steps_limit=max_n_steps
            )
        return CoupledStepperTrainLoss(ocean_loss, ice_loss, atmos_loss)

    def get_train_stepper(
        self,
        stepper_config: CoupledStepperConfig,
        dataset_info: CoupledDatasetInfo,
    ) -> "CoupledTrainStepper":
        """
        Build a CoupledTrainStepper from this configuration.

        Args:
            stepper_config: The CoupledStepper configuration.
            dataset_info: Information about the coupled training datasets.

        Returns:
            A CoupledTrainStepper wrapping the given or built stepper with
            training functionality.
        """
        loaders = self.parameter_init.build_weights_and_history_loaders()
        ocean_initializer = None
        ice_initializer = None
        atmosphere_initializer = None
        if self.atmosphere is None:
            ocean_initializer = self.ocean.parameter_init.build(
                load_weights_and_history=loaders.ocean,
            )
            ice_initializer = self.ice.parameter_init.build(
                load_weights_and_history=loaders.ice,
            )
        elif self.ice is None:
            ocean_initializer = self.ocean.parameter_init.build(
                load_weights_and_history=loaders.ocean,
            )
            atmosphere_initializer = self.atmosphere.parameter_init.build(
                load_weights_and_history=loaders.atmosphere,
            )
        elif self.ocean is None:
            ice_initializer = self.ice.parameter_init.build(
                load_weights_and_history=loaders.ice,
            )
            atmosphere_initializer = self.atmosphere.parameter_init.build(
                load_weights_and_history=loaders.atmosphere,
            )
        else:
            ocean_initializer = self.ocean.parameter_init.build(
                load_weights_and_history=loaders.ocean,
            )
            ice_initializer = self.ice.parameter_init.build(
                load_weights_and_history=loaders.ice,
            )
            atmosphere_initializer = self.atmosphere.parameter_init.build(
                load_weights_and_history=loaders.atmosphere,
            )
        stepper = stepper_config.get_stepper(
            dataset_info=dataset_info,
            ocean_parameter_initializer=ocean_initializer,
            ice_parameter_initializer=ice_initializer,
            atmosphere_parameter_initializer=atmosphere_initializer,
        )
        return CoupledTrainStepper(
            stepper=stepper,
            config=self,
        )


class CoupledTrainStepper(
    TrainStepperABC[
        CoupledPrognosticState,
        CoupledBatchData,
        CoupledBatchData,
        CoupledPairedData,
        CoupledTrainOutput,
    ],
):
    """Wrapper around CoupledStepper that adds training functionality.

    This class composes a CoupledStepper (for inference) with training-specific
    loss configuration and implements the train_on_batch method.

    Stochastic training assumptions (n_ensemble > 1):
        Ensemble training broadcasts each batch member into n_ensemble copies
        along the batch dimension; the copies receive identical inputs, so
        divergent outputs depend entirely on model stochasticity (e.g., noise
        conditioning).

        We currently assume that the atmosphere component is stochastic. If it
        is not, the first n_inner_steps atmosphere steps (before the first ocean
        step) will produce identical outputs across ensemble members, doubling
        compute with no benefit.

        On the other hand, even when the ocean component itself is deterministic,
        its optimization is effectively stochastic so long as the atmosphere is
        stochastic: the ocean receives atmosphere-averaged forcings that differ
        across ensemble members.

    """

    def __init__(
        self,
        stepper: CoupledStepper,
        config: CoupledTrainStepperConfig,
    ):
        """
        Args:
            stepper: The underlying coupled stepper for inference operations.
            config: The train stepper config.
        """
        self._stepper = stepper
        self._config = config
        self._loss = self._config._build_loss(stepper, config.n_coupled_steps)

    @property
    def ocean(self) -> Stepper:
        return self._stepper.ocean
    
    @property
    def ice(self) -> Stepper:
        return self._stepper.ice

    @property
    def atmosphere(self) -> Stepper:
        return self._stepper.atmosphere

    @property
    def effective_loss_scaling(self) -> CoupledTensorMapping:
        return self._loss.effective_loss_scaling

    @property
    def modules(self) -> nn.ModuleList:
        return self._stepper.modules

    @property
    def n_ic_timesteps(self) -> int:
        return self._stepper.n_ic_timesteps

    @property
    def n_inner_steps(self) -> int:
        """Number of atmosphere steps per ocean step."""
        return self._stepper.n_inner_steps

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> tuple[CoupledPairedData, CoupledPrognosticState]:
        return self._stepper.predict_paired(
            initial_condition, forcing, compute_derived_variables
        )

    def set_train(self):
        self._stepper.set_train()

    def set_eval(self):
        self._stepper.set_eval()

    def get_state(self) -> dict[str, Any]:
        return self._stepper.get_state()

    def load_state(self, state: dict[str, Any]):
        self._stepper.load_state(state)

    def update_training_history(self, training_job: TrainingJob) -> None:
        self._stepper.update_training_history(training_job)

    def _accumulate_step_loss(
        self,
        gen_step: ComponentStepPrediction,
        forward_data: TensorMapping,
        time_dim: int,
        n_ensemble: int,
        optimization: OptimizationABC,
        metrics: ComponentStepMetrics,
        output_list: list[ComponentEnsembleStepPrediction],
    ) -> None:
        target_step = {
            k: v.select(time_dim, gen_step.step) for k, v in forward_data.items()
        }
        ensemble_step = ComponentEnsembleStepPrediction(
            realm=gen_step.realm,
            data=unfold_ensemble_dim(gen_step.data, n_ensemble),
            step=gen_step.step,
        )
        target_step_ensemble = add_ensemble_dim(target_step)
        step_loss = self._loss(ensemble_step, target_step_ensemble)
        if step_loss is not None:
            label = f"loss/{gen_step.realm}_step_{gen_step.step}"
            metrics.add_metric(label, step_loss.detach(), gen_step.realm)
            optimization.accumulate_loss(step_loss)
        output_list.append(
            ensemble_step.detach_if_using_gradient_accumulation(
                optimization
            )  # eagerly detach
        )

    def _accumulate_loss(
        self,
        data: CoupledBatchData,
        optimization: OptimizationABC,
        metrics: ComponentStepMetrics,
        ocean_forward_data: BatchData | None = None,
        ice_forward_data: BatchData | None = None,
        atmos_forward_data: BatchData | None = None,
    ) -> list[ComponentEnsembleStepPrediction]:
        n_ensemble = self._config.n_ensemble
        if atmos_forward_data is None:
            data_ensemble = CoupledBatchData(
                ocean_data=data.ocean_data.broadcast_ensemble(n_ensemble),
                ice_data=data.ice_data.broadcast_ensemble(n_ensemble),
            )
            # get initial condition prognostic variables
            input_data = CoupledPrognosticState(
                ice_data=data_ensemble.ice_data.get_start(
                    self.ice.prognostic_names, self.n_ic_timesteps
                ),
                ocean_data=data_ensemble.ocean_data.get_start(
                    self.ocean.prognostic_names, self.n_ic_timesteps
                ),
            )
            output_generator = self._stepper.get_prediction_generator(
                input_data,
                data_ensemble,
                optimization,
            )
            output_iterator = iter(output_generator)
            output_list: list[ComponentEnsembleStepPrediction] = []
            n_outer_steps_data = data.ocean_data.n_timesteps - self.n_ic_timesteps
            # Clamp to at least 1 outer step so downstream gen_data is non-empty
            # in the rare case where stochastic samplers yield n_steps=0 for both
            # realms; that batch contributes zero loss but a valid TrainOutput.
            n_outer_steps = min(
                n_outer_steps_data,
                max(1, self._loss.n_required_outer_steps(self.n_inner_steps)),
            )
            for i_outer in range(n_outer_steps):
                for i_inner in range(self.n_inner_steps):
                    global_ice_step = i_outer * self.n_inner_steps + i_inner
                    optimize = self._loss.step_is_optimized(
                        "ice",
                        global_ice_step,
                    )
                    grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                    with grad_context:
                        gen_step = next(output_iterator)
                        assert (
                            gen_step.realm == "ice"
                            and gen_step.step == global_ice_step
                        )
                        self._accumulate_step_loss(
                            gen_step=gen_step,
                            forward_data=ice_forward_data.data,
                            time_dim=self.ice.TIME_DIM,
                            n_ensemble=n_ensemble,
                            optimization=optimization,
                            metrics=metrics,
                            output_list=output_list,
                        )
                optimize = self._loss.step_is_optimized("ocean", i_outer)
                grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                with grad_context:
                    gen_step = next(output_iterator)
                    assert gen_step.realm == "ocean" and gen_step.step == i_outer
                    self._accumulate_step_loss(
                        gen_step=gen_step,
                        forward_data=ocean_forward_data.data,
                        time_dim=self.ocean.TIME_DIM,
                        n_ensemble=n_ensemble,
                        optimization=optimization,
                        metrics=metrics,
                        output_list=output_list,
                    )
        elif ice_forward_data is None:
            data_ensemble = CoupledBatchData(
                ocean_data=data.ocean_data.broadcast_ensemble(n_ensemble),
                atmosphere_data=data.atmosphere_data.broadcast_ensemble(n_ensemble),
            )
            # get initial condition prognostic variables
            input_data = CoupledPrognosticState(
                atmosphere_data=data_ensemble.atmosphere_data.get_start(
                    self.atmosphere.prognostic_names, self.n_ic_timesteps
                ),
                ocean_data=data_ensemble.ocean_data.get_start(
                    self.ocean.prognostic_names, self.n_ic_timesteps
                ),
            )
            output_generator = self._stepper.get_prediction_generator(
                input_data,
                data_ensemble,
                optimization,
            )
            output_iterator = iter(output_generator)
            output_list: list[ComponentEnsembleStepPrediction] = []
            n_outer_steps_data = data.ocean_data.n_timesteps - self.n_ic_timesteps
            n_outer_steps = min(
                n_outer_steps_data,
                max(1, self._loss.n_required_outer_steps(self.n_inner_steps)),
            )
            for i_outer in range(n_outer_steps):
                for i_inner in range(self.n_inner_steps):
                    global_atmos_step = i_outer * self.n_inner_steps + i_inner
                    optimize = self._loss.step_is_optimized(
                        "atmosphere",
                        global_atmos_step,
                    )
                    grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                    with grad_context:
                        gen_step = next(output_iterator)
                        assert (
                            gen_step.realm == "atmosphere"
                            and gen_step.step == global_atmos_step
                        )
                        self._accumulate_step_loss(
                            gen_step=gen_step,
                            forward_data=atmos_forward_data.data,
                            time_dim=self.atmosphere.TIME_DIM,
                            n_ensemble=n_ensemble,
                            optimization=optimization,
                            metrics=metrics,
                            output_list=output_list,
                        )
                optimize = self._loss.step_is_optimized("ocean", i_outer)
                grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                with grad_context:
                    gen_step = next(output_iterator)
                    assert gen_step.realm == "ocean" and gen_step.step == i_outer
                    self._accumulate_step_loss(
                        gen_step=gen_step,
                        forward_data=ocean_forward_data.data,
                        time_dim=self.ocean.TIME_DIM,
                        n_ensemble=n_ensemble,
                        optimization=optimization,
                        metrics=metrics,
                        output_list=output_list,
                    )
        elif ocean_forward_data is None:
            data_ensemble = CoupledBatchData(
                ice_data=data.ice_data.broadcast_ensemble(n_ensemble),
                atmosphere_data=data.atmosphere_data.broadcast_ensemble(n_ensemble),
            )
            # get initial condition prognostic variables
            input_data = CoupledPrognosticState(
                atmosphere_data=data_ensemble.atmosphere_data.get_start(
                    self.atmosphere.prognostic_names, self.n_ic_timesteps
                ),
                ice_data=data_ensemble.ice_data.get_start(
                    self.ice.prognostic_names, self.n_ic_timesteps
                ),
            )
            output_generator = self._stepper.get_prediction_generator(
                input_data,
                data_ensemble,
                optimization,
            )
            output_iterator = iter(output_generator)
            output_list: list[ComponentEnsembleStepPrediction] = []
            n_outer_steps_data = data.ice_data.n_timesteps - self.n_ic_timesteps
            n_outer_steps = min(
                n_outer_steps_data,
                max(1, self._loss.n_required_outer_steps(self.n_inner_steps)),
            )
            for i_outer in range(n_outer_steps):
                for i_inner in range(self.n_inner_steps):
                    global_atmos_step = i_outer * self.n_inner_steps + i_inner
                    optimize = self._loss.step_is_optimized(
                        "atmosphere",
                        global_atmos_step,
                    )
                    grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                    with grad_context:
                        gen_step = next(output_iterator)
                        assert (
                            gen_step.realm == "atmosphere"
                            and gen_step.step == global_atmos_step
                        )
                        self._accumulate_step_loss(
                            gen_step=gen_step,
                            forward_data=atmos_forward_data.data,
                            time_dim=self.atmosphere.TIME_DIM,
                            n_ensemble=n_ensemble,
                            optimization=optimization,
                            metrics=metrics,
                            output_list=output_list,
                        )
                optimize = self._loss.step_is_optimized("ice", i_outer)
                grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                with grad_context:
                    gen_step = next(output_iterator)
                    assert gen_step.realm == "ice" and gen_step.step == i_outer
                    self._accumulate_step_loss(
                        gen_step=gen_step,
                        forward_data=ice_forward_data.data,
                        time_dim=self.ice.TIME_DIM,
                        n_ensemble=n_ensemble,
                        optimization=optimization,
                        metrics=metrics,
                        output_list=output_list,
                    )
        else:
            data_ensemble = CoupledBatchData(
                ocean_data=data.ocean_data.broadcast_ensemble(n_ensemble),
                ice_data=data.ice_data.broadcast_ensemble(n_ensemble),
                atmosphere_data=data.atmosphere_data.broadcast_ensemble(n_ensemble),
            )
            # get initial condition prognostic variables
            input_data = CoupledPrognosticState(
                atmosphere_data=data_ensemble.atmosphere_data.get_start(
                    self.atmosphere.prognostic_names, self.n_ic_timesteps
                ),
                ocean_data=data_ensemble.ocean_data.get_start(
                    self.ocean.prognostic_names, self.n_ic_timesteps
                ),
                ice_data=data_ensemble.ice_data.get_start(
                    self.ice.prognostic_names, self.n_ic_timesteps
                ),
            )
            output_generator = self._stepper.get_prediction_generator(
                input_data,
                data_ensemble,
                optimization,
            )
            output_iterator = iter(output_generator)
            output_list: list[ComponentEnsembleStepPrediction] = []
            n_outer_steps_data = data.ocean_data.n_timesteps - self.n_ic_timesteps
            n_outer_steps = min(
                n_outer_steps_data,
                max(1, self._loss.n_required_outer_steps(self.n_inner_steps)),
            )
            for i_outer in range(n_outer_steps):
                for i_inner in range(self.n_inner_steps):
                    global_atmos_step = i_outer * self.n_inner_steps + i_inner
                    optimize = self._loss.step_is_optimized(
                        "atmosphere",
                        global_atmos_step,
                    )
                    grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                    with grad_context:
                        gen_step = next(output_iterator)
                        assert (
                            gen_step.realm == "atmosphere"
                            and gen_step.step == global_atmos_step
                        )
                        self._accumulate_step_loss(
                            gen_step=gen_step,
                            forward_data=atmos_forward_data.data,
                            time_dim=self.atmosphere.TIME_DIM,
                            n_ensemble=n_ensemble,
                            optimization=optimization,
                            metrics=metrics,
                            output_list=output_list,
                        )
                    global_ice_step = i_outer * self.n_inner_steps + i_inner
                    optimize = self._loss.step_is_optimized(
                        "ice",
                        global_ice_step,
                    )
                    grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                    with grad_context:
                        gen_step = next(output_iterator)
                        assert (
                            gen_step.realm == "ice"
                            and gen_step.step == global_ice_step
                        )
                        self._accumulate_step_loss(
                            gen_step=gen_step,
                            forward_data=ice_forward_data.data,
                            time_dim=self.ice.TIME_DIM,
                            n_ensemble=n_ensemble,
                            optimization=optimization,
                            metrics=metrics,
                            output_list=output_list,
                        )
                optimize = self._loss.step_is_optimized("ocean", i_outer)
                grad_context = contextlib.nullcontext() if optimize else torch.no_grad()
                with grad_context:
                    gen_step = next(output_iterator)
                    assert gen_step.realm == "ocean" and gen_step.step == i_outer
                    self._accumulate_step_loss(
                        gen_step=gen_step,
                        forward_data=ocean_forward_data.data,
                        time_dim=self.ocean.TIME_DIM,
                        n_ensemble=n_ensemble,
                        optimization=optimization,
                        metrics=metrics,
                        output_list=output_list,
                    )

        return output_list

    def train_on_batch(
        self,
        data: CoupledBatchData,
        optimization: OptimizationABC,
        compute_derived_variables: bool = False,
    ) -> CoupledTrainOutput:
        """
        Args:
            data: The coupled batch data, consisting of separate batches for ocean and
                atmosphere with the same initial condition time.
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            compute_derived_variables: Whether to compute derived variables for the
                prediction and target atmosphere data.

        """
        atmos_forward_data = None
        if self.atmosphere is not None:
            atmos_forward_data = self.atmosphere.get_forward_data(
                data.atmosphere_data,
                compute_derived_variables=False,
            )
        ocean_forward_data = None
        if self.ocean is not None:
            ocean_forward_data = self.ocean.get_forward_data(
                data.ocean_data,
                compute_derived_variables=False,
            )
        ice_forward_data = None
        if self.ice is not None:
            ice_forward_data = self.ice.get_forward_data(
                data.ice_data,
                compute_derived_variables=False,
            )

        metrics = ComponentStepMetrics()
        self._loss.sample_n_steps()
        optimization.set_mode(self.modules)
        with optimization.autocast():
            output_list = self._accumulate_loss(
                data,
                optimization,
                metrics,
                ocean_forward_data=ocean_forward_data,
                atmos_forward_data=atmos_forward_data,
                ice_forward_data=ice_forward_data, 
            )

        loss = optimization.get_accumulated_loss().detach()
        optimization.step_weights()

        ocean_gen_data, atmos_gen_data, ice_gen_data = _process_ensemble_output_list(
                output_list
            )  
        
        atmos_stepped = None
        atmos_ic = None
        if atmos_forward_data is not None:
            atmos_stepped = TrainOutput(
                metrics=metrics.get_atmosphere_metrics(),
                gen_data=atmos_gen_data,
                target_data=add_ensemble_dim(dict(atmos_forward_data.data)),
                time=atmos_forward_data.time,
                normalize=self.atmosphere.normalizer.normalize,
                derive_func=self.atmosphere.derive_func,
            )
            atmos_data = data.atmosphere_data
            # TODO: different n_ic_timesteps for atmosphere?
            atmos_ic = atmos_data.get_start(
                set(atmos_data.data.keys()), self.n_ic_timesteps
            )
        ocean_stepped = None
        ocean_ic = None
        if ocean_forward_data is not None:
            ocean_stepped = TrainOutput(
                metrics=metrics.get_ocean_metrics(),
                gen_data=ocean_gen_data,
                target_data=add_ensemble_dim(dict(ocean_forward_data.data)),
                time=ocean_forward_data.time,
                normalize=self.ocean.normalizer.normalize,
                derive_func=self.ocean.derive_func,
            )
            ocean_data = data.ocean_data
            ocean_ic = ocean_data.get_start(
                set(ocean_data.data.keys()), self.n_ic_timesteps
            )
        ice_stepped = None
        ice_ic = None
        if ice_forward_data is not None:
            ice_stepped = TrainOutput(
                metrics=metrics.get_ice_metrics(),
                gen_data=ice_gen_data,
                target_data=add_ensemble_dim(dict(ice_forward_data.data)),
                time=ice_forward_data.time,
                normalize=self.ice.normalizer.normalize,
                derive_func=self.ice.derive_func,
            )
            ice_data = data.ice_data
            ice_ic = ice_data.get_start(
                set(ice_data.data.keys()), self.n_ic_timesteps
            )
        
        stepped = CoupledTrainOutput(
            total_metrics={"loss": loss},
            ocean=ocean_stepped,
            ice=ice_stepped,
            atmosphere=atmos_stepped,
        )

        # prepend initial conditions
        ic = CoupledPrognosticState(ocean_data=ocean_ic,
                                    ice_data=ice_ic,
                                    atmosphere_data=atmos_ic)
        stepped = stepped.prepend_initial_condition(ic)

        if compute_derived_variables:
            stepped = stepped.compute_derived_variables()

        return stepped


def load_coupled_stepper(checkpoint_path: str | pathlib.Path) -> CoupledStepper:
    logging.info(f"Loading trained coupled model checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    stepper = CoupledStepper.from_state(checkpoint["stepper"])

    return stepper
