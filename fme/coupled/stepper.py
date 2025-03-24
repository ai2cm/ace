import dataclasses
import datetime
import logging
from typing import Any, Dict, Generator, Iterable, List, Literal, Optional, Tuple

import dacite
import numpy as np
import pandas as pd
import torch
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.requirements import DataRequirements
from fme.ace.stepper import (
    SingleModuleStepperConfig,
    Stepper,
    TrainOutput,
    process_prediction_generator_list,
    stack_list_of_tensor_dicts,
)
from fme.ace.stepper.single_module import get_serialized_stepper_vertical_coordinate
from fme.core.coordinates import (
    DepthCoordinate,
    OptionalDepthCoordinate,
    OptionalHybridSigmaPressureCoordinate,
)
from fme.core.device import get_device
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.optimization import NullOptimization
from fme.core.timing import GlobalTimer
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.data_loading.data_typing import CoupledVerticalCoordinate
from fme.coupled.data_loading.gridded_data import InferenceGriddedData
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

    """

    timedelta: str
    stepper: SingleModuleStepperConfig


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

    """

    sea_ice_fraction_name: str
    land_fraction_name: str

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


@dataclasses.dataclass
class CoupledStepperConfig:
    """Configuration for a coupled atmosphere-ocean stepper. From a common
    initial condition time the atmosphere steps first and takes as many steps as
    fit in a single ocean step, while being forced by the ocean's initial
    condition SST. The ocean then steps forward once, receiving required
    forcings from the atmosphere-generated output as averages over its step
    window. This completes a single "coupled step". For subsequent coupled
    steps, the generated SST from the ocean forces the atmosphere's steps.

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

    """

    ocean: ComponentConfig
    atmosphere: ComponentConfig
    sst_name: str = "sst"
    ocean_fraction_prediction: Optional[CoupledOceanFractionConfig] = None

    def __post_init__(self):
        self._validate_component_configs()

        # this was already checked in _validate_component_configs, so an
        # assertion will do fine here to appease mypy
        assert self.atmosphere.stepper.ocean is not None
        self._atmosphere_ocean_config = self.atmosphere.stepper.ocean

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
        self._atmosphere_forcing_exogenous_names = list(
            set(self.atmosphere.stepper.input_only_names).difference(
                self.ocean.stepper.output_names
            )
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
        self._all_atmosphere_names = list(
            set(self.atmosphere.stepper.all_names).difference(
                self.ocean.stepper.output_names
            )
        )
        self._all_ocean_names = list(
            set(self.ocean.stepper.all_names).difference(self._all_atmosphere_names)
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
    def ocean_fraction_name(self) -> str:
        """Name of the ocean fraction field in the atmosphere data."""
        return self._atmosphere_ocean_config.ocean_fraction_name

    @property
    def surface_temperature_name(self) -> str:
        """Name of the surface temperature field in the atmosphere data."""
        return self._atmosphere_ocean_config.surface_temperature_name

    @property
    def ocean_next_step_forcing_names(self) -> List[str]:
        """Ocean next-step forcings."""
        return self.ocean.stepper.next_step_forcing_names

    @property
    def ocean_forcing_exogenous_names(self) -> List[str]:
        """Ocean forcing variables that are not outputs of the atmosphere."""
        return self._ocean_forcing_exogenous_names

    @property
    def atmosphere_forcing_exogenous_names(self) -> List[str]:
        """Atmosphere forcing variables that are not outputs of the ocean."""
        return self._atmosphere_forcing_exogenous_names

    @property
    def shared_forcing_exogenous_names(self) -> List[str]:
        """Exogenous forcing variables shared by both components. Must be
        present in the atmosphere data on disk. If time-varying, the ocean
        receives the atmosphere data forcings averaged over its step window.

        """
        return self._shared_forcing_exogenous_names

    @property
    def atmosphere_to_ocean_forcing_names(self) -> List[str]:
        """Ocean forcing variables that are outputs of the atmosphere."""
        return self._atmosphere_to_ocean_forcing_names

    @property
    def ocean_to_atmosphere_forcing_names(self) -> List[str]:
        """Atmosphere forcing variables that are outputs of the ocean."""
        return self._ocean_to_atmosphere_forcing_names

    def _validate_component_configs(self):
        # validate atmosphere's OceanConfig
        if self.atmosphere.stepper.ocean is None:
            raise ValueError(
                "The atmosphere stepper 'ocean' config is missing but must be set for "
                "coupled emulation."
            )
        if self.atmosphere.stepper.ocean.slab is not None:
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
            .difference(self.ocean.stepper.in_names)
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

    def _get_ocean_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: OptionalDepthCoordinate,
    ) -> Stepper:
        return self.ocean.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=self.ocean_timestep,
        )

    def _get_atmosphere_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: OptionalHybridSigmaPressureCoordinate,
    ) -> Stepper:
        return self.atmosphere.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=self.atmosphere_timestep,
        )

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        vertical_coordinate: CoupledVerticalCoordinate,
    ):
        logging.info("Initializing coupler")
        sst_mask = None
        if isinstance(vertical_coordinate.ocean, DepthCoordinate):
            sst_mask = vertical_coordinate.ocean.get_mask_level(0)
        return CoupledStepper(
            config=self,
            ocean=self._get_ocean_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                vertical_coordinate=vertical_coordinate.ocean,
            ),
            atmosphere=self._get_atmosphere_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                vertical_coordinate=vertical_coordinate.atmosphere,
            ),
            sst_mask=sst_mask,
        )

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "CoupledStepperConfig":
        state = cls.remove_deprecated_keys(state)
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @classmethod
    def remove_deprecated_keys(cls, state: Dict[str, Any]) -> Dict[str, Any]:
        state_copy = state.copy()
        if "sst_mask_name" in state_copy:
            del state_copy["sst_mask_name"]
        return state_copy


@dataclasses.dataclass
class CoupledTrainOutput(TrainOutputABC):
    metrics: TensorDict
    ocean_data: TrainOutput
    atmosphere_data: TrainOutput

    def remove_initial_condition(self, n_ic_timesteps: int) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            metrics=self.metrics,
            ocean_data=self.ocean_data.remove_initial_condition(n_ic_timesteps),
            atmosphere_data=self.atmosphere_data.remove_initial_condition(
                n_ic_timesteps
            ),
        )

    def copy(self) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            metrics=self.metrics,
            ocean_data=self.ocean_data.copy(),
            atmosphere_data=self.atmosphere_data.copy(),
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
            metrics=self.metrics,
            ocean_data=self.ocean_data.prepend_initial_condition(
                initial_condition.ocean_data,
            ),
            atmosphere_data=self.atmosphere_data.prepend_initial_condition(
                initial_condition.atmosphere_data,
            ),
        )

    def compute_derived_variables(self) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            metrics=self.metrics,
            ocean_data=self.ocean_data.compute_derived_variables(),
            atmosphere_data=self.atmosphere_data.compute_derived_variables(),
        )

    def get_metrics(self) -> TensorDict:
        return self.metrics


@dataclasses.dataclass
class ComponentStepPrediction:
    realm: Literal["ocean", "atmosphere"]
    data: TensorDict
    step: int

    def detach(self, optimizer: OptimizationABC) -> "ComponentStepPrediction":
        """Detach the data tensor map from the computation graph."""
        return ComponentStepPrediction(
            realm=self.realm,
            data=optimizer.detach_if_using_gradient_accumulation(self.data),
            step=self.step,
        )


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
        sst_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            config: The configuration.
            ocean: The ocean stepper.
            atmosphere: The atmosphere stepper.
            sst_mask: (Optional) The ocean surface mask tensor, with value 1 at
                valid ocean points and 0 elsewhere. If provided, ensures that
                the ocean fraction variable used by the atmosphere to determine
                where to prescribe the ocean's SST is 0 everywhere that the mask
                is 0.

        """
        if ocean.n_ic_timesteps != 1 or atmosphere.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported.")

        self.ocean = ocean
        self.atmosphere = atmosphere
        self._config = config
        self._sst_mask = sst_mask
        if self._sst_mask is not None:
            self._sst_mask = self._sst_mask.to(get_device())

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
        }

    def load_state(self, state: Dict[str, Any]):
        self.atmosphere.load_state(state["atmosphere_state"])
        self.ocean.load_state(state["ocean_state"])

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    def validate_inference_data(self, data: InferenceGriddedData):
        self.atmosphere.validate_inference_data(data.atmosphere_properties)
        self.ocean.validate_inference_data(data.ocean_properties)

    @property
    def n_inner_steps(self) -> int:
        """Number of atmosphere steps per ocean step."""
        return self._config.n_inner_steps

    @property
    def _ocean_forcing_exogenous_names(self) -> List[str]:
        return self._config.ocean_forcing_exogenous_names

    @property
    def _atmosphere_forcing_exogenous_names(self) -> List[str]:
        return self._config.atmosphere_forcing_exogenous_names

    @property
    def _shared_forcing_exogenous_names(self) -> List[str]:
        return self._config.shared_forcing_exogenous_names

    @property
    def _atmosphere_to_ocean_forcing_names(self) -> List[str]:
        return self._config.atmosphere_to_ocean_forcing_names

    @property
    def _ocean_to_atmosphere_forcing_names(self) -> List[str]:
        return self._config.ocean_to_atmosphere_forcing_names

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
            land_frac_name = self._config.ocean_fraction_prediction.land_fraction_name
            sea_ice_frac = forcings_from_ocean[
                self._config.ocean_fraction_prediction.sea_ice_fraction_name
            ]
            ocean_frac = 1 - atmos_forcing_data[land_frac_name] - sea_ice_frac
            # remove negative values just in case the ocean doesn't constrain
            # the sea ice
            forcings_from_ocean[ocean_frac_name] = torch.clip(ocean_frac, min=0)
        if self._sst_mask is not None:
            # enforce agreement between the ocean frac and the SST mask so that
            # ocean frac is 0 everywhere the ocean target data is NaN (i.e., where
            # the mask indicates there is no ocean)
            ocean_frac = forcings_from_ocean[ocean_frac_name]
            img_shape = list(ocean_frac.shape[-2:])
            expanded_shape = [1] * (ocean_frac.ndim - len(img_shape)) + img_shape
            forcings_from_ocean[ocean_frac_name] = torch.minimum(
                ocean_frac,
                self._sst_mask.expand(*expanded_shape),
            )
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
                )
            )

    def _process_prediction_generator_list(
        self,
        output_list: List[ComponentStepPrediction],
        forcing_data: CoupledBatchData,
    ) -> CoupledBatchData:
        atmos_data = process_prediction_generator_list(
            [x.data for x in output_list if x.realm == "atmosphere"],
            time_dim=self.atmosphere.TIME_DIM,
            time=forcing_data.atmosphere_data.time[:, self.atmosphere.n_ic_timesteps :],
            horizontal_dims=forcing_data.atmosphere_data.horizontal_dims,
        )
        ocean_data = process_prediction_generator_list(
            [x.data for x in output_list if x.realm == "ocean"],
            time_dim=self.ocean.TIME_DIM,
            time=forcing_data.ocean_data.time[:, self.ocean.n_ic_timesteps :],
            horizontal_dims=forcing_data.ocean_data.horizontal_dims,
        )
        return CoupledBatchData(ocean_data=ocean_data, atmosphere_data=atmos_data)

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> Tuple[CoupledPairedData, CoupledPrognosticState]:
        """
        Predict multiple steps forward given initial condition and reference data.
        """
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
        ocean_metrics = {}

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
                    step_loss = self.ocean.loss_obj(
                        gen_step.data,
                        target_step,
                    )
                    ocean_metrics[f"loss/ocean_step_{gen_step.step}"] = (
                        step_loss.detach()
                    )
                    optimization.accumulate_loss(step_loss)
                gen_step = gen_step.detach(optimization)
                output_list.append(gen_step)

        loss = optimization.get_accumulated_loss().detach()
        optimization.step_weights()

        gen_data = self._process_prediction_generator_list(output_list, data)

        ocean_stepped = TrainOutput(
            metrics={},
            gen_data=dict(gen_data.ocean_data.data),
            target_data=dict(ocean_forward_data.data),
            time=gen_data.ocean_data.time,
            normalize=self.ocean.normalizer.normalize,
            derive_func=self.ocean.derive_func,
        )
        atmos_stepped = TrainOutput(
            metrics={},
            gen_data=dict(gen_data.atmosphere_data.data),
            target_data=dict(atmos_forward_data.data),
            time=gen_data.atmosphere_data.time,
            normalize=self.atmosphere.normalizer.normalize,
            derive_func=self.atmosphere.derive_func,
        )

        ocean_loss: torch.Tensor = sum(ocean_metrics.values())
        stepped = CoupledTrainOutput(
            metrics={
                "loss": loss,
                "loss/ocean": ocean_loss,
                **ocean_metrics,
            },
            ocean_data=ocean_stepped,
            atmosphere_data=atmos_stepped,
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

    @classmethod
    def from_state(cls, state) -> "CoupledStepper":
        config = CoupledStepperConfig.from_state(state["config"])
        ocean = Stepper.from_state(state["ocean_state"])
        atmosphere = Stepper.from_state(state["atmosphere_state"])
        sst_mask = None
        ocean_vertical_coord = get_serialized_stepper_vertical_coordinate(
            state["ocean_state"]
        )
        if isinstance(ocean_vertical_coord, DepthCoordinate):
            sst_mask = ocean_vertical_coord.get_mask_level(0)
        return cls(config, ocean, atmosphere, sst_mask)
