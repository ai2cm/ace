import dataclasses
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import xarray as xr
from torch import nn

from fme.core.data_loading.batch_data import BatchData
from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.data_loading.requirements import DataRequirements
from fme.core.device import get_device
from fme.core.generics.optimization import OptimizationABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.stepper import (
    SingleModuleStepper,
    SingleModuleStepperConfig,
    StepperABC,
    TrainOutput,
    TrainOutputABC,
)
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import CoupledBatchData
from fme.coupled.data_loading.requirements import CoupledDataRequirements


@dataclasses.dataclass
class CoupledComponentConfig:
    """Configuration for one of the components (ocean or atmosphere) within a
    CoupledStepper.

    Attributes:
        timedelta: An ISO 8601 Duration string specifying the size of this component's
            stepper step.
        stepper: The single module stepper configuration for this component.
        surface_temperature_name: (optional) Name of the sea surface temperature
            field for this component, in case the name is different from the
            OceanConfig.surface_temperature_name specified in the atmosphere stepper.

    """

    timedelta: str
    stepper: SingleModuleStepperConfig
    surface_temperature_name: Optional[str] = None


@dataclasses.dataclass
class CoupledStepperConfig:
    """
    Configuration for a coupled stepper.

    Attributes:
        ocean: The ocean component configuration.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include 'ocean'.

    """

    ocean: CoupledComponentConfig
    atmosphere: CoupledComponentConfig

    def __post_init__(self):
        if self.atmosphere.stepper.ocean is None:
            raise ValueError(
                "The atmosphere stepper 'ocean' config is missing but must be set for "
                "coupled emulation."
            )
        self.atmosphere_surface_temperature_name = (
            self.atmosphere.stepper.ocean.surface_temperature_name
        )
        self._ocean_timestep = pd.Timedelta(self.ocean.timedelta).to_pytimedelta()
        self._atmosphere_timestep = pd.Timedelta(
            self.atmosphere.timedelta
        ).to_pytimedelta()
        if self.atmosphere_timestep > self.ocean_timestep:
            raise ValueError("Atmosphere timedelta must not be larger than ocean's.")
        n_inner_steps = self.ocean_timestep / self.atmosphere_timestep
        if n_inner_steps != int(n_inner_steps):
            raise ValueError("Ocean timedelta must be a multiple of the atmosphere's.")
        self.n_inner_steps = int(n_inner_steps)
        # calculate forcings
        self._ocean_forcing_exogenous_names = list(
            set(self.ocean.stepper.forcing_names).difference(
                self.atmosphere.stepper.out_names
            )
        )
        self._atmosphere_forcing_exogenous_names = list(
            set(self.atmosphere.stepper.forcing_names).difference(
                self.ocean.stepper.out_names
            )
        )
        self._atmosphere_to_ocean_forcing_names = list(
            set(self.ocean.stepper.forcing_names).intersection(
                self.atmosphere.stepper.out_names
            )
        )
        # include the ocean surface temperature variable as forcing for the atmosphere
        ocean_sfc_temp_name = self.ocean.surface_temperature_name
        if ocean_sfc_temp_name is None:
            ocean_sfc_temp_name = self.atmosphere.stepper.ocean.surface_temperature_name
        if ocean_sfc_temp_name not in self.ocean.stepper.out_names:
            raise ValueError(
                f"The variable {ocean_sfc_temp_name} is not in the ocean's output "
                "names but is required for coupling with the atmosphere."
            )
        self.ocean_surface_temperature_name = ocean_sfc_temp_name
        self._ocean_to_atmosphere_forcing_names = list(
            set(self.atmosphere.stepper.forcing_names)
            .intersection(self.ocean.stepper.out_names)
            .union([self.ocean_surface_temperature_name])
        )
        # calculate names for each component's data requirements
        self._all_ocean_names = list(
            set(self.ocean.stepper.all_names)
            .difference(self.atmosphere.stepper.out_names)
            .union([self.ocean_surface_temperature_name])
        )
        self._all_atmosphere_names = list(
            set(self.atmosphere.stepper.all_names)
            .difference(self.ocean.stepper.out_names)
            .union([self.atmosphere.stepper.ocean.surface_temperature_name])
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
    def ocean_forcing_exogenous_names(self) -> List[str]:
        """Ocean forcing variables that are not outputs of the atmosphere."""
        return self._ocean_forcing_exogenous_names

    @property
    def atmosphere_forcing_exogenous_names(self) -> List[str]:
        """Atmosphere forcing variables that are not outputs of the ocean."""
        return self._atmosphere_forcing_exogenous_names

    @property
    def atmosphere_to_ocean_forcing_names(self) -> List[str]:
        """Ocean forcing variables that are outputs of the atmosphere."""
        return self._atmosphere_to_ocean_forcing_names

    @property
    def ocean_to_atmosphere_forcing_names(self) -> List[str]:
        """Atmosphere forcing variables that are outputs of the ocean."""
        return self._ocean_to_atmosphere_forcing_names

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

    def get_data_requirements(self, n_coupled_steps: int) -> CoupledDataRequirements:
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

    def _get_ocean_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
    ) -> SingleModuleStepper:
        return self.ocean.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
            timestep=self.ocean_timestep,
        )

    def _get_atmosphere_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
    ) -> SingleModuleStepper:
        return self.atmosphere.stepper.get_stepper(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
            timestep=self.atmosphere_timestep,
        )

    def get_stepper(
        self,
        img_shape: Tuple[int, int],
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
    ):
        logging.info("Initializing coupler")
        return CoupledStepper(
            config=self,
            ocean=self._get_ocean_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                sigma_coordinates=sigma_coordinates,
            ),
            atmosphere=self._get_atmosphere_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                sigma_coordinates=sigma_coordinates,
            ),
        )


def _concat_list_of_dicts(dict_list: List[TensorMapping], dim: int) -> Dict[str, Any]:
    keys = next(iter(dict_list)).keys()
    concat_dict = {}
    for k in keys:
        concat_dict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return concat_dict


def _concat_list_of_batch_data(batch_data_list: List[BatchData], dim: int) -> BatchData:
    data_list = [batch_data.device_data for batch_data in batch_data_list]
    data_concat = _concat_list_of_dicts(data_list, dim=dim)
    times_list = [batch_data.times for batch_data in batch_data_list]
    times_concat = xr.concat(times_list, dim="time")
    return BatchData(data_concat, times=times_concat)


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

    def compute_derived_variables(self) -> "CoupledTrainOutput":
        return CoupledTrainOutput(
            metrics=self.metrics,
            ocean_data=self.ocean_data.compute_derived_variables(),
            atmosphere_data=self.atmosphere_data.compute_derived_variables(),
        )

    def get_metrics(self) -> TensorDict:
        return self.metrics


class CoupledStepper(StepperABC[CoupledBatchData, CoupledTrainOutput]):
    TIME_DIM = 1

    def __init__(
        self,
        config: CoupledStepperConfig,
        ocean: SingleModuleStepper,
        atmosphere: SingleModuleStepper,
    ):
        self.ocean = ocean
        self.atmosphere = atmosphere
        self._config = config

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([self.atmosphere.module, self.ocean.module])

    def get_state(self):
        """
        Returns:
            The state of the stepper.
        """
        return {
            "atmosphere_state": self.atmosphere.get_state(),
            "ocean_state": self.ocean.get_state(),
        }

    def load_state(self, state: Dict[str, Any]):
        self.atmosphere.load_state(state["atmosphere_state"])
        self.ocean.load_state(state["ocean_state"])

    @property
    def n_ic_timesteps(self) -> int:
        return 1

    @property
    def sigma_coordinates(self) -> SigmaCoordinates:
        return self.atmosphere.sigma_coordinates

    @property
    def timestep(self) -> datetime.timedelta:
        return self._config.timestep

    @property
    def n_inner_steps(self) -> int:
        """Number of atmosphere steps per ocean step."""
        return self._config.n_inner_steps

    @property
    def _ocean_forcing_exogenous_names(self) -> List[str]:
        """Ocean forcing variables that are not outputs of the atmosphere."""
        return self._config.ocean_forcing_exogenous_names

    @property
    def _atmosphere_forcing_exogenous_names(self) -> List[str]:
        """Atmosphere forcing variables that are not outputs of the ocean."""
        return self._config.atmosphere_forcing_exogenous_names

    @property
    def _atmosphere_to_ocean_forcing_names(self) -> List[str]:
        """Ocean forcing variables that are outputs of the atmosphere."""
        return self._config.atmosphere_to_ocean_forcing_names

    @property
    def _ocean_to_atmosphere_forcing_names(self) -> List[str]:
        """Atmosphere forcing variables that are outputs of the ocean."""
        return self._config.ocean_to_atmosphere_forcing_names

    def _select_step(
        self, data: TensorMapping, names: List[str], dim: int, idx: int = 0
    ):
        """Select and return the tensors in dict data with the specified names,
        slicing along the given dimension at the given index.

        """
        return {k: data[k].select(dim, idx) for k in names}

    def _get_atmosphere_forcings(
        self, data: TensorMapping, times: xr.DataArray
    ) -> BatchData:
        time_dim = self.atmosphere.TIME_DIM
        sizes = [-1] * len(next(iter(data.values())).shape)
        # treat ocean-to-atmosphere forcings as "next step" forcings
        sizes[time_dim] = self.n_inner_steps + 1
        # exogenous forcings are used as is
        forcing_data = {
            k: data[k][:, : self.n_inner_steps + 1]
            for k in self._atmosphere_forcing_exogenous_names
        }
        # forcings from ocean are constant during the fast atmosphere steps
        forcings_from_ocean = {
            k: data[k].select(time_dim, 0).unsqueeze(time_dim).expand(*sizes)
            for k in self._ocean_to_atmosphere_forcing_names
        }
        # rename the ocean surface temperature variable using the corresponding
        # name in the atmosphere
        forcings_from_ocean[
            self._config.atmosphere_surface_temperature_name
        ] = forcings_from_ocean.pop(self._config.ocean_surface_temperature_name)
        forcing_data.update(forcings_from_ocean)
        return BatchData(forcing_data, times=times)

    def _get_ocean_forcings(
        self, data: TensorMapping, times: xr.DataArray
    ) -> BatchData:
        time_dim = self.ocean.TIME_DIM
        # get one timestep of ocean exognous forcings
        forcing_data = {
            k: data[k].select(time_dim, 0).unsqueeze(time_dim)
            for k in self._ocean_forcing_exogenous_names
        }
        # get time-averaged forcings from atmosphere
        forcing_data.update(
            {
                k: data[k].mean(time_dim, keepdim=True)
                for k in self._atmosphere_to_ocean_forcing_names
            }
        )
        return BatchData(forcing_data, times=times)

    def _get_step_loss(
        self,
        gen_data: TensorMapping,
        target_data: TensorMapping,
        step: int,
        stepper: SingleModuleStepper,
    ):
        time_dim = stepper.TIME_DIM
        gen_step = {k: v.select(time_dim, step) for k, v in gen_data.items()}
        target_step = {k: v.select(time_dim, step) for k, v in target_data.items()}
        gen_norm_step = stepper.loss_normalizer.normalize(gen_step)
        target_norm_step = stepper.loss_normalizer.normalize(target_step)
        return stepper.loss_obj(gen_norm_step, target_norm_step)

    def train_on_batch(
        self,
        data: CoupledBatchData,
        optimization: OptimizationABC,
        keep_initial_condition: bool = False,
    ):
        """
        Args:
            data: The coupled batch data, consisting of separate batches for ocean and
                atmosphere with the same initial condition time.
            optimization: The optimization class to use for updating the module.
                Use `NullOptimization` to disable training.
            keep_initial_condition: Whether to keep the initial condition in the output.

        """

        # we'll use data_ocean and data_atmos to keep track of their respective
        # targets and forcings needed for future steps, removing whatever is
        # already used as we go
        data_ocean = data.ocean_data.device_data
        data_atmos = data.atmosphere_data.device_data

        loss = torch.tensor(0.0, device=get_device())
        metrics = {}

        gen_data_atmos = []
        gen_data_ocean = []

        # get initial condition prognostic variables
        atmos_prognostic = self._select_step(
            data_atmos,
            names=self.atmosphere.prognostic_names,
            dim=self.atmosphere.TIME_DIM,
        )
        ocean_prognostic = self._select_step(
            data_ocean,
            names=self.ocean.prognostic_names,
            dim=self.ocean.TIME_DIM,
        )
        # get initial condition atmosphere forcing variables
        atmos_forcings = self._get_atmosphere_forcings(
            {**data_atmos, **data_ocean}, times=data.atmosphere_data.times
        )
        n_outer_steps = list(data_ocean.values())[0].shape[1] - self.n_ic_timesteps

        with optimization.autocast():
            for step in range(n_outer_steps):
                # atmosphere steps

                # predict atmosphere forward n_inner_steps
                gen_data = self.atmosphere.predict(
                    atmos_prognostic, atmos_forcings, n_forward_steps=self.n_inner_steps
                )
                gen_data_atmos.append(gen_data)
                target_data = {
                    k: data_atmos[k][:, 1 : self.n_inner_steps + 1]
                    for k in gen_data.device_data
                }
                # compute inner step metrics
                for inner_step in range(self.n_inner_steps):
                    step_loss = self._get_step_loss(
                        gen_data.device_data, target_data, inner_step, self.atmosphere
                    )
                    loss += step_loss
                    metrics[f"loss_atmos_step_{step}.{inner_step}"] = step_loss.detach()
                # remove used initial condition times
                data_atmos = {
                    k: data_atmos[k][:, self.n_inner_steps :] for k in data_atmos
                }
                # get the initial atmosphere prognostic state for the next iter
                atmos_prognostic = self._select_step(
                    gen_data.device_data,
                    names=self.atmosphere.prognostic_names,
                    dim=self.atmosphere.TIME_DIM,
                    idx=-1,
                )

                # ocean step

                # gen_data here is from the atmosphere; we need the full
                # sequence of n_inner_steps for time averaging the atmosphere
                # over the ocean's timestep
                ocean_forcings = self._get_ocean_forcings(
                    {**data_ocean, **gen_data.device_data}, times=gen_data.times
                )
                # ocean always predicts forward one step at a time
                gen_data = self.ocean.predict(
                    ocean_prognostic, ocean_forcings, n_forward_steps=1
                )
                gen_data_ocean.append(gen_data)
                target_data = {k: data_ocean[k][:, :1] for k in data_ocean}
                # compute ocean step metrics
                step_loss = self._get_step_loss(
                    gen_data.device_data, target_data, 0, self.ocean
                )
                loss += step_loss
                metrics[f"loss_ocean_step_{step}"] = step_loss.detach()
                # remove used initial condition time
                data_ocean = {k: data_ocean[k][:, 1:] for k in data_ocean}
                # get generated prognostic variables for next iter
                ocean_prognostic = self._select_step(
                    gen_data.device_data,
                    names=self.ocean.prognostic_names,
                    dim=self.ocean.TIME_DIM,
                    idx=-1,
                )
                # get generated ocean-to-atmosphere forcings for next iter
                atmos_forcings = self._get_atmosphere_forcings(
                    {**data_atmos, **gen_data.device_data}, times=gen_data.times
                )

        gen_ocean = _concat_list_of_batch_data(gen_data_ocean, dim=self.ocean.TIME_DIM)
        gen_atmos = _concat_list_of_batch_data(
            gen_data_atmos, dim=self.atmosphere.TIME_DIM
        )

        target_ocean: TensorDict = {
            k: data.ocean_data.device_data[k][:, 1:] for k in gen_ocean.data
        }
        target_atmos: TensorDict = {
            k: data.atmosphere_data.device_data[k][:, 1:] for k in gen_atmos.data
        }

        metrics["loss"] = loss.detach()
        optimization.step_weights(loss)

        ocean_stepped = TrainOutput(
            metrics={},
            gen_data=dict(gen_ocean.device_data),
            target_data=target_ocean,
            times=gen_ocean.times,
            normalize=self.ocean.normalizer.normalize,
            derive_func=data.ocean_data.derive_func,
        )
        atmos_stepped = TrainOutput(
            metrics={},
            gen_data=dict(gen_atmos.device_data),
            target_data=target_atmos,
            times=gen_atmos.times,
            normalize=self.atmosphere.normalizer.normalize,
            derive_func=data.atmosphere_data.derive_func,
        )

        if keep_initial_condition:
            # prepend target/initial input data to both prediction and target data
            ic_ocean = self.ocean.get_initial_condition(data.ocean_data)
            ic_atmos = self.atmosphere.get_initial_condition(data.atmosphere_data)
            ocean_stepped = ocean_stepped.prepend_initial_condition(ic_ocean)
            atmos_stepped = atmos_stepped.prepend_initial_condition(ic_atmos)

        return CoupledTrainOutput(
            metrics=metrics,
            ocean_data=ocean_stepped,
            atmosphere_data=atmos_stepped,
        )
