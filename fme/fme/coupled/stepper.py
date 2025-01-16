import dataclasses
import datetime
import logging
from typing import Any, Dict, List, Tuple

import dacite
import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch import nn

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.ace.stepper import SingleModuleStepper, SingleModuleStepperConfig, TrainOutput
from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.dataset.requirements import DataRequirements
from fme.core.device import get_device
from fme.core.generics.inference import PredictFunction
from fme.core.generics.optimization import OptimizationABC
from fme.core.generics.train_stepper import TrainOutputABC, TrainStepperABC
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.batch_data import (
    CoupledBatchData,
    CoupledPairedData,
    CoupledPrognosticState,
)
from fme.coupled.requirements import (
    CoupledDataRequirements,
    CoupledPrognosticStateDataRequirements,
)


@dataclasses.dataclass
class ComponentConfig:
    """Configuration for one of the components (ocean or atmosphere) within a
    CoupledStepper.

    Parameters:
        timedelta: An ISO 8601 Duration string specifying the size of this component's
            stepper step.
        stepper: The single module stepper configuration for this component.

    """

    timedelta: str
    stepper: SingleModuleStepperConfig


@dataclasses.dataclass
class CoupledStepperConfig:
    """
    Configuration for a coupled stepper.

    Parameters:
        ocean: The ocean component configuration.
        atmosphere: The atmosphere component configuration. The stepper
            configuration must include 'ocean'.
        sst_name: Name of the sea surface temperature field in the ocean data.
    """

    ocean: ComponentConfig
    atmosphere: ComponentConfig
    sst_name: str = "sst"

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
        # check for overlapping names
        duplicate_outputs = set(self.ocean.stepper.out_names).intersection(
            self.atmosphere.stepper.out_names
        )
        if len(duplicate_outputs) > 0:
            raise ValueError(
                "Output variable names of CoupledStepper components cannot "
                f"overlap. Found the following duplicated names: {duplicate_outputs}"
            )
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
        if self.sst_name not in self.ocean.stepper.out_names:
            raise ValueError(
                f"The variable {self.sst_name} is not in the ocean's output "
                "names but is required for coupling with the atmosphere."
            )
        self._ocean_to_atmosphere_forcing_names = list(
            set(self.atmosphere.stepper.forcing_names)
            .intersection(self.ocean.stepper.out_names)
            .union([self.sst_name])
        )
        # calculate names for each component's data requirements
        self._all_ocean_names = list(
            set(self.ocean.stepper.all_names)
            .difference(self.atmosphere.stepper.out_names)
            .union([self.sst_name])
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
        vertical_coordinate: HybridSigmaPressureCoordinate,
    ) -> SingleModuleStepper:
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
        vertical_coordinate: HybridSigmaPressureCoordinate,
    ) -> SingleModuleStepper:
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
        vertical_coordinate: HybridSigmaPressureCoordinate,
    ):
        logging.info("Initializing coupler")
        return CoupledStepper(
            config=self,
            ocean=self._get_ocean_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                vertical_coordinate=vertical_coordinate,
            ),
            atmosphere=self._get_atmosphere_stepper(
                img_shape=img_shape,
                gridded_operations=gridded_operations,
                vertical_coordinate=vertical_coordinate,
            ),
        )

    def get_state(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_state(cls, state) -> "CoupledStepperConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


def _concat_list_of_dicts(dict_list: List[TensorMapping], dim: int) -> Dict[str, Any]:
    keys = next(iter(dict_list)).keys()
    concat_dict = {}
    for k in keys:
        concat_dict[k] = torch.cat([d[k] for d in dict_list], dim=dim)
    return concat_dict


def _concat_list_of_paired_data(
    paired_data_list: List[PairedData], dim: int
) -> PairedData:
    data_list = [paired_data.prediction for paired_data in paired_data_list]
    target_list = [paired_data.target for paired_data in paired_data_list]
    data_concat = _concat_list_of_dicts(data_list, dim=dim)
    target_concat = _concat_list_of_dicts(target_list, dim=dim)
    times_list = [paired_data.time for paired_data in paired_data_list]
    times_concat = xr.concat(times_list, dim="time")
    return PairedData(prediction=data_concat, target=target_concat, time=times_concat)


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
        ocean: SingleModuleStepper,
        atmosphere: SingleModuleStepper,
    ):
        if ocean.n_ic_timesteps != 1 or atmosphere.n_ic_timesteps != 1:
            raise ValueError("Only n_ic_timesteps = 1 is currently supported.")

        self.ocean = ocean
        self.atmosphere = atmosphere
        self._config = config

        _: PredictFunction[  # for type checking
            CoupledPrognosticState,
            CoupledBatchData,
            CoupledPairedData,
        ] = self.predict_paired

    @property
    def modules(self) -> nn.ModuleList:
        return nn.ModuleList([self.atmosphere.module, self.ocean.module])

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

    @property
    def vertical_coordinate(self) -> HybridSigmaPressureCoordinate:
        return self.atmosphere.vertical_coordinate

    @property
    def timestep(self) -> datetime.timedelta:
        return self._config.timestep

    @property
    def ocean_timestep(self) -> datetime.timedelta:
        return self.timestep

    @property
    def atmosphere_timestep(self) -> datetime.timedelta:
        return self.atmosphere.timestep

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
        self,
        atmos_data: BatchData,
        ocean_ic: PrognosticState,
    ) -> BatchData:
        """
        Get the forcings for the atmosphere component.

        Args:
            atmos_data: Atmosphere batch data, including initial condition and forward
                steps.
            ocean_ic: Ocean initial condition state.
        """
        data = atmos_data.data
        time_dim = self.atmosphere.TIME_DIM
        sizes = [-1] * len(next(iter(data.values())).shape)
        # treat ocean-to-atmosphere forcings as "next step" forcings
        sizes[time_dim] = self.n_inner_steps + 1
        # exogenous forcings are used as is
        forcing_data = {k: data[k] for k in self._atmosphere_forcing_exogenous_names}
        # forcings from ocean are constant during the fast atmosphere steps
        ocean_data = ocean_ic.as_batch_data()
        # NOTE: only n_ic_timesteps = 1 is currently supported
        assert ocean_data.time["time"].values.size == 1
        forcings_from_ocean = {
            k: ocean_data.data[k].expand(*sizes)
            for k in self._ocean_to_atmosphere_forcing_names
        }
        # rename the ocean surface temperature variable using the corresponding
        # name in the atmosphere
        forcings_from_ocean[self._config.atmosphere_surface_temperature_name] = (
            forcings_from_ocean.pop(self._config.sst_name)
        )
        forcing_data.update(forcings_from_ocean)
        return BatchData(forcing_data, time=atmos_data.time)

    def _get_ocean_forcings(
        self,
        ocean_data: BatchData,
        atmos_data: BatchData,
    ) -> BatchData:
        """
        Get the forcings for the ocean component.

        Args:
            ocean_data: Ocean batch data, including initial condition and forward
                steps.
            atmos_data: Atmosphere batch data, spanning the time covered by the ocean
                data forward steps.
        """
        data = ocean_data.data
        time_dim = self.ocean.TIME_DIM
        # get n_ic_timesteps of ocean exognous forcings
        forcing_data = {k: data[k] for k in self._ocean_forcing_exogenous_names}
        # get time-averaged forcings from atmosphere
        forcings_from_atmosphere = {
            k: atmos_data.data[k].mean(time_dim, keepdim=True)
            for k in self._atmosphere_to_ocean_forcing_names
        }
        # append nans to match external forcings which have forward step values
        # as there is no atmospheric forward-step value.
        # NOTE: only n_ic_timesteps = 1 is currently supported
        assert ocean_data.time["time"].values.size == 2
        forcings_from_atmosphere = {
            k: torch.cat([v, torch.full_like(v, fill_value=np.nan)], dim=time_dim)
            for k, v in forcings_from_atmosphere.items()
        }
        forcing_data.update(forcings_from_atmosphere)
        return BatchData(forcing_data, time=ocean_data.time)

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

    def predict_paired(
        self,
        initial_condition: CoupledPrognosticState,
        forcing: CoupledBatchData,
        compute_derived_variables: bool = False,
    ) -> Tuple[CoupledPairedData, CoupledPrognosticState]:
        """
        Predict multiple steps forward given initial condition and reference data.
        """
        output_atmos = []
        output_ocean = []

        atmos_prognostic = initial_condition.atmosphere_data
        ocean_prognostic = initial_condition.ocean_data

        n_outer_steps = (
            list(forcing.ocean_data.data.values())[0].shape[1] - self.n_ic_timesteps
        )

        for i_outer in range(n_outer_steps):
            # atmosphere steps

            # get the atmosphere window for the initial coupled step
            atmos_window = forcing.atmosphere_data.select_time_slice(
                slice(
                    i_outer * self.n_inner_steps,
                    (i_outer + 1) * self.n_inner_steps + self.atmosphere.n_ic_timesteps,
                )
            )
            atmos_forcings = self._get_atmosphere_forcings(
                atmos_window,
                ocean_prognostic,
            )
            # predict atmosphere forward n_inner_steps
            output, atmos_prognostic = self.atmosphere.predict(
                atmos_prognostic, atmos_forcings, compute_derived_variables
            )
            output_atmos.append(
                PairedData.from_batch_data(
                    prediction=output,
                    target=self.atmosphere.get_forward_data(
                        atmos_window,
                        compute_derived_variables=compute_derived_variables,
                    ),
                )
            )

            # ocean step

            ocean_window = forcing.ocean_data.select_time_slice(
                slice(i_outer, i_outer + self.n_ic_timesteps + 1)
            )
            # output here is from the atmosphere; we need the full
            # sequence of n_inner_steps for time averaging the atmosphere
            # over the ocean's timestep
            ocean_forcings = self._get_ocean_forcings(ocean_window, output)

            # ocean always predicts forward one step at a time
            output, ocean_prognostic = self.ocean.predict(
                ocean_prognostic,
                ocean_forcings,
                compute_derived_variables=False,
            )
            output_ocean.append(
                PairedData.from_batch_data(
                    prediction=output,
                    target=self.ocean.get_forward_data(
                        ocean_window,
                        compute_derived_variables=False,
                    ),
                )
            )

        return (
            CoupledPairedData(
                ocean_data=_concat_list_of_paired_data(
                    output_ocean, dim=self.ocean.TIME_DIM
                ),
                atmosphere_data=_concat_list_of_paired_data(
                    output_atmos, dim=self.atmosphere.TIME_DIM
                ),
            ),
            CoupledPrognosticState(
                ocean_data=ocean_prognostic, atmosphere_data=atmos_prognostic
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
        loss = torch.tensor(0.0, device=get_device())
        ocean_metrics = {}

        # get initial condition prognostic variables
        atmos_prognostic = data.atmosphere_data.get_start(
            self.atmosphere.prognostic_names, self.n_ic_timesteps
        )
        ocean_prognostic = data.ocean_data.get_start(
            self.ocean.prognostic_names, self.n_ic_timesteps
        )
        input_data = CoupledPrognosticState(
            atmosphere_data=atmos_prognostic, ocean_data=ocean_prognostic
        )

        optimization.set_mode(self.modules)
        with optimization.autocast():
            output, _ = self.predict_paired(input_data, data)
            n_outer_steps = output.ocean_data.time.shape[1]
            for outer_step in range(n_outer_steps):
                # compute ocean step metrics
                step_loss = self._get_step_loss(
                    output.ocean_data.prediction,
                    output.ocean_data.target,
                    0,
                    self.ocean,
                )
                loss += step_loss
                ocean_metrics[f"loss/ocean_step_{outer_step}"] = step_loss.detach()

        optimization.step_weights(loss)

        ocean_stepped = TrainOutput(
            metrics={},
            gen_data=dict(output.ocean_data.prediction),
            target_data=dict(output.ocean_data.target),
            time=output.ocean_data.time,
            normalize=self.ocean.normalizer.normalize,
            derive_func=self.ocean.derive_func,
        )
        atmos_stepped = TrainOutput(
            metrics={},
            gen_data=dict(output.atmosphere_data.prediction),
            target_data=dict(output.atmosphere_data.target),
            time=output.atmosphere_data.time,
            normalize=self.atmosphere.normalizer.normalize,
            derive_func=self.atmosphere.derive_func,
        )

        ocean_loss: torch.Tensor = sum(ocean_metrics.values())
        stepped = CoupledTrainOutput(
            metrics={
                "loss": loss.detach(),
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
        ocean = SingleModuleStepper.from_state(state["ocean_state"])
        atmosphere = SingleModuleStepper.from_state(state["atmosphere_state"])
        return cls(config, ocean, atmosphere)
