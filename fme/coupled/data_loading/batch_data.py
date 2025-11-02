import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.core.typing_ import TensorDict, TensorMapping
from fme.coupled.data_loading.data_typing import CoupledDatasetItem
from fme.coupled.requirements import CoupledPrognosticStateDataRequirements

SelfType = TypeVar("SelfType", bound="CoupledBatchData")


class CoupledPrognosticState:
    """
    Thin typing wrapper around CoupledBatchData to indicate that the data is
    a prognostic state, such as an initial condition or final state when
    evolving forward in time.
    """

    def __init__(self, ocean_data: PrognosticState, atmosphere_data: PrognosticState):
        self.ocean_data = ocean_data
        self.atmosphere_data = atmosphere_data

    def to_device(self) -> "CoupledPrognosticState":
        return CoupledPrognosticState(
            self.ocean_data.to_device(), self.atmosphere_data.to_device()
        )

    def as_batch_data(self) -> "CoupledBatchData":
        return CoupledBatchData(
            self.ocean_data.as_batch_data(), self.atmosphere_data.as_batch_data()
        )


@dataclasses.dataclass
class CoupledBatchData:
    ocean_data: BatchData
    atmosphere_data: BatchData

    def to_device(self) -> "CoupledBatchData":
        return self.__class__(
            ocean_data=self.ocean_data.to_device(),
            atmosphere_data=self.atmosphere_data.to_device(),
        )

    @classmethod
    def new_on_device(
        cls,
        ocean_data: BatchData,
        atmosphere_data: BatchData,
    ) -> "CoupledBatchData":
        ocean_batch = ocean_data.to_device()
        atmos_batch = atmosphere_data.to_device()
        return CoupledBatchData(ocean_data=ocean_batch, atmosphere_data=atmos_batch)

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData,
        atmosphere_data: BatchData,
    ) -> "CoupledBatchData":
        ocean_batch = ocean_data.to_cpu()
        atmos_batch = atmosphere_data.to_cpu()
        return CoupledBatchData(ocean_data=ocean_batch, atmosphere_data=atmos_batch)

    @classmethod
    def collate_fn(
        cls,
        samples: Sequence[CoupledDatasetItem],
        ocean_horizontal_dims: list[str],
        atmosphere_horizontal_dims: list[str],
        sample_dim_name: str = "sample",
    ) -> "CoupledBatchData":
        """
        Collate function for use with PyTorch DataLoader. Separates out ocean
        and atmosphere sample tuples and constructs BatchData instances for
        each of the two components.

        """
        ocean_data = BatchData.from_sample_tuples(
            [x.ocean for x in samples],
            horizontal_dims=ocean_horizontal_dims,
            sample_dim_name=sample_dim_name,
        )
        atmosphere_data = BatchData.from_sample_tuples(
            [x.atmosphere for x in samples],
            horizontal_dims=atmosphere_horizontal_dims,
            sample_dim_name=sample_dim_name,
        )
        return CoupledBatchData.new_on_cpu(ocean_data, atmosphere_data)

    def get_start(
        self: SelfType,
        requirements: CoupledPrognosticStateDataRequirements,
    ) -> CoupledPrognosticState:
        """
        Get the initial condition state.
        """
        return CoupledPrognosticState(
            ocean_data=self.ocean_data.get_start(
                requirements.ocean.names,
                requirements.ocean.n_timesteps,
            ),
            atmosphere_data=self.atmosphere_data.get_start(
                requirements.atmosphere.names,
                requirements.atmosphere.n_timesteps,
            ),
        )

    def prepend(self: SelfType, initial_condition: CoupledPrognosticState) -> SelfType:
        return self.__class__(
            ocean_data=self.ocean_data.prepend(initial_condition.ocean_data),
            atmosphere_data=self.atmosphere_data.prepend(
                initial_condition.atmosphere_data
            ),
        )

    def remove_initial_condition(
        self: SelfType,
        n_ic_timesteps_ocean: int,
        n_ic_timesteps_atmosphere: int,
    ) -> SelfType:
        return self.__class__(
            ocean_data=self.ocean_data.remove_initial_condition(n_ic_timesteps_ocean),
            atmosphere_data=self.atmosphere_data.remove_initial_condition(
                n_ic_timesteps_atmosphere
            ),
        )

    def compute_derived_variables(
        self: SelfType,
        ocean_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        atmosphere_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict],
        forcing_data: SelfType,
    ) -> SelfType:
        return self.__class__(
            ocean_data=self.ocean_data.compute_derived_variables(
                ocean_derive_func, forcing_data.ocean_data
            ),
            atmosphere_data=self.atmosphere_data.compute_derived_variables(
                atmosphere_derive_func, forcing_data.atmosphere_data
            ),
        )


@dataclasses.dataclass
class CoupledPairedData:
    """
    A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    ocean_data: PairedData
    atmosphere_data: PairedData

    @classmethod
    def from_coupled_batch_data(
        cls,
        prediction: CoupledBatchData,
        reference: CoupledBatchData,
    ) -> "CoupledPairedData":
        if not np.all(
            prediction.ocean_data.time.values == reference.ocean_data.time.values
        ):
            raise ValueError(
                "Prediction and target ocean time coordinate must be the same."
            )
        if not np.all(
            prediction.atmosphere_data.time.values
            == reference.atmosphere_data.time.values
        ):
            raise ValueError(
                "Prediction and target atmosphere time coordinate must be the same."
            )
        return CoupledPairedData(
            ocean_data=PairedData(
                prediction=prediction.ocean_data.data,
                reference=reference.ocean_data.data,
                time=prediction.ocean_data.time,
                labels=prediction.ocean_data.labels,
            ),
            atmosphere_data=PairedData(
                prediction=prediction.atmosphere_data.data,
                reference=reference.atmosphere_data.data,
                time=prediction.atmosphere_data.time,
                labels=prediction.atmosphere_data.labels,
            ),
        )
