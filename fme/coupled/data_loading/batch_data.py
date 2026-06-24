import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
from fme.core.coordinates import NullDeriveFn
from fme.core.labels import LabelEncoding
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

    def __init__(self,
                 ocean_data: PrognosticState | None = None, 
                 ice_data: PrognosticState | None = None, 
                 atmosphere_data: PrognosticState | None = None
        ):
        self.ocean_data = ocean_data
        self.ice_data = ice_data
        self.atmosphere_data = atmosphere_data

    def to_device(self) -> "CoupledPrognosticState":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if self.ocean_data is not None:
            ocean_device = self.ocean_data.to_device()
        if self.ice_data is not None:
            ice_device = self.ice_data.to_device()
        if self.atmosphere_data is not None:
            atmos_device = self.atmosphere_data.to_device()
        return CoupledPrognosticState(
            ocean_device, 
            ice_device, 
            atmos_device
        )

    def as_batch_data(self) -> "CoupledBatchData":
        ocean_batch = None
        ice_batch = None
        atmos_batch = None
        if self.ocean_data is not None:
            ocean_batch = self.ocean_data.as_batch_data()
        if self.ice_data is not None:
            ice_batch = self.ice_data.as_batch_data()
        if self.atmosphere_data is not None:
            atmos_batch = self.atmosphere_data.as_batch_data()
        return CoupledBatchData(
            ocean_batch,
            ice_batch,
            atmos_batch
        )


@dataclasses.dataclass
class CoupledBatchData:
    ocean_data: BatchData | None = None
    ice_data: BatchData | None = None
    atmosphere_data: BatchData | None = None

    def to_device(self) -> "CoupledBatchData":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if self.ocean_data is not None:
            ocean_device = self.ocean_data.to_device()
        if self.ice_data is not None:
            ice_device = self.ice_data.to_device()
        if self.atmosphere_data is not None:
            atmos_device = self.atmosphere_data.to_device()
        return self.__class__(
            ocean_data=ocean_device,
            ice_data=ice_device,
            atmosphere_data=atmos_device,
        )

    @classmethod
    def new_on_device(
        cls,
        ocean_data: BatchData | None = None,
        ice_data: BatchData | None = None,
        atmosphere_data: BatchData | None = None,
    ) -> "CoupledBatchData":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if ocean_data is not None:
            ocean_device = ocean_data.to_device()
        if ice_data is not None:
            ice_device = ice_data.to_device()
        if atmosphere_data is not None:
            atmos_device = atmosphere_data.to_device()
        return CoupledBatchData(
            ocean_data=ocean_device,
            ice_data=ice_device,
            atmosphere_data=atmos_device
        )

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData | None = None,
        ice_data: BatchData | None = None,
        atmosphere_data: BatchData | None = None,
    ) -> "CoupledBatchData":
        ocean_device = None
        ice_device = None
        atmos_device = None
        if ocean_data is not None:
            ocean_device = ocean_data.to_cpu()
        if ice_data is not None:
            ice_device = ice_data.to_cpu()
        if atmosphere_data is not None:
            atmos_device = atmosphere_data.to_cpu()
        return CoupledBatchData(
            ocean_data=ocean_device,
            ice_data=ice_device,
            atmosphere_data=atmos_device
        )

    @classmethod
    def collate_fn(
        cls,
        samples: Sequence[CoupledDatasetItem],
        sample_dim_name: str = "sample",
        ocean_horizontal_dims: list[str] | None = None,
        ice_horizontal_dims: list[str] | None = None,
        atmosphere_horizontal_dims: list[str] | None = None,
        ocean_label_encoding: LabelEncoding | None = None,
        ice_label_encoding: LabelEncoding | None = None,
        atmosphere_label_encoding: LabelEncoding | None = None,
    ) -> "CoupledBatchData":
        """
        Collate function for use with PyTorch DataLoader. Separates out ocean,
        ice, and atmosphere sample tuples and constructs BatchData instances for
        each component.

        """
        ocean_data = None
        if ocean_horizontal_dims is not None:
            ocean_data = BatchData.from_sample_tuples(
                [x.ocean for x in samples],
                horizontal_dims=ocean_horizontal_dims,
                sample_dim_name=sample_dim_name,
                label_encoding=ocean_label_encoding,
            )
        ice_data = None
        if ice_horizontal_dims is not None:
            ice_data = BatchData.from_sample_tuples(
                [x.ice for x in samples],
                horizontal_dims=ice_horizontal_dims,
                sample_dim_name=sample_dim_name,
                label_encoding=ice_label_encoding,
            )
        atmosphere_data = None
        if atmosphere_horizontal_dims is not None:
            atmosphere_data = BatchData.from_sample_tuples(
                [x.atmosphere for x in samples],
                horizontal_dims=atmosphere_horizontal_dims,
                sample_dim_name=sample_dim_name,
                label_encoding=atmosphere_label_encoding,
            )
        return CoupledBatchData.new_on_cpu(ocean_data, ice_data, atmosphere_data)

    def get_start(
        self: SelfType,
        requirements: CoupledPrognosticStateDataRequirements,
    ) -> CoupledPrognosticState:
        """
        Get the initial condition state.
        """
        ocean_data = None
        if self.ocean_data is not None:
            ocean_data=self.ocean_data.get_start(
                requirements.ocean.names,
                requirements.ocean.n_timesteps,
            )
        ice_data = None
        if self.ice_data is not None:
            ice_data=self.ice_data.get_start(
                requirements.ice.names,
                requirements.ice.n_timesteps,
            )
        atmosphere_data = None
        if self.atmosphere_data is not None:
            atmosphere_data=self.atmosphere_data.get_start(
                requirements.atmosphere.names,
                requirements.atmosphere.n_timesteps,
            )
        return CoupledPrognosticState(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data
        )

    def prepend(self: SelfType, initial_condition: CoupledPrognosticState) -> SelfType:
        ocean_data = None
        if self.ocean_data is not None:
            ocean_data=self.ocean_data.prepend(initial_condition.ocean_data)
        ice_data = None
        if self.ice_data is not None:
            ice_data=self.ice_data.prepend(initial_condition.ice_data)
        atmosphere_data = None
        if self.atmosphere_data is not None:
            atmosphere_data=self.atmosphere_data.prepend(
                initial_condition.atmosphere_data
            )
        return self.__class__(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data
        )

    def remove_initial_condition(
        self: SelfType,
        n_ic_timesteps_ocean: int | None = None,
        n_ic_timesteps_ice: int | None = None,
        n_ic_timesteps_atmosphere: int | None = None,
    ) -> SelfType:
        ocean_data = None
        if self.ocean_data is not None:
            ocean_data = self.ocean_data.remove_initial_condition(n_ic_timesteps_ocean)
        ice_data = None
        if self.ice_data is not None:
            ice_data=self.ice_data.remove_initial_condition(n_ic_timesteps_ice)
        atmosphere_data = None
        if self.atmosphere_data is not None:
            atmosphere_data=self.atmosphere_data.remove_initial_condition(
                n_ic_timesteps_atmosphere
            )
        return self.__class__(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data
        )

    def compute_derived_variables(
        self: SelfType,
        forcing_data: SelfType,
        ocean_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict] | None = None,
        ice_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict] | None = None,
        atmosphere_derive_func: Callable[[TensorMapping, TensorMapping], TensorDict] | None = None,
    ) -> SelfType:
        ocean_data = None
        if self.ocean_data is not None:
            ocean_data=self.ocean_data.compute_derived_variables(
                ocean_derive_func, forcing_data.ocean_data
            )
        ice_data = None
        if self.ice_data is not None:
            if isinstance(ice_derive_func, NullDeriveFn):
                ice_data = self.ice_data
            else:
                ice_data=self.ice_data.compute_derived_variables(
                    ice_derive_func, forcing_data.ice_data
                )
        atmosphere_data = None
        if self.atmosphere_data is not None:
            atmosphere_data=self.atmosphere_data.compute_derived_variables(
                atmosphere_derive_func, forcing_data.atmosphere_data
            )
        return self.__class__(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data
        )

    def pin_memory(self: SelfType) -> SelfType:
        if self.ocean_data is not None:
            self.ocean_data = self.ocean_data.pin_memory()
        if self.ice_data is not None:
            self.ice_data = self.ice_data.pin_memory()
        if self.atmosphere_data is not None:
            self.atmosphere_data = self.atmosphere_data.pin_memory()
        return self


@dataclasses.dataclass
class CoupledPairedData:
    """
    A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    ocean_data: PairedData | None = None
    ice_data: PairedData | None = None
    atmosphere_data: PairedData | None = None

    @classmethod
    def from_coupled_batch_data(
        cls,
        prediction: CoupledBatchData,
        reference: CoupledBatchData,
    ) -> "CoupledPairedData":
        ocean_data = None
        if prediction.ocean_data is not None:
            if not np.all(
                prediction.ocean_data.time.values == reference.ocean_data.time.values
            ):
                raise ValueError(
                    "Prediction and target ocean time coordinate must be the same."
                )
            ocean_data=PairedData(
                prediction=prediction.ocean_data.data,
                reference=reference.ocean_data.data,
                time=prediction.ocean_data.time,
                labels=prediction.ocean_data.labels,
            )
        ice_data = None
        if prediction.ice_data is not None:
            if not np.all(
                prediction.ice_data.time.values == reference.ice_data.time.values
            ):
                raise ValueError(
                    "Prediction and target ice time coordinate must be the same."
                )
            ice_data=PairedData(
                prediction=prediction.ice_data.data,
                reference=reference.ice_data.data,
                time=prediction.ice_data.time,
                labels=prediction.ice_data.labels,
            )
        atmosphere_data = None
        if prediction.atmosphere_data is not None:
            if not np.all(
                prediction.atmosphere_data.time.values
                == reference.atmosphere_data.time.values
            ):
                raise ValueError(
                    "Prediction and target atmosphere time coordinate must be the same."
                )
            atmosphere_data=PairedData(
                prediction=prediction.atmosphere_data.data,
                reference=reference.atmosphere_data.data,
                time=prediction.atmosphere_data.time,
                labels=prediction.atmosphere_data.labels,
            )
        return CoupledPairedData(
            ocean_data=ocean_data,
            ice_data=ice_data,
            atmosphere_data=atmosphere_data,
        )
