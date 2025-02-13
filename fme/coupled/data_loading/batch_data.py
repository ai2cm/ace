import dataclasses
from typing import List, Sequence, TypeVar

from fme.ace.data_loading.batch_data import BatchData, PairedData, PrognosticState
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
        ocean_batch = BatchData.new_on_device(
            ocean_data.data, ocean_data.time, ocean_data.horizontal_dims
        )
        atmos_batch = BatchData.new_on_device(
            atmosphere_data.data, atmosphere_data.time, atmosphere_data.horizontal_dims
        )
        return CoupledBatchData(ocean_data=ocean_batch, atmosphere_data=atmos_batch)

    @classmethod
    def new_on_cpu(
        cls,
        ocean_data: BatchData,
        atmosphere_data: BatchData,
    ) -> "CoupledBatchData":
        ocean_batch = BatchData.new_on_cpu(
            ocean_data.data, ocean_data.time, ocean_data.horizontal_dims
        )
        atmos_batch = BatchData.new_on_cpu(
            atmosphere_data.data, atmosphere_data.time, atmosphere_data.horizontal_dims
        )
        return CoupledBatchData(ocean_data=ocean_batch, atmosphere_data=atmos_batch)

    @classmethod
    def collate_fn(
        cls,
        samples: Sequence[CoupledDatasetItem],
        horizontal_dims: List[str],
        sample_dim_name: str = "sample",
    ) -> "CoupledBatchData":
        """
        Collate function for use with PyTorch DataLoader. Separates out ocean
        and atmosphere sample tuples and constructs BatchData instances for
        each of the two components.

        """
        ocean_data = BatchData.from_sample_tuples(
            [x.ocean for x in samples], sample_dim_name=sample_dim_name
        )
        atmosphere_data = BatchData.from_sample_tuples(
            [x.atmosphere for x in samples],
            horizontal_dims=horizontal_dims,
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


@dataclasses.dataclass
class CoupledPairedData:
    """
    A container for the data and time coordinates of a batch, with paired
    prediction and target data.
    """

    ocean_data: PairedData
    atmosphere_data: PairedData
