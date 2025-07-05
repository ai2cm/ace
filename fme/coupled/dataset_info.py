from typing import Any, Literal

from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.masking import HasGetMaskTensorFor
from fme.coupled.data_loading.data_typing import CoupledHorizontalCoordinates


class MissingCoupledDatasetInfo(ValueError):
    def __init__(self, info: str):
        super().__init__(
            f"Dataset used for initialization is missing required information: {info}"
        )


class CoupledDatasetInfo:
    def __init__(
        self,
        ocean: DatasetInfo,
        atmosphere: DatasetInfo,
    ):
        self.ocean = ocean
        self.atmosphere = atmosphere

    @property
    def ocean_mask_provider(self) -> HasGetMaskTensorFor:
        try:
            return self.ocean.mask_provider
        except MissingDatasetInfo as err:
            raise MissingCoupledDatasetInfo("ocean_mask_provider") from err

    def __eq__(self, other):
        if not isinstance(other, CoupledDatasetInfo):
            return False
        return self.ocean == other.ocean and self.atmosphere == other.atmosphere

    def to_state(self) -> dict[Literal["ocean", "atmosphere"], dict[str, Any]]:
        return {
            "ocean": self.ocean.to_state(),
            "atmosphere": self.atmosphere.to_state(),
        }

    @property
    def horizontal_coordinates(self) -> CoupledHorizontalCoordinates:
        return CoupledHorizontalCoordinates(
            ocean=self.ocean.horizontal_coordinates,
            atmosphere=self.atmosphere.horizontal_coordinates,
        )

    @classmethod
    def from_state(
        cls, state: dict[Literal["ocean", "atmosphere"], dict[str, Any]]
    ) -> "CoupledDatasetInfo":
        return cls(
            ocean=DatasetInfo.from_state(state["ocean"]),
            atmosphere=DatasetInfo.from_state(state["atmosphere"]),
        )

    def update_variable_metadata(
        self, variable_metadata: dict[str, Any]
    ) -> "CoupledDatasetInfo":
        """
        Update the variable metadata for both ocean and atmosphere datasets.
        """
        return CoupledDatasetInfo(
            ocean=self.ocean.update_variable_metadata(variable_metadata),
            atmosphere=self.atmosphere.update_variable_metadata(variable_metadata),
        )
