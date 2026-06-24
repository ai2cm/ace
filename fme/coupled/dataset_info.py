from typing import Any, Literal

from fme.core.dataset_info import DatasetInfo, MissingDatasetInfo
from fme.core.spatial_masking import HasGetSpatialMask
from fme.coupled.data_loading.data_typing import CoupledHorizontalCoordinates


class MissingCoupledDatasetInfo(ValueError):
    def __init__(self, info: str):
        super().__init__(
            f"Dataset used for initialization is missing required information: {info}"
        )


class CoupledDatasetInfo:
    def __init__(
        self,
        ocean: DatasetInfo | None = None,
        ice: DatasetInfo | None = None,
        atmosphere: DatasetInfo | None = None,
    ):
        self.ocean = ocean
        self.ice = ice
        self.atmosphere = atmosphere

    @property
    def ocean_spatial_mask_provider(self) -> HasGetSpatialMask:
        if self.ocean is not None:
            try:
                return self.ocean.spatial_mask_provider
            except MissingDatasetInfo as err:
                raise MissingCoupledDatasetInfo("ocean_spatial_mask_provider") from err
        else:
            return None
        
    @property
    def ice_spatial_mask_provider(self) -> HasGetSpatialMask:
        if self.ice is not None:
            try:
                return self.ice.spatial_mask_provider
            except MissingDatasetInfo as err:
                raise MissingCoupledDatasetInfo("ice_spatial_mask_provider") from err
        else:
            return None

    def __eq__(self, other):
        if not isinstance(other, CoupledDatasetInfo):
            return False
        ocean_check = self.ocean == other.ocean
        ice_check = self.ice == other.ice
        atmos_check = self.atmosphere == other.atmosphere
        return ocean_check and ice_check and atmos_check

    def get_state(self) -> dict[Literal["ocean", "ice", "atmosphere"], dict[str, Any]]:
        if self.atmosphere is None:
            ds = {
                "ocean": self.ocean.get_state(),
                "ice": self.ice.get_state(),
            }
        elif self.ice is None:
            ds = {
                "ocean": self.ocean.get_state(),
                "atmosphere": self.atmosphere.get_state(),
             }
        elif self.ocean is None:
            ds = {
                "ice": self.ice.get_state(),
                "atmosphere": self.atmosphere.get_state(),
            }
        else:
            ds = {
                "ocean": self.ocean.get_state(),
                "ice": self.ice.get_state(),
                "atmosphere": self.atmosphere.get_state(),
            }
        return ds

    @property
    def horizontal_coordinates(self) -> CoupledHorizontalCoordinates:
        if self.atmosphere is None:
            return CoupledHorizontalCoordinates(
                ocean=self.ocean.horizontal_coordinates,
                ice=self.ice.horizontal_coordinates,
            )
        elif self.ice is None:
            return CoupledHorizontalCoordinates(
                ocean=self.ocean.horizontal_coordinates,
                atmosphere=self.atmosphere.horizontal_coordinates,
            )
        elif self.ocean is None:
            return CoupledHorizontalCoordinates(
                ice=self.ice.horizontal_coordinates,
                atmosphere=self.atmosphere.horizontal_coordinates,
            )
        else:
            return CoupledHorizontalCoordinates(
                ocean=self.ocean.horizontal_coordinates,
                ice=self.ice.horizontal_coordinates,
                atmosphere=self.atmosphere.horizontal_coordinates,
            )

    @classmethod
    def from_state(
        cls, state: dict[Literal["ocean", "ice", "atmosphere"], dict[str, Any]]
    ) -> "CoupledDatasetInfo":
        ocean_state = None
        ice_state = None
        atmosphere_state = None
        if "ocean" in state:
            ocean_state = DatasetInfo.from_state(state["ocean"])
        if "ice" in state:
            ice_state = DatasetInfo.from_state(state["ice"])
        if "atmosphere" in state:
            atmosphere_state = DatasetInfo.from_state(state["atmosphere"])
        return cls(
            ocean=ocean_state,
            ice=ice_state,
            atmosphere=atmosphere_state,
        )

    def update_variable_metadata(
        self, variable_metadata: dict[str, Any]
    ) -> "CoupledDatasetInfo":
        """
        Update the variable metadata for ocean, ice, and atmosphere datasets.
        """
        ocean_metadata = None
        if self.ocean is not None:
            ocean_metadata = self.ocean.update_variable_metadata(variable_metadata)
        ice_metadata = None
        if self.ice is not None:
            ice_metadata = self.ice.update_variable_metadata(variable_metadata)
        atmos_metadata = None
        if self.atmosphere is not None:
            atmos_metadata = self.atmosphere.update_variable_metadata(variable_metadata)
        return CoupledDatasetInfo(
            ocean=ocean_metadata,
            ice=ice_metadata,
            atmosphere=atmos_metadata,
        )
