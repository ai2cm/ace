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
        self._components: dict[str, DatasetInfo] = {
            name: val
            for name, val in [
                ("ocean", ocean),
                ("ice", ice),
                ("atmosphere", atmosphere),
            ]
            if val is not None
        }

    @property
    def ocean(self) -> DatasetInfo | None:
        return self._components.get("ocean")

    @property
    def ice(self) -> DatasetInfo | None:
        return self._components.get("ice")

    @property
    def atmosphere(self) -> DatasetInfo | None:
        return self._components.get("atmosphere")

    def _get_spatial_mask_provider(self, name: str) -> HasGetSpatialMask | None:
        info = self._components.get(name)
        if info is None:
            return None
        try:
            return info.spatial_mask_provider
        except MissingDatasetInfo as err:
            raise MissingCoupledDatasetInfo(f"{name}_spatial_mask_provider") from err

    @property
    def ocean_spatial_mask_provider(self) -> HasGetSpatialMask | None:
        return self._get_spatial_mask_provider("ocean")

    @property
    def ice_spatial_mask_provider(self) -> HasGetSpatialMask | None:
        return self._get_spatial_mask_provider("ice")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CoupledDatasetInfo):
            return False
        return self._components == other._components

    def get_state(self) -> dict[Literal["ocean", "ice", "atmosphere"], dict[str, Any]]:
        return {
            name: info.get_state()  # type: ignore[misc]
            for name, info in self._components.items()
        }

    @property
    def horizontal_coordinates(self) -> CoupledHorizontalCoordinates:
        return CoupledHorizontalCoordinates(
            **{
                name: info.horizontal_coordinates
                for name, info in self._components.items()
            }
        )

    @classmethod
    def from_state(
        cls, state: dict[Literal["ocean", "ice", "atmosphere"], dict[str, Any]]
    ) -> "CoupledDatasetInfo":
        parsed = {name: DatasetInfo.from_state(s) for name, s in state.items()}
        return cls(
            ocean=parsed.get("ocean"),
            ice=parsed.get("ice"),
            atmosphere=parsed.get("atmosphere"),
        )

    def update_variable_metadata(
        self, variable_metadata: dict[str, Any]
    ) -> "CoupledDatasetInfo":
        return CoupledDatasetInfo(
            **{
                name: info.update_variable_metadata(variable_metadata)
                for name, info in self._components.items()
            }
        )
