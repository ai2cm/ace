import datetime
from typing import Any, Dict, Optional, Tuple

from fme.core.coordinates import SerializableVerticalCoordinate, VerticalCoordinate
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.gridded_ops import GriddedOperations
from fme.core.masking import HasGetMaskTensorFor


class MissingDatasetInfo(ValueError):
    def __init__(self, info: str):
        super().__init__(
            f"Dataset used for initialization is missing required information: {info}"
        )


class IncompatibleDatasetInfo(ValueError):
    pass


class DatasetInfo:
    """
    Information about a dataset.

    Generally this is used to provide static information about a training dataset
    when initializing a model, and to validate the compatibility of a model's training
    dataset against inference datasets.

    Its API is meant to support type-safe attribute access in cases where data
    may or may not be available. If data is not available, a MissingDatasetInfo
    exception is raised.
    """

    def __init__(
        self,
        img_shape: Optional[Tuple[int, int]] = None,
        gridded_operations: Optional[GriddedOperations] = None,
        vertical_coordinate: Optional[VerticalCoordinate] = None,
        timestep: Optional[datetime.timedelta] = None,
    ):
        self._img_shape = img_shape
        self._gridded_operations = gridded_operations
        self._vertical_coordinate = vertical_coordinate
        self._timestep = timestep

    def __eq__(self, other) -> bool:
        if not isinstance(other, DatasetInfo):
            return False
        return (
            self._img_shape == other._img_shape
            and self._gridded_operations == other._gridded_operations
            and self._vertical_coordinate == other._vertical_coordinate
            and self._timestep == other._timestep
        )

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(img_shape={self._img_shape}, "
            f"gridded_operations={self._gridded_operations}, "
            f"vertical_coordinate={self._vertical_coordinate}, "
            f"timestep={self._timestep})"
        )

    def assert_compatible_with(self, other: "DatasetInfo"):
        issues = []
        if self._img_shape != other._img_shape:
            issues.append(
                "img_shape is not compatible, {} != {}".format(
                    self._img_shape, other._img_shape
                )
            )
        if self._gridded_operations is not None:
            if self._gridded_operations != other._gridded_operations:
                issues.append(
                    "gridded_operations is not compatible, {} != {}".format(
                        self._gridded_operations, other._gridded_operations
                    )
                )
        if (
            self._vertical_coordinate is not None
            and other._vertical_coordinate is not None
        ):
            if self._vertical_coordinate != other._vertical_coordinate:
                issues.append(
                    "vertical_coordinate is not compatible, {} != {}".format(
                        self._vertical_coordinate, other._vertical_coordinate
                    )
                )
        if self._timestep is not None:
            if self._timestep != other._timestep:
                issues.append(
                    "timestep is not compatible, {} != {}".format(
                        self._timestep, other._timestep
                    )
                )
        if issues:
            raise IncompatibleDatasetInfo(
                "DatasetInfo is not compatible with other DatasetInfo:\n"
                + "\n".join(issues)
            )

    @property
    def img_shape(self) -> Tuple[int, int]:
        if self._img_shape is None:
            raise MissingDatasetInfo("img_shape")
        return self._img_shape

    @property
    def gridded_operations(self) -> GriddedOperations:
        if self._gridded_operations is None:
            raise MissingDatasetInfo("gridded_operations")
        return self._gridded_operations

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        if self._vertical_coordinate is None:
            raise MissingDatasetInfo("vertical_coordinate")
        return self._vertical_coordinate

    @property
    def mask_provider(self) -> HasGetMaskTensorFor:
        try:
            coord = self.vertical_coordinate
        except MissingDatasetInfo as err:
            raise MissingDatasetInfo("mask_provider") from err
        if not isinstance(coord, HasGetMaskTensorFor):
            raise MissingDatasetInfo("mask_provider")
        return coord

    @property
    def timestep(self) -> datetime.timedelta:
        if self._timestep is None:
            raise MissingDatasetInfo("timestep")
        return self._timestep

    def to_state(self) -> Dict[str, Any]:
        if self._gridded_operations is None:
            gridded_operations = None
        else:
            gridded_operations = self._gridded_operations.to_state()
        if self._vertical_coordinate is None:
            vertical_coordinate = None
        else:
            vertical_coordinate = self._vertical_coordinate.as_dict()
        if self._timestep is None:
            timestep = None
        else:
            timestep = encode_timestep(self._timestep)
        return {
            "img_shape": self._img_shape,
            "gridded_operations": gridded_operations,
            "vertical_coordinate": vertical_coordinate,
            "timestep": timestep,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "DatasetInfo":
        if state.get("img_shape") is not None:
            img_shape = state["img_shape"]
        else:
            img_shape = None
        if state.get("gridded_operations") is not None:
            gridded_operations = GriddedOperations.from_state(
                state["gridded_operations"]
            )
        else:
            gridded_operations = None
        if state.get("vertical_coordinate") is not None:
            vertical_coordinate = SerializableVerticalCoordinate.from_state(
                state["vertical_coordinate"]
            )
        else:
            vertical_coordinate = None
        if state.get("timestep") is not None:
            timestep = decode_timestep(state["timestep"])
        else:
            timestep = None
        return cls(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        )
