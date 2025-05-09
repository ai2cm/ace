import datetime
import logging
from collections.abc import Mapping
from typing import Any

from fme.core.coordinates import SerializableVerticalCoordinate, VerticalCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.gridded_ops import GriddedOperations
from fme.core.mask_provider import MaskProvider
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
        img_shape: tuple[int, int] | None = None,
        gridded_operations: GriddedOperations | None = None,
        vertical_coordinate: VerticalCoordinate | None = None,
        mask_provider: MaskProvider | None = None,
        timestep: datetime.timedelta | None = None,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        self._img_shape = img_shape
        self._gridded_operations = gridded_operations
        self._vertical_coordinate = vertical_coordinate
        self._mask_provider = mask_provider
        self._timestep = timestep
        self._variable_metadata = variable_metadata

    def __eq__(self, other) -> bool:
        if not isinstance(other, DatasetInfo):
            return False
        return (
            self._img_shape == other._img_shape
            and self._gridded_operations == other._gridded_operations
            and self._vertical_coordinate == other._vertical_coordinate
            and self._mask_provider == other._mask_provider
            and self._timestep == other._timestep
            and self._variable_metadata == other._variable_metadata
        )

    def __repr__(self) -> str:
        return (
            f"DatasetInfo(img_shape={self._img_shape}, "
            f"gridded_operations={self._gridded_operations}, "
            f"vertical_coordinate={self._vertical_coordinate}, "
            f"timestep={self._timestep}), "
            f"mask_provider={self._mask_provider}, "
            f"variable_metadata={self._variable_metadata})"
        )

    def assert_compatible_with(self, other: "DatasetInfo"):
        issues = []
        if self._img_shape != other._img_shape:
            issues.append(
                f"img_shape is not compatible, {self._img_shape} != {other._img_shape}"
            )
        if self._gridded_operations is not None:
            if self._gridded_operations != other._gridded_operations:
                issues.append(
                    f"gridded_operations is not compatible, "
                    f"{self._gridded_operations} != {other._gridded_operations}"
                )
        if (
            self._vertical_coordinate is not None
            and other._vertical_coordinate is not None
        ):
            if self._vertical_coordinate != other._vertical_coordinate:
                issues.append(
                    f"vertical_coordinate is not compatible, "
                    f"{self._vertical_coordinate} != {other._vertical_coordinate}"
                )
        if self._mask_provider is not None and other._mask_provider is not None:
            if self._mask_provider != other._mask_provider:
                issues.append(
                    f"mask_provider is not compatible, "
                    f"{self._mask_provider} != {other._mask_provider}"
                )
        if self._timestep is not None:
            if self._timestep != other._timestep:
                issues.append(
                    f"timestep is not compatible, {self._timestep} != {other._timestep}"
                )
        metadata_conflicts = get_keys_with_conflicts(
            self._variable_metadata, other._variable_metadata
        )
        for key, (a, b) in metadata_conflicts.items():
            logging.warning(
                "DatasetInfo has different metadata from other DatasetInfo for key "
                f"{key}: {a} != {b}"
            )
        if issues:
            raise IncompatibleDatasetInfo(
                "DatasetInfo is not compatible with other DatasetInfo:\n"
                + "\n".join(issues)
            )

    @property
    def img_shape(self) -> tuple[int, int]:
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
        if self._mask_provider is None:
            try:
                coord = self.vertical_coordinate
            except MissingDatasetInfo as err:
                raise MissingDatasetInfo("mask_provider") from err
            if not isinstance(coord, HasGetMaskTensorFor):
                raise MissingDatasetInfo("mask_provider")
            return coord
        return self._mask_provider

    @property
    def timestep(self) -> datetime.timedelta:
        if self._timestep is None:
            raise MissingDatasetInfo("timestep")
        return self._timestep

    @property
    def variable_metadata(self) -> Mapping[str, VariableMetadata]:
        if self._variable_metadata is None:
            return {}
        return self._variable_metadata

    def to_state(self) -> dict[str, Any]:
        if self._gridded_operations is None:
            gridded_operations = None
        else:
            gridded_operations = self._gridded_operations.to_state()
        if self._vertical_coordinate is None:
            vertical_coordinate = None
        else:
            vertical_coordinate = self._vertical_coordinate.as_dict()
        if self._mask_provider is None:
            mask_provider = None
        else:
            mask_provider = self._mask_provider.to_state()
        if self._timestep is None:
            timestep = None
        else:
            timestep = encode_timestep(self._timestep)
        return {
            "img_shape": self._img_shape,
            "gridded_operations": gridded_operations,
            "vertical_coordinate": vertical_coordinate,
            "mask_provider": mask_provider,
            "timestep": timestep,
            "variable_metadata": self._variable_metadata,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "DatasetInfo":
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
        if state.get("mask_provider") is not None:
            mask_provider = MaskProvider.from_state(state["mask_provider"])
        else:
            mask_provider = None
        if state.get("timestep") is not None:
            timestep = decode_timestep(state["timestep"])
        else:
            timestep = None
        variable_metadata = state.get("variable_metadata")
        return cls(
            img_shape=img_shape,
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            mask_provider=mask_provider,
            timestep=timestep,
            variable_metadata=variable_metadata,
        )


def get_keys_with_conflicts(
    a: Mapping[str, VariableMetadata] | None,
    b: Mapping[str, VariableMetadata] | None,
) -> Mapping[str, tuple[VariableMetadata, VariableMetadata]]:
    """
    Get the return the keys where the values in the two dictionaries are different.
    """
    a = a or {}
    b = b or {}
    keys_with_conflicts = {}
    for key in a.keys() & b.keys():
        if a[key] != b[key]:
            keys_with_conflicts[key] = (a[key], b[key])
    return keys_with_conflicts
