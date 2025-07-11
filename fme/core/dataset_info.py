import datetime
import logging
from collections.abc import Mapping
from typing import Any

from fme.core.coordinates import (
    HorizontalCoordinates,
    NullVerticalCoordinate,
    SerializableHorizontalCoordinates,
    SerializableVerticalCoordinate,
    VerticalCoordinate,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset.utils import decode_timestep, encode_timestep
from fme.core.gridded_ops import GriddedOperations
from fme.core.mask_provider import MaskProvider, MaskProviderABC, NullMaskProvider


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
        horizontal_coordinates: HorizontalCoordinates | None = None,
        vertical_coordinate: VerticalCoordinate | None = None,
        mask_provider: MaskProvider | None = None,
        timestep: datetime.timedelta | None = None,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        gridded_operations: GriddedOperations | None = None,
        img_shape: tuple[int, int] | None = None,
    ):
        self._horizontal_coordinates = horizontal_coordinates
        self._vertical_coordinate = vertical_coordinate
        self._mask_provider = mask_provider
        self._timestep = timestep
        self._variable_metadata = variable_metadata

        # The gridded_operations and img_shape arguments are only provided for backwards
        # compatibility with older serialized states that have these attributes instead
        # of horizontal_coordinates.
        if gridded_operations is not None and horizontal_coordinates is not None:
            msg = "Cannot provide both gridded_operations and horizontal_coordinates. "
            raise ValueError(msg)
        self._gridded_operations = gridded_operations
        if img_shape is not None and horizontal_coordinates is not None:
            msg = "Cannot provide both img_shape and horizontal_coordinates. "
            raise ValueError(msg)
        self._img_shape = img_shape

    def __eq__(self, other) -> bool:
        if not isinstance(other, DatasetInfo):
            return False
        return (
            self._img_shape == other._img_shape
            and self._gridded_operations == other._gridded_operations
            and self._horizontal_coordinates == other._horizontal_coordinates
            and self._vertical_coordinate == other._vertical_coordinate
            and self._mask_provider == other._mask_provider
            and self._timestep == other._timestep
            and self._variable_metadata == other._variable_metadata
        )

    def __repr__(self) -> str:
        return (
            f"DatasetInfo( "
            f"horizontal_coordinates={self._horizontal_coordinates}, "
            f"vertical_coordinate={self._vertical_coordinate}, "
            f"timestep={self._timestep}), "
            f"mask_provider={self._mask_provider}, "
            f"variable_metadata={self._variable_metadata})"
        )

    def assert_compatible_with(self, other: "DatasetInfo"):
        issues = []
        if (
            self._horizontal_coordinates is not None
            and other._horizontal_coordinates is not None
        ):
            if self._horizontal_coordinates != other._horizontal_coordinates:
                issues.append(
                    f"horizontal_coordinates is not compatible, "
                    f"{self._horizontal_coordinates} != {other._horizontal_coordinates}"
                )
        if (
            self._vertical_coordinate is not None
            and other._vertical_coordinate is not None
        ) and not (
            isinstance(self._vertical_coordinate, NullVerticalCoordinate)
            or isinstance(other._vertical_coordinate, NullVerticalCoordinate)
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
        if self._img_shape is not None:
            return self._img_shape
        if self._horizontal_coordinates is None:
            raise MissingDatasetInfo("horizontal_coordinates")
        result = self._horizontal_coordinates.shape[-2:]
        assert len(result) == 2  # for a happy mypy
        return result

    @property
    def gridded_operations(self) -> GriddedOperations:
        if self._gridded_operations is not None:
            return self._gridded_operations
        if self._horizontal_coordinates is None:
            raise MissingDatasetInfo("horizontal_coordinates")
        if self._mask_provider is None:
            mp: MaskProviderABC = NullMaskProvider
        else:
            mp = self._mask_provider
        return self._horizontal_coordinates.get_gridded_operations(mask_provider=mp)

    @property
    def horizontal_coordinates(self) -> HorizontalCoordinates:
        if self._horizontal_coordinates is None:
            raise MissingDatasetInfo("horizontal_coordinates")
        return self._horizontal_coordinates

    @property
    def vertical_coordinate(self) -> VerticalCoordinate:
        if self._vertical_coordinate is None:
            raise MissingDatasetInfo("vertical_coordinate")
        return self._vertical_coordinate

    @property
    def mask_provider(self) -> MaskProvider:
        if self._mask_provider is None:
            raise MissingDatasetInfo("mask_provider")
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

    def update_variable_metadata(
        self, new_metadata: Mapping[str, VariableMetadata] | None
    ) -> "DatasetInfo":
        """
        Return a new DatasetInfo with the variable metadata updated.
        """
        return DatasetInfo(
            horizontal_coordinates=self._horizontal_coordinates,
            vertical_coordinate=self._vertical_coordinate,
            mask_provider=self._mask_provider,
            timestep=self._timestep,
            variable_metadata=new_metadata,
            gridded_operations=self._gridded_operations,
            img_shape=self._img_shape,
        )

    def to_state(self) -> dict[str, Any]:
        if self._gridded_operations is not None:
            gridded_operations = self._gridded_operations.to_state()
        else:
            gridded_operations = None
        if self._img_shape is not None:
            img_shape = self._img_shape
        else:
            img_shape = None
        if self._horizontal_coordinates is None:
            horizontal_coordinates = None
        else:
            horizontal_coordinates = self._horizontal_coordinates.to_state()
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
            "horizontal_coordinates": horizontal_coordinates,
            "vertical_coordinate": vertical_coordinate,
            "mask_provider": mask_provider,
            "timestep": timestep,
            "variable_metadata": self._variable_metadata,
            "gridded_operations": gridded_operations,
            "img_shape": img_shape,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "DatasetInfo":
        if state.get("gridded_operations") is not None:
            # this is for backwards compatibility with older serialized states
            assert state.get("horizontal_coordinates") is None
            gridded_ops = GriddedOperations.from_state(state["gridded_operations"])
        else:
            gridded_ops = None
        if state.get("img_shape") is not None:
            # this is for backwards compatibility with older serialized states
            assert state.get("horizontal_coordinates") is None
            img_shape = state["img_shape"]
        else:
            img_shape = None
        if state.get("horizontal_coordinates") is not None:
            horizontal_coordinates = SerializableHorizontalCoordinates.from_state(
                state["horizontal_coordinates"]
            )
        else:
            horizontal_coordinates = None
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
            horizontal_coordinates=horizontal_coordinates,
            vertical_coordinate=vertical_coordinate,
            mask_provider=mask_provider,
            timestep=timestep,
            variable_metadata=variable_metadata,
            gridded_operations=gridded_ops,
            img_shape=img_shape,
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
