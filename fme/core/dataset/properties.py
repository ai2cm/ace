import datetime

from fme.core.coordinates import HorizontalCoordinates, VerticalCoordinate
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.device import get_device
from fme.core.mask_provider import MaskProvider


class DatasetProperties:
    def __init__(
        self,
        variable_metadata: dict[str, VariableMetadata],
        vertical_coordinate: VerticalCoordinate,
        horizontal_coordinates: HorizontalCoordinates,
        mask_provider: MaskProvider,
        timestep: datetime.timedelta,
        is_remote: bool,
    ):
        self.variable_metadata = variable_metadata
        self.vertical_coordinate = vertical_coordinate
        self.horizontal_coordinates = horizontal_coordinates
        self.mask_provider = mask_provider
        self.timestep = timestep
        self.is_remote = is_remote

    def to_device(self) -> "DatasetProperties":
        device = get_device()
        return DatasetProperties(
            self.variable_metadata,
            self.vertical_coordinate.to(device),
            self.horizontal_coordinates.to(device),
            self.mask_provider.to(device),
            self.timestep,
            self.is_remote,
        )

    def update(self, other: "DatasetProperties"):
        self.is_remote = self.is_remote or other.is_remote
        if self.timestep != other.timestep:
            raise ValueError("Inconsistent timesteps between datasets")
        if self.variable_metadata != other.variable_metadata:
            raise ValueError("Inconsistent metadata between datasets")
        if self.vertical_coordinate != other.vertical_coordinate:
            raise ValueError("Inconsistent vertical coordinates between datasets")
        if self.horizontal_coordinates != other.horizontal_coordinates:
            raise ValueError("Inconsistent horizontal coordinates between datasets")
        if self.mask_provider != other.mask_provider:
            raise ValueError("Inconsistent mask providers between datasets")

    def update_merged_dataset(self, other: "DatasetProperties"):
        if isinstance(self.variable_metadata, dict):
            self.variable_metadata.update(other.variable_metadata)
        self.mask_provider.update(other.mask_provider)
        self.is_remote = self.is_remote or other.is_remote
        if self.timestep != other.timestep:
            raise ValueError("Inconsistent timesteps between datasets")
        if self.horizontal_coordinates != other.horizontal_coordinates:
            raise ValueError("Inconsistent horizontal coordinates between datasets")
