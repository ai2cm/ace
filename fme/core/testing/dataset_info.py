import datetime

import torch

from fme.core.coordinates import (
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
    VerticalCoordinate,
)
from fme.core.dataset_info import DatasetInfo
from fme.core.spatial_mask_provider import SpatialMaskProvider

TIMESTEP = datetime.timedelta(hours=6)


def get_dataset_info(
    img_shape: tuple[int, int] = (5, 5),
    spatial_mask_provider: SpatialMaskProvider | None = None,
    vertical_coordinate: VerticalCoordinate | None = None,
    horizontal_coordinate: HorizontalCoordinates | None = None,
    timestep: datetime.timedelta = TIMESTEP,
    all_labels: set[str] | None = None,
    device: torch.device | None = None,
) -> DatasetInfo:
    """
    Create a DatasetInfo with placeholder coordinates for testing.
    """
    if horizontal_coordinate is None:
        horizontal_coordinate = LatLonCoordinates(
            lat=torch.zeros(img_shape[-2], device=device),
            lon=torch.zeros(img_shape[-1], device=device),
        )
    if vertical_coordinate is None:
        vertical_coordinate = HybridSigmaPressureCoordinate(
            ak=torch.arange(7, device=device), bk=torch.arange(7, device=device)
        )
    return DatasetInfo(
        horizontal_coordinates=horizontal_coordinate,
        vertical_coordinate=vertical_coordinate,
        timestep=timestep,
        spatial_mask_provider=spatial_mask_provider,
        all_labels=all_labels,
    )
