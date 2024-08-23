import abc
from typing import Optional

import torch

from fme.core import metrics
from fme.core.device import get_device


class GriddedOperations(abc.ABC):
    def area_weighted_mean(self, data: torch.Tensor) -> torch.Tensor:
        ...

    def area_weighted_mean_bias(
        self, predicted: torch.Tensor, truth: torch.Tensor
    ) -> torch.Tensor:
        return self.area_weighted_mean(predicted - truth)

    def area_weighted_rmse(
        self, predicted: torch.Tensor, truth: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self.area_weighted_mean((predicted - truth) ** 2))


class LatLonOperations(GriddedOperations):
    def __init__(self, area_weights: torch.Tensor):
        self._device_area = area_weights.to(get_device())
        self._cpu_area = area_weights.to("cpu")

    def area_weighted_mean(self, data: torch.Tensor) -> torch.Tensor:
        if data.device.type == "cpu":
            area_weights = self._cpu_area
        else:
            area_weights = self._device_area
        return metrics.weighted_mean(data, area_weights, dim=(-2, -1))


class HEALPixOperations(GriddedOperations):
    def area_weighted_mean(self, data: torch.Tensor) -> torch.Tensor:
        return data.mean(dim=(-3, -2, -1))


def get_gridded_operations(area_weights: Optional[torch.Tensor]) -> GriddedOperations:
    if area_weights is None:
        return HEALPixOperations()
    return LatLonOperations(area_weights)
