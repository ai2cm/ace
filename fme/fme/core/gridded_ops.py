import abc
from typing import Any, Dict, List, Type, TypeVar

import torch

from fme.core import metrics
from fme.core.device import get_device


class GriddedOperations(abc.ABC):
    @abc.abstractmethod
    def area_weighted_mean(
        self, data: torch.Tensor, keepdim: bool = False
    ) -> torch.Tensor: ...

    def area_weighted_mean_bias(
        self, truth: torch.Tensor, predicted: torch.Tensor
    ) -> torch.Tensor:
        return self.area_weighted_mean(predicted - truth)

    def area_weighted_rmse(
        self, truth: torch.Tensor, predicted: torch.Tensor
    ) -> torch.Tensor:
        return torch.sqrt(self.area_weighted_mean((predicted - truth) ** 2))

    def area_weighted_std(self, data: torch.Tensor, keepdim: bool = False):
        return self.area_weighted_mean(
            (data - self.area_weighted_mean(data, keepdim=True)) ** 2,
            keepdim=keepdim,
        ).sqrt()

    @abc.abstractmethod
    def area_weighted_gradient_magnitude_percent_diff(
        self, truth: torch.Tensor, predicted: torch.Tensor
    ): ...

    def to_state(self) -> Dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "state": self.get_initialization_kwargs(),
        }

    @abc.abstractmethod
    def get_initialization_kwargs(self) -> Dict[str, Any]:
        """
        Get the keyword arguments needed to initialize the instance.
        """
        ...

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "GriddedOperations":
        """
        Given a dictionary with a "type" key and a "state" key, return
        the GriddedOperations it describes.

        The "type" key should be the name of a subclass of GriddedOperations,
        and the "state" key should be a dictionary specific to
        that subclass.

        Args:
            state: A dictionary with a "type" key and a "state" key.

        Returns:
            An instance of the subclass.
        """
        if cls is not GriddedOperations:
            raise RuntimeError(
                "This method should be called on GriddedOperations, "
                "not on its subclasses."
            )
        subclasses = get_all_subclasses(cls)
        for subclass in subclasses:
            if subclass.__name__ == state["type"]:
                return subclass(**state["state"])
        raise ValueError(
            f"Unknown subclass type: {state['type']}, "
            f"available: {[s.__name__ for s in subclasses]}"
        )


T = TypeVar("T")


def get_all_subclasses(cls: Type[T]) -> List[Type[T]]:
    """
    Gets all subclasses of a given class, including their subclasses etc.
    """
    all_subclasses = []
    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))
    return all_subclasses


class LatLonOperations(GriddedOperations):
    HORIZONTAL_DIMS = (-2, -1)

    def __init__(self, area_weights: torch.Tensor):
        self._device_area = area_weights.to(get_device())
        self._cpu_area = area_weights.to("cpu")

    def area_weighted_mean(
        self, data: torch.Tensor, keepdim: bool = False
    ) -> torch.Tensor:
        if data.device.type == "cpu":
            area_weights = self._cpu_area
        else:
            area_weights = self._device_area
        return metrics.weighted_mean(
            data, area_weights, dim=self.HORIZONTAL_DIMS, keepdim=keepdim
        )

    def area_weighted_gradient_magnitude_percent_diff(
        self, truth: torch.Tensor, predicted: torch.Tensor
    ):
        if predicted.device.type == "cpu":
            area_weights = self._cpu_area
        else:
            area_weights = self._device_area
        return metrics.gradient_magnitude_percent_diff(
            truth, predicted, weights=area_weights, dim=self.HORIZONTAL_DIMS
        )

    def get_initialization_kwargs(self) -> Dict[str, Any]:
        return {"area_weights": self._cpu_area}


class HEALPixOperations(GriddedOperations):
    HORIZONTAL_DIMS = (-3, -2, -1)

    def area_weighted_mean(
        self, data: torch.Tensor, keepdim: bool = False
    ) -> torch.Tensor:
        return data.mean(dim=self.HORIZONTAL_DIMS, keepdim=keepdim)

    def area_weighted_gradient_magnitude_percent_diff(
        self, truth: torch.Tensor, predicted: torch.Tensor
    ):
        return metrics.gradient_magnitude_percent_diff(
            truth, predicted, weights=None, dim=self.HORIZONTAL_DIMS
        )

    def get_initialization_kwargs(self) -> Dict[str, Any]:
        return {}
