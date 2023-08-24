import dataclasses
from typing import Any, List, Literal, Mapping, Optional

import torch

from fme.core.device import get_device
from fme.fcn_training.utils.darcy_loss import LpLoss


class WeightedSum(torch.nn.Module):
    """
    A module which applies multiple loss-function modules (taking two inputs)
    to the same input and returns a tensor equal to the weighted sum of the
    outputs of the modules.
    """

    def __init__(self, modules: List[torch.nn.Module], weights: List[float]):
        """
        Args:
            modules: A list of modules, each of which takes two tensors and
                returns a scalar tensor.
            weights: A list of weights to apply to the outputs of the modules.
        """
        super().__init__()
        if len(modules) != len(weights):
            raise ValueError("modules and weights must have the same length")
        self._wrapped = modules
        self._weights = weights

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return sum(w * module(x, y) for w, module in zip(self._weights, self._wrapped))


class GlobalMeanLoss(torch.nn.Module):
    """
    A module which computes a loss on the global mean of each sample.
    """

    def __init__(self, area: torch.Tensor, loss: torch.nn.Module):
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
            loss: A loss function which takes two tensors of shape
                (n_samples, n_timesteps, n_channels) and returns a scalar
                tensor.
        """
        super().__init__()
        self.global_mean = GlobalMean(area)
        self.loss = loss

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.global_mean(x)
        y = self.global_mean(y)
        return self.loss(x, y)


class GlobalMean(torch.nn.Module):
    def __init__(self, area: torch.Tensor):
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
        """
        super().__init__()
        self.area_weights = area / area.sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor of shape (n_samples, n_timesteps, n_channels, n_lat, n_lon)
        """
        return (x * self.area_weights[None, None, None, :, :]).sum(dim=(3, 4))


@dataclasses.dataclass
class LossConfig:
    """
    A dataclass containing all the information needed to build a loss function,
    including the type of the loss function and the data needed to build it.

    Args:
        type: the type of the loss function
        kwargs: data for a loss function instance of the indicated type
        global_mean_type: the type of the loss function to apply to the global
            mean of each sample, by default no loss is applied
        global_mean_kwargs: data for a loss function instance of the indicated
            type to apply to the global mean of each sample
        global_mean_weight: the weight to apply to the global mean loss
            relative to the main loss
    """

    type: Literal["LpLoss"] = "LpLoss"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Optional[Literal["LpLoss"]] = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {}
    )
    global_mean_weight: float = 1.0

    def __post_init__(self):
        if self.type != "LpLoss":
            raise NotImplementedError()
        if self.global_mean_type is not None and self.global_mean_type != "LpLoss":
            raise NotImplementedError()

    def build(self, area: torch.Tensor) -> Any:
        """
        Args:
            area: A tensor of shape (n_lat, n_lon) containing the area of
                each grid cell.
        """
        main_loss = LpLoss(**self.kwargs)
        if self.global_mean_type is not None:
            global_mean_loss = GlobalMeanLoss(
                area=area.to(get_device()), loss=LpLoss(**self.global_mean_kwargs)
            )
            final_loss = WeightedSum(
                modules=[main_loss, global_mean_loss],
                weights=[1.0, self.global_mean_weight],
            )
        else:
            final_loss = main_loss
        return final_loss.to(device=get_device())
