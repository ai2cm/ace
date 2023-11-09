import dataclasses
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

import torch

from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device

from .climate_data import ClimateData, compute_dry_air_absolute_differences


def get_dry_air_nonconservation(
    data: Mapping[str, torch.Tensor],
    area_weights: torch.Tensor,
    sigma_coordinates: SigmaCoordinates,
):
    """
    Computes the time-average one-step absolute difference in surface pressure due to
    changes in globally integrated dry air.

    Args:
        data: A mapping from variable name to tensor of shape
            [sample, time, lat, lon], in physical units. specific_total_water in kg/kg
            and surface_pressure in Pa must be present.
        area_weights: The area of each grid cell as a [lat, lon] tensor, in m^2.
        sigma_coordinates: The sigma coordinates of the model.
    """
    return compute_dry_air_absolute_differences(
        ClimateData(data), area=area_weights, sigma_coordinates=sigma_coordinates
    ).mean()


class ConservationLoss:
    def __init__(
        self,
        dry_air_penalty: Optional[float],
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
    ):
        """
        Args:
            dry_air_penalty: A constant by which to multiply one-step non-conservation
                of surface pressure due to dry air in Pa as an L1 loss penalty. By
                default, no such loss will be included.
            area_weights: The area of each grid cell as a [lat, lon] tensor, in m^2.
            sigma_coordinates: The sigma coordinates of the model.
        """
        self._dry_air_penalty = dry_air_penalty
        self._area_weights = area_weights.to(get_device())
        self._sigma_coordinates = sigma_coordinates.to(get_device())

    def __call__(
        self, gen_data: Mapping[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Compute loss and metrics related to conservation.

        Args:
            gen_data: A mapping from variable name to tensor of shape
                [sample, time, lat, lon], in physical units.
        """
        conservation_metrics = {}
        loss = torch.tensor(0.0, device=get_device())
        if self._dry_air_penalty is not None:
            dry_air_loss = self._dry_air_penalty * get_dry_air_nonconservation(
                gen_data,
                area_weights=self._area_weights,
                sigma_coordinates=self._sigma_coordinates,
            )
            conservation_metrics["dry_air_loss"] = dry_air_loss.detach()
            loss += dry_air_loss
        return conservation_metrics, loss

    def get_state(self):
        return {
            "dry_air_penalty": self._dry_air_penalty,
            "sigma_coordinates": self._sigma_coordinates,
            "area_weights": self._area_weights,
        }

    @classmethod
    def from_state(cls, state) -> "ConservationLoss":
        return cls(**state)


@dataclasses.dataclass
class ConservationLossConfig:
    """
    Attributes:
        dry_air_penalty: A constant by which to multiply one-step non-conservation
            of surface pressure due to dry air in Pa as an L1 loss penalty. By
            default, no such loss will be included.
    """

    dry_air_penalty: Optional[float] = None

    def build(
        self, area_weights: torch.Tensor, sigma_coordinates: SigmaCoordinates
    ) -> ConservationLoss:
        return ConservationLoss(
            dry_air_penalty=self.dry_air_penalty,
            area_weights=area_weights,
            sigma_coordinates=sigma_coordinates,
        )


class LpLoss(torch.nn.Module):
    def __init__(self, p=2):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
        """
        super(LpLoss, self).__init__()

        if p <= 0:
            raise ValueError("Lp-norm type should be positive")

        self.p = p

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        return torch.mean(diff_norms / y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)


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
