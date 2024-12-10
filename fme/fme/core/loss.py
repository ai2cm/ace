import dataclasses
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional

import torch
import torch.linalg

from fme.core.device import get_device
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict


class NaNLoss(torch.nn.Module):
    def __init__(self):
        super(NaNLoss, self).__init__()

    def forward(self, input, target):
        return torch.tensor(torch.nan)


class MappingLoss:
    def __init__(self, loss: torch.nn.Module, packer: Packer, channel_dim: int = -3):
        self.loss = loss
        self.packer = packer
        self.channel_dim = channel_dim

    def __call__(
        self,
        predict_dict: TensorDict,
        target_dict: TensorDict,
    ):
        predict_tensors = self.packer.pack(predict_dict, axis=self.channel_dim)
        target_tensors = self.packer.pack(target_dict, axis=self.channel_dim)

        return self.loss(predict_tensors, target_tensors)


def _construct_weight_tensor(
    weights: Dict[str, float],
    out_names: List[str],
    n_dim: int = 4,
    channel_dim: int = -3,
) -> torch.Tensor:
    """Creates a packed weight tensor with the appropriate dimensions for
    broadcasting with generated or target output tensors. When used in
    the n_forward_steps loop in the stepper's run_on_batch, the channel dim is
    -3 and the n_dim is 4 (sample, channel, lat, lon).

    Args:
        weights: dict of variable names with individual weights to apply
            to their normalized loss
        out_names: list of output variable names
        n_dim: number of dimensions of the output tensor
        channel_dim: the channel dimension of the output tensor
    """
    missing_keys = set(weights.keys()) - set(out_names)
    if len(missing_keys) > 0:
        raise KeyError(
            f"Variables {missing_keys} in loss weights not in "
            f"output variables list."
        )
    weights_tensor = torch.tensor([weights.get(key, 1.0) for key in out_names])
    # positive index of the channel dimension
    _channel_dim = n_dim + channel_dim if channel_dim < 0 else channel_dim
    reshape_dim = (
        len(weights_tensor) if i == _channel_dim else 1 for i in range(n_dim)
    )
    return weights_tensor.reshape(*reshape_dim).to(get_device(), dtype=torch.float)


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

        diff_norms = torch.linalg.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), ord=self.p, dim=1
        )
        y_norms = torch.linalg.norm(y.reshape(num_examples, -1), ord=self.p, dim=1)

        return torch.mean(diff_norms / y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)


class AreaWeightedMSELoss(torch.nn.Module):
    def __init__(self, area_weighted_mean: Callable[[torch.Tensor], torch.Tensor]):
        super(AreaWeightedMSELoss, self).__init__()
        self._area_weighted_mean = area_weighted_mean

    def __call__(self, x, y):
        return torch.mean(self._area_weighted_mean((x - y) ** 2))


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

    def __init__(
        self,
        area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
        loss: torch.nn.Module,
    ):
        """
        Args:
            area_weighted_mean: Computes an area-weighted mean, removing the
                horizontal dimensions.
            loss: A loss function which takes two tensors of shape
                (n_samples, n_timesteps, n_channels) and returns a scalar
                tensor.
        """
        super().__init__()
        self.global_mean = GlobalMean(area_weighted_mean)
        self.loss = loss

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.global_mean(x)
        y = self.global_mean(y)
        return self.loss(x, y)


class GlobalMean(torch.nn.Module):
    def __init__(self, area_weighted_mean: Callable[[torch.Tensor], torch.Tensor]):
        """
        Args:
            area_weighted_mean: Computes an area-weighted mean, removing the
                horizontal dimensions.
        """
        super().__init__()
        self._area_weighted_mean = area_weighted_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: A tensor with spatial dimensions in shape (n_samples, n_timesteps,
             n_channels, n_lat, n_lon).
        """
        return self._area_weighted_mean(x)


class VariableWeightingLoss(torch.nn.Module):
    def __init__(self, weights: torch.Tensor, loss: torch.nn.Module):
        """
        Args:
            weights: A tensor of shape (n_samples, n_channels, n_lat, n_lon)
                containing the weights to apply to each channel.
            loss: A loss function which takes two tensors.
        """
        super().__init__()
        self.loss = loss
        self.weights = weights

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.loss(self.weights * x, self.weights * y)


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

    type: Literal["LpLoss", "L1", "MSE", "AreaWeightedMSE", "NaN"] = "MSE"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Optional[Literal["LpLoss"]] = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {}
    )
    global_mean_weight: float = 1.0

    def __post_init__(self):
        if self.type not in ("LpLoss", "L1", "MSE", "AreaWeightedMSE", "NaN"):
            raise NotImplementedError(self.type)
        if self.global_mean_type is not None and self.global_mean_type != "LpLoss":
            raise NotImplementedError(self.global_mean_type)

    def build(
        self,
        area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
        reduction: Literal["mean", "none"],
    ) -> Any:
        """
        Args:
            area_weighted_mean: Computes an area-weighted mean, removing the
                horizontal dimensions. Only used if the loss function is
                AreaWeightedMSE.
            reduction: The reduction to apply to the loss, either "mean" or "none".
                Only used if the loss function is L1, MSE, or LpLoss.
        """
        if self.type == "LpLoss":
            main_loss = LpLoss(**self.kwargs)
        elif self.type == "L1":
            main_loss = torch.nn.L1Loss(reduction=reduction)
        elif self.type == "MSE":
            main_loss = torch.nn.MSELoss(reduction=reduction)
        elif self.type == "AreaWeightedMSE":
            main_loss = AreaWeightedMSELoss(area_weighted_mean)
        elif self.type == "NaN":
            main_loss = NaNLoss()

        if self.global_mean_type is not None:
            global_mean_loss = GlobalMeanLoss(
                area_weighted_mean=area_weighted_mean,
                loss=LpLoss(**self.global_mean_kwargs),
            )
            final_loss = WeightedSum(
                modules=[main_loss, global_mean_loss],
                weights=[1.0, self.global_mean_weight],
            )
        else:
            final_loss = main_loss
        return final_loss.to(device=get_device())


@dataclasses.dataclass
class WeightedMappingLossConfig:
    """
    Loss configuration class that has the same fields as LossConfig but also
    has additional weights field. The build method will apply the weights to
    the inputs of the loss function. The loss returned by build will be a
    MappingLoss, which takes Dict[str, tensor] as inputs instead of packed
    tensors.

    Args:
        type: the type of the loss function
        kwargs: data for a loss function instance of the indicated type
        global_mean_type: the type of the loss function to apply to the global
            mean of each sample, by default no loss is applied
        global_mean_kwargs: data for a loss function instance of the indicated
            type to apply to the global mean of each sample
        global_mean_weight: the weight to apply to the global mean loss
            relative to the main loss
        weights: A dictionary of variable names with individual
            weights to apply to their normalized losses

    """

    type: Literal["LpLoss", "MSE", "AreaWeightedMSE"] = "MSE"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Optional[Literal["LpLoss"]] = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {}
    )
    global_mean_weight: float = 1.0
    weights: Dict[str, float] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        self.loss_config = LossConfig(
            type=self.type,
            kwargs=self.kwargs,
            global_mean_type=self.global_mean_type,
            global_mean_kwargs=self.global_mean_kwargs,
            global_mean_weight=self.global_mean_weight,
        )

    def build(
        self,
        area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
        out_names: List[str],
        channel_dim: int = -3,
    ) -> Any:
        loss = self.loss_config.build(area_weighted_mean, reduction="mean")
        weighted_loss = VariableWeightingLoss(
            weights=_construct_weight_tensor(
                self.weights, out_names, channel_dim=channel_dim
            ),
            loss=loss,
        )
        packer = Packer(out_names)
        return MappingLoss(loss=weighted_loss, packer=packer, channel_dim=channel_dim)
