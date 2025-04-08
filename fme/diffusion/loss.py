import dataclasses
from typing import Any, Callable, Dict, List, Literal, Mapping

import torch
import torch.linalg

from fme.core.device import get_device
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict


class NaNLoss(torch.nn.Module):
    def __init__(self):
        super(NaNLoss, self).__init__()

    def forward(self, input, target, batch_weights):
        return torch.tensor(torch.nan)


class WeightedMappingLoss:
    def __init__(
        self,
        loss: torch.nn.Module,
        weights: Dict[str, float],
        out_names: List[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ):
        """
        Args:
            loss: The loss function to apply, taking predict, target, and batch weight
                tensors as inputs.
            weights: A dictionary of variable names with individual
                weights to apply to their normalized losses
            out_names: The names of the output variables.
            normalizer: The normalizer to use.
            channel_dim: The channel dimension of the input tensors.
        """
        self._weight_tensor = _construct_weight_tensor(
            weights, out_names, channel_dim=channel_dim
        )
        self.loss = VariableWeightingLoss(
            weights=self._weight_tensor,
            loss=loss,
        )
        if self._weight_tensor.flatten().shape[0] != len(out_names):
            raise RuntimeError(
                "The number of weights must match the number of output names, "
                "behavior of _construct_weight_tensor has changed."
            )
        self.packer = Packer(out_names)
        self.channel_dim = channel_dim
        self.normalizer = normalizer

    def __call__(
        self,
        predict_dict: TensorDict,
        target_dict: TensorDict,
        batch_weights: torch.Tensor,
    ):
        """
        Args:
            predict_dict: A dictionary of predicted tensors.
            target_dict: A dictionary of target tensors.
            batch_weights: A tensor which can be broadcasted to the shape of
                the predicted and target tensors. This will be multiplied with each
                element (without normalizing the weights) before reducing the loss
                but after any nonlinear operations.
        """
        predict_tensors = self.packer.pack(
            self.normalizer.normalize(predict_dict), axis=self.channel_dim
        )
        target_tensors = self.packer.pack(
            self.normalizer.normalize(target_dict), axis=self.channel_dim
        )

        return self.loss(predict_tensors, target_tensors, batch_weights[:, None])

    def get_normalizer_state(self) -> Dict[str, float]:
        return self.normalizer.get_state()

    @property
    def effective_loss_scaling(self) -> Dict[str, float]:
        custom_weights = dict(zip(self.packer.names, self._weight_tensor.flatten()))
        loss_normalizer_stds = self.normalizer.stds
        return {
            k: loss_normalizer_stds[k] / custom_weights[k] for k in self.packer.names
        }


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
    weights_tensor = torch.tensor([weights.get(key, 1.0) for key in out_names])
    # positive index of the channel dimension
    _channel_dim = n_dim + channel_dim if channel_dim < 0 else channel_dim
    reshape_dim = (
        len(weights_tensor) if i == _channel_dim else 1 for i in range(n_dim)
    )
    return weights_tensor.reshape(*reshape_dim).to(get_device(), dtype=torch.float)


class AreaWeightedMSELoss(torch.nn.Module):
    def __init__(self, area_weighted_mean: Callable[[torch.Tensor], torch.Tensor]):
        super(AreaWeightedMSELoss, self).__init__()
        self._area_weighted_mean = area_weighted_mean

    def forward(self, x, y, weights):
        return torch.mean(self._area_weighted_mean(torch.square(x - y) * weights))


class MSELoss(torch.nn.Module):
    def forward(self, x, y, weights):
        return torch.mean(torch.square(x - y) * weights)


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

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        batch_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss(self.weights * x, self.weights * y, batch_weights)


@dataclasses.dataclass
class LossConfig:
    """
    A dataclass containing all the information needed to build a loss function,
    including the type of the loss function and the data needed to build it.

    Args:
        type: the type of the loss function
        kwargs: data for a loss function instance of the indicated type
    """

    type: Literal["MSE", "AreaWeightedMSE", "NaN"] = "MSE"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        if self.type not in ("MSE", "AreaWeightedMSE", "NaN"):
            raise NotImplementedError(self.type)

    def build(
        self,
        area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:
        """
        Args:
            area_weighted_mean: Computes an area-weighted mean, removing the
                horizontal dimensions. Only used if the loss function is
                AreaWeightedMSE.
        """
        if self.type == "MSE":
            loss = MSELoss()
        elif self.type == "AreaWeightedMSE":
            loss = AreaWeightedMSELoss(area_weighted_mean)
        elif self.type == "NaN":
            loss = NaNLoss()

        return loss.to(device=get_device())


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, area_weighted_mean: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.area_weighted_mean = area_weighted_mean

    def forward(self, x, y, weights):
        return torch.mean(self.area_weighted_mean(torch.square(x - y) * weights))


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
        weights: A dictionary of variable names with individual
            weights to apply to their normalized losses

    """

    type: Literal["MSE", "AreaWeightedMSE", "NaN"] = "MSE"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    weights: Dict[str, float] = dataclasses.field(default_factory=lambda: {})

    def __post_init__(self):
        self.loss_config = LossConfig(
            type=self.type,
            kwargs=self.kwargs,
        )

    def build(
        self,
        area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
        out_names: List[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ) -> Any:
        loss = self.loss_config.build(area_weighted_mean)
        weighted_loss = VariableWeightingLoss(
            weights=_construct_weight_tensor(
                self.weights, out_names, channel_dim=channel_dim
            ),
            loss=loss,
        )
        return WeightedMappingLoss(
            loss=weighted_loss,
            weights=self.weights,
            out_names=out_names,
            normalizer=normalizer,
            channel_dim=channel_dim,
        )
