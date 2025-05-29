import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
import torch.linalg

from fme.core.device import get_device
from fme.core.ensemble import get_crps, get_energy_score
from fme.core.gridded_ops import GriddedOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorMapping


class NaNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.tensor(torch.nan)


class WeightedMappingLoss:
    def __init__(
        self,
        loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        weights: dict[str, float],
        out_names: list[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ):
        """
        Args:
            loss: The loss function to apply.
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
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
    ):
        predict_tensors = self.packer.pack(
            self.normalizer.normalize(predict_dict), axis=self.channel_dim
        )
        target_tensors = self.packer.pack(
            self.normalizer.normalize(target_dict), axis=self.channel_dim
        )
        nan_mask = target_tensors.isnan()
        if nan_mask.any():
            predict_tensors = torch.where(nan_mask, 0.0, predict_tensors)
            target_tensors = torch.where(nan_mask, 0.0, target_tensors)

        return self.loss(predict_tensors, target_tensors)

    def get_normalizer_state(self) -> dict[str, float]:
        return self.normalizer.get_state()

    @property
    def effective_loss_scaling(self) -> dict[str, float]:
        custom_weights = dict(zip(self.packer.names, self._weight_tensor.flatten()))
        loss_normalizer_stds = self.normalizer.stds
        return {
            k: loss_normalizer_stds[k] / custom_weights[k] for k in self.packer.names
        }


def _construct_weight_tensor(
    weights: dict[str, float],
    out_names: list[str],
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


class LpLoss(torch.nn.Module):
    def __init__(self, p=2):
        """
        Args:
            p: Lp-norm type. For example, p=1 for L1-norm, p=2 for L2-norm.
        """
        super().__init__()

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
        super().__init__()
        self._area_weighted_mean = area_weighted_mean

    def __call__(self, x, y):
        return torch.mean(self._area_weighted_mean((x - y) ** 2))


class WeightedSum(torch.nn.Module):
    """
    A module which applies multiple loss-function modules (taking two inputs)
    to the same input and returns a tensor equal to the weighted sum of the
    outputs of the modules.
    """

    def __init__(self, modules: list[torch.nn.Module], weights: list[float]):
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


class EnergyScoreLoss(torch.nn.Module):
    """
    Compute the energy score over the complex-valued spectral coefficients.

    The energy score is defined as

    .. math::

        E[||X - y||^{beta}] - 1/2 E[||X - X'||^{beta}]

    where :math:`X` is the ensemble, :math:`y` is the target, and :math:`||.||`
    is the complex modulus. It is a proper scoring rule for beta in (0, 2). Here
    we use beta=1. See Gneiting and Raftery (2007) [1]_ Section 4.3 for more details.

    We use a scaling factor of 2 * sqrt(n_l * n_m) to bring its magnitude in
    line with the real-valued CRPS loss, and to prevent its value depending on domain
    size for Gaussian distributed random data where n_lon = 2 * n_lat.

    .. [1] https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
    """

    def __init__(self, sht: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.sht = sht
        self.scaling: torch.Tensor | None = None
        self.mode_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_hat = self.sht(x)
        y_hat = self.sht(y)
        if self.scaling is None:
            self.scaling = 2 * (x_hat.shape[-2] * x_hat.shape[-1]) ** 0.5
        if self.mode_weights is None:
            # we need to weight the modes properly,
            # with each m mode contributing twice for m!=0:
            H, W = x_hat.shape[-2:]
            self.mode_weights = 2 * torch.ones(
                (*([1] * (x_hat.ndim - 1)), H, W),
                device=x_hat.device,
            )
            self.mode_weights[..., 0] = 1
        return (
            (get_energy_score(x_hat, y_hat) * self.mode_weights)
            .sum(dim=(-2, -1))
            .mean()
        ) / self.scaling


class CRPSLoss(torch.nn.Module):
    """
    Compute the CRPS loss.

    Supports almost-fair modification to CRPS from
    https://arxiv.org/html/2412.15832v1, which claims to be helpful in
    avoiding numerical issues with fair CRPS.
    """

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return get_crps(x, y, alpha=self.alpha).mean()


class EnsembleLoss(torch.nn.Module):
    def __init__(
        self,
        crps_weight: float,
        energy_score_weight: float,
        sht: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        if crps_weight < 0 or energy_score_weight < 0:
            raise ValueError(
                "crps_weight and energy_score_weight must be non-negative, "
                f"got {crps_weight} and {energy_score_weight}"
            )
        if crps_weight + energy_score_weight == 0:
            raise ValueError(
                "crps_weight and energy_score_weight must sum to a positive value, "
                f"got {crps_weight} and {energy_score_weight}"
            )
        self.crps_loss = CRPSLoss(alpha=0.95)
        self.energy_score_loss = EnergyScoreLoss(sht=sht)

        self.crps_weight = crps_weight
        self.energy_score_weight = energy_score_weight

    def forward(
        self,
        gen_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ):
        if self.crps_weight > 0:
            crps = self.crps_weight * self.crps_loss(gen_norm, target_norm)
        else:
            crps = torch.tensor(0.0)
        if self.energy_score_weight > 0:
            energy_score_loss = self.energy_score_weight * self.energy_score_loss(
                gen_norm, target_norm
            )
        else:
            energy_score_loss = torch.tensor(0.0)
        return crps + energy_score_loss


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

    type: Literal["LpLoss", "L1", "MSE", "AreaWeightedMSE", "NaN", "EnsembleLoss"] = (
        "MSE"
    )
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Literal["LpLoss"] | None = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {}
    )
    global_mean_weight: float = 1.0

    def __post_init__(self):
        if self.type not in (
            "LpLoss",
            "L1",
            "MSE",
            "AreaWeightedMSE",
            "NaN",
            "EnsembleLoss",
        ):
            raise NotImplementedError(self.type)
        if self.global_mean_type is not None and self.global_mean_type != "LpLoss":
            raise NotImplementedError(self.global_mean_type)

    def build(
        self,
        reduction: Literal["mean", "none"],
        gridded_operations: GriddedOperations | None,
    ) -> Any:
        """
        Args:
            reduction: The reduction to apply to the loss, either "mean" or "none".
                Only used if the loss function is L1, MSE, or LpLoss.
            gridded_operations: The gridded operations to use in the case that
                the loss function requires use of the horizontal dimensions.
        """
        if self.type == "LpLoss":
            main_loss = LpLoss(**self.kwargs)
        elif self.type == "L1":
            main_loss = torch.nn.L1Loss(reduction=reduction)
        elif self.type == "MSE":
            main_loss = torch.nn.MSELoss(reduction=reduction)
        elif self.type == "AreaWeightedMSE":
            if gridded_operations is None:
                raise ValueError("gridded_operations is required for AreaWeightedMSE")
            main_loss = AreaWeightedMSELoss(gridded_operations.area_weighted_mean)
        elif self.type == "NaN":
            main_loss = NaNLoss()
        elif self.type == "EnsembleLoss":
            if gridded_operations is None:
                raise ValueError("gridded_operations is required for EnsembleLoss")
            kwargs = dict(self.kwargs)
            crps_weight = kwargs.pop("crps_weight", 1.0)
            energy_score_weight = kwargs.pop("energy_score_weight", 0.0)
            main_loss = EnsembleLoss(
                sht=gridded_operations.get_real_sht(),
                crps_weight=crps_weight,
                energy_score_weight=energy_score_weight,
                **kwargs,
            )

        if self.global_mean_type is not None:
            if gridded_operations is None:
                raise ValueError("gridded_operations is required for global mean loss")
            global_mean_loss = GlobalMeanLoss(
                area_weighted_mean=gridded_operations.area_weighted_mean,
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

    type: Literal["LpLoss", "MSE", "AreaWeightedMSE", "EnsembleLoss"] = "MSE"
    kwargs: Mapping[str, Any] = dataclasses.field(default_factory=lambda: {})
    global_mean_type: Literal["LpLoss"] | None = None
    global_mean_kwargs: Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: {}
    )
    global_mean_weight: float = 1.0
    weights: dict[str, float] = dataclasses.field(default_factory=lambda: {})

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
        gridded_ops: GriddedOperations | None,
        out_names: list[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ) -> WeightedMappingLoss:
        loss = self.loss_config.build(
            reduction="mean",
            gridded_operations=gridded_ops,
        )
        return WeightedMappingLoss(
            loss=loss,
            weights=self.weights,
            out_names=out_names,
            channel_dim=channel_dim,
            normalizer=normalizer,
        )
