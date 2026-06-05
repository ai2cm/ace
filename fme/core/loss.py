import abc
import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
import torch.linalg
import torch.nn.functional as F

from fme.core.device import get_device
from fme.core.ensemble import get_crps, get_energy_score
from fme.core.gridded_ops import GriddedOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorMapping


@dataclasses.dataclass
class ChannelLossInfo:
    """Per-channel loss value and the number of batch samples that contributed."""

    loss: torch.Tensor
    count: int


class LossComponent(abc.ABC):
    """A pre-weighted loss tensor that knows how to reduce itself to ``(B, C)``.

    All loss tensors are pre-weighted so that ``.mean()`` over trailing
    (non-batch, non-channel) dimensions gives the correct per-sample,
    per-channel loss. Subclasses encode the tensor layout (where the
    channel dimension lives) and implement :meth:`reduce_to_channel`.
    """

    def __init__(self, loss: torch.Tensor):
        """
        Args:
            loss: The loss tensor. Can be a scalar (``ndim == 0``),
                a partially-reduced tensor like ``(B, C)``, or a
                full element-wise tensor like ``(B, C, lat, lon)``.
        """
        self.loss = loss

    @abc.abstractmethod
    def reduce_to_channel(self) -> torch.Tensor:
        """Reduce to ``(B, C)`` by meaning over non-batch, non-channel dims."""


class StandardLoss(LossComponent):
    """Standard ``(B, C, ...)`` layout with channel at dim 1."""

    def reduce_to_channel(self) -> torch.Tensor:
        if self.loss.ndim <= 2:
            return self.loss
        return self.loss.mean(dim=tuple(range(2, self.loss.ndim)))


class EnsembleComponentLoss(LossComponent):
    """Ensemble ``(B, E, C, ...)`` layout with channel at dim 2."""

    def reduce_to_channel(self) -> torch.Tensor:
        dims = tuple(i for i in range(self.loss.ndim) if i not in (0, 2))
        return self.loss.mean(dim=dims) if dims else self.loss


class LossOutput:
    """Container for loss values returned by WeightedMappingLoss/StepLoss.

    Holds one or more :class:`LossComponent` instances and provides
    convenience methods for the scalar total and per-channel breakdowns.

    Reduction is computed once and cached: ``total()`` derives from
    the cached per-channel values so they are always consistent.
    """

    def __init__(
        self,
        losses: list[LossComponent],
        channel_names: list[str],
        mask: torch.Tensor | None = None,
    ):
        self._losses = losses
        self._channel_names = channel_names
        self._mask = mask
        self._per_channel: torch.Tensor | None = None
        self._counts: list[int] | None = None

    def _reduce(self) -> tuple[torch.Tensor, list[int]]:
        """Return ``(per_channel, counts)`` tensors, computed once.

        When a mask is present (shape ``(B, C)``), each channel's loss
        is averaged only over the batch samples where that channel is
        present, so masked-out variables never dilute the result.
        """
        if self._per_channel is None:
            bc = sum(c.reduce_to_channel() for c in self._losses)
            assert isinstance(bc, torch.Tensor)
            if bc.ndim == 0:
                self._per_channel = bc.expand(len(self._channel_names))
                self._counts = [1] * len(self._channel_names)
            elif self._mask is not None:
                masked_sum = (bc * self._mask).sum(dim=0)
                self._per_channel = masked_sum / self._mask.sum(dim=0).clamp(min=1)
                self._counts = [int(c.item()) for c in self._mask.sum(dim=0)]
            else:
                self._per_channel = bc.mean(dim=0)
                self._counts = [bc.shape[0]] * len(self._channel_names)
        assert self._per_channel is not None and self._counts is not None
        return self._per_channel, self._counts

    def total(self) -> torch.Tensor:
        """Scalar loss used as the optimization target.

        This is the mean of the per-channel losses across channels (over
        active channels only when a mask is present), not a sum. Adding
        or removing channels therefore does not change the scale of the
        returned value.
        """
        pc, _ = self._reduce()
        if self._mask is not None:
            active = self._mask.sum(dim=0) > 0
            if active.any():
                return pc[active].mean()
        return pc.mean()

    def get_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Per-channel mean losses with active-sample counts.

        Each :class:`ChannelLossInfo` carries the mean loss for that
        channel (averaged over active samples only) and the number of
        batch samples that contributed. Downstream aggregators should
        use the counts to compute properly weighted means across
        batches.
        """
        pc, counts = self._reduce()
        n_channels = len(self._channel_names)
        if pc.ndim > 0 and pc.shape[0] != n_channels:
            raise RuntimeError(
                f"Per-channel loss has {pc.shape[0]} elements but "
                f"{n_channels} channel names were provided."
            )
        return {
            name: ChannelLossInfo(loss=pc[i], count=counts[i])
            for i, name in enumerate(self._channel_names)
        }

    def scale(self, weight: float) -> "LossOutput":
        """Return a new ``LossOutput`` with every component scaled."""
        return LossOutput(
            [type(c)(c.loss * weight) for c in self._losses],
            self._channel_names,
            mask=self._mask,
        )


class _MSELoss(torch.nn.Module):
    """MSE with ``reduction="none"`` that returns ``list[LossComponent]``."""

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.MSELoss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        return [StandardLoss(self._loss(x, y))]


class _L1Loss(torch.nn.Module):
    """L1 with ``reduction="none"`` that returns ``list[LossComponent]``."""

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.L1Loss(reduction="none")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        return [StandardLoss(self._loss(x, y))]


class NaNLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> list[LossComponent]:
        return [StandardLoss(torch.tensor(torch.nan))]


class WeightedMappingLoss:
    def __init__(
        self,
        loss: Callable[
            [torch.Tensor, torch.Tensor], list[LossComponent] | torch.Tensor
        ],
        weights: dict[str, float],
        out_names: list[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ):
        """
        Args:
            loss: The loss function to apply. Should return a
                ``list[LossComponent]``. Element-wise losses (e.g.
                ``torch.nn.MSELoss``) that return a raw tensor are also
                accepted and will be wrapped automatically based on
                *channel_dim*.
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
        data_mask: TensorMapping | None = None,
    ) -> LossOutput:
        """
        Args:
            predict_dict: The predicted data.
            target_dict: The target data.
            data_mask: Optional per-variable boolean masks of shape
                ``[batch]`` indicating which samples have each variable
                present. Used to exclude masked channels from the loss
                average.

        Returns:
            A ``LossOutput`` wrapping pre-weighted loss component tensors.
        """
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

        result = self.loss(predict_tensors, target_tensors)
        input_ndim = predict_tensors.ndim
        cdim = (
            input_ndim + self.channel_dim if self.channel_dim < 0 else self.channel_dim
        )

        def _wrap_elementwise(t: torch.Tensor) -> StandardLoss:
            # Element-wise loss tensors have the same shape as the input;
            # the channel position depends on the data layout (ensemble,
            # tile, etc.). Reduce non-(batch, channel) dims here so the
            # downstream component carries a canonical ``(B, C)`` tensor.
            dims = tuple(i for i in range(t.ndim) if i not in (0, cdim))
            reduced = t.mean(dim=dims) if dims else t
            return StandardLoss(reduced)

        if isinstance(result, list):
            # Inner losses that return raw element-wise tensors (e.g. MSE,
            # L1) wrap themselves in StandardLoss but don't know the input
            # channel layout, so reduce around the actual channel dim here.
            losses = [
                _wrap_elementwise(c.loss)
                if c.loss.ndim == input_ndim and type(c) is StandardLoss
                else c
                for c in result
            ]
        else:
            losses = [_wrap_elementwise(result)]

        mask = None
        if data_mask is not None:
            batch_size = predict_tensors.shape[0]
            device = predict_tensors.device
            filled: dict[str, torch.Tensor] = {}
            for name in self.packer.names:
                if name in data_mask:
                    filled[name] = data_mask[name].to(device=device, dtype=torch.float)
                else:
                    filled[name] = torch.ones(
                        batch_size, device=device, dtype=torch.float
                    )
            mask = self.packer.pack(filled, axis=1)

        return LossOutput(
            losses=losses,
            channel_names=list(self.packer.names),
            mask=mask,
        )

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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        B, C = x.shape[0], x.shape[1]
        x_flat = x.reshape(B, C, -1)
        y_flat = y.reshape(B, C, -1)
        diff_norms = torch.linalg.norm(x_flat - y_flat, ord=self.p, dim=2)
        y_norms = torch.linalg.norm(y_flat, ord=self.p, dim=2)
        return [StandardLoss(diff_norms / y_norms)]


class AreaWeightedMSELoss(torch.nn.Module):
    def __init__(self, area_weighted_mean: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self._area_weighted_mean = area_weighted_mean

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        return [StandardLoss(self._area_weighted_mean((x - y) ** 2))]


class WeightedSum(torch.nn.Module):
    """
    A module which applies multiple loss-function modules (taking two inputs)
    and returns their weighted components as a flat list.
    """

    def __init__(self, modules: list[torch.nn.Module], weights: list[float]):
        """
        Args:
            modules: A list of modules, each of which takes two tensors and
                returns a ``list[LossComponent]``.
            weights: A list of weights to apply to the outputs of the modules.
        """
        super().__init__()
        if len(modules) != len(weights):
            raise ValueError("modules and weights must have the same length")
        self._wrapped = modules
        self._weights = weights

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        components: list[LossComponent] = []
        for w, module in zip(self._weights, self._wrapped):
            for c in module(x, y):
                components.append(type(c)(c.loss * w))
        return components


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
                (n_samples, n_channels) and returns a
                ``list[LossComponent]``.
        """
        super().__init__()
        self.global_mean = GlobalMean(area_weighted_mean)
        self.loss = loss

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
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

    Returns a pre-weighted ``(B, C, L, M)`` tensor where ``.mean(dim=(-2, -1))``
    reproduces the old scalar value per ``(B, C)`` pair.

    .. [1] https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf
    """

    def __init__(self, sht: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.sht = sht
        self.scaling: float | None = None
        self.n_spectral: int | None = None
        self.mode_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        x_hat = self.sht(x)
        y_hat = self.sht(y)
        n_l, n_m = x_hat.shape[-2], x_hat.shape[-1]
        if self.scaling is None:
            self.scaling = 2 * (n_l * n_m) ** 0.5
            self.n_spectral = n_l * n_m
        if self.mode_weights is None:
            self.mode_weights = 2 * torch.ones(
                (*([1] * (x_hat.ndim - 1)), n_l, n_m),
                device=x_hat.device,
            )
            self.mode_weights[..., 0] = 1
        assert self.n_spectral is not None
        es = get_energy_score(x_hat, y_hat) * self.mode_weights
        # Old path: .sum(dim=(-2,-1)).mean() / scaling
        # New path: StandardLoss does .mean(dim=(-2,-1)) i.e. sum/(L*M)
        # Multiply by L*M/scaling so mean gives the same result as sum/scaling.
        pre_weighted = es * (self.n_spectral / self.scaling)
        return [StandardLoss(pre_weighted)]


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

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        return [StandardLoss(get_crps(x, y, alpha=self.alpha))]


class FiniteDifferenceCRPSLoss(torch.nn.Module):
    """
    Computes the CRPS of the x and y finite differences of the input tensors,
    which helps with representations of horizontal stochastic structures.

    Returns a ``(B, C)`` tensor (spatial dims reduced internally because
    lat and lon diffs have incompatible shapes).
    """

    def __init__(self, alpha: float, levels: int = 1):
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be at least 1, got {levels}")
        self.alpha = alpha
        self.levels = levels

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        result = _get_finite_difference_crps_loss(x, y, self.alpha, levels=self.levels)
        return [StandardLoss(result / self.levels)]


def _reduce_spatial(t: torch.Tensor) -> torch.Tensor:
    """Reduce trailing (non-batch, non-channel) dims of a ``(B, C, ...)`` tensor."""
    if t.ndim <= 2:
        return t
    return t.mean(dim=tuple(range(2, t.ndim)))


def _get_finite_difference_crps_loss(
    x: torch.Tensor, y: torch.Tensor, alpha: float, levels: int
) -> torch.Tensor:
    """Returns a ``(B, C)`` tensor summing contributions from each level."""
    x_diff_lat = x[..., 1:, :] - x[..., :-1, :]
    y_diff_lat = y[..., 1:, :] - y[..., :-1, :]
    crps_lat = _reduce_spatial(get_crps(x_diff_lat, y_diff_lat, alpha=alpha))
    x_diff_lon = torch.roll(x, shifts=-1, dims=-1) - x
    y_diff_lon = torch.roll(y, shifts=-1, dims=-1) - y
    crps_lon = _reduce_spatial(get_crps(x_diff_lon, y_diff_lon, alpha=alpha))
    level_crps = 0.5 * (crps_lat + crps_lon)
    if levels > 1:
        x_flat = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
        y_flat = y.reshape(-1, 1, y.shape[-2], y.shape[-1])
        x_pooled = F.avg_pool2d(x_flat, kernel_size=2, stride=2, ceil_mode=True)
        y_pooled = F.avg_pool2d(y_flat, kernel_size=2, stride=2, ceil_mode=True)
        x_coarse = x_pooled.reshape(
            *x.shape[:-2], x_pooled.shape[-2], x_pooled.shape[-1]
        )
        y_coarse = y_pooled.reshape(
            *y.shape[:-2], y_pooled.shape[-2], y_pooled.shape[-1]
        )
        return level_crps + _get_finite_difference_crps_loss(
            x_coarse, y_coarse, alpha=alpha, levels=levels - 1
        )
    return level_crps


class EnsembleLoss(torch.nn.Module):
    def __init__(
        self,
        crps_weight: float,
        energy_score_weight: float,
        sht: Callable[[torch.Tensor], torch.Tensor],
        finite_difference_crps_weight: float = 0.0,
        finite_difference_crps_levels: int = 1,
        almost_fair_crps_alpha: float = 1.0,
    ):
        super().__init__()
        if crps_weight < 0 or energy_score_weight < 0:
            raise ValueError(
                "crps_weight and energy_score_weight must be non-negative, "
                f"got {crps_weight} and {energy_score_weight}"
            )
        if finite_difference_crps_weight < 0:
            raise ValueError(
                "finite_difference_crps_weight must be non-negative, "
                f"got {finite_difference_crps_weight}"
            )
        if crps_weight + energy_score_weight == 0:
            raise ValueError(
                "crps_weight and energy_score_weight must sum to a positive value, "
                f"got {crps_weight} and {energy_score_weight}"
            )
        self.crps_loss = CRPSLoss(alpha=almost_fair_crps_alpha)
        if finite_difference_crps_weight > 0:
            self.diff_crps_loss: FiniteDifferenceCRPSLoss | None = (
                FiniteDifferenceCRPSLoss(
                    alpha=almost_fair_crps_alpha,
                    levels=finite_difference_crps_levels,
                )
            )
        else:
            self.diff_crps_loss = None
        self.energy_score_loss = EnergyScoreLoss(sht=sht)

        self.crps_weight = crps_weight
        self.diff_crps_weight = finite_difference_crps_weight
        self.energy_score_weight = energy_score_weight

    def forward(
        self,
        gen_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> list[LossComponent]:
        components: list[LossComponent] = []
        if self.crps_weight > 0:
            for c in self.crps_loss(gen_norm, target_norm):
                components.append(type(c)(c.loss * self.crps_weight))
        if self.energy_score_weight > 0:
            for c in self.energy_score_loss(gen_norm, target_norm):
                components.append(type(c)(c.loss * self.energy_score_weight))
        if self.diff_crps_loss is not None:
            for c in self.diff_crps_loss(gen_norm, target_norm):
                components.append(type(c)(c.loss * self.diff_crps_weight))
        return components


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
        gridded_operations: GriddedOperations | None,
    ) -> Any:
        """
        Args:
            gridded_operations: The gridded operations to use in the case that
                the loss function requires use of the horizontal dimensions.
        """
        if self.type == "LpLoss":
            main_loss = LpLoss(**self.kwargs)
        elif self.type == "L1":
            main_loss = _L1Loss()
        elif self.type == "MSE":
            main_loss = _MSELoss()
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
class CorrectorRegularizationConfig:
    """Configuration for penalizing the magnitude of corrector adjustments.

    The configured loss is applied between the per-variable corrector
    corrections (post-corrector minus pre-corrector outputs) and a zero
    baseline, in the loss-normalized space. This produces an L1- or
    L2-style penalty on the size of the corrector's adjustments.

    Args:
        loss: The loss function configuration. Comparisons are gen-vs-gen,
            so ``EnsembleLoss``/``NaN`` are not supported and
            ``global_mean_type`` must be ``None``.
        weight: Scalar multiplier applied to the regularization term.
    """

    loss: LossConfig
    weight: float = 1.0

    def __post_init__(self):
        if self.loss.type in ("EnsembleLoss", "NaN"):
            raise ValueError(
                f"CorrectorRegularizationConfig does not support loss type "
                f"{self.loss.type!r}; corrections are compared against a "
                "zero baseline, so only standard pointwise losses make sense."
            )
        if self.loss.global_mean_type is not None:
            raise ValueError(
                "CorrectorRegularizationConfig does not support "
                "global_mean_type; corrections are compared against a zero "
                "baseline, so a global-mean component is ill-defined."
            )

    def build(
        self,
        gridded_ops: GriddedOperations | None,
        out_names: list[str],
        normalizer: StandardNormalizer,
        channel_dim: int = -3,
    ) -> "WeightedMappingLoss":
        bare_loss = self.loss.build(gridded_operations=gridded_ops)
        return WeightedMappingLoss(
            loss=bare_loss,
            weights={},
            out_names=out_names,
            channel_dim=channel_dim,
            normalizer=normalizer,
        )


class StepLoss(torch.nn.Module):
    def __init__(
        self, loss: WeightedMappingLoss, sqrt_loss_decay_constant: float = 0.0
    ):
        super().__init__()
        self.loss = loss
        self.sqrt_loss_decay_constant = sqrt_loss_decay_constant

    @property
    def _normalizer(self) -> StandardNormalizer:
        # private because this is only used in unit tests
        return self.loss.normalizer

    @property
    def effective_loss_scaling(self) -> dict[str, float]:
        return self.loss.effective_loss_scaling

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
    ) -> LossOutput:
        """
        Args:
            predict_dict: The predicted data.
            target_dict: The target data.
            step: The step number, indexed from 0 for the first step.
            data_mask: Optional per-variable boolean masks forwarded to
                the underlying :class:`WeightedMappingLoss`.

        Returns:
            A ``LossOutput`` wrapping the step-weighted loss tensor.
        """
        step_weight = (1.0 + self.sqrt_loss_decay_constant * step) ** (-0.5)
        return self.loss(predict_dict, target_dict, data_mask=data_mask).scale(
            step_weight
        )


@dataclasses.dataclass
class StepLossConfig:
    """
    Loss configuration class that has the same fields as LossConfig but also
    has additional weights field, and optional step loss decay.

    The build method will apply the weights to
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
        sqrt_loss_step_decay_constant: the constant to use for the square root
            loss step decay, alpha in 1/sqrt(1.0 + alpha * step) where step is
            indexed from 0 for the first step.
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
    sqrt_loss_step_decay_constant: float = 0.0
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
    ) -> StepLoss:
        loss = self.loss_config.build(
            gridded_operations=gridded_ops,
        )
        return StepLoss(
            WeightedMappingLoss(
                loss=loss,
                weights=self.weights,
                out_names=out_names,
                channel_dim=channel_dim,
                normalizer=normalizer,
            ),
            sqrt_loss_decay_constant=self.sqrt_loss_step_decay_constant,
        )
