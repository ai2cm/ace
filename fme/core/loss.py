import abc
import dataclasses
import logging
from collections.abc import Callable, Collection, Mapping
from typing import Any, Literal

import fsspec
import numpy as np
import torch
import torch.linalg
import torch.nn.functional as F
import xarray as xr

from fme.core.device import get_device
from fme.core.ensemble import (
    get_crps,
    get_energy_score,
    get_variogram_edge_offsets,
    get_variogram_score,
)
from fme.core.gridded_ops import GriddedOperations
from fme.core.name_and_prefix_matcher import NameAndPrefixSelection
from fme.core.normalizer import StandardNormalizer
from fme.core.packer import Packer
from fme.core.typing_ import TensorDict, TensorMapping


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

    def __init__(
        self,
        loss: torch.Tensor,
        term: str | None = None,
        weight: float = 1.0,
        report_only: bool = False,
    ):
        """
        Args:
            loss: The loss tensor. Can be a scalar (``ndim == 0``),
                a partially-reduced tensor like ``(B, C)``, or a
                full element-wise tensor like ``(B, C, lat, lon)``.
            term: Optional name of the loss term this component belongs to
                (e.g. ``"crps"``), used to report per-term values.
            weight: The product of the scalar weights already applied to
                ``loss`` (term weight, step weight, ...), so that
                ``loss / weight`` is the term's unweighted value.
            report_only: If True, ``loss`` is an unweighted value that is
                reported under ``term`` but not added to the loss total.
        """
        self.loss = loss
        self.term = term
        self.weight = weight
        self.report_only = report_only

    @abc.abstractmethod
    def reduce_to_channel(self) -> torch.Tensor:
        """Reduce to ``(B, C)`` by meaning over non-batch, non-channel dims."""

    def replace_loss(self, loss: torch.Tensor) -> "LossComponent":
        """A copy with the same metadata holding a different loss tensor."""
        return type(self)(
            loss, term=self.term, weight=self.weight, report_only=self.report_only
        )

    def scaled(self, factor: float) -> "LossComponent":
        """A copy with ``loss`` multiplied by ``factor``, and ``weight``
        tracking it. Report-only components are returned unchanged, since
        they carry an unweighted value.
        """
        if self.report_only:
            return self
        return type(self)(
            self.loss * factor, term=self.term, weight=self.weight * factor
        )

    def unweighted_channel_loss(self) -> torch.Tensor | None:
        """The detached ``(B, C)`` unweighted value of this component, or None
        if it is untagged or its weight is zero (the value is unrecoverable).
        """
        if self.term is None:
            return None
        bc = self.reduce_to_channel().detach()
        if self.report_only:
            return bc
        if self.weight == 0:
            return None
        return bc / self.weight


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
            bc = sum(c.reduce_to_channel() for c in self._losses if not c.report_only)
            assert isinstance(bc, torch.Tensor)
            self._per_channel, self._counts = self._reduce_batch(bc)
        assert self._per_channel is not None and self._counts is not None
        return self._per_channel, self._counts

    def _reduce_batch(self, bc: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
        """Reduce a ``(B, C)`` (or scalar) tensor to per-channel values."""
        if bc.ndim == 0:
            return bc.expand(len(self._channel_names)), [1] * len(self._channel_names)
        elif self._mask is not None:
            masked_sum = (bc * self._mask).sum(dim=0)
            per_channel = masked_sum / self._mask.sum(dim=0).clamp(min=1)
            return per_channel, [int(c.item()) for c in self._mask.sum(dim=0)]
        else:
            return bc.mean(dim=0), [bc.shape[0]] * len(self._channel_names)

    def _mean_over_channels(self, pc: torch.Tensor) -> torch.Tensor:
        if self._mask is not None:
            active = self._mask.sum(dim=0) > 0
            if active.any():
                return pc[active].mean()
        return pc.mean()

    def total(self) -> torch.Tensor:
        """Scalar loss used as the optimization target.

        This is the mean of the per-channel losses across channels (over
        active channels only when a mask is present), not a sum. Adding
        or removing channels therefore does not change the scale of the
        returned value.
        """
        pc, _ = self._reduce()
        return self._mean_over_channels(pc)

    def get_term_losses(self) -> dict[str, torch.Tensor]:
        """Detached scalar unweighted value of each named loss term.

        Each term is reduced like :meth:`total` (mean over active samples per
        channel, then over active channels), after dividing out the scalar
        weights applied to it (term and step weights; per-variable weights,
        which scale the loss inputs, are kept). Report-only terms, which do
        not enter :meth:`total`, are included.
        """
        by_term: dict[str, torch.Tensor] = {}
        for c in self._losses:
            bc = c.unweighted_channel_loss()
            if bc is None:
                continue
            assert c.term is not None
            if c.term in by_term:
                by_term[c.term] = by_term[c.term] + bc
            else:
                by_term[c.term] = bc
        return {
            term: self._mean_over_channels(self._reduce_batch(bc)[0])
            for term, bc in by_term.items()
        }

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
            [c.scaled(weight) for c in self._losses],
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

        def _reduce_elementwise(t: torch.Tensor) -> torch.Tensor:
            # Element-wise loss tensors have the same shape as the input;
            # the channel position depends on the data layout (ensemble,
            # tile, etc.). Reduce non-(batch, channel) dims here so the
            # downstream component carries a canonical ``(B, C)`` tensor.
            dims = tuple(i for i in range(t.ndim) if i not in (0, cdim))
            return t.mean(dim=dims) if dims else t

        losses: list[LossComponent]
        if isinstance(result, list):
            # Inner losses that return raw element-wise tensors (e.g. MSE,
            # L1) wrap themselves in StandardLoss but don't know the input
            # channel layout, so reduce around the actual channel dim here.
            losses = [
                c.replace_loss(_reduce_elementwise(c.loss))
                if c.loss.ndim == input_ndim and type(c) is StandardLoss
                else c
                for c in result
            ]
        else:
            losses = [StandardLoss(_reduce_elementwise(result))]

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
                components.append(c.scaled(w))
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


class SpectralWhitening:
    """Per-sample spectral-whitening reweight for :class:`EnergyScoreLoss`.

    ``factor`` returns a per-(sample, channel, degree) multiplier (broadcast over
    order m) computed from the *detached* target coefficients, so it only
    reweights the energy score across degrees and adds no gradient path. The
    unnormalized weight is ``(1 / amp_l) ** exponent`` where ``amp_l`` is the
    per-degree RMS amplitude over valid orders (m <= l), floored at ``eps_frac``
    of the per-sample mean amplitude to bound the boost at near-zero-power
    degrees. ``exponent`` gamma in (0, 1] sets the strength: gamma=1 fully
    flattens the spectrum, gamma<1 whitens partially. The factor is rescaled per
    (sample, channel) to preserve the amplitude-weighted total, so the overall
    energy-score magnitude (and the meaning of ``energy_score_weight``) is
    unchanged. A white-spectrum target yields a uniform factor (no-op).
    """

    def __init__(self, eps_frac: float = 0.02, exponent: float = 1.0):
        self.eps_frac = eps_frac
        self.exponent = exponent

    def factor(self, y_hat: torch.Tensor) -> torch.Tensor:
        # The target carries a singleton ensemble dim (B, 1, [C], L, M); drop it
        # up front so the per-(sample, [channel], degree) factor aligns with the
        # energy score, whose ensemble dim get_energy_score has already reduced.
        yt = y_hat.detach()[:, 0]  # (B, [C], L, M)
        amp_mode = yt.abs()  # (B, [C], L, M)
        real_dtype = amp_mode.dtype
        n_l, n_m = yt.shape[-2], yt.shape[-1]
        l_idx = torch.arange(n_l, device=yt.device).unsqueeze(-1)  # (L, 1)
        m_idx = torch.arange(n_m, device=yt.device).unsqueeze(0)  # (1, M)
        valid = (m_idx <= l_idx).to(real_dtype)  # (L, M); zero where m > l
        # real-SHT redundancy (m>0 counts double) x validity, clean (L, M) shape
        redundancy = 2.0 * torch.ones(n_l, n_m, device=yt.device, dtype=real_dtype)
        redundancy[:, 0] = 1.0
        w = redundancy * valid  # (L, M)
        tiny = torch.finfo(real_dtype).tiny
        # per-mode mean power at degree l (over valid m) -> per-mode RMS amplitude
        meanpow_l = (amp_mode**2 * w).sum(dim=-1) / w.sum(dim=-1).clamp_min(tiny)
        amp_l = torch.sqrt(meanpow_l)  # (B, [C], L)
        mean_amp = amp_l.mean(dim=-1, keepdim=True)
        f = 1.0 / torch.clamp(amp_l, min=self.eps_frac * mean_amp)
        if self.exponent != 1.0:
            f = f**self.exponent
        f_m = f.unsqueeze(-1)  # (B, [C], L, 1), broadcast over m
        # Magnitude preservation: the per-mode energy score scales ~ |y_hat|, so
        # rescale so sum_lm w * |y_hat| is unchanged by the reweight.
        num = (w * amp_mode).sum(dim=(-2, -1), keepdim=True)
        den = (w * f_m * amp_mode).sum(dim=(-2, -1), keepdim=True)
        alpha = num / (den + tiny)
        return alpha * f_m  # (B, [C], L, 1)


#: Default whitening strength when ``kind='per_sample'`` and ``exponent`` is
#: unset. A two-seed gamma in {0, 0.2, 0.5, 0.8} sweep selected 0.5 as the
#: stable knee: it captures most of the small-scale spectral gain while staying
#: rollout-stable, whereas full whitening (gamma=1) over-upweights
#: noise-dominated low-amplitude high-l degrees and destabilizes residual
#: rollouts. See the whitening-gamma-selection report linked from PR #1303.
_DEFAULT_WHITENING_EXPONENT = 0.5
#: Default per-degree amplitude floor (fraction of the per-sample mean degree
#: amplitude) when ``kind='per_sample'`` and ``eps_frac`` is unset.
_DEFAULT_WHITENING_EPS_FRAC = 0.02


@dataclasses.dataclass
class SpectralWhiteningConfig:
    """Configures per-sample spectral whitening of the energy score (see
    :class:`SpectralWhitening` for what the reweight does).

    Args:
        kind: ``'none'`` (the default) disables whitening; ``'per_sample'``
            enables the per-sample reweight.
        eps_frac: floor on the per-degree amplitude, as a fraction of the
            per-sample mean degree-amplitude. It bounds the boost applied to
            near-zero-power degrees (where ``1 / amp_l`` would blow up).
            Unset (``None``) defaults to ``0.02`` when whitening is enabled.
            Requires ``kind='per_sample'``.
        exponent: whitening strength gamma in (0, 1]; the per-degree weight is
            ``(1 / amp_l) ** gamma``. gamma=1 fully flattens the target
            spectrum; smaller gamma whitens partially, taming the upweighting of
            noise-dominated low-amplitude degrees. Unset (``None``) defaults to
            ``0.5`` when whitening is enabled -- the validated stable knee; full
            whitening (gamma=1) destabilizes residual rollouts, so it is opt-in
            rather than the default. Requires ``kind='per_sample'``.
    """

    kind: Literal["none", "per_sample"] = "none"
    eps_frac: float | None = None
    exponent: float | None = None

    def __post_init__(self):
        if self.kind not in ("none", "per_sample"):
            raise NotImplementedError(
                f"spectral whitening kind={self.kind!r} not supported; "
                "use 'none' or 'per_sample'."
            )
        if self.kind == "none":
            if self.eps_frac is not None or self.exponent is not None:
                raise ValueError(
                    "eps_frac and exponent require kind='per_sample'; "
                    "got kind='none'."
                )
            return
        # kind='per_sample': resolve unset fields to their validated defaults,
        # then validate. Storing the resolved values keeps build() total and the
        # config introspectable.
        if self.eps_frac is None:
            self.eps_frac = _DEFAULT_WHITENING_EPS_FRAC
        if self.exponent is None:
            self.exponent = _DEFAULT_WHITENING_EXPONENT
        if self.eps_frac <= 0:
            raise ValueError(f"eps_frac must be positive, got {self.eps_frac}")
        if not 0.0 < self.exponent <= 1.0:
            raise ValueError(f"exponent must be in (0, 1], got {self.exponent}")

    def build(self) -> SpectralWhitening | None:
        if self.kind == "none":
            return None
        assert self.eps_frac is not None and self.exponent is not None
        return SpectralWhitening(eps_frac=self.eps_frac, exponent=self.exponent)


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

    def __init__(
        self,
        sht: Callable[[torch.Tensor], torch.Tensor],
        whitening: SpectralWhitening | None = None,
    ):
        super().__init__()
        self.sht = sht
        # None (whitening disabled) makes forward() skip the reweight entirely.
        self._whitening = whitening
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
        if self._whitening is not None:
            es = es * self._whitening.factor(y_hat)
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


def load_variogram_scaling(
    path: str,
    names: list[str],
    offsets: list[tuple[int, int]],
) -> torch.Tensor:
    """Load per-variable, per-edge-kind increment scales from a netCDF file.

    The file (written by ``scripts/data_process/get_stats.py``) holds one
    variable per data variable with an edge dimension carrying integer
    coordinates ``di`` (latitude index step) and ``dj`` (longitude index
    step); each value is the RMS of the increment for that offset, in the
    variable's physical units.

    Args:
        path: Path to the netCDF file.
        names: The variable names, in channel order.
        offsets: The ``(di, dj)`` edge kinds to select, in order.

    Returns:
        A tensor of shape ``(len(offsets), len(names))`` in physical units.
    """
    with fsspec.open(path, "rb") as f:
        ds = xr.load_dataset(f)
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise ValueError(
            f"Variogram scaling file {path} is missing variables {missing}; "
            "it must contain every variable the loss is computed on."
        )
    di = ds["di"].values
    dj = ds["dj"].values
    indices = []
    for offset in offsets:
        match = np.nonzero((di == offset[0]) & (dj == offset[1]))[0]
        if len(match) != 1:
            raise ValueError(
                f"Variogram scaling file {path} has {len(match)} entries for "
                f"edge offset (di, dj) = {offset}, expected exactly 1."
            )
        indices.append(int(match[0]))
    values = np.stack(
        [ds[name].values[indices].astype(np.float64) for name in names], axis=-1
    )
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        bad = sorted(
            {
                names[j]
                for j in range(len(names))
                if not np.all(np.isfinite(values[:, j])) or np.any(values[:, j] <= 0)
            }
        )
        raise ValueError(
            f"Variogram scaling file {path} has non-positive or non-finite "
            f"scales for {bad}."
        )
    return torch.tensor(values, dtype=torch.float32)


class VariogramScoreLoss(torch.nn.Module):
    """
    Per-channel, grid-local variogram score of order ``p`` (Scheuerer and
    Hamill 2015) with the fair two-member estimator; see
    :func:`fme.core.ensemble.get_variogram_score` for its definition, pole
    and longitude handling, and reduction.

    Increments are divided by a per-channel, per-edge-kind scale before the
    power. The scales are given in physical units and converted to the
    normalized units the loss is computed in: normalized data is
    ``(x - mean_c) / std_c``, so a normalized increment is the physical one
    divided by ``std_c`` (centering cancels), and the scale in normalized
    units is ``scale_phys / std_c``.

    Per-variable loss weights multiply the loss inputs, so they scale the
    score by ``weight ** (2 * p)`` (linearly, like CRPS, for ``p = 0.5``).

    Assumes a ``[..., n_lat, n_lon]`` layout with periodic longitude (a
    lat-lon grid).

    Returns a ``(B, C)`` tensor.
    """

    def __init__(
        self,
        scale: torch.Tensor,
        window_size: int,
        p: float = 0.5,
    ):
        """
        Args:
            scale: The increment scales in normalized units, of shape
                ``(n_edges, n_channels)`` with edge kinds ordered as
                :func:`get_variogram_edge_offsets` returns them for
                ``window_size``.
            window_size: Odd width of the window of edges in grid points.
            p: Order of the variogram score.
        """
        super().__init__()
        self.offsets = get_variogram_edge_offsets(window_size)
        if scale.ndim != 2 or scale.shape[0] != len(self.offsets):
            raise ValueError(
                f"scale must have shape ({len(self.offsets)}, n_channels) for "
                f"window_size {window_size}, got {tuple(scale.shape)}"
            )
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}")
        self.window_size = window_size
        self.p = p
        self.scale = scale.to(get_device())

    @property
    def term_name(self) -> str:
        return f"variogram_score_w{self.window_size}"

    @classmethod
    def from_file(
        cls,
        path: str,
        names: list[str],
        normalizer: StandardNormalizer,
        window_size: int,
        p: float = 0.5,
    ) -> "VariogramScoreLoss":
        """Build from a physical-units scaling file, converting the scales to
        the normalized units of ``normalizer``.

        Args:
            path: Path to the variogram scaling netCDF file.
            names: The variable names, in the channel order of the loss
                inputs.
            normalizer: The normalizer that produced the loss inputs.
            window_size: Odd width of the window of edges in grid points.
            p: Order of the variogram score.
        """
        offsets = get_variogram_edge_offsets(window_size)
        scale_phys = load_variogram_scaling(path, names, offsets)
        stds = torch.stack(
            [normalizer.stds[name].detach().float().cpu().reshape(()) for name in names]
        )
        return cls(scale_phys / stds[None, :], window_size=window_size, p=p)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> list[LossComponent]:
        return [
            StandardLoss(
                get_variogram_score(
                    x,
                    y,
                    self.scale.to(device=x.device, dtype=x.dtype),
                    self.offsets,
                    p=self.p,
                )
            )
        ]


class EnsembleLoss(torch.nn.Module):
    def __init__(
        self,
        crps_weight: float,
        energy_score_weight: float,
        sht: Callable[[torch.Tensor], torch.Tensor],
        finite_difference_crps_weight: float = 0.0,
        finite_difference_crps_levels: int = 1,
        almost_fair_crps_alpha: float = 1.0,
        energy_score_whitening: SpectralWhitening | None = None,
        variogram_score_weight: float = 0.0,
        variogram_score_loss: VariogramScoreLoss | None = None,
        variogram_score_report_losses: list[VariogramScoreLoss] | None = None,
    ):
        """
        Args:
            crps_weight: Weight of the CRPS term.
            energy_score_weight: Weight of the spectral energy score term.
            sht: The real spherical harmonic transform for the energy score.
            finite_difference_crps_weight: Weight of the finite-difference
                CRPS term.
            finite_difference_crps_levels: Number of coarsening levels of the
                finite-difference CRPS.
            almost_fair_crps_alpha: The almost-fair CRPS alpha.
            energy_score_whitening: Optional spectral whitening of the energy
                score.
            variogram_score_weight: Weight of the variogram score term.
            variogram_score_loss: The variogram score term. Required when
                variogram_score_weight is positive; with a zero weight it is
                computed and reported but not added to the loss.
            variogram_score_report_losses: Further variogram score terms that
                are computed and reported but not added to the loss (e.g. other
                window sizes).
        """
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
        if variogram_score_weight < 0:
            raise ValueError(
                "variogram_score_weight must be non-negative, "
                f"got {variogram_score_weight}"
            )
        if variogram_score_weight > 0 and variogram_score_loss is None:
            raise ValueError(
                "variogram_score_loss is required when variogram_score_weight "
                "is positive"
            )
        total_weight = (
            crps_weight
            + energy_score_weight
            + finite_difference_crps_weight
            + variogram_score_weight
        )
        if total_weight == 0:
            raise ValueError(
                "crps_weight, energy_score_weight, finite_difference_crps_weight "
                "and variogram_score_weight must sum to a positive value, "
                f"got {crps_weight}, {energy_score_weight}, "
                f"{finite_difference_crps_weight} and {variogram_score_weight}"
            )
        report_losses = list(variogram_score_report_losses or [])
        if variogram_score_loss is not None and variogram_score_weight == 0:
            report_losses.insert(0, variogram_score_loss)
            variogram_score_loss = None
        self.variogram_score_weight = variogram_score_weight
        self.variogram_score_loss = variogram_score_loss
        term_names = [loss.term_name for loss in report_losses]
        if variogram_score_loss is not None:
            term_names.append(variogram_score_loss.term_name)
        if len(set(term_names)) != len(term_names):
            raise ValueError(
                f"variogram score terms must have distinct window sizes, got "
                f"{term_names}"
            )
        # A plain list, not a ModuleList: these terms hold no parameters and
        # keep their scales on the device themselves.
        self.variogram_score_report_losses = report_losses
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
        self.energy_score_loss = EnergyScoreLoss(
            sht=sht,
            whitening=energy_score_whitening,
        )

        self.crps_weight = crps_weight
        self.diff_crps_weight = finite_difference_crps_weight
        self.energy_score_weight = energy_score_weight

    def forward(
        self,
        gen_norm: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> list[LossComponent]:
        """Weighted components of the active terms, each tagged with its term
        name, plus unweighted report-only components for zero-weight
        variogram score terms.
        """
        components: list[LossComponent] = []

        def _add(
            term: str, weight: float, term_components: list[LossComponent]
        ) -> None:
            for c in term_components:
                components.append(
                    type(c)(c.loss * weight, term=term, weight=c.weight * weight)
                )

        if self.crps_weight > 0:
            _add("crps", self.crps_weight, self.crps_loss(gen_norm, target_norm))
        if self.energy_score_weight > 0:
            _add(
                "energy_score",
                self.energy_score_weight,
                self.energy_score_loss(gen_norm, target_norm),
            )
        if self.diff_crps_loss is not None:
            _add(
                "finite_difference_crps",
                self.diff_crps_weight,
                self.diff_crps_loss(gen_norm, target_norm),
            )
        if self.variogram_score_loss is not None:
            _add(
                self.variogram_score_loss.term_name,
                self.variogram_score_weight,
                self.variogram_score_loss(gen_norm, target_norm),
            )
        for report_loss in self.variogram_score_report_losses:
            with torch.no_grad():
                report_components = report_loss(gen_norm, target_norm)
            for c in report_components:
                components.append(
                    type(c)(
                        c.loss.detach(),
                        term=report_loss.term_name,
                        report_only=True,
                    )
                )
        return components


@dataclasses.dataclass
class _VariogramScoreKwargs:
    """The variogram score settings among the (flat) EnsembleLoss kwargs.

    Parameters:
        weight: ``variogram_score_weight``, the weight of the variogram score
            term. With a zero weight and a scaling path, the term is still
            computed and reported (unweighted), but not added to the loss.
        window_size: ``variogram_score_window_size``, the odd window width
            (3 or 5) of the weighted term.
        p: ``variogram_score_p``, the order of the variogram score.
        scaling_path: ``variogram_score_scaling_path``, the netCDF file of
            per-variable, per-edge-kind increment scales in physical units
            (``variogram-scaling.nc`` from ``scripts/data_process/get_stats.py``).
        report_window_sizes: ``variogram_score_report_window_sizes``, further
            window sizes whose variogram score is computed and reported but not
            added to the loss.
    """

    weight: float = 0.0
    window_size: int = 3
    p: float = 0.5
    scaling_path: str | None = None
    report_window_sizes: list[int] = dataclasses.field(default_factory=list)

    SUPPORTED_WINDOW_SIZES = (3, 5)

    def __post_init__(self):
        # kwargs are untyped, so an unfilled config placeholder arrives here
        if not isinstance(self.weight, int | float) or isinstance(self.weight, bool):
            raise TypeError(
                f"variogram_score_weight must be a number, got {self.weight!r}"
            )
        if self.weight < 0:
            raise ValueError(
                f"variogram_score_weight must be non-negative, got {self.weight}"
            )
        for size in [self.window_size, *self.report_window_sizes]:
            if size not in self.SUPPORTED_WINDOW_SIZES:
                raise ValueError(
                    f"variogram score window sizes must be one of "
                    f"{self.SUPPORTED_WINDOW_SIZES}, got {size}"
                )
        if len(set(self.report_window_sizes)) != len(self.report_window_sizes):
            raise ValueError(
                "variogram_score_report_window_sizes must be distinct, got "
                f"{self.report_window_sizes}"
            )
        if self.p <= 0:
            raise ValueError(f"variogram_score_p must be positive, got {self.p}")
        if self.scaling_path is None and (
            self.weight > 0 or len(self.report_window_sizes) > 0
        ):
            raise ValueError(
                "variogram_score_scaling_path is required when "
                "variogram_score_weight is positive or "
                "variogram_score_report_window_sizes is given"
            )
        if self.weight > 0 and self.window_size in self.report_window_sizes:
            raise ValueError(
                f"window size {self.window_size} is both the weighted variogram "
                "score and a report-only one"
            )

    @classmethod
    def pop_from(cls, kwargs: dict[str, Any]) -> "_VariogramScoreKwargs":
        """Pop the ``variogram_score_*`` entries from ``kwargs``."""
        return cls(
            weight=kwargs.pop("variogram_score_weight", 0.0),
            window_size=kwargs.pop("variogram_score_window_size", 3),
            p=kwargs.pop("variogram_score_p", 0.5),
            scaling_path=kwargs.pop("variogram_score_scaling_path", None),
            report_window_sizes=list(
                kwargs.pop("variogram_score_report_window_sizes", [])
            ),
        )

    def build(
        self,
        out_names: list[str] | None,
        normalizer: StandardNormalizer | None,
    ) -> tuple["VariogramScoreLoss | None", list["VariogramScoreLoss"]]:
        """Build the weighted (or, at zero weight, reported) term and the
        report-only terms. Nothing is built without a scaling path.
        """
        if self.scaling_path is None:
            return None, []
        if out_names is None or normalizer is None:
            raise ValueError(
                "out_names and normalizer are required to build the variogram score"
            )
        path = self.scaling_path

        def _build(window_size: int) -> VariogramScoreLoss:
            return VariogramScoreLoss.from_file(
                path,
                names=out_names,
                normalizer=normalizer,
                window_size=window_size,
                p=self.p,
            )

        report = [
            _build(size)
            for size in self.report_window_sizes
            if size != self.window_size
        ]
        return _build(self.window_size), report


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
        if self.type == "EnsembleLoss":
            # validate the variogram score kwargs without reading the file
            _VariogramScoreKwargs.pop_from(dict(self.kwargs))

    def build(
        self,
        gridded_operations: GriddedOperations | None,
        out_names: list[str] | None = None,
        normalizer: StandardNormalizer | None = None,
    ) -> Any:
        """
        Args:
            gridded_operations: The gridded operations to use in the case that
                the loss function requires use of the horizontal dimensions.
            out_names: The names of the loss channels, in channel order.
                Required by the EnsembleLoss variogram score.
            normalizer: The normalizer applied to the loss inputs. Required by
                the EnsembleLoss variogram score.
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
            # kwargs is opaque (Mapping[str, Any]), so dacite does not descend
            # into it; build the nested whitening config from its dict here, then
            # pass the built operator (or None) down to EnsembleLoss.
            whitening_config = kwargs.pop("energy_score_whitening", None)
            if isinstance(whitening_config, Mapping):
                whitening_config = SpectralWhiteningConfig(**whitening_config)
            whitening = (
                whitening_config.build() if whitening_config is not None else None
            )
            variogram = _VariogramScoreKwargs.pop_from(kwargs)
            vs_loss, vs_report_losses = variogram.build(out_names, normalizer)
            main_loss = EnsembleLoss(
                sht=gridded_operations.get_real_sht(),
                crps_weight=crps_weight,
                energy_score_weight=energy_score_weight,
                energy_score_whitening=whitening,
                variogram_score_weight=variogram.weight,
                variogram_score_loss=vs_loss,
                variogram_score_report_losses=vs_report_losses,
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

    type: Literal["LpLoss", "L1", "MSE", "AreaWeightedMSE", "EnsembleLoss"] = "MSE"
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
            out_names=out_names,
            normalizer=normalizer,
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


def require_matched_entries(
    selection: NameAndPrefixSelection,
    names: Collection[str],
    feature: str,
    subject: str,
) -> list[str]:
    """The names the selection matches, sorted; raises if any entry matches none.

    The one validation primitive of corrector-loss name selection, used both at
    build against the names the loss covers and at runtime against the
    corrector's actual delta keys.

    Args:
        selection: The configured entries.
        names: The names to match against.
        feature: The config field the entries came from, for the error text.
        subject: A plural noun phrase for what ``names`` are, for the error
            text (e.g. "variables the loss covers").
    """
    unmatched = selection.unmatched_entries(names)
    if unmatched:
        raise ValueError(
            f"{feature} selects entries that match none of the {subject}: "
            + "; ".join(f"{entry!r} matches none" for entry in unmatched)
            + f". The {subject} are {sorted(names)}."
        )
    return selection.matched(names)


class CorrectorRegularizer(torch.nn.Module):
    """A penalty pushing selected correction deltas toward zero.

    Holds the three things the penalty is: the selection of deltas it is taken
    over, a factory for the loss over them, and the weight it enters the step
    total with. The names and the loss are fixed by :meth:`resolve`, because
    which names the selection covers is not knowable until the corrector's
    first non-empty delta arrives.
    """

    def __init__(
        self,
        selection: NameAndPrefixSelection,
        build_loss: Callable[[list[str]], WeightedMappingLoss],
        weight: float,
    ):
        """
        Args:
            selection: The configured selection of correction deltas.
            build_loss: Builds the loss applied to the normalized deltas
                against zeros, over the names it is given.
            weight: The weight applied to the penalty in the step total.
        """
        super().__init__()
        self._selection = selection
        self._build_loss = build_loss
        self._loss: WeightedMappingLoss | None = None
        self._names: list[str] = []
        self._weight = weight

    @property
    def weight(self) -> float:
        return self._weight

    @property
    def names(self) -> list[str]:
        """The names the penalty is taken over, empty before :meth:`resolve`."""
        return list(self._names)

    def resolve(self, delta_names: Collection[str]) -> None:
        """Fix the penalty's channels to the selection's matches among
        ``delta_names`` and build the loss over them.

        The loss is built over the matched delta keys rather than over the
        matched loss names, so the penalty is not taken over levels the
        corrector never touched.
        """
        self._names = require_matched_entries(
            self._selection,
            delta_names,
            "regularization",
            "correction deltas the corrector produced",
        )
        self._loss = self._build_loss(self._names)

    def forward(
        self, deltas: TensorMapping, data_mask: TensorMapping | None = None
    ) -> LossOutput:
        """Penalty over the selected deltas, per channel.

        The deltas are compared against zeros in loss-normalized space, so with
        an affine normalizer the means cancel and this penalizes ``delta/std``.
        The mask is the main loss's, so both halves of the step total average
        over the same samples per channel.

        NaN-filled delta points are zeroed on both sides by
        ``WeightedMappingLoss``, matching how the main loss treats a NaN
        target: they enter the channel mean contributing zero, so the penalty
        is diluted by the masked fraction rather than renormalized over the
        kept points. With ``fill_nans_on_normalize`` set on the loss
        normalizer the NaN is filled before that mask is taken, and no point
        is zeroed at all.
        """
        if self._loss is None:
            raise RuntimeError(
                "the penalty was taken before its channels were resolved; "
                "CorrectorLoss.resolve_names must run first."
            )
        selected: TensorDict = {}
        targets: TensorDict = {}
        for name in self._names:
            _require_delta(deltas, name, "regularization")
            delta = deltas[name]
            selected[name] = delta
            # NaN target where the delta is NaN-filled; see the docstring for
            # what the downstream loss does with it.
            targets[name] = torch.where(
                delta.isnan(),
                torch.full_like(delta, torch.nan),
                torch.zeros_like(delta),
            )
        return self._loss(selected, targets, data_mask)


class CorrectorLoss(torch.nn.Module):
    """Loss for corrector optimization.

    Owns both features that consume the correction deltas of a ``StepOutput``:
    pre-corrector optimization and corrector regularization.
    """

    def __init__(
        self,
        precorrector_selection: NameAndPrefixSelection | None,
        regularizer: CorrectorRegularizer | None,
    ):
        """
        Args:
            precorrector_selection: Selection of the variables whose main-loss
                prediction is the pre-corrector network output, or None when
                the feature is off.
            regularizer: The penalty over the selected deltas, or None when
                the feature is off.
        """
        super().__init__()
        self._precorrector_selection = precorrector_selection
        self._precorrector_names: list[str] | None = None
        self._regularizer = regularizer
        self._resolved = False

    @property
    def penalty_weight(self) -> float:
        """The weight the penalty enters the step total with, 1.0 with no penalty."""
        if self._regularizer is None:
            return 1.0
        return self._regularizer.weight

    def resolve_names(self, delta_names: Collection[str]) -> None:
        """Validate both features' entries against the corrector's delta keys
        and fix the names each acts on. A no-op after the first call.

        Called on the first step whose deltas are non-empty, which is the first
        point at which the keys the corrector really produces are observable.
        Empty-delta steps -- every train-mode step of an
        ``EpochScheduledCorrector``'s disabled epochs -- do not consume it: the
        corrector is always applied in eval mode, so the end-of-epoch
        validation pass resolves at the latest.
        """
        if self._resolved:
            return
        subject = "correction deltas the corrector produced"
        if self._precorrector_selection is not None:
            self._precorrector_names = require_matched_entries(
                self._precorrector_selection,
                delta_names,
                "precorrector_optimization",
                subject,
            )
        if self._regularizer is not None:
            self._regularizer.resolve(delta_names)
        self._resolved = True
        if self._precorrector_names is not None:
            logging.info(
                "corrector_loss: optimizing pre-corrector outputs for "
                f"{self._precorrector_names}"
            )
        if self._regularizer is not None:
            logging.info(
                f"corrector_loss: penalizing deltas for {self._regularizer.names} "
                f"with weight {self._regularizer.weight}"
            )

    def pre_corrector_outputs(
        self, predict_dict: TensorMapping, deltas: TensorMapping
    ) -> TensorDict:
        """``predict_dict[k] - deltas[k]`` for the selected names, else
        ``predict_dict[k]``; raises when a selected name is missing.
        """
        net_output = dict(predict_dict)
        if self._precorrector_selection is None or len(deltas) == 0:
            return net_output
        if self._precorrector_names is None:
            raise RuntimeError(
                "the pre-corrector outputs were taken before their names were "
                "resolved; CorrectorLoss.resolve_names must run first."
            )
        for name in self._precorrector_names:
            _require_delta(deltas, name, "precorrector_optimization")
            net_output[name] = predict_dict[name] - deltas[name]
        return net_output

    def penalty(
        self, deltas: TensorMapping, data_mask: TensorMapping | None = None
    ) -> LossOutput | None:
        """The regularizer's penalty, or None when the feature is off or the
        deltas are empty.
        """
        if self._regularizer is None or len(deltas) == 0:
            return None
        return self._regularizer(deltas, data_mask)


def _require_delta(deltas: TensorMapping, name: str, feature: str) -> None:
    if name not in deltas:
        raise ValueError(
            f"{feature} selects {name!r}, but the corrector produced no delta "
            f"for it; it produced deltas for {sorted(deltas)}. An active "
            "corrector must produce deltas for every selected name."
        )


@dataclasses.dataclass
class StepOutputLossOutput:
    """The loss of one step, main term plus the corrector penalty.

    Parameters:
        main: The main ``StepLoss`` output.
        corrector_penalty: The penalty's own per-channel ``LossOutput``, or
            None when there is no penalty.
        corrector_penalty_weight: The weight applied to the penalty in
            ``total()``.
    """

    main: LossOutput
    corrector_penalty: LossOutput | None = None
    corrector_penalty_weight: float = 1.0

    def total(self) -> torch.Tensor:
        """``main.total() + weight * corrector_penalty.total()``."""
        # The penalty rides the per-step total, so one backward() call
        # carries it; a second would double-backward under accumulation.
        total = self.main.total()
        if self.corrector_penalty is not None:
            total = (
                total + self.corrector_penalty_weight * self.corrector_penalty.total()
            )
        return total

    def get_channel_losses(self) -> dict[str, ChannelLossInfo]:
        """Per-channel main-loss values; the penalty is in ``total()`` only."""
        return self.main.get_channel_losses()

    def get_term_losses(self) -> dict[str, torch.Tensor]:
        """Unweighted values of the main loss's named terms."""
        return self.main.get_term_losses()


class StepOutputLoss(torch.nn.Module):
    """``StepLoss`` plus the corrector-delta terms of a ``StepOutput``.

    Deltas come from the ``StepOutput``, never the inference-only
    ``StepDiagnostics`` carriage. The penalty takes no per-step decay, and the
    two corrector features may be enabled together.
    """

    def __init__(self, step_loss: StepLoss, corrector_loss: CorrectorLoss | None):
        super().__init__()
        self.step_loss = step_loss
        self.corrector_loss = corrector_loss

    def forward(
        self,
        predict_dict: TensorMapping,
        target_dict: TensorMapping,
        step: int,
        data_mask: TensorMapping | None = None,
        deltas: TensorMapping | None = None,
    ) -> StepOutputLossOutput:
        """
        Args:
            predict_dict: The predicted (corrected) data.
            target_dict: The target data.
            step: The step number, indexed from 0 for the first step.
            data_mask: Optional per-variable boolean masks forwarded to the
                main loss.
            deltas: The corrector's per-variable correction deltas, empty or
                None when the corrector was inactive.
        """
        if self.corrector_loss is None or deltas is None or len(deltas) == 0:
            # Inert path: exactly the StepLoss result. An epoch-disabled
            # corrector lands here.
            return StepOutputLossOutput(
                main=self.step_loss(predict_dict, target_dict, step, data_mask)
            )
        self.corrector_loss.resolve_names(deltas.keys())
        # Pre-corrector outputs first, so StepLoss never sees a delta.
        net_output = self.corrector_loss.pre_corrector_outputs(predict_dict, deltas)
        main = self.step_loss(net_output, target_dict, step, data_mask)
        return StepOutputLossOutput(
            main=main,
            corrector_penalty=self.corrector_loss.penalty(deltas, data_mask),
            corrector_penalty_weight=self.corrector_loss.penalty_weight,
        )
