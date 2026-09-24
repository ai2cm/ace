"""Optimized derived variables: fields computed from the output variables that
enter the training loss as extra channels, for both prediction and target.

    x' = x ∪ derive(x)            for x in (prediction, target)
    L  = StepLoss over out_names ∪ derived names, weight w_d per derived name

Nothing is added to the dataset or to the stepper checkpoint; the feature is
configured on the training side only.

Registered variables:

    rho_wright97:  rho_wright97_k = EOS(so_k, thetao_k, p_k) - RHO_0,   k in levels
                   EOS = Wright (1997) reduced range (fme.core.ocean_eos)
                   p_k = RHO_0 * G_EARTH * (idepth[k] + idepth[k+1]) / 2
                   NaN where mask_k == 0 or so_k / thetao_k is NaN

Loss scale of a derived name, unless given in ``stds``, is the linearization

    s_k = sqrt((d rho/dT * std(thetao_k))^2 + (d rho/dS * std(so_k))^2)

at (mean(thetao_k), mean(so_k), p_k), with means from the network normalizer
and stds from the loss normalizer, so ``s_k`` follows whichever loss scaling
(full-field or residual) the stepper uses for ``thetao_k`` and ``so_k``.
"""

import dataclasses
from collections.abc import Callable
from typing import Literal

import torch

from fme.core.coordinates import DepthCoordinate, VerticalCoordinate
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean_eos import (
    RHO_0,
    boussinesq_pressure,
    interface_to_center_depth,
    wright97_anomaly,
)
from fme.core.typing_ import TensorDict, TensorMapping

# Values substituted for masked or NaN inputs before the EOS, so its gradient
# is finite there; the output is NaN at those points either way.
_SAFE_SALINITY = 35.0
_SAFE_THETA = 10.0


@dataclasses.dataclass
class OptimizedDerivedVariableConfig:
    """One registered derived variable to include in the training loss.

    Parameters:
        name: The registered variable. ``"rho_wright97"`` produces
            ``rho_wright97_{k}``, the
            Wright (1997) in-situ density anomaly ``rho - 1035 kg/m^3`` of
            ``so_{k}``, ``thetao_{k}`` at the level-centre Boussinesq pressure
            of the depth coordinate.
        weight: Loss weight of every name this variable produces.
        levels: Levels ``k`` to produce; all levels of the depth coordinate by
            default.
        stds: Per-name loss scale overriding the linearized default.
    """

    name: Literal["rho_wright97"] = "rho_wright97"
    weight: float = 1.0
    levels: list[int] | None = None
    stds: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.levels is not None and (
            len(self.levels) == 0
            or any(k < 0 for k in self.levels)
            or len(set(self.levels)) != len(self.levels)
        ):
            raise ValueError(
                f"levels must be distinct non-negative ints, got {self.levels}"
            )
        bad = {k: v for k, v in self.stds.items() if not v > 0}
        if bad:
            raise ValueError(f"stds must be positive, got {bad}")


class OptimizedDerivedVariables:
    """The built feature: names, loss weights, loss scales, and ``derive``."""

    def __init__(
        self,
        derivations: list[Callable[[TensorMapping], TensorDict]],
        weights: dict[str, float],
        means: dict[str, float],
        stds: dict[str, float],
    ):
        self._derivations = derivations
        self.names = list(weights)
        self.weights = weights
        self.means = means
        self.stds = stds

    def __call__(self, data: TensorMapping) -> TensorDict:
        """The derived fields only, computed from ``data``."""
        out: TensorDict = {}
        for derivation in self._derivations:
            out.update(derivation(data))
        return out

    def extend_normalizer(self, normalizer: StandardNormalizer) -> StandardNormalizer:
        """``normalizer`` with the derived names' means and stds added."""
        overlap = set(self.names).intersection(normalizer.means)
        if overlap:
            raise ValueError(
                f"optimized derived variables {sorted(overlap)} already have "
                "loss normalization constants; derived names must be new."
            )
        return StandardNormalizer(
            means={
                **normalizer.means,
                **{k: torch.tensor(v) for k, v in self.means.items()},
            },
            stds={
                **normalizer.stds,
                **{k: torch.tensor(v) for k, v in self.stds.items()},
            },
            fill_nans_on_normalize=normalizer.fill_nans_on_normalize,
            fill_nans_on_denormalize=normalizer.fill_nans_on_denormalize,
        )


class _RhoDerivation:
    """``rho_wright97_k`` from ``so_k``, ``thetao_k`` on a depth coordinate."""

    def __init__(self, levels: list[int], idepth: torch.Tensor, mask: torch.Tensor):
        self.levels = levels
        self.pressure = boussinesq_pressure(
            interface_to_center_depth(idepth.to(torch.float64))
        )
        self._mask = mask > 0
        self._mask_by_device: dict[torch.device, torch.Tensor] = {}

    def _level_mask(self, k: int, device: torch.device) -> torch.Tensor:
        if device not in self._mask_by_device:
            self._mask_by_device[device] = self._mask.to(device)
        return self._mask_by_device[device][..., k]

    def __call__(self, data: TensorMapping) -> TensorDict:
        out: TensorDict = {}
        for k in self.levels:
            S = data[f"so_{k}"]
            T = data[f"thetao_{k}"]
            valid = self._level_mask(k, S.device) & S.isfinite() & T.isfinite()
            p = self.pressure[k].to(dtype=S.dtype, device=S.device)
            rho = wright97_anomaly(
                torch.where(valid, S, _SAFE_SALINITY),
                torch.where(valid, T, _SAFE_THETA),
                p,
                RHO_0,
            )
            out[f"rho_wright97_{k}"] = torch.where(valid, rho, torch.nan)
        return out

    def linearized_std(
        self, k: int, network: StandardNormalizer, loss: StandardNormalizer
    ) -> float:
        S0, T0 = (
            torch.tensor(float(network.means[n]), dtype=torch.float64).requires_grad_()
            for n in (f"so_{k}", f"thetao_{k}")
        )
        dS, dT = torch.autograd.grad(
            wright97_anomaly(S0, T0, self.pressure[k], RHO_0), (S0, T0)
        )
        sS, sT = (float(loss.stds[n]) for n in (f"so_{k}", f"thetao_{k}"))
        return float(torch.sqrt((dT * sT) ** 2 + (dS * sS) ** 2))


def build_optimized_derived_variables(
    configs: list[OptimizedDerivedVariableConfig],
    vertical_coordinate: VerticalCoordinate,
    network_normalizer: StandardNormalizer,
    loss_normalizer: StandardNormalizer,
    loss_names: list[str],
) -> OptimizedDerivedVariables:
    """Validate the configs against the stepper and build the feature.

    Args:
        configs: The configured derived variables.
        vertical_coordinate: The stepper's vertical coordinate; ``rho_wright97`` needs a
            ``DepthCoordinate``.
        network_normalizer: Source of the linearization point (means).
        loss_normalizer: Source of the linearized scale (stds).
        loss_names: The stepper's loss names, which must contain every input
            of every derived variable and none of its outputs.
    """
    derivations: list[Callable[[TensorMapping], TensorDict]] = []
    weights: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for config in configs:
        if config.name != "rho_wright97":
            raise ValueError(f"unknown optimized derived variable {config.name!r}")
        # Only a DepthCoordinate carries the idepth and mask rho_wright97 needs; the
        # VerticalCoordinate interface has no depth accessor to use instead.
        if not isinstance(vertical_coordinate, DepthCoordinate):
            raise ValueError(
                "optimized derived variable 'rho_wright97' needs a DepthCoordinate, "
                f"got {type(vertical_coordinate).__name__}."
            )
        n_levels = len(vertical_coordinate) - 1
        levels = list(range(n_levels)) if config.levels is None else config.levels
        bad = [k for k in levels if not 0 <= k < n_levels]
        if bad:
            raise ValueError(
                f"optimized derived variable 'rho_wright97' levels {bad} are outside "
                f"the depth coordinate's {n_levels} levels."
            )
        missing = [
            n
            for k in levels
            for n in (f"so_{k}", f"thetao_{k}")
            if n not in loss_names
            or n not in network_normalizer.means
            or n not in loss_normalizer.stds
        ]
        if missing:
            raise ValueError(
                "optimized derived variable 'rho_wright97' needs these inputs among "
                f"the loss names, with normalization constants: {missing}."
            )
        derivation = _RhoDerivation(
            levels, vertical_coordinate.idepth, vertical_coordinate.mask
        )
        names = [f"rho_wright97_{k}" for k in levels]
        unknown = sorted(set(config.stds) - set(names))
        if unknown:
            raise ValueError(
                f"optimized derived variable 'rho_wright97' stds names {unknown} are "
                f"not among the names it produces, {names}."
            )
        for k, name in zip(levels, names):
            if name in weights or name in loss_names:
                raise ValueError(f"optimized derived name {name!r} is not unique.")
            weights[name] = config.weight
            means[name] = 0.0
            stds[name] = config.stds.get(
                name,
                derivation.linearized_std(k, network_normalizer, loss_normalizer),
            )
        derivations.append(derivation)
    return OptimizedDerivedVariables(derivations, weights, means, stds)
