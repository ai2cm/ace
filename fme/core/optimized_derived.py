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
                   inputs clamped to thetao_clamp x so_clamp before the EOS

    pbo_wright97:            P - <P>,  P = RHO_0 * G_EARTH * zos + G_EARTH * C  [Pa]
    steric_height_wright97:  -(C - <C>) / RHO_0                                [m]
                   C   = sum_k rho_wright97_k * dz_k   (all levels; 0 where rho NaN)
                   dz  = DepthCoordinate.dz            (partial bottom cells, deptho)
                   <x> = area-weighted mean over mask_0 (zos finite for pbo)
                   NaN where mask_0 == 0

pbo_wright97 is a proxy for MOM6 ``pbo`` (bottom pressure) up to the static
``RHO_0 * G_EARTH * deptho`` and the global mean; zos is demeaned in the data,
so both terms are demeaned here and a global offset of predicted zos or of C
does not enter this loss.

Loss scale of pbo_wright97 / steric_height_wright97, unless given in ``stds``,
is ``PBO_WRIGHT97_STD`` / ``STERIC_HEIGHT_WRIGHT97_STD`` (target data). Loss
scale of rho_wright97_k, unless given in ``stds``, is the linearization

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
from fme.core.gridded_ops import GriddedOperations
from fme.core.normalizer import StandardNormalizer
from fme.core.ocean_eos import (
    G_EARTH,
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

# Default EOS input clamp box [degC], [PSU]: contains the training data's
# thetao/so range with margin, and the Wright (1997) denominator stays far from
# its zero here (.scratch/2026-09-24-0145-rho-eos-loss/06a_clamp_bounds.yaml
# box_recommended, from 06a_clamp_bounds.py). Outside it the rho term gives no
# gradient.
THETAO_CLAMP_RANGE = (-5.0, 42.0)
SO_CLAMP_RANGE = (0.0, 73.0)

# Default loss scales of the column variables: std of the target field's
# per-cell time anomaly, area-weighted over wet cells, over the training windows
# (.scratch/2026-09-24-0145-rho-eos-loss/09_bottom_pressure_magnitudes.yaml
# default_stds, from 09_bottom_pressure_magnitudes.py). The full-field std is
# dominated by the static compressibility part of C, which cancels in
# prediction - target.
PBO_WRIGHT97_STD = 375.0  # [Pa]
STERIC_HEIGHT_WRIGHT97_STD = 0.05  # [m]

_COLUMN_NAMES = ("pbo_wright97", "steric_height_wright97")
_DEFAULT_COLUMN_STDS = {
    "pbo_wright97": PBO_WRIGHT97_STD,
    "steric_height_wright97": STERIC_HEIGHT_WRIGHT97_STD,
}


@dataclasses.dataclass
class OptimizedDerivedVariableConfig:
    """One registered derived variable to include in the training loss.

    Parameters:
        name: The registered variable. ``"rho_wright97"`` produces
            ``rho_wright97_{k}``, the
            Wright (1997) in-situ density anomaly ``rho - 1035 kg/m^3`` of
            ``so_{k}``, ``thetao_{k}`` at the level-centre Boussinesq pressure
            of the depth coordinate. ``"pbo_wright97"`` produces
            ``pbo_wright97``, the globally demeaned hydrostatic bottom pressure
            anomaly ``RHO_0 g zos + g sum_k rho_wright97_k dz_k`` [Pa].
            ``"steric_height_wright97"`` produces ``steric_height_wright97``,
            the globally demeaned ``-(1/RHO_0) sum_k rho_wright97_k dz_k`` [m].
        weight: Loss weight of every name this variable produces.
        levels: Levels ``k`` to produce; all levels of the depth coordinate by
            default. Must be unset for the column variables, which use every
            level.
        stds: Per-name loss scale overriding the default (linearized for
            ``rho_wright97``, ``PBO_WRIGHT97_STD`` /
            ``STERIC_HEIGHT_WRIGHT97_STD`` for the column variables).
        thetao_clamp: ``[min, max]`` [degC] ``thetao_k`` is clamped to before
            the EOS, for prediction and target.
        so_clamp: ``[min, max]`` [PSU] ``so_k`` is clamped to before the EOS.
    """

    name: Literal["rho_wright97", "pbo_wright97", "steric_height_wright97"] = (
        "rho_wright97"
    )
    weight: float = 1.0
    levels: list[int] | None = None
    stds: dict[str, float] = dataclasses.field(default_factory=dict)
    thetao_clamp: list[float] = dataclasses.field(
        default_factory=lambda: list(THETAO_CLAMP_RANGE)
    )
    so_clamp: list[float] = dataclasses.field(
        default_factory=lambda: list(SO_CLAMP_RANGE)
    )

    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"weight must be non-negative, got {self.weight}")
        if self.name in _COLUMN_NAMES and self.levels is not None:
            raise ValueError(
                f"levels must be unset for {self.name!r}, which integrates every "
                f"level, got {self.levels}"
            )
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
        for field in ("thetao_clamp", "so_clamp"):
            bounds = getattr(self, field)
            if len(bounds) != 2 or not bounds[0] < bounds[1]:
                raise ValueError(f"{field} must be [min, max], min < max, got {bounds}")


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

    def __init__(
        self,
        levels: list[int],
        idepth: torch.Tensor,
        mask: torch.Tensor,
        thetao_clamp: tuple[float, float] = THETAO_CLAMP_RANGE,
        so_clamp: tuple[float, float] = SO_CLAMP_RANGE,
    ):
        self.levels = levels
        self.thetao_clamp = thetao_clamp
        self.so_clamp = so_clamp
        self.pressure = boussinesq_pressure(
            interface_to_center_depth(idepth.to(torch.float64))
        )
        self._mask = mask > 0
        self._mask_by_device: dict[torch.device, torch.Tensor] = {}

    def _level_mask(self, k: int, device: torch.device) -> torch.Tensor:
        if device not in self._mask_by_device:
            self._mask_by_device[device] = self._mask.to(device)
        return self._mask_by_device[device][..., k]

    def level(self, data: TensorMapping, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rho, valid)`` at level ``k``; ``rho`` is finite everywhere, and
        meaningful only where ``valid``.
        """
        S = data[f"so_{k}"]
        T = data[f"thetao_{k}"]
        valid = self._level_mask(k, S.device) & S.isfinite() & T.isfinite()
        p = self.pressure[k].to(dtype=S.dtype, device=S.device)
        rho = wright97_anomaly(
            torch.where(valid, S, _SAFE_SALINITY).clamp(*self.so_clamp),
            torch.where(valid, T, _SAFE_THETA).clamp(*self.thetao_clamp),
            p,
            RHO_0,
        )
        return rho, valid

    def __call__(self, data: TensorMapping) -> TensorDict:
        out: TensorDict = {}
        for k in self.levels:
            rho, valid = self.level(data, k)
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


class _ColumnDerivation:
    """``pbo_wright97`` or ``steric_height_wright97`` from the column of
    ``rho_wright97_k`` (all levels), ``zos`` and the depth coordinate.
    """

    def __init__(
        self,
        name: str,
        rho: _RhoDerivation,
        dz: torch.Tensor,
        surface_mask: torch.Tensor,
        gridded_operations: GriddedOperations,
    ):
        if name not in _COLUMN_NAMES:
            raise ValueError(f"unknown column variable {name!r}")
        self.name = name
        self.rho = rho
        self._dz = dz
        self._wet = surface_mask > 0
        self._ops = gridded_operations
        self._statics_by_device: dict[
            torch.device, tuple[torch.Tensor, torch.Tensor]
        ] = {}

    def _statics(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if device not in self._statics_by_device:
            self._statics_by_device[device] = (
                self._dz.to(device),
                self._wet.to(device),
            )
        return self._statics_by_device[device]

    def column(self, data: TensorMapping) -> torch.Tensor:
        """``C = sum_k rho_wright97_k * dz_k`` [kg m-2], 0 where rho is invalid."""
        C: torch.Tensor | None = None
        for k in self.rho.levels:
            rho, valid = self.rho.level(data, k)
            dz, _ = self._statics(rho.device)
            term = torch.where(valid, rho * dz[..., k].to(rho.dtype), 0.0)
            C = term if C is None else C + term
        assert C is not None
        return C

    def __call__(self, data: TensorMapping) -> TensorDict:
        C = self.column(data)
        _, wet = self._statics(C.device)
        wet = wet.expand(C.shape)
        if self.name == "pbo_wright97":
            zos = data["zos"]
            wet = wet & zos.isfinite()
            x = RHO_0 * G_EARTH * torch.where(wet, zos, 0.0) + G_EARTH * C
        else:
            x = -C / RHO_0
        mean = self._ops.regional_area_weighted_mean(
            x, regional_weights=wet.to(x.dtype), keepdim=True
        )
        return {self.name: torch.where(wet, x - mean, torch.nan)}


def build_optimized_derived_variables(
    configs: list[OptimizedDerivedVariableConfig],
    vertical_coordinate: VerticalCoordinate,
    network_normalizer: StandardNormalizer,
    loss_normalizer: StandardNormalizer,
    loss_names: list[str],
    gridded_operations: GriddedOperations | None = None,
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
        gridded_operations: The stepper's gridded operations, for the global
            means of the column variables; required for those.
    """
    derivations: list[Callable[[TensorMapping], TensorDict]] = []
    weights: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for config in configs:
        if config.name != "rho_wright97" and config.name not in _COLUMN_NAMES:
            raise ValueError(f"unknown optimized derived variable {config.name!r}")
        # Only a DepthCoordinate carries the idepth and mask rho_wright97 needs; the
        # VerticalCoordinate interface has no depth accessor to use instead.
        if not isinstance(vertical_coordinate, DepthCoordinate):
            raise ValueError(
                f"optimized derived variable {config.name!r} needs a DepthCoordinate, "
                f"got {type(vertical_coordinate).__name__}."
            )
        n_levels = len(vertical_coordinate) - 1
        levels = list(range(n_levels)) if config.levels is None else config.levels
        bad = [k for k in levels if not 0 <= k < n_levels]
        if bad:
            raise ValueError(
                f"optimized derived variable {config.name!r} levels {bad} are outside "
                f"the depth coordinate's {n_levels} levels."
            )
        required = [n for k in levels for n in (f"so_{k}", f"thetao_{k}")]
        if config.name == "pbo_wright97":
            required.append("zos")
        missing = [
            n
            for n in required
            if n not in loss_names
            or (
                n != "zos"
                and (n not in network_normalizer.means or n not in loss_normalizer.stds)
            )
        ]
        if missing:
            raise ValueError(
                f"optimized derived variable {config.name!r} needs these inputs among "
                f"the loss names, with normalization constants: {missing}."
            )
        rho_derivation = _RhoDerivation(
            levels,
            vertical_coordinate.idepth,
            vertical_coordinate.mask,
            thetao_clamp=(config.thetao_clamp[0], config.thetao_clamp[1]),
            so_clamp=(config.so_clamp[0], config.so_clamp[1]),
        )
        derivation: Callable[[TensorMapping], TensorDict]
        if config.name == "rho_wright97":
            names = [f"rho_wright97_{k}" for k in levels]
            derivation = rho_derivation
        else:
            if gridded_operations is None:
                raise ValueError(
                    f"optimized derived variable {config.name!r} needs the gridded "
                    "operations for its global mean."
                )
            names = [config.name]
            derivation = _ColumnDerivation(
                config.name,
                rho_derivation,
                dz=vertical_coordinate.dz,
                surface_mask=vertical_coordinate.mask[..., 0],
                gridded_operations=gridded_operations,
            )
        unknown = sorted(set(config.stds) - set(names))
        if unknown:
            raise ValueError(
                f"optimized derived variable {config.name!r} stds names {unknown} are "
                f"not among the names it produces, {names}."
            )
        for k, name in zip(levels, names):
            if name in weights or name in loss_names:
                raise ValueError(f"optimized derived name {name!r} is not unique.")
            weights[name] = config.weight
            means[name] = 0.0
            if name in config.stds:
                stds[name] = config.stds[name]
            elif config.name == "rho_wright97":
                stds[name] = rho_derivation.linearized_std(
                    k, network_normalizer, loss_normalizer
                )
            else:
                stds[name] = _DEFAULT_COLUMN_STDS[name]
        derivations.append(derivation)
    return OptimizedDerivedVariables(derivations, weights, means, stds)
