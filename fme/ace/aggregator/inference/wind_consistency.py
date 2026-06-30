"""Companion diagnostic: log-pressure self-consistency of pressure-surface winds.

This is a cleanly separated, config-gated inference aggregator (disabled by
default). It checks whether the model's two representations of the horizontal
wind agree: the directly-predicted pressure-surface winds (``UGRD{lvl}`` /
``VGRD{lvl}``) versus the model-level winds (``eastward_wind_*`` /
``northward_wind_*``) interpolated to the same pressure surface, linear in
log-pressure. The residual between the two is reported as signed components, the
magnitude of the residual vector (``residual_speed``), and the difference of the
two wind speeds (``speed_residual``) -- the latter two are distinct quantities
(equal only when the vectors are parallel), so both are reported.

Columns where the target pressure lies outside the model-level range (e.g. an
850 hPa surface below ground over high terrain) are NaN-masked and excluded from
the area mean via a NaN-aware weighted mean, which is safe here because this
aggregator owns its own metric reduction.
"""

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import field
from typing import Any

import numpy as np
import torch
import xarray as xr

from fme.core.coordinates import (
    HorizontalCoordinates,
    HybridSigmaPressureCoordinate,
    NullVerticalCoordinate,
)
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.distributed import Distributed
from fme.core.gridded_ops import GriddedOperations
from fme.core.metrics import spherical_area_weights
from fme.core.typing_ import TensorMapping

from .build_context import MetricBuildContext, MetricNotSupportedError
from .data import InferenceBatchData, MetricBuildResult

# Quantities reported for each pressure surface, in a fixed order.
_QUANTITIES = (
    "eastward_residual",
    "northward_residual",
    "residual_speed",
    "speed_residual",
)


def interpolate_to_pressure_log_linear(
    field: torch.Tensor,
    level_pressure: torch.Tensor,
    target_pressure_pa: float,
) -> torch.Tensor:
    """Interpolate a model-level field to a pressure surface, linear in log(p).

    Args:
        field: values on model levels, shape ``(..., n_level)`` (level last).
        level_pressure: pressure in Pa at each model level, same shape as
            ``field``, strictly increasing along the last axis (level 0 is the
            model top / lowest pressure).
        target_pressure_pa: the target pressure surface in Pa.

    Returns:
        The interpolated field with the level axis removed (shape ``(...)``).
        Columns whose pressure range does not bracket ``target_pressure_pa``
        (the surface is above the top level or below the bottom level) are set
        to NaN.
    """
    log_p = torch.log(level_pressure)
    target = math.log(target_pressure_pa)
    n_level = log_p.shape[-1]

    target_col = torch.full_like(log_p[..., :1], target)
    # idx is the first level whose log-pressure is >= target; the bracketing
    # levels are (idx - 1, idx). idx == 0 means the target is above the top
    # level, idx == n_level means it is below the bottom level: both out of range.
    idx = torch.searchsorted(log_p.contiguous(), target_col)
    out_of_range = (idx == 0) | (idx == n_level)

    upper = idx.clamp(1, n_level - 1)
    lower = upper - 1
    log_p_lower = torch.gather(log_p, -1, lower)
    log_p_upper = torch.gather(log_p, -1, upper)
    field_lower = torch.gather(field, -1, lower)
    field_upper = torch.gather(field, -1, upper)

    weight = (target - log_p_lower) / (log_p_upper - log_p_lower)
    interpolated = field_lower + weight * (field_upper - field_lower)
    interpolated = torch.where(
        out_of_range, torch.full_like(interpolated, float("nan")), interpolated
    )
    return interpolated.squeeze(-1)


def _count_model_levels(stream: TensorMapping) -> int:
    """Number of contiguous ``eastward_wind_{i}`` levels present (0, 1, ...)."""
    n = 0
    while f"eastward_wind_{n}" in stream:
        n += 1
    return n


def _stack_model_levels(
    stream: TensorMapping, prefix: str, n_level: int
) -> torch.Tensor:
    """Stack ``{prefix}{0..n_level-1}`` along a new last axis (level ascending)."""
    return torch.stack([stream[f"{prefix}{i}"] for i in range(n_level)], dim=-1)


class _NanAwareWeightedMean:
    """Streaming NaN-aware area-weighted mean over (sample, time, lat, lon).

    Accumulates running numerator ``sum(X * w)`` and denominator ``sum(w)`` over
    all cells with the contributions of NaN cells dropped, so the final mean
    ``numerator / denominator`` excludes NaN cells. Float64 sums are used for
    numerical stability across many batches.
    """

    def __init__(self, device: torch.device):
        self._num = torch.zeros((), dtype=torch.float64, device=device)
        self._den = torch.zeros((), dtype=torch.float64, device=device)

    def add(self, x: torch.Tensor, weights: torch.Tensor) -> None:
        """Accumulate a per-cell field ``x`` with broadcastable ``weights``.

        Args:
            x: field of shape (sample, time, lat, lon), possibly containing NaN.
            weights: area weights of shape (lat, lon), broadcast over ``x``.
        """
        x64 = x.to(torch.float64)
        w = weights.to(torch.float64).broadcast_to(x64.shape)
        is_nan = torch.isnan(x64)
        zero = torch.zeros((), dtype=torch.float64, device=x64.device)
        self._num = self._num + torch.where(is_nan, zero, x64 * w).sum()
        self._den = self._den + torch.where(is_nan, zero, w).sum()

    def mean(self) -> torch.Tensor:
        """Reduce across ranks and return the finalized mean (NaN if empty)."""
        dist = Distributed.get_instance()
        num = dist.reduce_sum(self._num.clone())
        den = dist.reduce_sum(self._den.clone())
        if den == 0:
            return torch.full((), float("nan"), dtype=torch.float64, device=den.device)
        return num / den


class WindConsistencyAggregator:
    """Log-pressure self-consistency of pressure-surface winds.

    For each configured pressure surface and each stream (generated, target),
    the model-level winds (``eastward_wind_*`` / ``northward_wind_*``) are
    interpolated -- linear in log-pressure -- to that surface using the
    surface-pressure-derived midpoint pressures, and compared to the directly
    predicted pressure-surface winds (``UGRD{p}`` / ``VGRD{p}``). Four residual
    quantities are accumulated per surface with a streaming, NaN-aware,
    area-weighted mean (columns whose pressure range does not bracket the
    surface are NaN-masked and excluded).
    """

    def __init__(
        self,
        gridded_operations: GriddedOperations,
        horizontal_coordinates: HorizontalCoordinates,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        surfaces_hpa: list[int],
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
    ):
        """
        Args:
            gridded_operations: Computes gridded (area-weighted) operations.
            horizontal_coordinates: Provides the latitudes used for area weights.
            vertical_coordinate: Provides ``interface_pressure(surface_pressure)``.
            surfaces_hpa: Pressure surfaces (hPa) to evaluate, e.g. [200, 500, 850].
            variable_metadata: Optional variable metadata (unused, kept for parity
                with other aggregators).
        """
        self._ops = gridded_operations
        self._vertical_coordinate = vertical_coordinate
        self._surfaces_hpa = list(surfaces_hpa)
        self._variable_metadata = variable_metadata or {}
        # Area weights (lat, lon); built on CPU now, moved to data device lazily.
        lat = horizontal_coordinates.lat_1d
        n_lon = len(horizontal_coordinates.coords[horizontal_coordinates.dims[1]])
        self._area_weights = spherical_area_weights(lat.cpu(), n_lon)
        self._area_weights_on_device: torch.Tensor | None = None
        # Per (surface, quantity) accumulators for gen, target and squared diff.
        self._gen: dict[tuple[int, str], _NanAwareWeightedMean] = {}
        self._target: dict[tuple[int, str], _NanAwareWeightedMean] = {}
        self._sq_diff: dict[tuple[int, str], _NanAwareWeightedMean] = {}
        self._device: torch.device | None = None

    def _accumulator(
        self, store: dict[tuple[int, str], _NanAwareWeightedMean], key: tuple[int, str]
    ) -> _NanAwareWeightedMean:
        if key not in store:
            assert self._device is not None
            store[key] = _NanAwareWeightedMean(self._device)
        return store[key]

    @staticmethod
    def _residual_quantities(
        u_surface: torch.Tensor,
        v_surface: torch.Tensor,
        u_interp: torch.Tensor,
        v_interp: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eastward_residual = u_surface - u_interp
        northward_residual = v_surface - v_interp
        residual_speed = torch.sqrt(eastward_residual**2 + northward_residual**2)
        speed_residual = torch.sqrt(u_surface**2 + v_surface**2) - torch.sqrt(
            u_interp**2 + v_interp**2
        )
        return {
            "eastward_residual": eastward_residual,
            "northward_residual": northward_residual,
            "residual_speed": residual_speed,
            "speed_residual": speed_residual,
        }

    def _stream_quantities(
        self, stream: TensorMapping, surface_hpa: int, n_level: int
    ) -> dict[str, torch.Tensor] | None:
        """Per-cell residual quantities for one stream at one surface, or None
        if the directly-predicted surface winds are absent for this surface.
        """
        u_key, v_key = f"UGRD{surface_hpa}", f"VGRD{surface_hpa}"
        if u_key not in stream or v_key not in stream:
            return None
        u_levels = _stack_model_levels(stream, "eastward_wind_", n_level)
        v_levels = _stack_model_levels(stream, "northward_wind_", n_level)
        interfaces = self._vertical_coordinate.interface_pressure(stream["PRESsfc"])
        midpoints = 0.5 * (interfaces[..., :-1] + interfaces[..., 1:])
        target_pa = surface_hpa * 100.0
        u_interp = interpolate_to_pressure_log_linear(u_levels, midpoints, target_pa)
        v_interp = interpolate_to_pressure_log_linear(v_levels, midpoints, target_pa)
        return self._residual_quantities(
            stream[u_key], stream[v_key], u_interp, v_interp
        )

    @torch.no_grad()
    def record_batch(self, data: InferenceBatchData) -> None:
        # Mirror TimeMeanAggregator/trend: skip the initial condition when it is
        # the first timestep of the first batch so it is not double-counted.
        time_slice = slice(1, None) if data.i_time_start == 0 else slice(None)

        def sliced(stream: TensorMapping) -> dict[str, torch.Tensor]:
            return {name: tensor[:, time_slice] for name, tensor in stream.items()}

        gen = sliced(data.prediction)
        if next(iter(gen.values())).shape[1] == 0:
            return
        # Presence-gate: model-level winds and surface pressure must be present.
        n_level = _count_model_levels(gen)
        if n_level == 0 or "PRESsfc" not in gen:
            return

        if self._device is None:
            self._device = next(iter(gen.values())).device
        if self._area_weights_on_device is None:
            sample = next(iter(gen.values()))
            self._area_weights_on_device = self._area_weights.to(
                device=sample.device, dtype=sample.dtype
            )
        w = self._area_weights_on_device

        target: dict[str, torch.Tensor] | None = (
            sliced(data.target) if data.has_target else None
        )

        for p in self._surfaces_hpa:
            gen_q = self._stream_quantities(gen, p, n_level)
            if gen_q is None:
                continue
            for quantity, value in gen_q.items():
                self._accumulator(self._gen, (p, quantity)).add(value, w)
            if target is None:
                continue
            target_q = self._stream_quantities(target, p, n_level)
            if target_q is None:
                continue
            for quantity, value in target_q.items():
                self._accumulator(self._target, (p, quantity)).add(value, w)
            for quantity in gen_q:
                diff = gen_q[quantity] - target_q[quantity]
                self._accumulator(self._sq_diff, (p, quantity)).add(diff**2, w)

    def _means(self) -> dict[tuple[int, str, str], float]:
        """Finalize all (surface, quantity, stat) scalars (reduces across ranks).

        All collectives are issued in a deterministic order on every rank.
        """
        results: dict[tuple[int, str, str], float] = {}
        for p in self._surfaces_hpa:
            for quantity in _QUANTITIES:
                gen_key = (p, quantity)
                gen = (
                    self._gen[gen_key].mean().item()
                    if gen_key in self._gen
                    else float("nan")
                )
                target = (
                    self._target[gen_key].mean().item()
                    if gen_key in self._target
                    else float("nan")
                )
                if gen_key in self._sq_diff:
                    rmse = math.sqrt(self._sq_diff[gen_key].mean().item())
                else:
                    rmse = float("nan")
                bias = gen - target
                results[(p, quantity, "gen")] = gen
                results[(p, quantity, "target")] = target
                results[(p, quantity, "bias")] = bias
                results[(p, quantity, "rmse")] = rmse
        return results

    @torch.no_grad()
    def get_logs(self, label: str) -> dict[str, Any]:
        means = self._means()
        if len(label) > 0:
            label = label + "/"
        logs: dict[str, Any] = {}
        for (p, quantity, stat), value in means.items():
            logs[f"{label}{quantity}_{p}/{stat}"] = float(value)
        return logs

    def get_dataset(self) -> xr.Dataset:
        means = self._means()
        data_vars: dict[str, Any] = {}
        for p in self._surfaces_hpa:
            for quantity in _QUANTITIES:
                gen = means[(p, quantity, "gen")]
                target = means[(p, quantity, "target")]
                data_vars[f"{quantity}_{p}"] = (
                    ("source",),
                    np.array([target, gen], dtype=np.float64),
                )
        return xr.Dataset(data_vars, coords={"source": ["target", "prediction"]})


@dataclasses.dataclass
class WindConsistencyMetricConfig:
    """Configuration for the wind-consistency inference metric.

    Attributes:
        surfaces_hpa: Pressure surfaces (hPa) to evaluate.
        name: Name used to label the metric's logs and diagnostics.
        enabled: Whether the metric is computed. Disabled by default.
        strict: If True, raise rather than skip when the metric is not
            supported for the current configuration.
    """

    surfaces_hpa: list[int] = field(default_factory=lambda: [200, 500, 850])
    name: str = "wind_consistency"
    enabled: bool = False
    strict: bool = False

    def get_name(self) -> str:
        return self.name

    def build(self, ctx: MetricBuildContext) -> MetricBuildResult:
        vc = ctx.vertical_coordinate
        if (
            vc is None
            or isinstance(vc, NullVerticalCoordinate)
            or not hasattr(vc, "interface_pressure")
            or not isinstance(vc, HybridSigmaPressureCoordinate)
        ):
            raise MetricNotSupportedError(
                "wind_consistency metric requires a vertical coordinate with "
                "interface_pressure (e.g. HybridSigmaPressureCoordinate)."
            )
        return MetricBuildResult(
            aggregator=WindConsistencyAggregator(
                gridded_operations=ctx.ops,
                horizontal_coordinates=ctx.horizontal_coordinates,
                vertical_coordinate=vc,
                surfaces_hpa=self.surfaces_hpa,
                variable_metadata=ctx.variable_metadata,
            )
        )
