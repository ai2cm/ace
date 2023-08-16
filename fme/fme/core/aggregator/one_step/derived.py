"""Derived metrics take the global state as input and usually output a new
variable, e.g. dry air mass."""

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Protocol, Tuple

import torch

from fme.core import metrics
from fme.core.aggregator.climate_data import CLIMATE_FIELD_NAME_PREFIXES, ClimateData
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device


@dataclass
class _TargetGenPair:
    target: torch.Tensor
    gen: torch.Tensor


class DerivedMetric(Protocol):
    """Derived metrics are computed from the global state and usually output a
    new variable, e.g. dry air tendencies."""

    def record(self, target: ClimateData, gen: ClimateData) -> None:
        ...

    def get(self) -> _TargetGenPair:
        """Returns the derived metric applied to the target and data generated
        by the model."""
        ...


class DryAir(DerivedMetric):
    """Computes absolute value of the dry air tendency of the first time step,
    averaged over the batch. If the data does not contain the required fields,
    then returns NaN."""

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        device: torch.device,
        spatial_dims=(2, 3),
    ):
        self._area_weights = area_weights
        self._sigma_coordinates = sigma_coordinates
        self._dry_air_target_total: Optional[torch.Tensor] = None
        self._dry_air_gen_total: Optional[torch.Tensor] = None
        self._device = device
        self._spatial_dims: Tuple[int, int] = spatial_dims

    def record(self, target: ClimateData, gen: ClimateData) -> None:
        def _compute_dry_air_helper(climate_data: ClimateData) -> torch.Tensor:
            water = climate_data.specific_total_water
            pressure = climate_data.surface_pressure
            if water is None or pressure is None:
                return torch.tensor(torch.nan)
            return (
                metrics.weighted_mean(
                    metrics.surface_pressure_due_to_dry_air(
                        water[:, 0:2, ...],  # (sample, time, y, x, level)
                        pressure[:, 0:2, ...],
                        self._sigma_coordinates.ak,
                        self._sigma_coordinates.bk,
                    ),
                    self._area_weights,
                    dim=(2, 3),
                )
                .diff(dim=-1)
                .abs()
                .mean()
            )

        dry_air_target = _compute_dry_air_helper(target)
        dry_air_gen = _compute_dry_air_helper(gen)

        # initialize
        if self._dry_air_target_total is None:
            self._dry_air_target_total = torch.zeros_like(
                dry_air_target, device=self._device
            )
        if self._dry_air_gen_total is None:
            self._dry_air_gen_total = torch.zeros_like(dry_air_gen, device=self._device)

        self._dry_air_target_total += dry_air_target
        self._dry_air_gen_total += dry_air_gen

    def get(self) -> _TargetGenPair:
        if self._dry_air_target_total is None or self._dry_air_gen_total is None:
            raise ValueError("No batches have been recorded.")
        return _TargetGenPair(
            target=self._dry_air_target_total, gen=self._dry_air_gen_total
        )


class DerivedMetricsAggregator:
    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        climate_field_name_prefixes: Mapping[str, str] = CLIMATE_FIELD_NAME_PREFIXES,
    ):
        self.area_weights = area_weights
        self.sigma_coordinates = sigma_coordinates
        self.climate_field_name_prefixes = climate_field_name_prefixes
        device = get_device()
        self._derived_metrics: Dict[str, DerivedMetric] = {
            "surface_pressure_due_to_dry_air": DryAir(
                self.area_weights, self.sigma_coordinates, device=device
            )
        }
        self._n_batches = 0

    @torch.no_grad()
    def record_batch(
        self,
        loss: float,
        target_data: Mapping[str, torch.Tensor],
        gen_data: Mapping[str, torch.Tensor],
        target_data_norm: Mapping[str, torch.Tensor],
        gen_data_norm: Mapping[str, torch.Tensor],
    ):
        del loss, target_data_norm, gen_data_norm  # unused
        target = ClimateData(target_data, self.climate_field_name_prefixes)
        gen = ClimateData(gen_data, self.climate_field_name_prefixes)

        for metric_fn in self._derived_metrics.values():
            metric_fn.record(target, gen)

        # only increment n_batches if we actually recorded a batch
        self._n_batches += 1

    def get_logs(self, label: str):
        logs = dict()
        for metric_name in self._derived_metrics:
            values = self._derived_metrics[metric_name].get()
            logs[f"{label}/{metric_name}/target"] = values.target / self._n_batches
            logs[f"{label}/{metric_name}/gen"] = values.gen / self._n_batches
        return logs
