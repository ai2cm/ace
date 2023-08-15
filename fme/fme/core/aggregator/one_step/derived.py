"""Derived metrics take the global state as input and usually output a new
variable, e.g. dry air mass."""

import logging
from typing import Dict, Mapping, Optional, Protocol, Tuple

import torch

from fme.core import metrics
from fme.core.aggregator.climate_data import CLIMATE_FIELD_NAME_PREFIXES, ClimateData
from fme.core.data_loading.typing import SigmaCoordinates
from fme.core.device import get_device


class DerivedMetric(Protocol):
    """Derived metrics are computed from the global state and usually output a
    new variable, e.g. dry air mass."""

    def record(self, target: ClimateData, gen: ClimateData) -> None:
        ...

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the derived metric applied to the target and data generated
        by the model."""
        ...


class DryAirMass(DerivedMetric):
    """Computes the dry air mass tendency of the first time step, averaged over
    the batch. If the data does not contain the required fields, then returns
    NaN."""

    def __init__(
        self,
        area_weights: torch.Tensor,
        sigma_coordinates: SigmaCoordinates,
        device: torch.device,
        spatial_dims=(2, 3),
    ):
        self._area_weights = area_weights
        self._sigma_coordinates = sigma_coordinates
        self._dry_air_mass_target_total: Optional[torch.Tensor] = None
        self._dry_air_mass_gen_total: Optional[torch.Tensor] = None
        self._device = device
        self._spatial_dims: Tuple[int, int] = spatial_dims

    def _validate_data(self, data: ClimateData) -> bool:
        """Checks that the data contains the required atmospheric fields."""
        try:
            data.specific_total_water
        except ValueError:
            return False
        try:
            data.surface_pressure
        except KeyError:
            return False
        return True

    def record(self, target: ClimateData, gen: ClimateData) -> None:
        if not self._validate_data(target) or not self._validate_data(gen):
            self._dry_air_mass_target_total = torch.tensor(torch.nan)
            self._dry_air_mass_gen_total = torch.tensor(torch.nan)
            logging.warning(
                f"Could not compute dry air mass due to missing atmospheric fields."
            )
            return

        dry_air_mass_target = (
            metrics.compute_dry_air_mass(
                target.specific_total_water[:, 0:2, ...],  # (sample, time, y, x, level)
                target.surface_pressure[:, 0:2, ...],
                self._sigma_coordinates.ak,
                self._sigma_coordinates.bk,
                self._area_weights,
            )
            .sum(self._spatial_dims)
            .diff(dim=-1)  # (sample, time)
            .mean()
        )

        dry_air_mass_gen = (
            metrics.compute_dry_air_mass(
                gen.specific_total_water[:, 0:2, ...],
                gen.surface_pressure[:, 0:2, ...],
                self._sigma_coordinates.ak,
                self._sigma_coordinates.bk,
                self._area_weights,
            )
            .sum(self._spatial_dims)
            .diff(dim=-1)  # (sample, time)
            .mean()
        )

        if self._dry_air_mass_gen_total is None:
            self._dry_air_mass_gen_total = torch.zeros_like(
                dry_air_mass_gen, device=self._device
            )
        if self._dry_air_mass_target_total is None:
            self._dry_air_mass_target_total = torch.zeros_like(
                dry_air_mass_target, device=self._device
            )

        self._dry_air_mass_target_total += dry_air_mass_target
        self._dry_air_mass_gen_total += dry_air_mass_gen

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            self._dry_air_mass_target_total is None
            or self._dry_air_mass_gen_total is None
        ):
            raise ValueError("No batches have been recorded.")
        return self._dry_air_mass_target_total, self._dry_air_mass_gen_total


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
            "dry_air_mass": DryAirMass(
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
            target, gen = self._derived_metrics[metric_name].get()
            logs[f"{label}/{metric_name}/target"] = target / self._n_batches
            logs[f"{label}/{metric_name}/gen"] = gen / self._n_batches
        return logs
