"""Inference aggregator for SST-perturbation warming-response diagnostics.

This aggregator is used by the perturbation-response inline-inference type. A
single rollout carries several *groups* of batch members that differ only in an
SST perturbation: group 0 is the unperturbed baseline, groups 1.. are perturbed
(e.g. a uniform +4 K offset). The aggregator accumulates a per-group time-mean,
differences each perturbed group against the baseline to form a warming
*response* field, and logs:

- a 2D response map (perturbed minus baseline time-mean) for **every** predicted
  field -- the headline diagnostic for inspecting where the model warms;
- near-surface land/ocean warming ratio, area-weighted, by latitude band;
- free-troposphere/surface warming ratio over tropical ocean;
- the global-mean column warming profile, level by level.

No external (e.g. c96-SHiELD) reference is involved: the diagnostics are
computed entirely from the model's own baseline and perturbed climates.
"""

import dataclasses
import logging
import os
from collections.abc import Mapping, Sequence

import torch
import xarray as xr

from fme.ace.data_loading.batch_data import PairedData, PrognosticState
from fme.core.coordinates import HorizontalCoordinates, LatLonCoordinates
from fme.core.dataset.data_typing import VariableMetadata
from fme.core.dataset_info import DatasetInfo
from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.generics.aggregator import (
    InferenceAggregatorABC,
    InferenceLogs,
    InferenceSummary,
)
from fme.core.gridded_ops import GriddedOperations
from fme.core.wandb import Image

from ..plotting import plot_paneled_data


@dataclasses.dataclass
class LatitudeBand:
    """An absolute-latitude band (hemispherically folded).

    Parameters:
        name: Label used in log keys.
        lat_min: Minimum absolute latitude in degrees (inclusive).
        lat_max: Maximum absolute latitude in degrees (exclusive).
    """

    name: str
    lat_min: float
    lat_max: float

    def __post_init__(self):
        if not 0.0 <= self.lat_min < self.lat_max <= 90.0:
            raise ValueError(
                "LatitudeBand requires 0 <= lat_min < lat_max <= 90, got "
                f"lat_min={self.lat_min}, lat_max={self.lat_max}."
            )


def _default_bands() -> list[LatitudeBand]:
    # Matches the land-amplification diagnosis report's bands (folded).
    return [
        LatitudeBand("tropics", 0.0, 15.0),
        LatitudeBand("subtropics", 15.0, 30.0),
        LatitudeBand("midlatitudes", 30.0, 50.0),
    ]


def _default_column_temperature_names() -> list[str]:
    # Hybrid-sigma model levels, top (0) to surface (7), for the 8-level configs.
    return [f"air_temperature_{i}" for i in range(8)]


@dataclasses.dataclass
class PerturbationResponseAggregatorConfig:
    """Configuration for the perturbation-response aggregator.

    Parameters:
        near_surface_temperature_name: Variable whose land/ocean warming ratio
            is reported by latitude band.
        column_temperature_names: Vertical temperature variables, ordered from
            model top to surface, used for the vertical warming ratio and the
            global-mean column profile.
        vertical_surface_index: Index into ``column_temperature_names`` for the
            surface level of the free-troposphere/surface ratio.
        vertical_upper_index: Index into ``column_temperature_names`` for the
            free-tropospheric level (e.g. ~200 hPa).
        latitude_bands: Absolute-latitude bands for the land/ocean ratio.
        tropical_lat_max: Absolute-latitude bound for the tropical-ocean region
            used by the vertical warming ratio.
        ocean_fraction_name: Forcing variable giving the ocean fraction.
        ocean_fraction_cutoff: Cells with ocean fraction strictly greater than
            this are ocean; the rest are land.
        response_map_variables: Predicted variables to log a 2D response map
            (perturbed minus baseline time-mean) for. If ``None``, a map is
            logged for every predicted field.
    """

    near_surface_temperature_name: str = "air_temperature_7"
    column_temperature_names: list[str] = dataclasses.field(
        default_factory=_default_column_temperature_names
    )
    vertical_surface_index: int = 7
    vertical_upper_index: int = 2
    latitude_bands: list[LatitudeBand] = dataclasses.field(
        default_factory=_default_bands
    )
    tropical_lat_max: float = 15.0
    ocean_fraction_name: str = "ocean_fraction"
    ocean_fraction_cutoff: float = 0.5
    response_map_variables: list[str] | None = None

    def __post_init__(self):
        n = len(self.column_temperature_names)
        for label, index in (
            ("vertical_surface_index", self.vertical_surface_index),
            ("vertical_upper_index", self.vertical_upper_index),
        ):
            if not 0 <= index < n:
                raise ValueError(
                    f"{label}={index} is out of range for "
                    f"{n} column_temperature_names."
                )
        if not 0.0 < self.tropical_lat_max <= 90.0:
            raise ValueError(
                f"tropical_lat_max must be in (0, 90], got {self.tropical_lat_max}."
            )

    @property
    def scalar_diagnostic_names(self) -> list[str]:
        """Variables required by the scalar (ratio/profile) diagnostics.

        These must be present among the predicted fields. The per-group
        time-mean is accumulated for *all* predicted fields (so response maps
        can be drawn for each); this is the subset the scalar diagnostics index.
        """
        names = list(self.column_temperature_names)
        if self.near_surface_temperature_name not in names:
            names.append(self.near_surface_temperature_name)
        return names

    def build(
        self,
        dataset_info: DatasetInfo,
        perturbation_labels: Sequence[str],
        group_onehot: torch.Tensor,
        output_dir: str | None = None,
        save_diagnostics: bool = False,
    ) -> "PerturbationResponseAggregator":
        return PerturbationResponseAggregator(
            ops=dataset_info.gridded_operations,
            horizontal_coordinates=dataset_info.horizontal_coordinates,
            config=self,
            perturbation_labels=perturbation_labels,
            group_onehot=group_onehot,
            variable_metadata=dataset_info.variable_metadata,
            output_dir=output_dir,
            save_diagnostics=save_diagnostics,
        )


def _validate_one_hot(group_onehot: torch.Tensor) -> torch.Tensor:
    """Validate a [n_members, n_groups] one-hot encoding and return group indices.

    Group 0 is the baseline; groups 1.. are perturbations. More than two groups
    (i.e. more than one perturbation) is not yet supported.
    """
    if group_onehot.ndim != 2:
        raise ValueError(
            "group_onehot must be a [n_members, n_groups] tensor, got shape "
            f"{tuple(group_onehot.shape)}."
        )
    n_groups = group_onehot.shape[1]
    if n_groups > 2:
        raise NotImplementedError(
            "perturbation-response evaluation currently supports a single "
            "perturbation (two groups: baseline + perturbed); got "
            f"{n_groups} groups."
        )
    row_sums = group_onehot.sum(dim=1)
    if not torch.all((group_onehot == 0) | (group_onehot == 1)) or not torch.all(
        row_sums == 1
    ):
        raise ValueError("group_onehot rows must each be one-hot (exactly one 1).")
    return group_onehot.argmax(dim=1)


class PerturbationResponseAggregator(
    InferenceAggregatorABC[PrognosticState, PairedData]
):
    def __init__(
        self,
        ops: GriddedOperations,
        horizontal_coordinates: HorizontalCoordinates,
        config: PerturbationResponseAggregatorConfig,
        perturbation_labels: Sequence[str],
        group_onehot: torch.Tensor,
        variable_metadata: Mapping[str, VariableMetadata] | None = None,
        output_dir: str | None = None,
        save_diagnostics: bool = False,
    ):
        """
        Parameters:
            ops: Gridded operations for area-weighted means.
            horizontal_coordinates: Provides the latitude grid for band masks.
            config: Aggregator configuration.
            perturbation_labels: Labels for the perturbed groups (groups 1..);
                length must be ``n_groups - 1``.
            group_onehot: [n_members, n_groups] one-hot mapping each local batch
                member to its group. Group 0 is the baseline. The member order
                must match the order of the batch dimension in recorded data.
            variable_metadata: Optional per-variable metadata for response-map
                captions (long name and units).
            output_dir: Directory for diagnostics netCDF; required if
                ``save_diagnostics``.
            save_diagnostics: Whether to write the response fields to netCDF.
        """
        if not isinstance(horizontal_coordinates, LatLonCoordinates):
            raise NotImplementedError(
                "PerturbationResponseAggregator currently supports only "
                "lat/lon coordinates."
            )
        if save_diagnostics and output_dir is None:
            raise ValueError("output_dir must be set to save diagnostics.")
        self._ops = ops
        self._config = config
        self._variable_metadata = variable_metadata or {}
        self._group_index = _validate_one_hot(group_onehot)
        self._n_groups = group_onehot.shape[1]
        if len(perturbation_labels) != self._n_groups - 1:
            raise ValueError(
                f"expected {self._n_groups - 1} perturbation label(s) for "
                f"{self._n_groups} groups, got {len(perturbation_labels)}."
            )
        self._labels = ["baseline", *perturbation_labels]
        self._output_dir = output_dir
        self._save_diagnostics = save_diagnostics

        # Localize the latitude grid to this rank's spatial chunk so the band
        # and land/ocean masks line up with the (possibly spatially scattered)
        # recorded data and the localized area weights. Identity when there is
        # no spatial (model) parallelism.
        lats, _ = horizontal_coordinates.localize().meshgrid
        self._abs_lat = torch.abs(lats)  # [local_lat, local_lon]

        # Names of all predicted fields, captured from the first recorded batch.
        # The per-group time-mean is accumulated for every one of them so a
        # response map can be drawn for each.
        self._field_names: list[str] | None = None

        # Per-group running spatial sums (summed over members and time) and the
        # per-group count of (member, timestep) samples. Reduced across ranks at
        # summary time, so groups split unevenly across data-parallel ranks are
        # handled correctly.
        self._sums: list[dict[str, torch.Tensor] | None] = [None] * self._n_groups
        self._counts = torch.zeros(self._n_groups, dtype=torch.float64)
        self._ocean_fraction: torch.Tensor | None = None

    def _initialize_field_names(self, prediction: Mapping[str, torch.Tensor]) -> None:
        self._field_names = sorted(prediction.keys())
        missing = [
            name
            for name in self._config.scalar_diagnostic_names
            if name not in self._field_names
        ]
        if missing:
            raise KeyError(
                "perturbation-response scalar diagnostics require predicted "
                f"variables {missing} which are not among the predicted fields "
                f"{self._field_names}."
            )

    @torch.no_grad()
    def record_batch(self, data: PairedData) -> InferenceLogs:
        prediction = data.prediction
        if len(prediction) == 0:
            raise ValueError("data is empty")
        if self._field_names is None:
            self._initialize_field_names(prediction)
        assert self._field_names is not None
        example = prediction[self._field_names[0]]
        n_members = example.shape[0]
        if n_members != self._group_index.shape[0]:
            raise ValueError(
                f"recorded batch has {n_members} members but group encoding has "
                f"{self._group_index.shape[0]}."
            )
        n_time = example.shape[1]

        if self._ocean_fraction is None:
            self._ocean_fraction = self._extract_ocean_fraction(data)

        group_index = self._group_index.to(example.device)
        for g in range(self._n_groups):
            member_mask = group_index == g
            n_in_group = int(member_mask.sum().item())
            if n_in_group == 0:
                continue
            group_sums = self._sums[g]
            if group_sums is None:
                group_sums = {}
                self._sums[g] = group_sums
            for name in self._field_names:
                # [n_members, n_time, lat, lon] -> [lat, lon], summed over the
                # group's members and all timesteps in this window.
                contribution = prediction[name][member_mask].sum(dim=1).sum(dim=0)
                if name in group_sums:
                    group_sums[name] = group_sums[name] + contribution
                else:
                    group_sums[name] = contribution
            self._counts[g] += n_in_group * n_time
        return []

    def _extract_ocean_fraction(self, data: PairedData) -> torch.Tensor:
        name = self._config.ocean_fraction_name
        source: Mapping[str, torch.Tensor]
        if name in data.reference:
            source = data.reference
        elif name in data.prediction:
            source = data.prediction
        else:
            raise KeyError(
                f"ocean fraction variable {name!r} not found in recorded data; "
                "set ocean_fraction_name to a variable present in the forcing."
            )
        # Geography is identical across members and time; take the first.
        return source[name][0, 0]

    def record_initial_condition(
        self, initial_condition: PrognosticState
    ) -> InferenceLogs:
        return []

    def _group_time_mean(self) -> list[dict[str, torch.Tensor]]:
        if self._field_names is None:
            raise ValueError("no data has been recorded yet.")
        dist = Distributed.get_instance()
        # Counts must be on the compute device for the distributed reduce
        # (e.g. NCCL all_reduce rejects CPU tensors); the group sums already are.
        counts = dist.reduce_sum(self._counts.clone().to(get_device())).cpu()
        means: list[dict[str, torch.Tensor]] = []
        for g in range(self._n_groups):
            group_sums = self._sums[g]
            count = float(counts[g].item())
            if group_sums is None or count == 0:
                raise ValueError(f"no data recorded for group {self._labels[g]!r}.")
            group_mean = {}
            for name in self._field_names:
                total = dist.reduce_sum(group_sums[name].clone())
                group_mean[name] = total / count
            means.append(group_mean)
        return means

    def _response_map_names(self) -> list[str]:
        assert self._field_names is not None
        if self._config.response_map_variables is None:
            return self._field_names
        return [n for n in self._config.response_map_variables if n in self._field_names]

    def _map_caption(self, label: str, name: str) -> str:
        meta = self._variable_metadata.get(name)
        if meta is not None:
            long_name = meta.display_long_name(name)
            units = meta.display_units("unknown units")
        else:
            long_name, units = name, "unknown units"
        return (
            f"{long_name} {label} response (perturbed - baseline time-mean) "
            f"[{units}]"
        )

    def _regional_mean(self, field: torch.Tensor, weights: torch.Tensor) -> float:
        value = self._ops.regional_area_weighted_mean(
            field, regional_weights=weights.to(field.device)
        )
        return float(value.item())

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        # The warming ratio is ill-defined when the baseline-relative ocean
        # warming is ~zero (e.g. a checkpoint with no response yet). Return NaN
        # rather than crash the inline eval.
        if abs(denominator) < 1e-12:
            return float("nan")
        return numerator / denominator

    def get_summary(self) -> InferenceSummary:
        means = self._group_time_mean()
        assert self._ocean_fraction is not None
        assert self._field_names is not None
        ocean_fraction = self._ocean_fraction
        device = ocean_fraction.device
        abs_lat = self._abs_lat.to(device)
        cfg = self._config

        ocean_mask = (ocean_fraction > cfg.ocean_fraction_cutoff).to(torch.float64)
        land_mask = 1.0 - ocean_mask

        logs: dict[str, float | Image] = {}
        baseline = means[0]
        for g in range(1, self._n_groups):
            # Keyed by the perturbation label only; the inline-inference callback
            # adds the task-name prefix (e.g. "perturbation_response/").
            prefix = self._labels[g]
            response = {
                name: (means[g][name] - baseline[name]).to(torch.float64)
                for name in self._field_names
            }

            # 2D response map (perturbed - baseline time-mean) for each field.
            for name in self._response_map_names():
                logs[f"{prefix}/response_map/{name}"] = plot_paneled_data(
                    [[response[name].cpu().numpy()]],
                    diverging=True,
                    caption=self._map_caption(prefix, name),
                )

            # Near-surface land/ocean warming ratio by latitude band.
            near = response[cfg.near_surface_temperature_name]
            for band in cfg.latitude_bands:
                band_mask = ((abs_lat >= band.lat_min) & (abs_lat < band.lat_max)).to(
                    torch.float64
                )
                land_warming = self._regional_mean(near, land_mask * band_mask)
                ocean_warming = self._regional_mean(near, ocean_mask * band_mask)
                logs[f"{prefix}/land_warming/{band.name}"] = land_warming
                logs[f"{prefix}/ocean_warming/{band.name}"] = ocean_warming
                logs[f"{prefix}/land_ocean_warming_ratio/{band.name}"] = self._ratio(
                    land_warming, ocean_warming
                )

            # Free-troposphere/surface warming ratio over tropical ocean.
            tropical_ocean = ocean_mask * (abs_lat < cfg.tropical_lat_max).to(
                torch.float64
            )
            surf_name = cfg.column_temperature_names[cfg.vertical_surface_index]
            upper_name = cfg.column_temperature_names[cfg.vertical_upper_index]
            surf_warming = self._regional_mean(response[surf_name], tropical_ocean)
            upper_warming = self._regional_mean(response[upper_name], tropical_ocean)
            logs[f"{prefix}/tropical_ocean_surface_warming"] = surf_warming
            logs[f"{prefix}/tropical_ocean_upper_warming"] = upper_warming
            logs[f"{prefix}/vertical_warming_ratio_tropical_ocean"] = self._ratio(
                upper_warming, surf_warming
            )

            # Global-mean column warming profile, level by level.
            for name in cfg.column_temperature_names:
                global_warming = float(
                    self._ops.area_weighted_mean(response[name]).item()
                )
                logs[f"{prefix}/column_warming/{name}"] = global_warming

        return InferenceSummary(logs=logs, loss=None)

    def flush_diagnostics(self, subdir: str | None) -> None:
        if not self._save_diagnostics:
            return
        assert self._output_dir is not None
        assert self._field_names is not None
        # _group_time_mean is a collective (all-reduce) and must run on every
        # rank; only the root rank then writes the (rank-identical) result, to
        # avoid a multi-writer race on the shared path.
        means = self._group_time_mean()
        if not Distributed.get_instance().is_root():
            return
        baseline = means[0]
        data_vars = {}
        for g in range(1, self._n_groups):
            label = self._labels[g]
            for name in self._field_names:
                response = (means[g][name] - baseline[name]).cpu().numpy()
                data_vars[f"{label}__{name}"] = (("lat", "lon"), response)
        dataset = xr.Dataset(data_vars)
        directory = self._output_dir
        if subdir is not None:
            directory = os.path.join(directory, subdir)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "perturbation_response_diagnostics.nc")
        logging.info(f"Writing perturbation-response diagnostics to {path}")
        dataset.to_netcdf(path)
