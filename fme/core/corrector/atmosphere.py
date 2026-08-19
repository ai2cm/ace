import dataclasses
import datetime
from collections.abc import Callable
from typing import Literal, Protocol

import torch

import fme
from fme.core.atmosphere_data import (
    AtmosphereData,
    HasAtmosphereVerticalIntegral,
    compute_layer_thickness,
)
from fme.core.constants import GRAVITY, SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME
from fme.core.corrector.output import CorrectorOutput
from fme.core.corrector.registry import (
    Correction,
    CorrectionSequence,
    CorrectorConfigABC,
)
from fme.core.corrector.state import CorrectorState
from fme.core.corrector.utils import ForcePositive
from fme.core.dataset_info import DatasetInfo
from fme.core.gridded_ops import GriddedOperations
from fme.core.registry.corrector import CorrectorSelector
from fme.core.typing_ import TensorMapping


class AreaWeightedMean(Protocol):
    def __call__(
        self, data: torch.Tensor, keepdim: bool = False, name: str | None = None
    ) -> torch.Tensor: ...


@dataclasses.dataclass
class EnergyBudgetConfig:
    """Configuration for total energy budget correction.

    Parameters:
        method: Method to use for total energy budget correction. The available option
            is "constant_temperature", which enforces conservation of total energy by
            imposing a vertically and horizontally uniform air temperature correction.
        constant_unaccounted_heating: Column-integrated heating in W/m**2 to be added
            to the energy flux into the atmosphere when conserving total energy.
            This can be useful for correcting errors in energy budget in target data.
            The same additional heating is imposed at all time steps and grid cells.
    """

    method: Literal["constant_temperature"]
    constant_unaccounted_heating: float = 0.0


@dataclasses.dataclass
class ConserveDryAir:
    """Correction that pins the global-mean dry-air mass to its IC value.

    Bundles the operators needed to apply the dry-air conservation step. The
    reference dry-air mass is seeded from ``input_data`` the first time it runs
    and carried across steps via ``corrector_state``.
    """

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasAtmosphereVerticalIntegral | None
    precision: torch.dtype

    def __call__(
        self,
        input_data: TensorMapping,
        forcing_data: TensorMapping,
        accumulated_output: CorrectorOutput,
    ) -> CorrectorOutput:
        """Apply dry-air conservation as a delta on the accumulated output."""
        if self.vertical_coordinate is None:
            raise ValueError(
                "conserve_dry_air is set to True, but no vertical coordinate is "
                "available."
            )
        corrector_state = _seed_global_dry_air_mass(
            input_data=input_data,
            corrector_state=accumulated_output.corrector_state,
            area_weighted_mean=self.area_weighted_mean,
            vertical_coordinate=self.vertical_coordinate,
            precision=self.precision,
        )
        assert corrector_state.global_dry_air_mass is not None
        gen = AtmosphereData.for_correction(
            accumulated_output, self.vertical_coordinate
        )
        _adjust_gen_dry_air_to_target(
            gen,
            target_global_dry_air=corrector_state.global_dry_air_mass,
            area_weighted_mean=self.area_weighted_mean,
            precision=self.precision,
        )
        return gen.result().with_state(corrector_state)


@dataclasses.dataclass
class ZeroGlobalMeanMoistureAdvection:
    """Correction that forces zero global-mean moisture advection."""

    area_weighted_mean: AreaWeightedMean

    def __call__(
        self,
        input_data: TensorMapping,
        forcing_data: TensorMapping,
        accumulated_output: CorrectorOutput,
    ) -> CorrectorOutput:
        """Apply zero global-mean moisture advection as a delta."""
        gen = AtmosphereData.for_correction(accumulated_output)
        _force_zero_global_mean_moisture_advection(
            gen=gen,
            area_weighted_mean=self.area_weighted_mean,
        )
        return gen.result()


@dataclasses.dataclass
class MoistureBudgetCorrection:
    """Correction that closes the moisture budget via the configured terms.

    When ``clip_frozen_precipitation`` is True, after closing the moisture budget
    the frozen precipitation rate (``total_frozen_precipitation_rate``) is clipped
    to the -- possibly corrected -- total precipitation rate when frozen
    precipitation is predicted, since frozen precipitation is a component of total
    precipitation and cannot exceed it.
    """

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasAtmosphereVerticalIntegral | None
    timestep_seconds: float
    terms_to_modify: Literal[
        "precipitation",
        "evaporation",
        "advection_and_precipitation",
        "advection_and_evaporation",
    ]
    clip_frozen_precipitation: bool = False

    def __call__(
        self,
        input_data: TensorMapping,
        forcing_data: TensorMapping,
        accumulated_output: CorrectorOutput,
    ) -> CorrectorOutput:
        """Apply moisture budget correction as deltas."""
        if self.vertical_coordinate is None:
            raise ValueError(
                "Moisture budget correction is turned on, but no vertical "
                "coordinate is available."
            )
        gen = AtmosphereData.for_correction(
            accumulated_output, self.vertical_coordinate
        )
        _force_conserve_moisture(
            input_data=input_data,
            gen=gen,
            area_weighted_mean=self.area_weighted_mean,
            timestep_seconds=self.timestep_seconds,
            terms_to_modify=self.terms_to_modify,
        )
        if self.clip_frozen_precipitation:
            _clip_frozen_precipitation(gen)
        return gen.result()


@dataclasses.dataclass
class TotalEnergyBudgetCorrection:
    """Correction that conserves an idealized total energy budget."""

    area_weighted_mean: AreaWeightedMean
    vertical_coordinate: HasAtmosphereVerticalIntegral | None
    timestep_seconds: float
    method: Literal["constant_temperature"]
    unaccounted_heating: float

    def __call__(
        self,
        input_data: TensorMapping,
        forcing_data: TensorMapping,
        accumulated_output: CorrectorOutput,
    ) -> CorrectorOutput:
        """Apply total energy budget correction as deltas."""
        if self.vertical_coordinate is None:
            raise ValueError(
                "Energy budget correction is turned on, but no vertical coordinate"
                " is available."
            )
        gen = AtmosphereData.for_correction(
            accumulated_output, self.vertical_coordinate
        )
        _force_conserve_total_energy(
            input_data=input_data,
            gen=gen,
            forcing_data=forcing_data,
            area_weighted_mean=self.area_weighted_mean,
            timestep_seconds=self.timestep_seconds,
            method=self.method,
            unaccounted_heating=self.unaccounted_heating,
        )
        return gen.result()


@CorrectorSelector.register("atmosphere_corrector")
@dataclasses.dataclass
class AtmosphereCorrectorConfig(CorrectorConfigABC):
    r"""
    Configuration for the post-step state corrector.

    ``conserve_dry_air`` enforces the constraint that:

    .. math::

        global\_dry\_air = global\_mean(ps -
        sum_k((ak\_diff + bk\_diff \* ps) \* wat_k))

    in the generated data is equal to its value at the initial condition. The
    reference is captured the first time the corrector runs (using the
    ``input_data`` it sees, which is the IC during the first step of inference
    or training rollout) and threaded across step calls via ``CorrectorState``.
    The correction is applied by adding a globally-constant offset to the
    surface pressure in each column. As per-mass values such as mixing ratios
    of water are unchanged, this can cause changes in total water or energy.
    Note all global means here are area-weighted.

    ``zero_global_mean_moisture_advection`` enforces the constraint that:

    .. math::

        global\_mean(tendency\_of\_total\_water\_path\_due\_to\_advection) = 0

    in the generated data. This is done by adding a globally-constant correction
    to the moisture advection tendency in each column.

    ``moisture_budget_correction`` enforces closure of the moisture budget equation:

    .. math::

        tendency\_of\_total\_water\_path = (evaporation\_rate - precipitation\_rate
        \\\\ + tendency\_of\_total\_water\_path\_due\_to\_advection)

    in the generated data, where ``tendency_of_total_water_path`` is the difference
    between the total water path at the current timestep and the previous
    timestep divided by the time difference. This is done by modifying the
    precipitation, evaporation, and/or moisture advection tendency fields as
    described in the ``moisture_budget_correction`` attribute. When
    advection tendency is modified, this budget equation is enforced in each
    column, while when only precipitation or evaporation are modified, only
    the global mean of the budget equation is enforced.

    When enforcing moisture budget closure, we assume the global mean moisture
    advection is zero. Therefore ``zero_global_mean_moisture_advection`` must be
    True if using a ``moisture_budget_correction`` option other than ``None``.

    Parameters:
        conserve_dry_air: If True, pin the global-mean dry-air mass of the
            generated data to its value at the initial condition. The reference
            is captured the first time the corrector runs (from ``input_data``,
            which is the IC during the first step of an inference or training
            rollout) and threaded across step calls via ``CorrectorState``.
            The correction is applied by adding a globally-constant offset to
            the dry-air pressure of each column and solving for the consistent
            surface pressure. As per-mass values such as mixing ratios of water
            are unchanged, this can cause changes in total water or energy.
        zero_global_mean_moisture_advection: If True, force the generated data to
            have zero global mean moisture advection by subtracting a constant
            offset from the moisture advection tendency of each column.
        moisture_budget_correction: If not "None", force the generated data to
            conserve global or column-local moisture by modifying budget fields.
            Options are:

            - ``precipitation``: multiply precipitation by a scale factor
              to close the global moisture budget.
            - ``evaporation``: multiply evaporation by a scale factor
              to close the global moisture budget.
            - ``advection_and_precipitation``: after applying the "precipitation"
              global-mean correction above, recompute the column-integrated
              advective tendency as the budget residual,
              ensuring column budget closure.
            - ``advection_and_evaporation``: after applying the "evaporation"
              global-mean correction above, recompute the column-integrated
              advective tendency as the budget residual,
              ensuring column budget closure.

        force_positive_names: Names of fields that should be forced to be greater
            than or equal to zero. This is useful for fields like precipitation.
        total_energy_budget_correction: If not None, force the generated data to
            conserve an idealized version of total energy using the provided
            configuration.
        keep_gradient_through_clamps: If True, apply the ``force_positive_names``
            clamp with a straight-through estimator: the forward value is still
            clamped to be non-negative, but gradient flows as if the clamp had
            not happened, so clamped-negative cells still get a learning signal.
        clip_frozen_precipitation: If True, and ``moisture_budget_correction`` is
            enabled and the frozen precipitation rate
            (``total_frozen_precipitation_rate``) is predicted, clip it to be less
            than or equal to the -- possibly corrected -- total precipitation rate
            (``PRATEsfc``) in each grid cell, since frozen precipitation is a
            component of total precipitation. The clip runs as part of the
            moisture budget correction, before any ``total_energy_budget_correction``,
            since frozen precipitation contributes to the surface energy flux via
            the latent heat of freezing. Defaults to False so that previously
            trained checkpoints, which did not apply this clip, are unaffected.
    """

    conserve_dry_air: bool = False
    zero_global_mean_moisture_advection: bool = False
    moisture_budget_correction: (
        Literal[
            "precipitation",
            "evaporation",
            "advection_and_precipitation",
            "advection_and_evaporation",
        ]
        | None
    ) = None
    force_positive_names: list[str] = dataclasses.field(default_factory=list)
    total_energy_budget_correction: EnergyBudgetConfig | None = None
    keep_gradient_through_clamps: bool = False
    clip_frozen_precipitation: bool = False

    def _get_corrector(
        self,
        dataset_info: DatasetInfo,
    ) -> "AtmosphereCorrector":
        return self._build(
            dataset_info.gridded_operations,
            dataset_info.atmosphere_vertical_coordinate,
            dataset_info.timestep,
        )

    def _build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HasAtmosphereVerticalIntegral | None,
        timestep: datetime.timedelta,
    ) -> "AtmosphereCorrector":
        area_weighted_mean = gridded_operations.area_weighted_mean
        timestep_seconds = timestep.total_seconds()
        corrections: list[Correction] = []

        # Fields that will be diagnosed (replaced wholesale) by later
        # corrections.  A ForcePositive delta on such a field would be
        # erased by the diagnosis, so a collision is rejected at build time.
        from fme.core.atmosphere_data import ATMOSPHERE_FIELD_NAME_PREFIXES

        diagnosed_prefixes: dict[str, str] = {}  # prefix -> source config key
        if self.moisture_budget_correction is not None:
            if self.moisture_budget_correction.endswith("precipitation"):
                for p in ATMOSPHERE_FIELD_NAME_PREFIXES["precipitation_rate"]:
                    diagnosed_prefixes[p] = "moisture_budget_correction"
            elif self.moisture_budget_correction.endswith("evaporation"):
                for p in ATMOSPHERE_FIELD_NAME_PREFIXES["latent_heat_flux"]:
                    diagnosed_prefixes[p] = "moisture_budget_correction"
            if self.moisture_budget_correction.startswith("advection"):
                for p in ATMOSPHERE_FIELD_NAME_PREFIXES[
                    "tendency_of_total_water_path_due_to_advection"
                ]:
                    diagnosed_prefixes[p] = "moisture_budget_correction"
        if self.clip_frozen_precipitation:
            for p in ATMOSPHERE_FIELD_NAME_PREFIXES.get(
                "frozen_precipitation_rate", []
            ):
                diagnosed_prefixes[p] = "clip_frozen_precipitation"
        advection_recomputed = (
            self.moisture_budget_correction is not None
            and self.moisture_budget_correction.startswith("advection")
        )
        if self.zero_global_mean_moisture_advection and not advection_recomputed:
            for p in ATMOSPHERE_FIELD_NAME_PREFIXES[
                "tendency_of_total_water_path_due_to_advection"
            ]:
                diagnosed_prefixes[p] = "zero_global_mean_moisture_advection"

        collision = set(self.force_positive_names) & set(diagnosed_prefixes)
        if collision:
            sources = sorted({diagnosed_prefixes[n] for n in collision})
            raise ValueError(
                f"force_positive_names {sorted(collision)} overlap with fields "
                f"diagnosed by {', '.join(sources)}: remove them from "
                f"force_positive_names or disable the diagnosing correction"
            )
        if len(self.force_positive_names) > 0:
            corrections.append(
                ForcePositive(
                    self.force_positive_names,
                    keep_gradient=self.keep_gradient_through_clamps,
                )
            )
        if self.conserve_dry_air:
            if fme.get_device() == torch.device("mps", 0):
                precision = torch.float32
            else:
                precision = torch.float64
            corrections.append(
                ConserveDryAir(area_weighted_mean, vertical_coordinate, precision)
            )
        if self.zero_global_mean_moisture_advection and not advection_recomputed:
            # Skip when the moisture budget correction recomputes advection
            # from scratch (a diagnosis), since that supersedes the
            # zero-global-mean correction.
            corrections.append(ZeroGlobalMeanMoistureAdvection(area_weighted_mean))
        if self.moisture_budget_correction is not None:
            corrections.append(
                MoistureBudgetCorrection(
                    area_weighted_mean,
                    vertical_coordinate,
                    timestep_seconds,
                    self.moisture_budget_correction,
                    clip_frozen_precipitation=self.clip_frozen_precipitation,
                )
            )
        if self.total_energy_budget_correction is not None:
            corrections.append(
                TotalEnergyBudgetCorrection(
                    area_weighted_mean,
                    vertical_coordinate,
                    timestep_seconds,
                    self.total_energy_budget_correction.method,
                    self.total_energy_budget_correction.constant_unaccounted_heating,
                )
            )
        return AtmosphereCorrector(corrections)


class AtmosphereCorrector(CorrectionSequence):
    pass


def _seed_global_dry_air_mass(
    input_data: TensorMapping,
    corrector_state: CorrectorState | None,
    area_weighted_mean: AreaWeightedMean,
    vertical_coordinate: HasAtmosphereVerticalIntegral,
    precision: torch.dtype,
) -> CorrectorState:
    """Return a CorrectorState whose ``global_dry_air_mass`` field is set.

    If ``corrector_state`` already carries a non-None ``global_dry_air_mass``
    (e.g. seeded by a prior step's call), it is returned unchanged. Otherwise
    the reference is computed from ``input_data`` — during the first step of
    a rollout this is the initial condition.
    """
    if corrector_state is not None and corrector_state.global_dry_air_mass is not None:
        return corrector_state
    ic = AtmosphereData(input_data, vertical_coordinate)
    if ic.surface_pressure is None:
        raise ValueError("surface_pressure is required to pin the global dry-air mass")
    target = area_weighted_mean(
        ic.surface_pressure_due_to_dry_air.to(precision),
        keepdim=True,
    )
    return CorrectorState(global_dry_air_mass=target)


def _adjust_gen_dry_air_to_target(
    gen: AtmosphereData,
    target_global_dry_air: torch.Tensor,
    area_weighted_mean: AreaWeightedMean,
    precision: torch.dtype,
) -> None:
    """Adjust *gen*'s surface pressure so its global mean dry-air mass
    matches ``target_global_dry_air``.

    The correction is recorded through ``gen.correct_surface_pressure``;
    the caller is responsible for calling ``gen.result()`` to produce the
    ``CorrectorOutput``.
    """
    if gen.surface_pressure is None:
        raise ValueError("surface_pressure is required to force dry air conservation")
    vertical_coordinate = gen._vertical_coordinate
    assert vertical_coordinate is not None
    gen_dry_air = gen.surface_pressure_due_to_dry_air
    global_gen_dry_air = area_weighted_mean(gen_dry_air.to(precision), keepdim=True)
    error = global_gen_dry_air - target_global_dry_air.to(precision)
    new_gen_dry_air = gen_dry_air.to(precision) - error
    try:
        wat = gen.specific_total_water.to(precision)
    except KeyError:
        raise ValueError("specific_total_water is required for conservation")
    ak_diff = vertical_coordinate.get_ak().diff().to(precision)
    bk_diff = vertical_coordinate.get_bk().diff().to(precision)
    new_pressure = (new_gen_dry_air + (ak_diff * wat).sum(-1)) / (
        1 - (bk_diff * wat).sum(-1)
    )
    gen.correct_surface_pressure(new_pressure.to(dtype=gen.surface_pressure.dtype))


def _force_zero_global_mean_moisture_advection(
    gen: AtmosphereData,
    area_weighted_mean: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    """Update *gen* so advection conserves moisture.

    Does so by adding a constant offset to the moisture advective tendency.
    The correction is recorded through ``gen.correct_*``; the caller is
    responsible for calling ``gen.result()``.
    """
    mean_moisture_advection = area_weighted_mean(
        gen.tendency_of_total_water_path_due_to_advection,
    )
    gen.diagnose_tendency_of_total_water_path_due_to_advection(
        gen.tendency_of_total_water_path_due_to_advection
        - mean_moisture_advection[..., None, None]
    )


def _clip_frozen_precipitation(gen: AtmosphereData) -> None:
    """Clip the frozen precipitation rate to be at most the total precipitation
    rate.  A no-op when the frozen precipitation rate is not among *gen*'s
    fields.  The correction is recorded through ``gen.correct_*``.
    """
    if "total_frozen_precipitation_rate" not in gen.data:
        return
    gen.diagnose_frozen_precipitation_rate(
        torch.minimum(gen.frozen_precipitation_rate, gen.precipitation_rate)
    )


def _force_conserve_moisture(
    input_data: TensorMapping,
    gen: AtmosphereData,
    area_weighted_mean: AreaWeightedMean,
    timestep_seconds: float,
    terms_to_modify: Literal[
        "precipitation",
        "evaporation",
        "advection_and_precipitation",
        "advection_and_evaporation",
    ],
) -> None:
    """Update *gen* to conserve moisture.

    Corrections are recorded through ``gen.correct_*``; the caller is
    responsible for calling ``gen.result()``.  *input_data* is read-only.
    """
    input = AtmosphereData(input_data, gen._vertical_coordinate)

    gen_total_water_path = gen.total_water_path
    twp_total_tendency = (
        gen_total_water_path - input.total_water_path
    ) / timestep_seconds
    twp_tendency_global_mean = area_weighted_mean(twp_total_tendency, keepdim=True)
    evaporation_global_mean = area_weighted_mean(gen.evaporation_rate, keepdim=True)
    precipitation_global_mean = area_weighted_mean(gen.precipitation_rate, keepdim=True)
    if terms_to_modify.endswith("precipitation"):
        new_precipitation_global_mean = (
            evaporation_global_mean - twp_tendency_global_mean
        )
        gen.diagnose_precipitation_rate(
            gen.precipitation_rate
            * (new_precipitation_global_mean / precipitation_global_mean)
        )
    elif terms_to_modify.endswith("evaporation"):
        new_evaporation_global_mean = (
            twp_tendency_global_mean + precipitation_global_mean
        )
        gen.diagnose_evaporation_rate(
            gen.evaporation_rate
            * (new_evaporation_global_mean / evaporation_global_mean)
        )
    if terms_to_modify.startswith("advection"):
        new_advection = twp_total_tendency - (
            gen.evaporation_rate - gen.precipitation_rate
        )
        gen.diagnose_tendency_of_total_water_path_due_to_advection(new_advection)


def _force_conserve_total_energy(
    input_data: TensorMapping,
    gen: AtmosphereData,
    forcing_data: TensorMapping,
    area_weighted_mean: AreaWeightedMean,
    timestep_seconds: float,
    method: Literal["constant_temperature"] = "constant_temperature",
    unaccounted_heating: float = 0.0,
) -> None:
    """Apply a correction to *gen* to conserve total energy.

    Corrections are recorded through ``gen.correct_all_levels``; the caller
    is responsible for calling ``gen.result()``.
    """
    if method != "constant_temperature":
        raise NotImplementedError(
            f"Method {method} not implemented for total energy conservation"
        )
    vertical_coordinate = gen._vertical_coordinate
    assert vertical_coordinate is not None
    input = AtmosphereData(input_data, vertical_coordinate)
    forcing = AtmosphereData(forcing_data)
    required_forcing = {
        "DSWRFtoa": forcing.toa_down_sw_radiative_flux,
        "HGTsfc": forcing.surface_height,
    }
    # Temporarily inject forcing fields so the energy computation can read
    # them; they are NOT correction targets.
    for name, tensor in required_forcing.items():
        gen._data[name] = tensor

    gen_energy_path = gen.total_energy_ace2_path
    input_energy_path = input.total_energy_ace2_path
    predicted_energy_flux_into_atmosphere = gen.net_energy_flux_into_atmosphere

    gen_energy_path_global_mean = area_weighted_mean(gen_energy_path, keepdim=True)
    input_energy_path_global_mean = area_weighted_mean(input_energy_path, keepdim=True)
    energy_flux_global_mean = area_weighted_mean(
        predicted_energy_flux_into_atmosphere, keepdim=True
    )

    desired_energy_path_global_mean = (
        input_energy_path_global_mean
        + (energy_flux_global_mean + unaccounted_heating) * timestep_seconds
    )

    energy_correction = desired_energy_path_global_mean - gen_energy_path_global_mean
    energy_to_temperature_factor = _energy_correction_factor(gen, vertical_coordinate)
    energy_to_temp_factor_gm = area_weighted_mean(energy_to_temperature_factor, True)
    temperature_correction = energy_correction / energy_to_temp_factor_gm

    # temperature_correction is (batch, 1, 1); unsqueeze to broadcast
    # against the stacked (batch, lat, lon, n_levels) air temperature.
    gen.correct_all_levels(
        "air_temperature",
        gen.air_temperature + temperature_correction.unsqueeze(-1),
    )


def _energy_correction_factor(
    gen: AtmosphereData, vertical_coordinate: HasAtmosphereVerticalIntegral
) -> torch.Tensor:
    """
    Compute the factor to get a vertically-uniform temperature correction that
    will lead to a desired change in the globally-averaged total energy.

    See https://www.overleaf.com/read/dqjjcvzxnfvn#d525aa.
    """
    interface_pressure = vertical_coordinate.interface_pressure(gen.surface_pressure)
    q_times_dlogp = (
        compute_layer_thickness(
            interface_pressure, gen.air_temperature, gen.specific_total_water
        )
        * GRAVITY
        / gen.air_temperature
    )
    cumulative = torch.cumsum(q_times_dlogp.flip(dims=(-1,)), dim=-1).flip(dims=(-1,))
    total_integrand = (
        SPECIFIC_HEAT_OF_DRY_AIR_CONST_VOLUME - 0.5 * q_times_dlogp + cumulative
    )
    correction_factor = vertical_coordinate.vertical_integral(
        total_integrand, gen.surface_pressure
    )
    return correction_factor
