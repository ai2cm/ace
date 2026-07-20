"""Saturation-normalized (RH-like) humidity representation.

A paired forward/inverse transform, sibling to ``GlobalMeanRemoval``, that
expresses humidity fields as ``q / q_sat`` — a relative-humidity-like quantity
that is dimensionless and O(1) at every level, latitude, and climate, and so
stays in-sample as the climate warms.  See the saturation-normalized-humidity
investigation for the (climate-invariance) rationale.

Saturation specific humidity ``q_sat`` is computed per model level from the raw
(pre-global-mean-removal) input temperature and surface pressure together with
the hybrid sigma-pressure vertical coordinate, using the Bolton (1980) helper in
``fme.core.humidity``.

The RH-like representation is carried through the network under a *derived*
channel name (``relative_humidity_<level>``) rather than the source humidity
name.  These derived names are internal: they appear in the input/output packers
and in the network and loss normalizers (with identity statistics, since
``q / q_sat`` is already O(1) — 0 means dry, 1 means saturated — and needs no
further normalization), but they are never exposed in the public ``in_names`` /
``out_names`` / prognostic-name lists.  ``forward_transform`` rewrites the source
humidity into its derived name in the network input; ``inverse_transform``
rewrites the derived prediction back to the source humidity, so the step's public
output stays in physical ``q`` space.

Using a distinct derived name (rather than overloading the source field's name
and statistics) keeps the input and prediction sides independent: a field can be
RH on the input, on the output, both, or neither.

Each configured rule (:class:`SaturationNormalizationConfig`) carries three
independently settable knobs:

- ``names``: which humidity fields to act on (fnmatch wildcards, resolved by
  ``fme.core.match_names.match_names``, which errors on zero matches).
- ``prediction``: whether the network predicts the field in ``q / q_sat`` space.
  With residual prediction the residual is then taken in the transformed
  variable, since the residual add happens in normalized space under the derived
  name.
- ``input``: ``none`` | ``replace`` | ``append`` — whether the network input
  carries the RH-like channel, and if so whether it replaces the raw ``q`` input
  channel or rides alongside it as an additional channel.
"""

import dataclasses
from collections.abc import Iterable
from typing import Literal

import torch

from fme.core.atmosphere_data import (
    ATMOSPHERE_FIELD_NAME_PREFIXES,
    HasAtmosphereVerticalIntegral,
)
from fme.core.humidity import saturation_specific_humidity
from fme.core.match_names import match_names
from fme.core.normalizer import StandardNormalizer
from fme.core.stacker import Stacker
from fme.core.typing_ import TensorDict, TensorMapping

_TEMPERATURE_PREFIXES = ATMOSPHERE_FIELD_NAME_PREFIXES["air_temperature"]
_SURFACE_PRESSURE_NAMES = ATMOSPHERE_FIELD_NAME_PREFIXES["surface_pressure"]

_RELATIVE_HUMIDITY_PREFIX = "relative_humidity_"


def _relative_humidity_name(level: int) -> str:
    return f"{_RELATIVE_HUMIDITY_PREFIX}{level}"


def _parse_level(name: str) -> int | None:
    match = Stacker.LEVEL_PATTERN.search(name)
    return int(match.group(1)) if match is not None else None


def _resolve_temperature_name(level: int, in_names: list[str]) -> str:
    for prefix in _TEMPERATURE_PREFIXES:
        candidate = f"{prefix}{level}"
        if candidate in in_names:
            return candidate
    raise ValueError(
        f"saturation_normalization needs an input temperature at level {level} "
        f"(one of {[f'{p}{level}' for p in _TEMPERATURE_PREFIXES]}), but none is "
        f"in in_names."
    )


def _resolve_surface_pressure_name(in_names: list[str]) -> str:
    for name in _SURFACE_PRESSURE_NAMES:
        if name in in_names:
            return name
    raise ValueError(
        f"saturation_normalization needs a surface pressure input (one of "
        f"{_SURFACE_PRESSURE_NAMES}), but none is in in_names: {in_names}."
    )


@dataclasses.dataclass(frozen=True)
class _SaturationField:
    """A resolved humidity field and how it is carried in network space."""

    name: str  # source humidity field (public)
    rh_name: str  # derived relative-humidity channel (internal)
    level: int
    temperature_name: str
    rh_in: bool  # the network input for this field is q / q_sat (replaces q)
    rh_extra: bool  # an extra q / q_sat input channel rides alongside q
    rh_out: bool  # the network predicts this field in q / q_sat space


@dataclasses.dataclass
class SaturationNormalizationState:
    """Opaque token from ``forward_transform`` consumed by ``inverse_transform``
    on the same instance.

    Carries the per-field ``q_sat`` (computed from the input-step temperature
    and pressure) used to convert a predicted RH field back to ``q``.
    """

    qsat: dict[str, torch.Tensor]


@dataclasses.dataclass
class SaturationNormalizationConfig:
    """One saturation-normalization rule.

    Parameters:
        names: Humidity field names or fnmatch patterns to act on (e.g.
            ``["specific_total_water_*"]``).  Each pattern must match at least
            one input/output field.
        prediction: Whether the network predicts the matched fields in
            ``q / q_sat`` space.
        input: Whether the network input carries the RH-like channel:
            ``none`` (raw ``q`` only), ``replace`` (feed ``q / q_sat`` in place
            of ``q``), or ``append`` (an extra ``q / q_sat`` channel alongside
            ``q``).
    """

    names: list[str]
    prediction: bool = False
    input: Literal["none", "replace", "append"] = "none"

    def validate_names(self, in_names: list[str], out_names: list[str]) -> None:
        resolve_entry_fields(self, in_names, out_names)


def resolve_entry_fields(
    config: SaturationNormalizationConfig,
    in_names: list[str],
    out_names: list[str],
) -> list[_SaturationField]:
    """Resolve one config entry to per-field descriptors, validating coherence.

    Raises ``ValueError`` for: patterns matching nothing, fields without a level
    suffix, input/prediction modes referencing a side the field is not on, and
    a derived ``relative_humidity_<level>`` name colliding with a real field.
    """
    candidate_names = list(dict.fromkeys(list(in_names) + list(out_names)))
    matched = match_names(config.names, candidate_names)
    rh_in = config.input == "replace"
    rh_extra = config.input == "append"
    rh_out = config.prediction
    fields: list[_SaturationField] = []
    for name in matched:
        level = _parse_level(name)
        if level is None:
            raise ValueError(
                f"saturation_normalization field '{name}' must end in '_<level>' "
                "so its level and temperature can be resolved."
            )
        in_input = name in in_names
        in_output = name in out_names
        if config.input != "none" and not in_input:
            raise ValueError(
                f"saturation_normalization input='{config.input}' for '{name}', "
                f"but it is not an input field: {in_names}."
            )
        if rh_out and not in_output:
            raise ValueError(
                f"saturation_normalization prediction=True for '{name}', but it "
                f"is not an output field: {out_names}."
            )
        rh_name = _relative_humidity_name(level)
        if rh_name in candidate_names:
            raise ValueError(
                f"saturation_normalization derived channel '{rh_name}' for "
                f"'{name}' collides with an existing field name."
            )
        temperature_name = _resolve_temperature_name(level, in_names)
        fields.append(
            _SaturationField(
                name=name,
                rh_name=rh_name,
                level=level,
                temperature_name=temperature_name,
                rh_in=rh_in,
                rh_extra=rh_extra,
                rh_out=rh_out,
            )
        )
    return fields


def resolve_all_fields(
    configs: Iterable[SaturationNormalizationConfig],
    in_names: list[str],
    out_names: list[str],
) -> list[_SaturationField]:
    """Resolve every config entry and reject a source field or derived name
    appearing in more than one entry.
    """
    fields: list[_SaturationField] = []
    seen_source: set[str] = set()
    seen_rh: set[str] = set()
    for config in configs:
        for field in resolve_entry_fields(config, in_names, out_names):
            if field.name in seen_source:
                raise ValueError(
                    f"saturation_normalization field '{field.name}' appears in "
                    "more than one entry."
                )
            if field.rh_name in seen_rh:
                raise ValueError(
                    f"saturation_normalization derived channel '{field.rh_name}' "
                    "is produced by more than one field."
                )
            seen_source.add(field.name)
            seen_rh.add(field.rh_name)
            fields.append(field)
    return fields


def derived_normalization_names(fields: Iterable[_SaturationField]) -> list[str]:
    """Derived RH channel names that the normalizer/scaler must carry (identity).

    A field contributes its derived name if it is RH on the input (replace or
    append) or on the output (prediction).
    """
    names: list[str] = []
    for field in fields:
        if field.rh_in or field.rh_extra or field.rh_out:
            names.append(field.rh_name)
    return names


def add_identity_names(
    normalizer: StandardNormalizer, names: Iterable[str]
) -> StandardNormalizer:
    """Return a copy of ``normalizer`` with ``names`` added at mean 0, std 1.

    Derived RH channels are O(1) and are fed to / produced by the network
    without further normalization, so the (network and loss) normalizers carry
    them with identity statistics.
    """
    names = list(names)
    if not names:
        return normalizer
    means = dict(normalizer.means)
    stds = dict(normalizer.stds)
    for name in names:
        means[name] = torch.zeros(())
        stds[name] = torch.ones(())
    return StandardNormalizer(
        means=means,
        stds=stds,
        fill_nans_on_normalize=normalizer.fill_nans_on_normalize,
        fill_nans_on_denormalize=normalizer.fill_nans_on_denormalize,
    )


class SaturationNormalization:
    """Forward/inverse ``q <-> q / q_sat`` transform over a set of fields."""

    def __init__(
        self,
        fields: list[_SaturationField],
        surface_pressure_name: str,
        vertical_coordinate: HasAtmosphereVerticalIntegral,
    ):
        self._fields = fields
        self._surface_pressure_name = surface_pressure_name
        self._vertical_coordinate = vertical_coordinate

    @property
    def derived_names(self) -> list[str]:
        return derived_normalization_names(self._fields)

    def input_packer_names(self, in_names: list[str]) -> list[str]:
        """The network input channel names: ``replace`` fields swap ``q`` for the
        derived RH name, ``append`` fields add the derived RH name after the
        inputs.
        """
        replace_map = {f.name: f.rh_name for f in self._fields if f.rh_in}
        names = [replace_map.get(name, name) for name in in_names]
        names += [f.rh_name for f in self._fields if f.rh_extra]
        return names

    def output_packer_names(self, out_names: list[str]) -> list[str]:
        """The network output channel names: ``prediction`` fields swap ``q`` for
        the derived RH name.
        """
        predict_map = {f.name: f.rh_name for f in self._fields if f.rh_out}
        return [predict_map.get(name, name) for name in out_names]

    def residual_names(self, prognostic_names: list[str]) -> list[str]:
        """Translate prognostic names to network space for residual prediction:
        ``prediction`` fields use the derived RH name.
        """
        predict_map = {f.name: f.rh_name for f in self._fields if f.rh_out}
        return [predict_map.get(name, name) for name in prognostic_names]

    def _compute_qsat(self, raw_input: TensorMapping) -> dict[str, torch.Tensor]:
        if self._surface_pressure_name not in raw_input:
            raise ValueError(
                f"surface pressure '{self._surface_pressure_name}' is required by "
                "saturation_normalization but is not present in the input."
            )
        surface_pressure = raw_input[self._surface_pressure_name]
        # interface_pressure appends the vertical dimension as the last axis,
        # length n_layers + 1; layer k spans interfaces k and k + 1.
        interface_pressure = self._vertical_coordinate.interface_pressure(
            surface_pressure
        )
        qsat: dict[str, torch.Tensor] = {}
        for field in self._fields:
            if field.temperature_name not in raw_input:
                raise ValueError(
                    f"temperature '{field.temperature_name}' is required by "
                    f"saturation_normalization for '{field.name}' but is not "
                    "present in the input."
                )
            temperature = raw_input[field.temperature_name]
            layer_pressure = 0.5 * (
                interface_pressure[..., field.level]
                + interface_pressure[..., field.level + 1]
            )
            qsat[field.name] = saturation_specific_humidity(temperature, layer_pressure)
        return qsat

    def forward_transform(
        self,
        raw_input: TensorMapping,
        network_input: TensorMapping,
    ) -> tuple[TensorDict, SaturationNormalizationState]:
        """Rewrite humidity fields into their derived ``q / q_sat`` channel.

        ``q_sat`` is computed from ``raw_input`` (the raw, pre-global-mean-removal
        input) so it reflects the true input-step temperature and pressure.  The
        derived RH channel is written for ``replace`` and ``append`` fields; for
        ``replace`` the source ``q`` channel is removed.
        """
        qsat = self._compute_qsat(raw_input)
        result = dict(network_input)
        for field in self._fields:
            if not (field.rh_in or field.rh_extra):
                continue
            # Saturation fields are forbidden from overlapping global mean
            # removal, so the network_input value here is the raw q.
            result[field.rh_name] = result[field.name] / qsat[field.name]
            if field.rh_in:
                del result[field.name]
        return result, SaturationNormalizationState(qsat=qsat)

    def inverse_transform(
        self,
        output: TensorDict,
        state: SaturationNormalizationState,
    ) -> TensorDict:
        """Rewrite predicted derived ``q / q_sat`` channels back to ``q``.

        Uses the input-step ``q_sat`` cached in ``state`` (the same anchoring as
        the forward transform).
        """
        result = dict(output)
        for field in self._fields:
            if field.rh_out and field.rh_name in result:
                result[field.name] = result.pop(field.rh_name) * state.qsat[field.name]
        return result


def build_saturation_normalization(
    configs: list[SaturationNormalizationConfig] | None,
    vertical_coordinate: HasAtmosphereVerticalIntegral | None,
    in_names: list[str],
    out_names: list[str],
) -> SaturationNormalization | None:
    """Build the transform from config, or return ``None`` when not configured."""
    if not configs:
        return None
    fields = resolve_all_fields(configs, in_names, out_names)
    if vertical_coordinate is None:
        raise ValueError(
            "saturation_normalization requires a vertical coordinate to compute "
            "saturation specific humidity, but none is available."
        )
    n_layers = len(vertical_coordinate.get_ak()) - 1
    for field in fields:
        if not 0 <= field.level < n_layers:
            raise ValueError(
                f"saturation_normalization field '{field.name}' is at level "
                f"{field.level}, outside the {n_layers} model layers."
            )
    surface_pressure_name = _resolve_surface_pressure_name(in_names)
    return SaturationNormalization(
        fields=fields,
        surface_pressure_name=surface_pressure_name,
        vertical_coordinate=vertical_coordinate,
    )
