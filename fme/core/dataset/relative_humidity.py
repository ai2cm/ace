"""Relative humidity derived at load time from temperature and total water.

``relative_humidity_<k>`` is ``specific_total_water_<k> / q_sat`` where
``q_sat`` is the Bolton (1980) saturation specific humidity over liquid water
at ``air_temperature_<k>`` and the layer-mean pressure (the mean of the hybrid
sigma-pressure interface pressures bounding layer ``k``). All inputs are taken
at the same time step. No clipping is applied: total water includes
condensate, so values above 1 are expected.
"""

import dataclasses
import re
from collections.abc import Sequence

from fme.core.coordinates import HybridSigmaPressureCoordinate, VerticalCoordinate
from fme.core.humidity import saturation_specific_humidity
from fme.core.typing_ import TensorDict

from .data_typing import VariableMetadata

RELATIVE_HUMIDITY_PREFIX = "relative_humidity_"
TEMPERATURE_PREFIX = "air_temperature_"
TOTAL_WATER_PREFIX = "specific_total_water_"
SURFACE_PRESSURE_NAME = "PRESsfc"

_RH_PATTERN = re.compile(rf"^{RELATIVE_HUMIDITY_PREFIX}(\d+)$")

RELATIVE_HUMIDITY_METADATA = VariableMetadata(
    units="1",
    long_name=(
        "relative humidity (specific total water over Bolton saturation "
        "specific humidity over liquid water)"
    ),
)


@dataclasses.dataclass(frozen=True)
class DerivedRelativeHumidity:
    """Derives the ``relative_humidity_<k>`` fields requested of a dataset.

    Parameters:
        levels: Vertical layer indices of the requested relative humidity
            fields.
        extra_names: Source fields loaded only to derive relative humidity
            (not themselves requested), dropped after derivation.
    """

    levels: tuple[int, ...]
    extra_names: frozenset[str]

    @classmethod
    def from_requested_names(cls, names: Sequence[str]) -> "DerivedRelativeHumidity":
        levels = []
        for name in names:
            match = _RH_PATTERN.match(name)
            if match is not None:
                levels.append(int(match.group(1)))
        levels = sorted(levels)
        sources = cls._source_names(levels)
        extra = frozenset(sources) - set(names)
        return cls(levels=tuple(levels), extra_names=extra)

    @staticmethod
    def _source_names(levels: Sequence[int]) -> list[str]:
        if len(levels) == 0:
            return []
        names = [SURFACE_PRESSURE_NAME]
        for k in levels:
            names.append(f"{TEMPERATURE_PREFIX}{k}")
            names.append(f"{TOTAL_WATER_PREFIX}{k}")
        return names

    @property
    def source_names(self) -> list[str]:
        """Names of the fields needed to derive relative humidity."""
        return self._source_names(self.levels)

    @property
    def names(self) -> list[str]:
        """Names of the derived relative humidity fields."""
        return [f"{RELATIVE_HUMIDITY_PREFIX}{k}" for k in self.levels]

    def load_names(self, names: Sequence[str]) -> list[str]:
        """Names to load from the data files: the requested names minus the
        derived ones, plus the source fields needed to derive them.
        """
        derived = set(self.names)
        result = [n for n in names if n not in derived]
        for name in self.source_names:
            if name not in result:
                result.append(name)
        return result

    def validate(self, vertical_coordinate: VerticalCoordinate):
        if len(self.levels) == 0:
            return
        if not isinstance(vertical_coordinate, HybridSigmaPressureCoordinate):
            raise ValueError(
                "Deriving relative humidity requires a hybrid sigma-pressure "
                f"vertical coordinate, got {type(vertical_coordinate).__name__}."
            )
        n_layers = len(vertical_coordinate.ak) - 1
        for k in self.levels:
            if k >= n_layers:
                raise ValueError(
                    f"Cannot derive {RELATIVE_HUMIDITY_PREFIX}{k}: the vertical "
                    f"coordinate only has {n_layers} layers."
                )

    def apply(
        self, tensors: TensorDict, vertical_coordinate: VerticalCoordinate
    ) -> TensorDict:
        """Add the derived relative humidity fields and drop the source fields
        that were loaded only to derive them.
        """
        if len(self.levels) == 0:
            return tensors
        assert isinstance(vertical_coordinate, HybridSigmaPressureCoordinate)
        surface_pressure = tensors[SURFACE_PRESSURE_NAME]
        interface_pressure = vertical_coordinate.interface_pressure(surface_pressure)
        for k in self.levels:
            layer_pressure = 0.5 * (
                interface_pressure[..., k] + interface_pressure[..., k + 1]
            )
            qsat = saturation_specific_humidity(
                tensors[f"{TEMPERATURE_PREFIX}{k}"], layer_pressure
            )
            rh = tensors[f"{TOTAL_WATER_PREFIX}{k}"] / qsat
            tensors[f"{RELATIVE_HUMIDITY_PREFIX}{k}"] = rh.to(
                tensors[f"{TOTAL_WATER_PREFIX}{k}"].dtype
            )
        for name in self.extra_names:
            del tensors[name]
        return tensors
