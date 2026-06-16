from collections.abc import Mapping
from typing import NamedTuple


class VariableMetadata(NamedTuple):
    units: str | None = None
    long_name: str | None = None

    @classmethod
    def from_attrs(cls, attrs: Mapping[str, str]) -> "VariableMetadata":
        return cls(units=attrs.get("units"), long_name=attrs.get("long_name"))

    def as_attrs(self) -> dict[str, str]:
        result: dict[str, str] = {}
        if self.units is not None:
            result["units"] = self.units
        if self.long_name is not None:
            result["long_name"] = self.long_name
        return result

    def display_long_name(self, fallback: str) -> str:
        return self.long_name if self.long_name is not None else fallback

    def display_units(self, fallback: str = "unknown_units") -> str:
        return self.units if self.units is not None else fallback
