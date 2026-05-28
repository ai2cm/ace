from typing import NamedTuple


class VariableMetadata(NamedTuple):
    units: str | None = None
    long_name: str | None = None
