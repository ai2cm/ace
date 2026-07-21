import dataclasses


@dataclasses.dataclass
class CoupledNames:
    ocean: list[str]
    atmosphere: list[str]


@dataclasses.dataclass
class CoupledOptionalInt:
    ocean: int | None
    atmosphere: int | None
