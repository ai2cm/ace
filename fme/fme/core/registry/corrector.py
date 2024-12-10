import dataclasses
import datetime
from typing import Any, Callable, ClassVar, Mapping, Type, TypeVar

import dacite

from fme.core.coordinates import HybridSigmaPressureCoordinate
from fme.core.corrector.registry import CorrectorConfigProtocol
from fme.core.gridded_ops import GriddedOperations

from .registry import Registry

CT = TypeVar("CT", bound=Type[CorrectorConfigProtocol])


@dataclasses.dataclass
class CorrectorSelector:
    """
    A dataclass containing all the information needed to build a
    CorrectorConfigProtocol, including the type of the CorrectorConfigProtocol
    and the data needed to build it.

    This is helpful as CorrectorSelector can be serialized and deserialized
    without any additional information, whereas to load a
    CorrectorConfigProtocol you would need to know the type of the
    CorrectorConfigProtocol being loaded.

    It is also convenient because CorrectorSelector is a single class that can
    be used to represent any CorrectorConfigProtocol, whereas
    CorrectorConfigProtocol is a protocol that can be implemented by many
    different classes.

    Parameters:
        type: the type of the CorrectorConfigProtocol
        config: data for a CorrectorConfigProtocol instance of the indicated type

    """

    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry] = Registry()

    def __post__init(self):
        if self.registry is not Registry():
            raise ValueError("CorrectorSelector.registry should not be set manually")

    @classmethod
    def register(cls, type_name) -> Callable[[CT], CT]:
        return cls.registry.register(type_name)

    def build(
        self,
        gridded_operations: GriddedOperations,
        vertical_coordinate: HybridSigmaPressureCoordinate,
        timestep: datetime.timedelta,
    ):
        instance = self.registry.from_dict(self.get_state())
        return instance.build(
            gridded_operations=gridded_operations,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        )

    def get_state(self) -> Mapping[str, Any]:
        """
        Get a dictionary containing all the information needed to build a
        CorrectorConfigProtocol.

        """
        return {"type": self.type, "config": self.config}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CorrectorSelector":
        """
        Create a CorrectorSelector from a dictionary containing all the information
        needed to build a CorrectorConfigProtocol.
        """
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Correctors."""
        return cls(type="", config={}).registry._types.keys()
