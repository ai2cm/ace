import abc
import dataclasses
import datetime
from typing import Any, Callable, Dict, Mapping, Protocol, Type

import dacite

from fme.core.data_loading.data_typing import SigmaCoordinates
from fme.core.gridded_ops import GriddedOperations
from fme.core.typing_ import TensorMapping


class CorrectorConfigProtocol(Protocol):
    def build(
        self,
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ) -> "CorrectorABC":
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "CorrectorConfigProtocol":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        ...


CORRECTOR_REGISTRY: Dict[str, Type[CorrectorConfigProtocol]] = {}


def get_available_corrector_types():
    return CORRECTOR_REGISTRY.keys()


def register_corrector(
    name: str,
) -> Callable[[Type[CorrectorConfigProtocol]], Type[CorrectorConfigProtocol]]:
    """
    Register a new ModuleConfig type with the CORRECTOR_REGISTRY.

    This is useful for adding new ModuleConfig types to the registry from
    other modules.

    Args:
        name: name of the ModuleConfig type to register

    Returns:
        a decorator which registers the decorated class with the CORRECTOR_REGISTRY
    """

    def decorator(cls: Type[CorrectorConfigProtocol]) -> Type[CorrectorConfigProtocol]:
        CORRECTOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_from_registry(name) -> Type[CorrectorConfigProtocol]:
    return CORRECTOR_REGISTRY[name]


class CorrectorABC(abc.ABC):
    @abc.abstractmethod
    def __call__(
        self,
        input_data: TensorMapping,
        gen_data: TensorMapping,
    ) -> TensorMapping:
        ...


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

    Attributes:
        type: the type of the CorrectorConfigProtocol
        config: data for a CorrectorConfigProtocol instance of the indicated type

    """

    type: str
    config: Mapping[str, Any]

    def __post_init__(self):
        try:
            self._config = get_from_registry(self.type).from_state(self.config)
        except KeyError:
            raise ValueError(
                f"unknown corrector type {self.type}, "
                f"known corrector types are {list(CORRECTOR_REGISTRY.keys())}"
            )

    def build(
        self,
        gridded_operations: GriddedOperations,
        sigma_coordinates: SigmaCoordinates,
        timestep: datetime.timedelta,
    ) -> CorrectorABC:
        return self._config.build(
            gridded_operations=gridded_operations,
            sigma_coordinates=sigma_coordinates,
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
