import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of CorrectorSelector
from typing import Any, ClassVar, Type, TypeVar  # noqa: UP035

from .registry import Registry

# Note we can either type hint register, which prevents registering
# from overwriting the type with a generic type or Any, _or_ we can
# type hint the registry attribute below, which prevents `instance`
# in `build` from being typed as `Any`. We cannot do both, unfortunately.
T = TypeVar("T")


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
    # Note this registry is no longer used, but is kept for backwards compatibility.
    # In practice the corrector is selected based on the type of the vertical
    # coordinate in the loaded data.
    registry: ClassVar[Registry] = Registry()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("CorrectorSelector.registry should not be set manually")

    @classmethod
    def register(cls, type_name) -> Callable[[Type[T]], Type[T]]:  # noqa: UP006
        return cls.registry.register(type_name)

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Correctors."""
        return cls(type="", config={}).registry._types.keys()
