import dataclasses
from typing import Any, Callable, Dict, Mapping, Protocol, Tuple, Type

import dacite
from torch import nn


@dataclasses.dataclass
class ModuleConfig(Protocol):
    """
    A protocol for a class that can build a nn.Module given information about the input
    and output channels and the image shape.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleConfig":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


NET_REGISTRY: Dict[str, Type[ModuleConfig]] = {}


def register(name: str) -> Callable[[Type[ModuleConfig]], Type[ModuleConfig]]:
    """
    Register a new ModuleConfig type with the NET_REGISTRY.

    This is useful for adding new ModuleConfig types to the registry from
    other modules.

    Args:
        name: name of the ModuleConfig type to register

    Returns:
        a decorator which registers the decorated class with the NET_REGISTRY
    """
    if not isinstance(name, str):
        raise TypeError(
            f"name must be a string, got {name}, "
            "make sure to use as @register('module_name')"
        )

    def decorator(cls: Type[ModuleConfig]) -> Type[ModuleConfig]:
        NET_REGISTRY[name] = cls
        return cls

    return decorator


def get_from_registry(name) -> Type[ModuleConfig]:
    return NET_REGISTRY[name]


@dataclasses.dataclass
class ModuleSelector:
    """
    A dataclass containing all the information needed to build a ModuleConfig,
    including the type of the ModuleConfig and the data needed to build it.

    This is helpful as ModuleSelector can be serialized and deserialized
    without any additional information, whereas to load a ModuleConfig you
    would need to know the type of the ModuleConfig being loaded.

    It is also convenient because ModuleSelector is a single class that can be
    used to represent any ModuleConfig, whereas ModuleConfig is a protocol
    that can be implemented by many different classes.

    Attributes:
        type: the type of the ModuleConfig
        config: data for a ModuleConfig instance of the indicated type
    """

    type: str
    config: Mapping[str, Any]

    def __post_init__(self):
        try:
            self._config = NET_REGISTRY[self.type].from_state(self.config)
        except KeyError:
            raise ValueError(
                f"unknown module type {self.type}, "
                f"known module types are {list(NET_REGISTRY.keys())}"
            )

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        return self._config.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )

    def get_state(self) -> Mapping[str, Any]:
        """
        Get a dictionary containing all the information needed to build a ModuleConfig.
        """
        return {"type": self.type, "config": self.config}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleSelector":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        return dacite.from_dict(
            data_class=ModuleSelector, data=state, config=dacite.Config(strict=True)
        )
