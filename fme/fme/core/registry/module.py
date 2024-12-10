import abc
import dataclasses
from typing import Any, Callable, ClassVar, Mapping, Tuple, TypeVar, Union

import dacite
from torch import nn

from .registry import Registry


@dataclasses.dataclass
class ModuleConfig(abc.ABC):
    """
    Builds a nn.Module given information about the input
    and output channels and the image shape.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    @abc.abstractmethod
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
            img_shape: shape of last two dimensions of data, e.g. latitude and
                longitude.

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


MT = TypeVar("MT", bound=nn.Module)


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

    Parameters:
        type: the type of the ModuleConfig
        config: data for a ModuleConfig instance of the indicated type
    """

    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry] = Registry()

    def __post__init(self):
        if self.registry is not Registry():
            raise ValueError("ModuleSelector.registry should not be set manually")

    @classmethod
    def register(cls, type_name) -> Callable[[MT], MT]:
        return cls.registry.register(type_name)

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
        instance = self.registry.from_dict(self.get_state())
        return instance.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )

    def get_state(self) -> Union[Mapping[str, Any], dict]:
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
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )

    @classmethod
    def from_dict(cls, config: dict):
        instance = cls.registry.from_dict(config)
        return cls(config=instance, type=config["type"])

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Modules."""
        return cls(type="", config={}).registry._types.keys()
