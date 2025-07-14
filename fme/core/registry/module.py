import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of ModuleSelector
from typing import Any, ClassVar, Type  # noqa: UP035

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
        img_shape: tuple[int, int],
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
    registry: ClassVar[Registry[ModuleConfig]] = Registry[ModuleConfig]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("ModuleSelector.registry should not be set manually")
        self._instance = self.registry.get(self.type, self.config)

    @classmethod
    def register(
        cls, type_name: str
    ) -> Callable[[Type[ModuleConfig]], Type[ModuleConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
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
        return self._instance.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Modules."""
        module = nn.Identity()
        return cls(type="prebuilt", config={"module": module}).registry._types.keys()
