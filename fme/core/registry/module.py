import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of ModuleSelector
from typing import Any, ClassVar, Type  # noqa: UP035

import dacite
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.labels import BatchLabels, LabelEncoding

from .registry import Registry


@dataclasses.dataclass
class ModuleConfig(abc.ABC):
    """
    Builds a nn.Module given information about the input and output channels
    and dataset information.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    @abc.abstractmethod
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the dataset.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            dataset_info: Information about the dataset, including img_shape,
                horizontal coordinates, vertical coordinate, etc.

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


CONDITIONAL_BUILDERS = [
    "NoiseConditionedSFNO",
]


class Module:
    def __init__(self, module: nn.Module, label_encoding: LabelEncoding | None):
        self._module = module
        self._label_encoding = label_encoding

    def __call__(
        self, input: torch.Tensor, labels: BatchLabels | None = None
    ) -> torch.Tensor:
        if labels is not None and self._label_encoding is None:
            raise TypeError("Labels are not allowed for unconditional models")

        if self._label_encoding is not None:
            if labels is None:
                raise TypeError("Labels are required for conditional models")
            encoded_labels = labels.conform_to_encoding(self._label_encoding)
            return self._module(input, labels=encoded_labels.tensor)
        else:
            return self._module(input)

    @property
    def torch_module(self) -> nn.Module:
        return self._module

    def get_state(self) -> dict[str, Any]:
        if self._label_encoding is not None:
            label_encoder_state = self._label_encoding.get_state()
        else:
            label_encoder_state = None
        return {
            **self._module.state_dict(),
            "label_encoding": label_encoder_state,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        state = state.copy()
        if state.get("label_encoding") is not None:
            if self._label_encoding is None:
                self._label_encoding = LabelEncoding.from_state(
                    state.pop("label_encoding")
                )
            else:
                self._label_encoding.conform_to_state(state.pop("label_encoding"))
        state.pop("label_encoding", None)
        self._module.load_state_dict(state)

    def wrap_module(self, callable: Callable[[nn.Module], nn.Module]) -> "Module":
        return Module(callable(self._module), self._label_encoding)

    def to(self, device: torch.device) -> "Module":
        return Module(self._module.to(device), self._label_encoding)


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
        conditional: whether to condition the predictions on batch labels.
    """

    type: str
    config: Mapping[str, Any]
    conditional: bool = False
    registry: ClassVar[Registry[ModuleConfig]] = Registry[ModuleConfig]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("ModuleSelector.registry should not be set manually")
        if self.conditional and self.type not in CONDITIONAL_BUILDERS:
            raise ValueError(
                "Conditional predictions require a conditional builder, "
                f"got {self.type} (available: {CONDITIONAL_BUILDERS})"
            )
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
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the dataset.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            dataset_info: Information about the dataset, including img_shape
                (shape of last two dimensions of data, e.g. latitude and
                longitude), horizontal coordinates, vertical coordinate, etc.

        Returns:
            a Module object
        """
        if self.conditional and len(dataset_info.all_labels) == 0:
            raise ValueError("Conditional predictions require labels")
        if self.conditional:
            label_encoding = LabelEncoding(sorted(list(dataset_info.all_labels)))
        else:
            label_encoding = None
        module = self._instance.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=dataset_info,
        )
        return Module(module, label_encoding)

    @classmethod
    def get_available_types(cls):
        """This class method is used to expose all available types of Modules."""
        return cls.registry._types.keys()
