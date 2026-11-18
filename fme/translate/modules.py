"""Registry of transform module builders spanning an input and an output grid.

The generic :class:`fme.core.registry.module.ModuleConfig` contract builds a
module against a single ``DatasetInfo``, which cannot express a
resolution-changing operator (a 1°→2° encoder needs both grids). Transform
builders registered here receive the input *and* output domains'
``DatasetInfo``s. Two builders ship with the skeleton:

- ``same_grid`` wraps any existing :class:`ModuleSelector` entry (SFNO,
  prebuilt, ...) for transforms whose input and output grids match.
- ``interpolate`` is the trivial resolution-change operator: a 1x1 convolution
  for the channel mapping followed by interpolation to the output grid. It
  exists as the simplest registry member exercising the two-grid contract (and
  a baseline); learned operators (SHT truncation, transformer resizers) are
  later registry entries.
"""

import abc
import dataclasses
from collections.abc import Callable, Mapping

# we use Type to distinguish from type attr of TransformSelector
from typing import Any, ClassVar, Type  # noqa: UP035

import dacite
import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.registry.module import Module, ModuleSelector
from fme.core.registry.registry import Registry

__all__ = [
    "InterpolateTransformConfig",
    "SameGridTransformConfig",
    "TransformModuleConfig",
    "TransformSelector",
]


@dataclasses.dataclass
class TransformModuleConfig(abc.ABC):
    """Builds a transform module given both domains' channel counts and grids."""

    @abc.abstractmethod
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        """Build a module mapping the input domain to the output domain.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            in_dataset_info: information about the input domain's dataset,
                including its img_shape.
            out_dataset_info: information about the output domain's dataset;
                the built module must map to this domain's img_shape.

        Returns:
            a Module object
        """
        ...

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "TransformModuleConfig":
        return dacite.from_dict(
            data_class=cls, data=state, config=dacite.Config(strict=True)
        )


@dataclasses.dataclass
class TransformSelector:
    """Selects and configures a registered :class:`TransformModuleConfig`.

    Parameters:
        type: the type of the TransformModuleConfig
        config: data for a TransformModuleConfig instance of the indicated type
    """

    type: str
    config: Mapping[str, Any]
    registry: ClassVar[Registry[TransformModuleConfig]] = Registry[
        TransformModuleConfig
    ]()

    def __post_init__(self):
        if not isinstance(self.registry, Registry):
            raise ValueError("TransformSelector.registry should not be set manually")
        self._instance = self.registry.get(self.type, self.config)

    @classmethod
    def register(
        cls, type_name: str
    ) -> Callable[[Type[TransformModuleConfig]], Type[TransformModuleConfig]]:  # noqa: UP006
        return cls.registry.register(type_name)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        return self._instance.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            in_dataset_info=in_dataset_info,
            out_dataset_info=out_dataset_info,
        )


@TransformSelector.register("same_grid")
@dataclasses.dataclass
class SameGridTransformConfig(TransformModuleConfig):
    """Adapts a generic :class:`ModuleSelector` entry as a same-grid transform.

    Parameters:
        module: The module builder; built against the (shared) input grid.
    """

    module: ModuleSelector

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        if in_dataset_info.img_shape != out_dataset_info.img_shape:
            raise ValueError(
                "A same_grid transform requires matching input and output "
                f"grids, got img_shapes {in_dataset_info.img_shape} and "
                f"{out_dataset_info.img_shape}. Use a resolution-changing "
                "transform type instead."
            )
        return self.module.build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            dataset_info=in_dataset_info,
        )


class _ConvInterpolate(nn.Module):
    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        out_shape: tuple[int, int],
        mode: str,
    ):
        super().__init__()
        self.conv = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=1)
        self.out_shape = out_shape
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return nn.functional.interpolate(x, size=self.out_shape, mode=self.mode)


@TransformSelector.register("interpolate")
@dataclasses.dataclass
class InterpolateTransformConfig(TransformModuleConfig):
    """The trivial resolution-change operator: 1x1 conv + interpolation.

    Maps channels with a pointwise convolution, then interpolates to the
    output domain's grid. Interpolation is planar (it ignores spherical
    geometry); this is the baseline / test operator, not a recommended
    resolution-change operator for experiments.

    Parameters:
        mode: Interpolation mode passed to ``torch.nn.functional.interpolate``.
    """

    mode: str = "bilinear"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        in_dataset_info: DatasetInfo,
        out_dataset_info: DatasetInfo,
    ) -> Module:
        return Module(
            _ConvInterpolate(
                n_in_channels=n_in_channels,
                n_out_channels=n_out_channels,
                out_shape=out_dataset_info.img_shape,
                mode=self.mode,
            ),
            label_encoding=None,
        )
