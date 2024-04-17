import dataclasses
from typing import Tuple

from torch import nn

from fme.ace.registry.registry import ModuleConfig, register


@register("prebuilt")
@dataclasses.dataclass
class PreBuiltBuilder(ModuleConfig):
    """
    A simple module configuration which returns a pre-defined module.

    Used mainly for testing.
    """

    module: nn.Module

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        return self.module
