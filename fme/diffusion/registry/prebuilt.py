import dataclasses
from typing import Tuple

from torch import nn

from fme.diffusion.registry.module import ModuleConfig, ModuleSelector


@ModuleSelector.register("prebuilt")
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
        n_sigma_embedding_channels: int,
    ) -> nn.Module:
        return self.module
