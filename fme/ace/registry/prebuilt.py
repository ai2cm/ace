import dataclasses

from torch import nn

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo


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
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        return self.module
