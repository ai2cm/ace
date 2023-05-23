# this package is installed in models/FourCastNet
from fourcastnet.networks.afnonet import AFNONetBuilder
from torch import nn
from typing import Mapping, Protocol

# this package is installed in models/fcn-mip
from networks.geometric_v1.sfnonet import (
    FourierNeuralOperatorBuilder,
)


class ModuleBuilder(Protocol):
    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape_x: int,
        img_shape_y: int,
    ) -> nn.Module:
        ...


NET_REGISTRY: Mapping[str, ModuleBuilder] = {
    "afno": AFNONetBuilder,  # using short acronym for backwards compatibility
    "FourierNeuralOperatorNet": FourierNeuralOperatorBuilder,
}
