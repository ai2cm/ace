import enum
import math

import torch
import torch.nn as nn
from torch import Tensor


class SirenLayerType(enum.Enum):
    FIRST = enum.auto()
    HIDDEN = enum.auto()
    LAST = enum.auto()


class SirenLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SirenLayerType = SirenLayerType.HIDDEN,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        self.omega_0 = omega_0

        self.linear = nn.Linear(in_features, out_features, bias=True)

        self.apply_activation = layer_type in {
            SirenLayerType.FIRST,
            SirenLayerType.HIDDEN,
        }

        self.reset_parameters()

    def reset_parameters(self) -> None:
        weight_ranges = {
            SirenLayerType.FIRST: 1.0 / self.in_features,
            SirenLayerType.HIDDEN: math.sqrt(6.0 / self.in_features) / self.omega_0,
            SirenLayerType.LAST: math.sqrt(6.0 / self.in_features),
        }
        weight_range = weight_ranges[self.layer_type]
        nn.init.uniform_(self.linear.weight, -weight_range, weight_range)

        k_sqrt = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.linear.bias, -k_sqrt, k_sqrt)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.apply_activation:
            x = torch.sin(self.omega_0 * x)
        return x
