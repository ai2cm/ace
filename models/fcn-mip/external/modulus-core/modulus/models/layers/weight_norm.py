import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightNormLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_g = nn.Parameter(torch.empty((out_features, 1)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.weight_g, 1.0)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, input: Tensor) -> Tensor:
        norm = self.weight.norm(dim=1, p=2, keepdim=True)
        weight = self.weight_g * self.weight / norm
        return F.linear(input, weight, self.bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
