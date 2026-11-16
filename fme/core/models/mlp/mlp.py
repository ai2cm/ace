import dataclasses
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.registry import ModuleSelector
from fme.core.registry.module import ModuleConfig


@ModuleSelector.register("MLP")
@dataclasses.dataclass
class MLPConfig(ModuleConfig):
    """
    Configuration for an MLP network.

    Parameters:
        hidden_dim: Number of hidden units in the MLP.
        depth: Number of layers in the MLP.
    """

    hidden_dim: int = 256
    depth: int = 2

    @classmethod
    def remove_deprecated_keys(cls, state: Mapping[str, Any]) -> dict[str, Any]:
        return dict(state)

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,  # Not needed for MLP
    ) -> nn.Module:
        """Build the MLP network."""
        return _build_mlp(
            in_dim=n_in_channels,
            out_dim=n_out_channels,
            n_hidden=self.hidden_dim,
            depth=self.depth,
        )


class MLP(nn.Sequential):
    """An ``nn.Sequential`` whose forward is defined in fme rather than torch.

    Distributed data parallel runs its wrapped module in a frame that
    ``torch.compile`` (dynamo) starts afresh, and dynamo skips such a frame,
    running it eagerly, when its code is in torch itself, as
    ``nn.Sequential.forward`` is. Overriding forward here gives dynamo a frame
    to compile. The state dict is the same as the parent ``nn.Sequential``'s.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input)


def _build_mlp(in_dim: int, out_dim: int, n_hidden: int, depth: int) -> MLP:
    """Build an MLP network with the given parameters."""
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth}")

    layers: list[nn.Module] = []
    if depth == 1:
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
    else:
        layers.append(nn.Conv2d(in_dim, n_hidden, kernel_size=1))
        layers.append(nn.GELU())
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_hidden, n_hidden, kernel_size=1))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(n_hidden, out_dim, kernel_size=1))

    return MLP(*layers)
