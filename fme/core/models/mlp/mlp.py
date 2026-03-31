import dataclasses

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


def _build_mlp(in_dim: int, out_dim: int, n_hidden: int, depth: int) -> nn.Sequential:
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

    return nn.Sequential(*layers)
