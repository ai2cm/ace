"""Secondary decoder for computing additional diagnostic variables."""

import dataclasses
from collections.abc import Callable

import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.packer import Packer
from fme.core.registry import ModuleSelector
from fme.core.registry.module import Module, ModuleConfig
from fme.core.typing_ import TensorDict


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


_VALID_NETWORK_TYPES = {"MLP"}


@dataclasses.dataclass
class SecondaryDecoderConfig:
    """
    Configuration for the secondary decoder that computes additional diagnostic
    variables from the main module's outputs.

    Parameters:
        secondary_diagnostic_names: Names of additional diagnostic variables, to be
            diagnosed directly from outputs without access to latent variables.
        network: Configuration for the decoder network.
    """

    secondary_diagnostic_names: list[str]
    network: ModuleSelector

    def __post_init__(self):
        if self.network.type not in _VALID_NETWORK_TYPES:
            raise ValueError(
                f"Invalid network type '{self.network.type}'. "
                f"Valid types are: {_VALID_NETWORK_TYPES}"
            )


class SecondaryDecoder:
    """
    A decoder for computing additional diagnostic variables from module outputs.

    This decoder wraps an nn.Module and a Packer to transform the main module's
    output channels into additional diagnostic variables.
    """

    CHANNEL_DIM = -3

    def __init__(
        self,
        in_dim: int,
        out_names: list[str],
        network: ModuleSelector,
    ):
        """
        Args:
            in_dim: Number of input channels (should match main module output).
            out_names: Names of the diagnostic variables this decoder produces.
            network: ModuleSelector specifying the network architecture.
        """
        out_dim = len(out_names)
        self._module: Module = network.build(
            n_in_channels=in_dim,
            n_out_channels=out_dim,
            dataset_info=DatasetInfo(),  # Not needed for MLP
        )
        self._packer = Packer(out_names)

    @property
    def torch_module(self) -> nn.Module:
        """The underlying nn.Module."""
        return self._module.torch_module

    @property
    def module(self) -> Module:
        return self._module

    @module.setter
    def module(self, value: nn.Module) -> None:
        self._module = value

    def wrap_module(
        self, callable: Callable[[nn.Module], nn.Module]
    ) -> "SecondaryDecoder":
        self._module = self._module.wrap_module(callable)
        return self

    def to(self, device) -> "SecondaryDecoder":
        """Move the module to the specified device."""
        self._module = self._module.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        return self._module(x)

    def __call__(self, x: torch.Tensor) -> TensorDict:
        """Call the decoder and unpack the outputs."""
        out = self.forward(x)
        return self._unpack(out, axis=self.CHANNEL_DIM)

    def _unpack(self, tensor: torch.Tensor, axis: int) -> TensorDict:
        return self._packer.unpack(tensor, axis=axis)

    def get_module_state(self) -> dict:
        """Return the state dict of the underlying module."""
        return self._module.get_state()

    def load_module_state(self, state_dict: dict) -> None:
        """Load the state dict into the underlying module."""
        self._module.load_state(state_dict)
