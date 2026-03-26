"""Secondary decoder for computing additional diagnostic variables."""

import dataclasses
from collections.abc import Callable

import torch
from torch import nn

from fme.core.dataset_info import DatasetInfo
from fme.core.packer import Packer
from fme.core.registry import ModuleSelector
from fme.core.registry.module import Module
from fme.core.typing_ import TensorDict

_VALID_NETWORK_TYPES = {"MLP"}


@dataclasses.dataclass
class SecondaryDecoderConfig:
    """
    Configuration for the secondary decoder that computes additional diagnostic
    variables from the main module's outputs.

    Parameters:
        secondary_diagnostic_names: Names of additional diagnostic variables, to be
            diagnosed directly from outputs without access to latent variables (i.e.,
            column-locally).
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

    def build(
        self,
        n_in_channels: int,
    ) -> "SecondaryDecoder":
        return SecondaryDecoder(
            in_dim=n_in_channels,
            out_names=self.secondary_diagnostic_names,
            network=self.network,
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
    def torch_modules(self) -> nn.ModuleList:
        """A list of the underlying nn.Module(s)."""
        return nn.ModuleList([self._module.torch_module])

    def wrap_module(
        self, wrapper: Callable[[nn.Module], nn.Module]
    ) -> "SecondaryDecoder":
        self._module = self._module.wrap_module(wrapper)
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


class NoSecondaryDecoder:
    """A placeholder for when no secondary decoder is used."""

    def __init__(self):
        pass

    def __call__(self, x: torch.Tensor) -> TensorDict:
        """Return an empty dictionary."""
        return {}

    def to(self, device) -> "NoSecondaryDecoder":
        """No-op for device transfer."""
        return self

    def detach(self) -> "NoSecondaryDecoder":
        """No-op for detach."""
        return self

    def wrap_module(
        self, wrapper: Callable[[nn.Module], nn.Module]
    ) -> "NoSecondaryDecoder":
        """No-op for wrapping module."""
        return self

    @property
    def torch_modules(self) -> nn.ModuleList:
        """No underlying module."""
        return nn.ModuleList()

    def get_module_state(self) -> dict:
        """Return an empty state dict."""
        return {}

    def load_module_state(self, state_dict: dict) -> None:
        """No-op for loading state dict."""
        pass
