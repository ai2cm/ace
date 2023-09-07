import dataclasses
from typing import Any, Literal, Mapping, Optional, Protocol, Tuple, Type

import torch_harmonics as harmonics

# this package is installed in models/FourCastNet
from fourcastnet.networks.afnonet import AFNONetBuilder
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet
from torch import nn


class ModuleConfig(Protocol):
    """
    A protocol for a class that can build a nn.Module given information about the input
    and output channels and the image shape.

    This is a "Config" as in practice it is a dataclass loaded directly from yaml,
    allowing us to specify details of the network architecture in a config file.
    """

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        ...


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@dataclasses.dataclass
class SphericalFourierNeuralOperatorBuilder(ModuleConfig):
    spectral_transform: str = "sht"
    filter_type: str = "non-linear"
    operator_type: str = "diagonal"
    scale_factor: int = 16
    embed_dim: int = 256
    num_layers: int = 12
    num_blocks: int = 16
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    factorization: Optional[str] = None
    separable: bool = False
    complex_network: bool = True
    complex_activation: str = "real"
    spectral_layers: int = 1
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ):
        sfno_net = SphericalFourierNeuralOperatorNet(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
        )

        # Patch in the grid that our data lies on rather than the one which is
        # hard-coded in the modulus codebase [1]. Duplicate the code to compute
        # the number of SHT modes determined by hard_thresholding_fraction. Note
        # that this does not handle the distributed case which is handled by
        # L518 [2] in their codebase.

        # [1] https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py  # noqa: E501
        # [2] https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L518  # noqa: E501
        nlat, nlon = img_shape
        modes_lat = int(nlat * self.hard_thresholding_fraction)
        modes_lon = int((nlon // 2 + 1) * self.hard_thresholding_fraction)
        sht = harmonics.RealSHT(
            nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=self.data_grid
        ).float()
        isht = harmonics.InverseRealSHT(
            nlat, nlon, lmax=modes_lat, mmax=modes_lon, grid=self.data_grid
        ).float()

        sfno_net.trans_down = sht
        sfno_net.itrans_up = isht

        return sfno_net


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


NET_REGISTRY: Mapping[str, Type[ModuleConfig]] = {
    "afno": AFNONetBuilder,  # using short acronym for backwards compatibility
    "SphericalFourierNeuralOperatorNet": SphericalFourierNeuralOperatorBuilder,  # type: ignore  # noqa: E501
    "prebuilt": PreBuiltBuilder,
}


@dataclasses.dataclass
class ModuleSelector:
    """
    A dataclass containing all the information needed to build a ModuleConfig,
    including the type of the ModuleConfig and the data needed to build it.

    This is helpful as ModuleSelector can be serialized and deserialized
    without any additional information, whereas to load a ModuleConfig you
    would need to know the type of the ModuleConfig being loaded.

    It is also convenient because ModuleSelector is a single class that can be
    used to represent any ModuleConfig, whereas ModuleConfig is a protocol
    that can be implemented by many different classes.

    Attributes:
        type: the type of the ModuleConfig
        config: data for a ModuleConfig instance of the indicated type
    """

    type: Literal[
        "afno",
        "SphericalFourierNeuralOperatorNet",
        "prebuilt",
    ]
    config: Mapping[str, Any]

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ) -> nn.Module:
        """
        Build a nn.Module given information about the input and output channels
        and the image shape.

        Args:
            n_in_channels: number of input channels
            n_out_channels: number of output channels
            img_shape: last two dimensions of data, corresponding to lat and
                lon when using FourCastNet conventions

        Returns:
            a nn.Module
        """
        return NET_REGISTRY[self.type](**self.config).build(
            n_in_channels=n_in_channels,
            n_out_channels=n_out_channels,
            img_shape=img_shape,
        )

    def get_state(self) -> Mapping[str, Any]:
        """
        Get a dictionary containing all the information needed to build a ModuleConfig.
        """
        return {"type": self.type, "config": self.config}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "ModuleSelector":
        """
        Create a ModuleSelector from a dictionary containing all the information
        needed to build a ModuleConfig.
        """
        return cls(**state)
