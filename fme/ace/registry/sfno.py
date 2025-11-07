import dataclasses
from typing import Literal

from fme.ace.models.makani.sfnonet import (
    SphericalFourierNeuralOperatorNet as MakaniSFNO,
)
from fme.ace.models.modulus.sfnonet import SphericalFourierNeuralOperatorNet, SFNO
from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.distributed import Distributed

# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@ModuleSelector.register("SphericalFourierNeuralOperatorNet")
@dataclasses.dataclass
class SphericalFourierNeuralOperatorBuilder(ModuleConfig):
    """
    Configuration for the SFNO architecture used in FourCastNet-SFNO.
    """

    spectral_transform: str = "sht"
    filter_type: str = "linear"
    operator_type: str = "diagonal"
    scale_factor: int = 1
    residual_filter_factor: int = 1
    embed_dim: int = 256
    num_layers: int = 12
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    rank: float = 1.0
    factorization: str | None = None
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
        img_shape: tuple[int, int],
    ):
      dist= Distributed.get_instance()
      if dist.spatial_parallelism:
        sfno_net = SFNO(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,)
      else:
        sfno_net = SphericalFourierNeuralOperatorNet(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,)

      return sfno_net


@ModuleSelector.register("SFNO-v0.1.0")
@dataclasses.dataclass
class SFNO_V0_1_0(ModuleConfig):
    """
    Configuration for the SFNO architecture in modulus-makani version 0.1.0.
    """

    spectral_transform: str = "sht"
    filter_type: Literal["linear"] = "linear"
    operator_type: str = "dhconv"
    scale_factor: int = 16
    embed_dim: int = 256
    num_layers: int = 12
    repeat_layers: int = 1
    hard_thresholding_fraction: float = 1.0
    normalization_layer: str = "instance_norm"
    use_mlp: bool = True
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: Literal["none", "direct", "frequency"] = "direct"
    big_skip: bool = True
    rank: float = 1.0
    factorization: str | None = None
    separable: bool = False
    complex_activation: str = "real"
    spectral_layers: int = 1
    checkpointing: int = 0
    data_grid: Literal["legendre-gauss", "equiangular", "healpix"] = "legendre-gauss"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: tuple[int, int],
    ):
        return MakaniSFNO(
            inp_chans=n_in_channels,
            out_chans=n_out_channels,
            inp_shape=img_shape,
            out_shape=img_shape,
            model_grid_type=self.data_grid,
            spectral_transform=self.spectral_transform,
            filter_type=self.filter_type,
            operator_type=self.operator_type,
            scale_factor=self.scale_factor,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            repeat_layers=self.repeat_layers,
            hard_thresholding_fraction=self.hard_thresholding_fraction,
            normalization_layer=self.normalization_layer,
            use_mlp=self.use_mlp,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            big_skip=self.big_skip,
            rank=self.rank,
            factorization=self.factorization,
            separable=self.separable,
            complex_activation=self.complex_activation,
            spectral_layers=self.spectral_layers,
            checkpointing=self.checkpointing,
        )
