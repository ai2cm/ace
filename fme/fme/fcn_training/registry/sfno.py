import dataclasses
from typing import Literal, Optional, Tuple

from ai2modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet

from fme.fcn_training.registry.registry import ModuleConfig, register


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@register("SphericalFourierNeuralOperatorNet")
@dataclasses.dataclass
class SphericalFourierNeuralOperatorBuilder(ModuleConfig):
    spectral_transform: str = "sht"
    filter_type: str = "non-linear"
    operator_type: str = "diagonal"
    scale_factor: int = 16
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

        return sfno_net
