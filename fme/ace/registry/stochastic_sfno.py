import dataclasses
from typing import Literal, Optional, Tuple

import torch

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.models.conditional_sfno.sfnonet import Context, ContextConfig
from fme.core.models.conditional_sfno.sfnonet import (
    SphericalFourierNeuralOperatorNet as ConditionalSFNO,
)


class NoiseConditionedSFNO(torch.nn.Module):
    def __init__(self, conditional_model: ConditionalSFNO, embed_dim: int):
        super().__init__()
        self.conditional_model = conditional_model
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(
            [*x.shape[:-3], self.embed_dim, *x.shape[-2:]],
            device=x.device,
            dtype=x.dtype,
        )
        embedding_scalar = torch.zeros(
            [*x.shape[:-3], 0], device=x.device, dtype=x.dtype
        )
        return self.conditional_model(
            x, Context(embedding_scalar=embedding_scalar, embedding_2d=noise)
        )


# this is based on the call signature of SphericalFourierNeuralOperatorNet at
# https://github.com/NVIDIA/modulus/blob/b8e27c5c4ebc409e53adaba9832138743ede2785/modulus/models/sfno/sfnonet.py#L292  # noqa: E501
@ModuleSelector.register("NoiseConditionedSFNO")
@dataclasses.dataclass
class NoiseConditionedSFNOBuilder(ModuleConfig):
    """
    Configuration for a noise-conditioned SFNO model.

    Noise is provided as conditioning input to conditional layer normalization.
    """

    spectral_transform: str = "sht"
    filter_type: str = "non-linear"
    operator_type: str = "diagonal"
    residual_filter_factor: int = 1
    embed_dim: int = 256
    noise_embed_dim: int = 256
    global_layer_norm: bool = False
    num_layers: int = 12
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
    # healpix not supported due to assumptions about number of spatial dims
    data_grid: Literal["legendre-gauss", "equiangular"] = "legendre-gauss"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        img_shape: Tuple[int, int],
    ):
        sfno_net = ConditionalSFNO(
            params=self,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=img_shape,
            context_config=ContextConfig(
                embed_dim_scalar=0,
                embed_dim_2d=self.noise_embed_dim,
            ),
        )
        return NoiseConditionedSFNO(sfno_net, self.noise_embed_dim)
