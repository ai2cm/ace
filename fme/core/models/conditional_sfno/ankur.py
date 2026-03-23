import dataclasses
import math

import torch
import torch.nn as nn

from fme.core.distributed import Distributed

from .initialization import trunc_normal_
from .layers import Context, ContextConfig
from .sfnonet import _compute_cutoff_radius


@dataclasses.dataclass
class AnkurLocalNetConfig:
    """Configuration parameters for AnkurLocalNet.

    Replicates the diagnostic MLP architecture from Ankur's
    ColumnDiagnosticSphericalFourierNeuralOperatorNet.

    Attributes:
        embed_dim: Dimension of the hidden layers.
        use_disco_encoder: Whether to use a DISCO convolution for the first
            layer instead of a 1x1 convolution.
        disco_kernel_size: Kernel size for the DISCO convolution if used.
        pos_embed: Whether to add a learned positional embedding after the
            first layer.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        data_grid: Grid type for DISCO convolutions
            ('equiangular', 'legendre-gauss').
    """

    embed_dim: int = 256
    use_disco_encoder: bool = False
    disco_kernel_size: int = 3
    pos_embed: bool = False
    activation_function: str = "gelu"
    data_grid: str = "equiangular"


class GroupedDiscreteContinuousConvS2(nn.Module):
    """DISCO convolution using groups=gcd(in_chans, out_chans)."""

    def __init__(self, in_chans, out_chans, img_shape, kernel_size, data_grid):
        super().__init__()
        nlat, nlon = img_shape
        kernel_shape = (kernel_size, kernel_size)
        theta_cutoff = _compute_cutoff_radius(
            nlat=nlat,
            kernel_shape=kernel_shape,
            basis_type="morlet",
        )
        dist = Distributed.get_instance()
        self.conv = dist.get_disco_conv_s2(
            in_chans,
            out_chans,
            in_shape=img_shape,
            out_shape=img_shape,
            kernel_shape=kernel_shape,
            basis_type="morlet",
            basis_norm_mode="mean",
            groups=math.gcd(in_chans, out_chans),
            grid_in=data_grid,
            grid_out=data_grid,
            bias=False,
            theta_cutoff=theta_cutoff,
        )

    def forward(self, x):
        return self.conv(x)


class AddPosEmbed(nn.Module):
    def __init__(self, embed_dim, img_shape):
        super().__init__()
        h, w = img_shape
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, h, w))
        self.pos_embed.is_shared_mp = ["matmul"]
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        return x + self.pos_embed


def get_lat_lon_ankur_localnet(
    params: AnkurLocalNetConfig,
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    context_config: ContextConfig = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=0,
        embed_dim_pos=0,
    ),
) -> "AnkurLocalNet":
    return AnkurLocalNet(
        params,
        img_shape=img_shape,
        in_chans=in_chans,
        out_chans=out_chans,
    )


class AnkurLocalNet(nn.Module):
    """Local network replicating Ankur's diagnostic MLP architecture.

    A simple sequential network with 3 hidden layers, optionally using a
    DISCO convolution for the first layer and a learned positional embedding.
    This is a drop-in replacement for LocalNet with the same forward signature.

    Args:
        params: Model configuration. See ``AnkurLocalNetConfig`` for details.
        img_shape: Spatial dimensions (lat, lon) of the input data.
        in_chans: Number of input channels.
        out_chans: Number of output channels.
    """

    def __init__(
        self,
        params: AnkurLocalNetConfig,
        img_shape: tuple[int, int],
        in_chans: int,
        out_chans: int,
    ):
        super().__init__()

        activation_functions = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if params.activation_function not in activation_functions:
            raise ValueError(
                f"Unknown activation function {params.activation_function}"
            )
        act_layer = activation_functions[params.activation_function]

        hidden_dim = params.embed_dim
        current_dim = in_chans
        modules: list[nn.Module] = []
        for i in range(3):
            if i == 0 and params.use_disco_encoder:
                modules.append(
                    GroupedDiscreteContinuousConvS2(
                        current_dim,
                        hidden_dim,
                        img_shape=img_shape,
                        kernel_size=params.disco_kernel_size,
                        data_grid=params.data_grid,
                    )
                )
            else:
                modules.append(nn.Conv2d(current_dim, hidden_dim, 1, bias=True))
            if i == 0 and params.pos_embed:
                modules.append(AddPosEmbed(hidden_dim, img_shape))
            modules.append(act_layer())
            current_dim = hidden_dim
        modules.append(nn.Conv2d(current_dim, out_chans, 1, bias=False))
        self.mlp = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor, context: Context):
        return self.mlp(x)
