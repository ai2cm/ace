import dataclasses
from collections.abc import Callable
from typing import Literal, get_args

import torch
import torch.nn as nn

from fme.core.benchmark.timer import NullTimer, Timer
from fme.core.distributed import Distributed

from .initialization import trunc_normal_
from .layers import MLP, ConditionalLayerNorm, Context, ContextConfig
from .lora import LoRAConv2d
from .sfnonet import DiscreteContinuousConvS2, NoLayerNorm, _compute_cutoff_radius

BlockType = Literal["disco", "conv1x1"]


@dataclasses.dataclass
class LocalNetConfig:
    """Configuration parameters for LocalNet.

    Attributes:
        embed_dim: Dimension of the embeddings.
        block_types: List of filter types for each block ('disco', 'conv1x1').
            The length determines the number of blocks.
        global_layer_norm: Whether to reduce along the spatial domain when
            applying layer normalization.
        use_mlp: Whether to use an MLP in each block.
        mlp_ratio: Ratio of MLP hidden dimension to the embedding dimension.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        encoder_layers: Number of convolutional layers in the encoder/decoder.
        pos_embed: Whether to use a learned positional embedding.
        big_skip: Whether to use a big skip connection from input to decoder.
        normalize_big_skip: Whether to normalize the big skip connection.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        lora_rank: Rank of LoRA adaptations. 0 disables LoRA.
        lora_alpha: Strength of LoRA adaptations. Defaults to lora_rank
            if None.
        data_grid: Grid type for DISCO convolutions
            ('equiangular', 'legendre-gauss').
    """

    embed_dim: int = 256
    block_types: list[BlockType] = dataclasses.field(
        default_factory=lambda: ["disco"] * 12
    )
    global_layer_norm: bool = False
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    big_skip: bool = True
    normalize_big_skip: bool = False
    affine_norms: bool = False
    lora_rank: int = 0
    lora_alpha: float | None = None
    data_grid: str = "equiangular"

    def __post_init__(self):
        valid = get_args(BlockType)
        for i, bt in enumerate(self.block_types):
            if bt not in valid:
                raise ValueError(
                    f"Invalid block type {bt!r} at index {i}, "
                    f"must be one of {valid}"
                )


class Conv1x1Filter(nn.Module):
    """1x1 convolution used as a local filter."""

    def __init__(self, embed_dim, lora_rank=0, lora_alpha=None):
        super().__init__()
        self.conv = LoRAConv2d(
            embed_dim,
            embed_dim,
            1,
            bias=True,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def forward(self, x, timer: Timer = NullTimer()):
        return self.conv(x), x


class LocalFilterLayer(nn.Module):
    """Local filter layer using either DISCO convolution or 1x1 convolution."""

    def __init__(
        self,
        embed_dim,
        img_shape: tuple[int, int],
        filter_type="disco",
        data_grid="equiangular",
        lora_rank: int = 0,
        lora_alpha: float | None = None,
    ):
        super().__init__()

        if filter_type == "disco":
            nlat, nlon = img_shape
            theta_cutoff = 2 * _compute_cutoff_radius(
                nlat=nlat,
                kernel_shape=(3, 3),
                basis_type="morlet",
            )
            self.filter = DiscreteContinuousConvS2(
                embed_dim,
                embed_dim,
                in_shape=img_shape,
                out_shape=img_shape,
                kernel_shape=(3, 3),
                basis_type="morlet",
                basis_norm_mode="mean",
                groups=1,
                grid_in=data_grid,
                grid_out=data_grid,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
        elif filter_type == "conv1x1":
            self.filter = Conv1x1Filter(
                embed_dim,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        else:
            raise NotImplementedError(f"Unknown filter type: {filter_type}")

    def forward(self, x, timer: Timer = NullTimer()):
        return self.filter(x, timer=timer)


class LocalBlock(nn.Module):
    """Block using local (non-spectral) filters."""

    def __init__(
        self,
        embed_dim,
        img_shape: tuple[int, int],
        context_config: ContextConfig,
        filter_type="disco",
        data_grid="equiangular",
        global_layer_norm: bool = False,
        mlp_ratio=2.0,
        act_layer=nn.GELU,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=False,
        affine_norms=False,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
    ):
        super().__init__()

        self.input_shape_loc = img_shape
        self.output_shape_loc = img_shape

        # norm layer
        self.norm0 = ConditionalLayerNorm(
            embed_dim,
            img_shape=self.input_shape_loc,
            global_layer_norm=global_layer_norm,
            context_config=context_config,
            elementwise_affine=affine_norms,
        )

        # local filter
        self.filter = LocalFilterLayer(
            embed_dim,
            img_shape=img_shape,
            filter_type=filter_type,
            data_grid=data_grid,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

        if inner_skip == "linear":
            self.inner_skip = LoRAConv2d(
                embed_dim,
                embed_dim,
                1,
                1,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        if filter_type == "conv1x1":
            self.act_layer = act_layer()

        # norm layer
        self.norm1 = ConditionalLayerNorm(
            embed_dim,
            img_shape=self.output_shape_loc,
            global_layer_norm=global_layer_norm,
            context_config=context_config,
            elementwise_affine=affine_norms,
        )

        if use_mlp:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )

        if outer_skip == "linear":
            self.outer_skip = LoRAConv2d(
                embed_dim,
                embed_dim,
                1,
                1,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

    def forward(self, x, context_embedding, timer: Timer = NullTimer()):
        with timer.child("norm0") as norm0_timer:
            x_norm = torch.zeros_like(x)
            x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = (
                self.norm0(
                    x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]],
                    context_embedding,
                    timer=norm0_timer,
                )
            )
        with timer.child("filter") as filter_timer:
            x, residual = self.filter(x_norm, timer=filter_timer)
        if hasattr(self, "inner_skip"):
            with timer.child("inner_skip"):
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            with timer.child("activation"):
                x = self.act_layer(x)

        with timer.child("norm1") as norm1_timer:
            x_norm = torch.zeros_like(x)
            x_norm[..., : self.output_shape_loc[0], : self.output_shape_loc[1]] = (
                self.norm1(
                    x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]],
                    context_embedding,
                    timer=norm1_timer,
                )
            )
            x = x_norm

        if hasattr(self, "mlp"):
            with timer.child("mlp"):
                x = self.mlp(x)

        if hasattr(self, "outer_skip"):
            with timer.child("outer_skip"):
                x = x + self.outer_skip(residual)

        return x


def get_lat_lon_localnet(
    params: LocalNetConfig,
    in_chans: int,
    out_chans: int,
    img_shape: tuple[int, int],
    context_config: ContextConfig = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=0,
        embed_dim_pos=0,
    ),
) -> "LocalNet":
    h, w = img_shape

    def get_pos_embed():
        pos_embed = nn.Parameter(torch.zeros(1, params.embed_dim, h, w))
        pos_embed.is_shared_mp = ["matmul"]
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    net = LocalNet(
        params,
        img_shape=img_shape,
        in_chans=in_chans,
        out_chans=out_chans,
        context_config=context_config,
        get_pos_embed=get_pos_embed,
    )
    return net


class LocalNet(torch.nn.Module):
    """Local Neural Operator Network.

    Uses only local operations (DISCO convolutions and 1x1 convolutions)
    without any spectral transforms.

    Args:
        params: Model configuration. See ``LocalNetConfig`` for details.
        img_shape: Spatial dimensions (lat, lon) of the input data.
        get_pos_embed: Factory function that returns a learned positional
            embedding parameter.
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        context_config: Configuration for conditional context embeddings
            (scalar, noise, positional, labels).
    """

    def __init__(
        self,
        params: LocalNetConfig,
        img_shape: tuple[int, int],
        get_pos_embed: Callable[[], nn.Parameter],
        in_chans: int,
        out_chans: int,
        context_config: ContextConfig = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=0,
            embed_dim_pos=0,
        ),
    ):
        super().__init__()

        self.block_types = params.block_types
        self.mlp_ratio = params.mlp_ratio
        self.img_shape = img_shape
        self._spatial_h_slice, self._spatial_w_slice = (
            Distributed.get_instance().get_local_slices(self.img_shape)
        )
        self.global_layer_norm = params.global_layer_norm
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = params.embed_dim
        self.num_layers = len(params.block_types)
        self.use_mlp = params.use_mlp
        self.encoder_layers = params.encoder_layers
        self._use_pos_embed = params.pos_embed
        self.big_skip = params.big_skip
        self.affine_norms = params.affine_norms
        self.lora_rank = params.lora_rank
        self.lora_alpha = params.lora_alpha
        self.data_grid = params.data_grid

        # determine activation function
        activation_functions = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        if params.activation_function not in activation_functions:
            raise ValueError(
                f"Unknown activation function {params.activation_function}"
            )
        act_layer = activation_functions[params.activation_function]

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(
                LoRAConv2d(
                    current_dim,
                    encoder_hidden_dim,
                    1,
                    bias=True,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                )
            )
            encoder_modules.append(act_layer())
            current_dim = encoder_hidden_dim
        encoder_modules.append(
            LoRAConv2d(
                current_dim,
                self.embed_dim,
                1,
                bias=False,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
        )
        self.encoder = nn.Sequential(*encoder_modules)

        # blocks
        self.blocks = nn.ModuleList([])
        for block_type in self.block_types:
            inner_skip = "linear"
            outer_skip = "identity"

            block = LocalBlock(
                self.embed_dim,
                img_shape=self.img_shape,
                context_config=context_config,
                filter_type=block_type,
                data_grid=self.data_grid,
                global_layer_norm=self.global_layer_norm,
                mlp_ratio=self.mlp_ratio,
                act_layer=act_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                affine_norms=self.affine_norms,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )

            self.blocks.append(block)

        # decoder
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(
                LoRAConv2d(
                    current_dim,
                    decoder_hidden_dim,
                    1,
                    bias=True,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                )
            )
            decoder_modules.append(act_layer())
            current_dim = decoder_hidden_dim
        decoder_modules.append(
            LoRAConv2d(
                current_dim,
                self.out_chans,
                1,
                bias=False,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
        )
        self.decoder = nn.Sequential(*decoder_modules)

        # learned position embedding
        if self._use_pos_embed:
            self.pos_embed = get_pos_embed()
        else:
            self.pos_embed = None

        if params.normalize_big_skip:
            self.norm_big_skip = ConditionalLayerNorm(
                in_chans,
                img_shape=self.img_shape,
                global_layer_norm=self.global_layer_norm,
                context_config=context_config,
                elementwise_affine=self.affine_norms,
            )
        else:
            self.norm_big_skip = NoLayerNorm()

    def _forward_features(self, x: torch.Tensor, context: Context):
        for blk in self.blocks:
            x = blk(x, context)

        return x

    def forward(self, x: torch.Tensor, context: Context):
        # save big skip
        if self.big_skip:
            residual = self.norm_big_skip(x, context=context)

        x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[..., self._spatial_h_slice, self._spatial_w_slice]

        x = self._forward_features(x, context)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)

        return x
