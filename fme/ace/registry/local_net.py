import dataclasses
from typing import Literal

import torch
from torch import nn

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.models.conditional_sfno.ankur import (
    AnkurLocalNetConfig,
    get_lat_lon_ankur_localnet,
)
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.conditional_sfno.localnet import (
    BlockType,
    LocalNetConfig,
    get_lat_lon_localnet,
)


class _ContextWrappedModule(nn.Module):
    """Wraps a module that takes (x, context: Context) to accept (x, labels=None).

    This adapts the conditional_sfno forward signature to the interface
    expected by the Module registry wrapper.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        context = Context(
            embedding_scalar=None,
            embedding_pos=None,
            labels=labels,
            noise=None,
        )
        return self.module(x, context)


class NoiseConditionedModule(nn.Module):
    """Wraps a context-based module with gaussian noise conditioning.

    Generates gaussian noise and optional positional embeddings (with
    label-position interaction), then calls the wrapped module with a
    fully populated Context.

    Args:
        module: An nn.Module with forward signature (x, context: Context).
        img_shape: Global spatial dimensions (lat, lon) of the input data.
        embed_dim_noise: Dimension of gaussian noise channels.
        embed_dim_pos: Dimension of learned positional embedding. 0 disables.
        embed_dim_labels: Dimension of label embeddings. 0 disables.
    """

    def __init__(
        self,
        module: nn.Module,
        img_shape: tuple[int, int],
        embed_dim_noise: int = 256,
        embed_dim_pos: int = 0,
        embed_dim_labels: int = 0,
    ):
        super().__init__()
        self.module = module
        self.embed_dim_noise = embed_dim_noise
        self.img_shape = img_shape
        self.label_pos_embed: nn.Parameter | None = None
        if embed_dim_pos != 0:
            self.pos_embed: nn.Parameter | None = nn.Parameter(
                torch.zeros(
                    1, embed_dim_pos, img_shape[0], img_shape[1], requires_grad=True
                )
            )
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            if embed_dim_labels > 0:
                self.label_pos_embed = nn.Parameter(
                    torch.zeros(
                        embed_dim_labels,
                        embed_dim_pos,
                        img_shape[0],
                        img_shape[1],
                        requires_grad=True,
                    )
                )
                nn.init.trunc_normal_(self.label_pos_embed, std=0.02)
        else:
            self.pos_embed = None

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = x.reshape(-1, *x.shape[-3:])
        noise = torch.randn(
            [x.shape[0], self.embed_dim_noise, *x.shape[-2:]],
            device=x.device,
            dtype=x.dtype,
        )

        h_slice, w_slice = Distributed.get_instance().get_local_slices(self.img_shape)

        embedding_pos: torch.Tensor | None = None
        if self.pos_embed is not None:
            pos_local = self.pos_embed[..., h_slice, w_slice]
            embedding_pos = pos_local.repeat(x.shape[0], 1, 1, 1)
            if self.label_pos_embed is not None and labels is not None:
                label_local = self.label_pos_embed[..., h_slice, w_slice]
                label_embedding_pos = torch.einsum(
                    "bl, lpxy -> bpxy", labels, label_local
                )
                embedding_pos = embedding_pos + label_embedding_pos

        return self.module(
            x,
            Context(
                embedding_scalar=None,
                embedding_pos=embedding_pos,
                labels=labels,
                noise=noise,
            ),
        )


@ModuleSelector.register("AnkurLocalNet")
@dataclasses.dataclass
class AnkurLocalNetBuilder(ModuleConfig):
    """Configuration for the AnkurLocalNet architecture.

    A simple 3-hidden-layer MLP that optionally uses a DISCO convolution
    for the first layer and a learned positional embedding.

    Attributes:
        embed_dim: Dimension of the hidden layers.
        use_disco_encoder: Whether to use a DISCO convolution for the first
            layer instead of a 1x1 convolution.
        disco_kernel_size: Kernel size for the DISCO convolution if used.
        pos_embed: Whether to add a learned positional embedding after the
            first layer.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        data_grid: Grid type for spherical harmonic transforms used by
            DISCO convolutions.
    """

    embed_dim: int = 256
    use_disco_encoder: bool = False
    disco_kernel_size: int = 3
    pos_embed: bool = False
    activation_function: str = "gelu"
    data_grid: Literal["legendre-gauss", "equiangular"] = "equiangular"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        params = AnkurLocalNetConfig(
            embed_dim=self.embed_dim,
            use_disco_encoder=self.use_disco_encoder,
            disco_kernel_size=self.disco_kernel_size,
            pos_embed=self.pos_embed,
            activation_function=self.activation_function,
        )
        context_config = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_noise=0,
            embed_dim_labels=len(dataset_info.all_labels),
            embed_dim_pos=0,
        )
        net = get_lat_lon_ankur_localnet(
            params=params,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=context_config,
        )
        return _ContextWrappedModule(net)


@ModuleSelector.register("LocalNet")
@dataclasses.dataclass
class LocalNetBuilder(ModuleConfig):
    """Configuration for the LocalNet architecture.

    A noise-conditioned local neural operator network using DISCO convolutions
    and/or 1x1 convolutions, with encoder/decoder structure and optional skip
    connections. Supports label conditioning when used with conditional=True
    on the ModuleSelector.

    Attributes:
        embed_dim: Dimension of the embeddings.
        noise_embed_dim: Dimension of the gaussian noise conditioning channels.
        context_pos_embed_dim: Dimension of the learned positional embedding
            used for conditioning. 0 disables.
        block_types: List of filter types for each block ('disco', 'conv1x1').
            The length determines the number of blocks.
        global_layer_norm: Whether to reduce along the spatial domain when
            applying layer normalization.
        use_mlp: Whether to use an MLP in each block.
        mlp_ratio: Ratio of MLP hidden dimension to the embedding dimension.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        encoder_layers: Number of convolutional layers in the encoder/decoder.
        pos_embed: Whether to use a learned positional embedding inside the
            LocalNet (distinct from context_pos_embed_dim).
        big_skip: Whether to use a big skip connection from input to decoder.
        normalize_big_skip: Whether to normalize the big skip connection.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        lora_rank: Rank of LoRA adaptations. 0 disables LoRA.
        lora_alpha: Strength of LoRA adaptations. Defaults to lora_rank
            if None.
        data_grid: Grid type for spherical harmonic transforms used by
            DISCO convolutions.
    """

    embed_dim: int = 256
    noise_embed_dim: int = 256
    context_pos_embed_dim: int = 0
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
    data_grid: Literal["legendre-gauss", "equiangular"] = "equiangular"

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        params = LocalNetConfig(
            embed_dim=self.embed_dim,
            block_types=self.block_types,
            global_layer_norm=self.global_layer_norm,
            use_mlp=self.use_mlp,
            mlp_ratio=self.mlp_ratio,
            activation_function=self.activation_function,
            encoder_layers=self.encoder_layers,
            pos_embed=self.pos_embed,
            big_skip=self.big_skip,
            normalize_big_skip=self.normalize_big_skip,
            affine_norms=self.affine_norms,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
        )
        embed_dim_labels = len(dataset_info.all_labels)
        context_config = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_labels=embed_dim_labels,
            embed_dim_pos=self.context_pos_embed_dim,
        )
        net = get_lat_lon_localnet(
            params=params,
            in_chans=n_in_channels,
            out_chans=n_out_channels,
            img_shape=dataset_info.img_shape,
            data_grid=self.data_grid,
            context_config=context_config,
        )
        return NoiseConditionedModule(
            net,
            img_shape=dataset_info.img_shape,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            embed_dim_labels=embed_dim_labels,
        )
