import dataclasses
from typing import Literal

import torch
from torch import nn

from fme.ace.registry.registry import ModuleConfig, ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedModel
from fme.core.dataset_info import DatasetInfo
from fme.core.distributed import Distributed
from fme.core.models.conditional_sfno.ankur import (
    AnkurLocalNetConfig,
    get_lat_lon_ankur_localnet,
)
from fme.core.models.conditional_sfno.layers import Context, ContextConfig
from fme.core.models.conditional_sfno.localnet import (
    BasisType,
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
    """

    embed_dim: int = 256
    use_disco_encoder: bool = False
    disco_kernel_size: int = 3
    pos_embed: bool = False
    activation_function: str = "gelu"

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
            data_grid="legendre-gauss",
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
        kernel_shape: Shape of the DISCO convolution filter basis, passed
            to the filter basis constructor. For the "piecewise linear" and
            "morlet" basis types this is a two-element tuple
            (n_radial_modes, n_azimuthal_modes). When n_azimuthal_modes is
            1, the "piecewise linear" basis produces isotropic (radially
            symmetric) filters. Only affects 'disco' blocks.
        basis_type: Type of filter basis for the DISCO convolution
            ('morlet', 'piecewise linear', or 'zernike'). Only affects
            'disco' blocks.
        noise_embed_dim: Dimension of the noise conditioning channels.
        noise_type: Type of noise for conditioning ('gaussian' or 'isotropic').
            Isotropic noise is generated via inverse spherical harmonic
            transform.
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
    """

    embed_dim: int = 256
    kernel_shape: tuple[int, int] = (3, 3)
    basis_type: BasisType = "morlet"
    noise_embed_dim: int = 256
    noise_type: Literal["gaussian", "isotropic"] = "gaussian"
    context_pos_embed_dim: int = 0
    block_types: list[BlockType] = dataclasses.field(
        default_factory=lambda: [
            "disco",
            "disco",
            "disco",
            "disco",
            "conv1x1",
            "conv1x1",
            "conv1x1",
            "conv1x1",
        ]
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

    def build(
        self,
        n_in_channels: int,
        n_out_channels: int,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        params = LocalNetConfig(
            embed_dim=self.embed_dim,
            kernel_shape=self.kernel_shape,
            basis_type=self.basis_type,
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
            data_grid="legendre-gauss",
            context_config=context_config,
        )
        img_shape = dataset_info.img_shape
        if self.noise_type == "isotropic":
            dist = Distributed.get_instance()
            inverse_sht = dist.get_isht(*img_shape, grid="legendre-gauss")
            lmax = inverse_sht.lmax
            mmax = inverse_sht.mmax
        else:
            inverse_sht = None
            lmax = 0
            mmax = 0
        return NoiseConditionedModel(
            net,
            img_shape=img_shape,
            embed_dim_noise=self.noise_embed_dim,
            embed_dim_pos=self.context_pos_embed_dim,
            embed_dim_labels=embed_dim_labels,
            inverse_sht=inverse_sht,
            lmax=lmax,
            mmax=mmax,
        )
