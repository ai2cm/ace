# flake8: noqa
# Copied from https://github.com/ai2cm/modulus/commit/22df4a9427f5f12ff6ac891083220e7f2f54d229
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import math
from typing import Callable, Tuple

import torch
import torch.nn as nn

from torch.utils.checkpoint import checkpoint

from fme.core.distributed import Distributed

from fme.core.benchmark.timer import Timer, NullTimer

from .initialization import trunc_normal_

# wrap fft, to unify interface to spectral transforms
# import global convolution and non-linear spectral layers
# helpers
from .layers import (
    MLP,
    ConditionalLayerNorm,
    Context,
    ContextConfig,
    DropPath,
)
from .lora import LoRAConv2d
from .s2convolutions import SpectralConvS2
from .makani.spectral_convolution import SpectralConv


@dataclasses.dataclass
class SFNONetConfig:
    """Configuration parameters for SphericalFourierNeuralOperatorNet.

    Attributes:
        embed_dim: Dimension of the embeddings.
        filter_type: Type of spectral filter to use ('linear', 'makani-linear',
            'local').
        scale_factor: Scale factor for input/output resolution. Must be 1
            (other values are not implemented for conditional layer norm).
        global_layer_norm: Whether to reduce along the spatial domain when
            applying layer normalization.
        num_layers: Number of Fourier Neural Operator blocks.
        use_mlp: Whether to use an MLP in each FNO block.
        mlp_ratio: Ratio of MLP hidden dimension to the embedding dimension.
        activation_function: Activation function name ('relu', 'gelu', 'silu').
        encoder_layers: Number of convolutional layers in the encoder/decoder.
        pos_embed: Whether to use a learned positional embedding.
        drop_rate: Dropout rate.
        drop_path_rate: Stochastic depth rate (drop path).
        hard_thresholding_fraction: Fraction of spectral modes to retain.
        big_skip: Whether to use a big skip connection from input to decoder.
        checkpointing: Gradient checkpointing level (0=none, 1=encoder/decoder,
            3=all blocks).
        filter_num_groups: Number of groups in grouped spectral convolutions.
        filter_residual: Whether to filter residual connections through a
            SHT round-trip.
        filter_output: Whether to filter the output through a SHT round-trip.
        local_blocks: List of block indices to use local (DISCO) filters
            instead of spectral filters.
        normalize_big_skip: Whether to normalize the big skip connection.
        affine_norms: Whether to use element-wise affine parameters in the
            normalization layers.
        lora_rank: Rank of LoRA adaptations outside spectral convolutions.
            0 disables LoRA.
        lora_alpha: Strength of LoRA adaptations outside spectral convolutions.
            Defaults to lora_rank if None.
        spectral_lora_rank: Rank of LoRA adaptations for spectral convolutions.
            0 disables LoRA.
        spectral_lora_alpha: Strength of LoRA adaptations for spectral
            convolutions. Defaults to spectral_lora_rank if None.
    """

    embed_dim: int = 256
    filter_type: str = "linear"
    scale_factor: int = 1
    global_layer_norm: bool = False
    num_layers: int = 12
    use_mlp: bool = True
    mlp_ratio: float = 2.0
    activation_function: str = "gelu"
    encoder_layers: int = 1
    pos_embed: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    hard_thresholding_fraction: float = 1.0
    big_skip: bool = True
    checkpointing: int = 0
    filter_num_groups: int = 1
    filter_residual: bool = False
    filter_output: bool = False
    local_blocks: list[int] | None = None
    normalize_big_skip: bool = False
    affine_norms: bool = False
    lora_rank: int = 0
    lora_alpha: float | None = None
    spectral_lora_rank: int = 0
    spectral_lora_alpha: float | None = None


# heuristic for finding theta_cutoff
def _compute_cutoff_radius(nlat, kernel_shape, basis_type):
    theta_cutoff_factor = {
        "piecewise linear": 0.5,
        "morlet": 0.5,
        "zernike": math.sqrt(2.0),
    }

    return (
        (kernel_shape[0] + 1)
        * theta_cutoff_factor[basis_type]
        * math.pi
        / float(nlat - 1)
    )


class DiscreteContinuousConvS2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        dist = Distributed.get_instance()
        self.conv = dist.get_disco_conv_s2(*args, **kwargs)

    def forward(self, x, timer: Timer = NullTimer()):
        return self.conv(x), x


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        num_groups=1,
        filter_residual=False,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
    ):
        super(SpectralFilterLayer, self).__init__()

        if lora_rank != 0 and filter_type != "linear":
            raise NotImplementedError("LoRA is only supported for linear filter type.")

        if filter_type == "non-linear":
            raise NotImplementedError("Non-linear spectral filters are not supported.")

        # spectral transform is passed to the module
        elif filter_type == "linear":
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                bias=True,
                filter_residual=filter_residual,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                num_groups=num_groups,
            )
        elif filter_type == "makani-linear":
            self.filter = SpectralConv(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type="dhconv",
                num_groups=num_groups,
                bias=False,
                gain=1.0,
            )

        elif filter_type == "local":
            # heuristic for finding theta_cutoff
            theta_cutoff = 2 * _compute_cutoff_radius(
                nlat=forward_transform.nlat,
                kernel_shape=(3, 3),
                basis_type="morlet",
            )
            self.filter = DiscreteContinuousConvS2(
                embed_dim,
                embed_dim,
                in_shape=(forward_transform.nlat, forward_transform.nlon),
                out_shape=(inverse_transform.nlat, inverse_transform.nlon),
                kernel_shape=(3, 3),
                basis_type="morlet",
                basis_norm_mode="mean",
                groups=1,
                grid_in=forward_transform.grid,
                grid_out=inverse_transform.grid,
                bias=False,
                theta_cutoff=theta_cutoff,
            )
        else:
            raise (NotImplementedError)

    def forward(self, x, timer: Timer = NullTimer()):
        return self.filter(x, timer=timer)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        img_shape: Tuple[int, int],
        context_config: ContextConfig,
        filter_type="linear",
        global_layer_norm: bool = False,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        checkpointing=0,
        filter_residual=False,
        affine_norms=False,
        filter_num_groups: int = 1,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        spectral_lora_rank: int = 0,
        spectral_lora_alpha: float | None = None,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

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

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            filter_residual=filter_residual,
            num_groups=filter_num_groups,
            lora_rank=spectral_lora_rank,
            lora_alpha=spectral_lora_alpha,
        )

        if inner_skip == "linear":
            self.inner_skip = LoRAConv2d(
                embed_dim, embed_dim, 1, 1, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = LoRAConv2d(
                2 * embed_dim,
                embed_dim,
                1,
                bias=False,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = ConditionalLayerNorm(
            embed_dim,
            img_shape=self.output_shape_loc,
            global_layer_norm=global_layer_norm,
            context_config=context_config,
            elementwise_affine=affine_norms,
        )

        if use_mlp == True:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )

        if outer_skip == "linear":
            self.outer_skip = LoRAConv2d(
                embed_dim, embed_dim, 1, 1, lora_rank=lora_rank, lora_alpha=lora_alpha
            )
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = LoRAConv2d(
                2 * embed_dim,
                embed_dim,
                1,
                bias=False,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )

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
                if self.concat_skip:
                    x = torch.cat((x, self.inner_skip(residual)), dim=1)
                    x = self.inner_skip_conv(x)
                else:
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

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            with timer.child("outer_skip"):
                if self.concat_skip:
                    x = torch.cat((x, self.outer_skip(residual)), dim=1)
                    x = self.outer_skip_conv(x)
                else:
                    x = x + self.outer_skip(residual)

        return x


class NoLayerNorm(nn.Module):
    def forward(self, x, context: Context):
        return x


def get_lat_lon_sfnonet(
    params: SFNONetConfig,
    in_chans: int,
    out_chans: int,
    img_shape: Tuple[int, int],
    data_grid: str = "equiangular",
    context_config: ContextConfig = ContextConfig(
        embed_dim_scalar=0,
        embed_dim_noise=0,
        embed_dim_labels=0,
        embed_dim_pos=0,
    ),
) -> "SphericalFourierNeuralOperatorNet":
    h, w = img_shape
    modes_lat = int(h * params.hard_thresholding_fraction)
    modes_lon = int((w // 2 + 1) * params.hard_thresholding_fraction)

    dist = Distributed.get_instance()

    trans_down = dist.get_sht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
    )
    itrans_up = dist.get_isht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
    )
    trans = dist.get_sht(
        *img_shape, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
    )
    itrans = dist.get_isht(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss")

    def get_pos_embed():
        pos_embed = nn.Parameter(torch.zeros(1, params.embed_dim, h, w))
        pos_embed.is_shared_mp = ["matmul"]
        trunc_normal_(pos_embed, std=0.02)
        return pos_embed

    net = SphericalFourierNeuralOperatorNet(
        params,
        img_shape=img_shape,
        in_chans=in_chans,
        out_chans=out_chans,
        context_config=context_config,
        trans_down=trans_down,
        itrans_up=itrans_up,
        trans=trans,
        itrans=itrans,
        get_pos_embed=get_pos_embed,
    )
    return net


class SphericalFourierNeuralOperatorNet(torch.nn.Module):
    """Spherical Fourier Neural Operator Network.

    Args:
        params: Model configuration. See ``SFNONetConfig`` for details.
        img_shape: Spatial dimensions (lat, lon) of the input data.
        get_pos_embed: Factory function that returns a learned positional
            embedding parameter.
        trans_down: Spherical harmonic transform from input grid to spectral
            space (used for the first layer).
        itrans_up: Inverse spherical harmonic transform from spectral space
            to the output grid (used for the last layer).
        trans: Spherical harmonic transform for intermediate layers.
        itrans: Inverse spherical harmonic transform for intermediate layers.
        in_chans: Number of input channels.
        out_chans: Number of output channels.
        context_config: Configuration for conditional context embeddings
            (scalar, noise, positional, labels).
    """

    def __init__(
        self,
        params: SFNONetConfig,
        img_shape: Tuple[int, int],
        get_pos_embed: Callable[[], nn.Parameter],
        trans_down: nn.Module,
        itrans_up: nn.Module,
        trans: nn.Module,
        itrans: nn.Module,
        in_chans: int,
        out_chans: int,
        context_config: ContextConfig = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_labels=0,
            embed_dim_noise=0,
            embed_dim_pos=0,
        ),
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.filter_type = params.filter_type
        self.filter_residual = params.filter_residual
        self.filter_output = params.filter_output
        self.mlp_ratio = params.mlp_ratio
        self.img_shape = img_shape
        self._spatial_h_slice, self._spatial_w_slice = (
            Distributed.get_instance().get_local_slices(self.img_shape)
        )
        if params.scale_factor != 1:
            raise NotImplementedError(
                "scale factor must be 1 as it is not implemented for "
                "conditional layer normalization"
            )
        self.global_layer_norm = params.global_layer_norm
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.embed_dim = params.embed_dim
        self.num_layers = params.num_layers
        self.use_mlp = params.use_mlp
        self.encoder_layers = params.encoder_layers
        self._use_pos_embed = params.pos_embed
        self.big_skip = params.big_skip
        self.checkpointing = params.checkpointing
        if params.local_blocks is not None:
            self.local_blocks = [
                i for i in range(self.num_layers) if i in params.local_blocks
            ]
        else:
            self.local_blocks = []
        self.affine_norms = params.affine_norms
        self.filter_num_groups = params.filter_num_groups
        self.lora_rank = params.lora_rank
        self.lora_alpha = params.lora_alpha
        self.spectral_lora_rank = params.spectral_lora_rank
        self.spectral_lora_alpha = params.spectral_lora_alpha

        self.trans_down = trans_down
        self.itrans_up = itrans_up
        self.trans = trans
        self.itrans = itrans

        if self.filter_residual:
            self.residual_filter_down = self.trans_down
            self.residual_filter_up = self.itrans_up
        else:
            self.residual_filter_down = nn.Identity()
            self.residual_filter_up = nn.Identity()

        if self.filter_output:
            self.filter_output_down = self.trans_down
            self.filter_output_up = self.itrans_up
        else:
            self.filter_output_down = nn.Identity()
            self.filter_output_up = nn.Identity()

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

        # dropout
        self.pos_drop = (
            nn.Dropout(p=params.drop_rate) if params.drop_rate > 0.0 else nn.Identity()
        )
        dpr = [
            x.item() for x in torch.linspace(0, params.drop_path_rate, self.num_layers)
        ]

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            if i in self.local_blocks:
                block_filter_type = "local"
            else:
                block_filter_type = self.filter_type

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                img_shape=self.img_shape,
                context_config=context_config,
                filter_type=block_filter_type,
                global_layer_norm=self.global_layer_norm,
                mlp_ratio=self.mlp_ratio,
                drop_rate=params.drop_rate,
                drop_path=dpr[i],
                act_layer=act_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                checkpointing=self.checkpointing,
                filter_residual=self.filter_residual,
                affine_norms=self.affine_norms,
                filter_num_groups=self.filter_num_groups,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                spectral_lora_rank=self.spectral_lora_rank,
                spectral_lora_alpha=self.spectral_lora_alpha,
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

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        """Helper"""
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x: torch.Tensor, context: Context):
        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x, context)
            else:
                x = blk(x, context)

        return x

    def forward(self, x: torch.Tensor, context: Context):
        # save big skip
        if self.big_skip:
            residual = self.residual_filter_up(self.residual_filter_down(x))
            residual = self.norm_big_skip(residual, context=context)

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if self.pos_embed is not None:
            x = x + self.pos_embed[..., self._spatial_h_slice, self._spatial_w_slice]

        # maybe clean the padding just in case

        x = self.pos_drop(x)

        x = self._forward_features(x, context)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)

        x = self.filter_output_up(self.filter_output_down(x))

        return x
