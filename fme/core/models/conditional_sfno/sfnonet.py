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

from functools import partial
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd
from torch.utils.checkpoint import checkpoint

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
    RealFFT2,
    SpectralAttention2d,
)
from .s2convolutions import SpectralAttentionS2, SpectralConvS2


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
        self.conv = th.DiscreteContinuousConvS2(*args, **kwargs)

    def forward(self, x):
        return self.conv(x), x


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="block-diagonal",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=1,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
        filter_residual=False,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                operator_type=operator_type,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "non-linear" and (isinstance(forward_transform, RealFFT2)):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        # spectral transform is passed to the module
        elif filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=True,
                use_tensorly=False if factorization is None else True,
                filter_residual=filter_residual,
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

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        context_config: ContextConfig,
        filter_type="linear",
        operator_type="diagonal",
        global_layer_norm: bool = False,
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
        filter_residual=False,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
        self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = ConditionalLayerNorm(
            embed_dim,
            img_shape=self.input_shape_loc,
            global_layer_norm=global_layer_norm,
            context_config=context_config,
        )

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
            filter_residual=filter_residual,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

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
        )

        if use_mlp == True:
            MLPH = MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x, context_embedding):
        x_norm = torch.zeros_like(x)
        x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
            x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]],
            context_embedding,
        )
        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x_norm = torch.zeros_like(x)
        x_norm[..., : self.output_shape_loc[0], : self.output_shape_loc[1]] = (
            self.norm1(
                x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]],
                context_embedding,
            )
        )
        x = x_norm

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(torch.nn.Module):
    """
    Spherical Fourier Neural Operator Network

    Parameters
    ----------
    params : dict
        Dictionary of parameters
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    filter_type : str, optional
        Type of filter to use ('linear', 'non-linear'), by default "non-linear"
    operator_type : str, optional
        Type of operator to use ('diaginal', 'dhconv'), by default "diagonal"
    img_shape : tuple, optional
        Shape of the input channels, by default (721, 1440)
    scale_factor : int, optional
        Scale factor to use, by default 16
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    context_config : ContextConfig, optional
        Context configuration, by default
        ContextConfig(embed_dim_scalar=0, embed_dim_2d=0)
    num_layers : int, optional
        Number of layers in the network, by default 12
    use_mlp : int, optional
        Whether to use MLP, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    num_blocks : int, optional
        Number of blocks in the network, by default 16
    sparsity_threshold : float, optional
        Threshold for sparsity, by default 0.0
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding to apply, by default 1.0
    use_complex_kernels : bool, optional
        Whether to use complex kernels, by default True
    big_skip : bool, optional
        Whether to use big skip connections, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    complex_network : bool, optional
        Whether to use a complex network architecture, by default True
    complex_activation : str, optional
        Type of complex activation function to use, by default "real"
    spectral_layers : int, optional
        Number of spectral layers, by default 3
    checkpointing : int, optional
        Number of checkpointing segments, by default 0
    local_blocks: List[int], optional
        List of blocks to use local filters, by default []

    Example:
    --------
    >>> from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet as SFNO
    >>> model = SFNO(
    ...         params={},
    ...         img_shape=(8, 16),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=2,
    ...         encoder_layers=1,
    ...         num_blocks=4,
    ...         spectral_layers=2,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 8, 16)).shape
    torch.Size([1, 2, 8, 16])
    """

    def __init__(
        self,
        params,
        spectral_transform: str = "sht",
        filter_type: str = "linear",
        operator_type: str = "diagonal",
        img_shape: Tuple[int, int] = (721, 1440),
        scale_factor: int = 1,
        residual_filter_factor: int = 1,
        in_chans: int = 2,
        out_chans: int = 2,
        embed_dim: int = 256,
        context_config: ContextConfig = ContextConfig(
            embed_dim_scalar=0,
            embed_dim_2d=0,
        ),
        global_layer_norm: bool = False,
        num_layers: int = 12,
        use_mlp: int = True,
        mlp_ratio: float = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: bool = True,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.0,
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Any = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        checkpointing: int = 0,
        filter_residual: bool = False,
        filter_output: bool = False,
        local_blocks: Optional[List[int]] = None,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.params = params
        self.spectral_transform = (
            params.spectral_transform
            if hasattr(params, "spectral_transform")
            else spectral_transform
        )
        self.filter_type = (
            params.filter_type if hasattr(params, "filter_type") else filter_type
        )
        self.filter_residual = (
            params.filter_residual
            if hasattr(params, "filter_residual")
            else filter_residual
        )
        self.filter_output = (
            params.filter_output if hasattr(params, "filter_output") else filter_output
        )
        self.mlp_ratio = params.mlp_ratio if hasattr(params, "mlp_ratio") else mlp_ratio
        self.operator_type = (
            params.operator_type if hasattr(params, "operator_type") else operator_type
        )
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else img_shape
        )
        self.scale_factor = (
            params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        )
        self.residual_filter_factor = (
            params.residual_filter_factor
            if hasattr(params, "residual_filter_factor")
            else residual_filter_factor
        )
        if self.scale_factor != 1:
            raise NotImplementedError(
                "scale factor must be 1 as it is not implemented for "
                "conditional layer normalization"
            )
        self.global_layer_norm = (
            params.global_layer_norm
            if hasattr(params, "global_layer_norm")
            else global_layer_norm
        )
        self.in_chans = (
            params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        )
        self.out_chans = (
            params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        )
        self.embed_dim = self.num_features = (
            params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        )
        self.num_layers = (
            params.num_layers if hasattr(params, "num_layers") else num_layers
        )
        self.num_blocks = (
            params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        )
        self.hard_thresholding_fraction = (
            params.hard_thresholding_fraction
            if hasattr(params, "hard_thresholding_fraction")
            else hard_thresholding_fraction
        )
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.activation_function = (
            params.activation_function
            if hasattr(params, "activation_function")
            else activation_function
        )
        self.encoder_layers = (
            params.encoder_layers
            if hasattr(params, "encoder_layers")
            else encoder_layers
        )
        self.pos_embed = params.pos_embed if hasattr(params, "pos_embed") else pos_embed
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.factorization = (
            params.factorization if hasattr(params, "factorization") else factorization
        )
        self.separable = params.separable if hasattr(params, "separable") else separable
        self.complex_network = (
            params.complex_network
            if hasattr(params, "complex_network")
            else complex_network
        )
        self.complex_activation = (
            params.complex_activation
            if hasattr(params, "complex_activation")
            else complex_activation
        )
        self.spectral_layers = (
            params.spectral_layers
            if hasattr(params, "spectral_layers")
            else spectral_layers
        )
        self.checkpointing = (
            params.checkpointing if hasattr(params, "checkpointing") else checkpointing
        )
        local_blocks = (
            params.local_blocks if hasattr(params, "local_blocks") else local_blocks
        )
        if local_blocks is not None:
            self.local_blocks = [i for i in range(self.num_layers) if i in local_blocks]
        else:
            self.local_blocks = []

        data_grid = params.data_grid if hasattr(params, "data_grid") else "equiangular"
        # self.pretrain_encoding = params.pretrain_encoding if hasattr(params, "pretrain_encoding") else False

        # compute the downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)
        modes_lat_residual = int(self.img_shape[0] // self.residual_filter_factor)
        modes_lon_residual = int(
            self.img_shape[1] // self.residual_filter_factor // 2 + 1
        )

        # no global padding because we removed the horizontal distributed code
        self.padding = (0, 0)

        if residual_filter_factor == 1 and not self.filter_residual:
            self.residual_filter_down = nn.Identity()
            self.residual_filter_up = nn.Identity()
        else:
            self.residual_filter_down = th.RealSHT(
                *self.img_shape,
                lmax=modes_lat_residual,
                mmax=modes_lon_residual,
                grid=data_grid,
            ).float()
            self.residual_filter_up = th.InverseRealSHT(
                *self.img_shape,
                lmax=modes_lat_residual,
                mmax=modes_lon_residual,
                grid=data_grid,
            ).float()

        if self.filter_output:
            self.filter_output_down = th.RealSHT(
                *self.img_shape,
                lmax=modes_lat,
                mmax=modes_lon,
                grid=data_grid,
            ).float()
            self.filter_output_up = th.InverseRealSHT(
                *self.img_shape,
                lmax=modes_lat,
                mmax=modes_lon,
                grid=data_grid,
            ).float()
        else:
            self.filter_output_down = nn.Identity()
            self.filter_output_up = nn.Identity()

        # prepare the spectral transforms
        if self.spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # set up
            self.trans_down = sht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.itrans_up = isht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # effective image size:
            self.img_shape_eff = (
                self.img_shape[0] + self.padding[0],
                self.img_shape[1] + self.padding[1],
            )
            self.img_shape_loc = (
                self.img_shape_eff[0],
                self.img_shape_eff[1],
            )

            self.trans_down = fft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.trans = fft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans = ifft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        self.img_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
        self.img_shape_eff = (self.trans_down.nlat, self.trans_down.nlon)
        self.h_loc = self.itrans.nlat
        self.w_loc = self.itrans.nlon

        # determine activation function
        if self.activation_function == "relu":
            self.activation_function = nn.ReLU
        elif self.activation_function == "gelu":
            self.activation_function = nn.GELU
        elif self.activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {self.activation_function}")

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(
                nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            )
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

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
                context_config=context_config,
                filter_type=block_filter_type,
                operator_type=self.operator_type,
                mlp_ratio=self.mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                sparsity_threshold=sparsity_threshold,
                global_layer_norm=self.global_layer_norm,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                rank=self.rank,
                factorization=self.factorization,
                separable=self.separable,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
                filter_residual=self.filter_residual,
            )

            self.blocks.append(block)

        # decoder
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(
                nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            )
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)

        # learned position embedding
        if self.pos_embed:
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.embed_dim, self.img_shape_loc[0], self.img_shape_loc[1]
                )
            )
            # self.pos_embed = nn.Parameter( torch.zeros(1, self.embed_dim, self.img_shape_eff[0], self.img_shape_eff[1]) )
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, ConditionalLayerNorm):
            m.reset_parameters()

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

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            # old way of treating unequally shaped weights
            if self.img_shape_loc != self.img_shape_eff:
                xp = torch.zeros_like(x)
                xp[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] = (
                    x[..., : self.img_shape_loc[0], : self.img_shape_loc[1]]
                    + self.pos_embed
                )
                x = xp
            else:
                x = x + self.pos_embed

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
