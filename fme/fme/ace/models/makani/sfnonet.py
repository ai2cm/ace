# flake8: noqa
# Copied from tag v0.1.0 https://github.com/NVIDIA/modulus-makani/tree/0218658e925a3c88c6513022a33ecc1647735c2e
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

# for annotation of models
import torch
import torch.nn as nn

# get spectral transforms from torch_harmonics
import torch_harmonics as th
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint

from .layers import MLP, DropPath, EncoderDecoder, InverseRealFFT2, RealFFT2
from .spectral_convolution import FactorizedSpectralConv, SpectralConv


class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        hidden_size_factor=1,
        factorization=None,
        rank=1.0,
        separable=False,
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        drop_rate=0.0,
        gain=1.0,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear":
            raise NotImplementedError("legacy nonlinear complex filter is removed")

        elif filter_type == "linear" and factorization is None:
            self.filter = SpectralConv(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                separable=separable,
                bias=bias,
                gain=gain,
            )

        elif filter_type == "linear" and factorization is not None:
            self.filter = FactorizedSpectralConv(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=bias,
                gain=gain,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        mlp_drop_rate=0.0,
        path_drop_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.Identity, nn.Identity),
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,
        use_mlp=False,
        comm_feature_inp_name=None,
        comm_feature_hidden_name=None,
        complex_activation="real",
        spectral_layers=1,
        bias=False,
        final_activation=False,
        checkpointing=0,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        # determine some shapes
        self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
        self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        if act_layer == nn.Identity:
            gain_factor = 1.0
        else:
            gain_factor = 2.0

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
            gain_factor /= 2.0
            nn.init.normal_(
                self.inner_skip.weight, std=math.sqrt(gain_factor / embed_dim)
            )
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()
            gain_factor /= 2.0
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            hidden_size_factor=mlp_ratio,
            factorization=factorization,
            rank=rank,
            separable=separable,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            bias=bias,
            drop_rate=path_drop_rate,
            gain=gain_factor,
        )

        self.act_layer0 = act_layer()

        # norm layer
        self.norm1 = norm_layer[1]()

        if final_activation and act_layer != nn.Identity:
            gain_factor = 2.0
        else:
            gain_factor = 1.0

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
            gain_factor /= 2.0
            torch.nn.init.normal_(
                self.outer_skip.weight, std=math.sqrt(gain_factor / embed_dim)
            )
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
            gain_factor /= 2.0
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")

        if use_mlp:
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLP(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=mlp_drop_rate,
                drop_type="features",
                comm_inp_name=comm_feature_inp_name,
                comm_hidden_name=comm_feature_hidden_name,
                checkpointing=checkpointing,
                gain=gain_factor,
            )

        # dropout
        self.drop_path = (
            DropPath(path_drop_rate) if path_drop_rate > 0.0 else nn.Identity()
        )

        if final_activation:
            self.act_layer1 = act_layer()

    def forward(self, x):
        """
        Updated FNO block
        """

        x, residual = self.filter(x)

        x = self.norm0(x)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer0"):
            x = self.act_layer0(x)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.norm1(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        if hasattr(self, "act_layer1"):
            x = self.act_layer1(x)

        return x


class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SFNO implementation as in Bonev et al.; Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    """

    def __init__(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        filter_type="linear",
        operator_type="dhconv",
        inp_shape=(721, 1440),
        out_shape=(721, 1440),
        scale_factor=8,
        inp_chans=2,
        out_chans=2,
        embed_dim=32,
        num_layers=4,
        repeat_layers=1,
        use_mlp=True,
        mlp_ratio=2.0,
        encoder_ratio=1,
        decoder_ratio=1,
        activation_function="gelu",
        encoder_layers=1,
        pos_embed="none",
        pos_drop_rate=0.0,
        path_drop_rate=0.0,
        mlp_drop_rate=0.0,
        normalization_layer="instance_norm",
        max_modes=None,
        hard_thresholding_fraction=1.0,
        big_skip=True,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_activation="real",
        spectral_layers=3,
        bias=False,
        checkpointing=0,
        **kwargs,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.repeat_layers = repeat_layers
        self.big_skip = big_skip
        self.checkpointing = checkpointing

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # initialize spectral transforms
        self._init_spectral_transforms(
            spectral_transform,
            model_grid_type,
            sht_grid_type,
            hard_thresholding_fraction,
            max_modes,
        )

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # set up encoder
        self.encoder = EncoderDecoder(
            num_layers=encoder_layers,
            input_dim=self.inp_chans,
            output_dim=self.embed_dim,
            hidden_dim=int(encoder_ratio * self.embed_dim),
            act_layer=activation_function,
            input_format="nchw",
        )
        fblock_mlp_inp_name = "fin"
        fblock_mlp_hidden_name = "fout"

        # dropout
        self.pos_drop = (
            nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0.0 else nn.Identity()
        )
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # pick norm layer
        if normalization_layer == "layer_norm":
            raise NotImplementedError("requires makani distributed libraries")
        elif normalization_layer == "instance_norm":
            norm_layer_inp = partial(
                nn.InstanceNorm2d,
                num_features=embed_dim,
                eps=1e-6,
                affine=True,
                track_running_stats=False,
            )
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(
                f"Error, normalization {normalization_layer} not implemented."
            )

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):
            first_layer = i == 0
            last_layer = i == num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "none"
            outer_skip = "linear"

            if first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_mid, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                mlp_drop_rate=mlp_drop_rate,
                path_drop_rate=dpr[i],
                act_layer=activation_function,
                norm_layer=norm_layer,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=use_mlp,
                comm_feature_inp_name=fblock_mlp_inp_name,
                comm_feature_hidden_name=fblock_mlp_hidden_name,
                rank=rank,
                factorization=factorization,
                separable=separable,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                bias=bias,
                checkpointing=checkpointing,
            )

            self.blocks.append(block)

        self.decoder = EncoderDecoder(
            num_layers=encoder_layers,
            input_dim=embed_dim,
            output_dim=self.out_chans,
            hidden_dim=int(decoder_ratio * embed_dim),
            act_layer=activation_function,
            gain=0.5 if self.big_skip else 1.0,
            input_format="nchw",
        )

        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(
                self.inp_chans, self.out_chans, 1, bias=False
            )
            self.residual_transform.weight.is_shared_mp = ["spatial"]
            self.residual_transform.weight.sharded_dims_mp = [None, None, None, None]
            scale = math.sqrt(0.5 / self.inp_chans)
            nn.init.normal_(self.residual_transform.weight, mean=0.0, std=scale)

        # learned position embedding
        if pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1])
            )
            # information about how tensors are shared / sharded across ranks
            self.pos_embed.is_shared_mp = (
                []
            )  # no reduction required since pos_embed is already serial
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            with torch.no_grad():
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        elif pos_embed == "frequency":
            lmax_loc = self.itrans_up.lmax
            mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(
                torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc), diagonal=0)
            )
            ccoeffs = nn.Parameter(
                torch.tril(
                    torch.randn(1, embed_dim, lmax_loc, mmax_loc - 1), diagonal=-1
                )
            )
            with torch.no_grad():
                nn.init.trunc_normal_(rcoeffs, std=0.02)
                nn.init.trunc_normal_(ccoeffs, std=0.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"
            self.pos_embed.is_shared_mp = []
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]

        elif pos_embed == "none" or pos_embed == "None" or pos_embed is None:
            pass
        else:
            raise ValueError("Unknown position embedding type")

    @torch.jit.ignore
    def _init_spectral_transforms(
        self,
        spectral_transform="sht",
        model_grid_type="equiangular",
        sht_grid_type="legendre-gauss",
        hard_thresholding_fraction=1.0,
        max_modes=None,
    ):
        """
        Initialize the spectral transforms based on the maximum number of modes to keep. Handles the computation
        of local image shapes and domain parallelism, based on the
        """

        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

        # prepare the spectral transforms
        if spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # set up
            self.trans_down = sht_handle(
                *self.inp_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type
            ).float()
            self.itrans_up = isht_handle(
                *self.out_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type
            ).float()

        elif spectral_transform == "fft":
            fft_handle = RealFFT2
            ifft_handle = InverseRealFFT2

            self.trans_down = fft_handle(
                *self.inp_shape, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.out_shape, lmax=modes_lat, mmax=modes_lon
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
        self.inp_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
        self.out_shape_loc = (self.itrans_up.nlat, self.itrans_up.nlon)
        self.h_loc = self.itrans.nlat
        self.w_loc = self.itrans.nlon

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for r in range(self.repeat_layers):
            for blk in self.blocks:
                if self.checkpointing >= 3:
                    x = checkpoint(blk, x, use_reentrant=False)
                else:
                    x = blk(x)

        return x

    def forward(self, x):
        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x.to(torch.float32)
                with amp.autocast(enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    residual = self.itrans_up(residual)
                    residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x, use_reentrant=False)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            if self.pos_embed.type == "frequency":
                pos_embed = torch.stack(
                    [
                        self.pos_embed[0],
                        nn.functional.pad(self.pos_embed[1], (1, 0), "constant", 0),
                    ],
                    dim=-1,
                )
                with amp.autocast(enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # add pos embed
            x = x + pos_embed

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x, use_reentrant=False)
        else:
            x = self.decoder(x)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        return x
