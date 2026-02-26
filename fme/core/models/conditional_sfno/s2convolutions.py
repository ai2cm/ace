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

# import FactorizedTensor from tensorly for tensorized operations
import math
import torch
import torch.nn as nn

import torch_harmonics as th
import torch_harmonics.distributed as thd

from fme.core.benchmark.timer import NullTimer, Timer

# import convenience functions for factorized tensors
from .activations import ComplexReLU

# for the experimental module
from .contractions import (
    _contract_localconv_fwd,
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
    real_mul2d_fwd,
    real_muladd2d_fwd,
)


@torch.jit.script
def _contract_lora(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    x: torch.Tensor,
):
    """
    Performs LoRA update contraction.

    Args:
        lora_A: LoRA A matrix of shape (group, in_channels, rank, nlat, 2)
        lora_B: LoRA B matrix of shape (group, rank, out_channels, nlat, 2)
        x: Complex input tensor of shape
            (batch_size, group, in_channels, nlat, nlon)

    Returns:
        Complex output tensor of shape (batch_size, group, out_channels, nlat, nlon)
    """
    lora_A = torch.view_as_complex(lora_A)
    lora_B = torch.view_as_complex(lora_B)
    return torch.einsum("girx,grox,bgixy->bgoxy", lora_A, lora_B, x)


@torch.jit.script
def _contract_lora_lowmem(
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    x: torch.Tensor,
):
    """
    Performs LoRA update contraction.

    Args:
        lora_A: LoRA A matrix of shape (group, nlat, rank, in_channels, 2)
        lora_B: LoRA B matrix of shape (group, nlat, out_channels, rank, 2)
        x: Complex input tensor of shape
            (batch_size, group, in_channels, nlat, nlon)

    Returns:
        Complex output tensor of shape (batch_size, group, out_channels, nlat, nlon)
    """
    lora_A = torch.view_as_complex(lora_A)
    lora_B = torch.view_as_complex(lora_B)
    tmp = torch.einsum("girx,bgixy->bgxry", lora_A, x)
    out = torch.einsum("grox,bgxry->bgoxy", lora_B, tmp)
    return out


@torch.jit.script
def _contract_dhconv(
    xc: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """
    Performs a complex Driscoll-Healy style convolution operation between two tensors
    'a' and 'b'.

    Args:
        xc: Complex input tensor of shape (batch_size, group, in_channels, nlat, nlon)
        weight: Weight tensor of shape (group, in_channels, out_channels, nlat, 2)

    Returns:
        Complex output tensor of shape (batch_size, group, out_channels, nlat, nlon)
    """
    wc = torch.view_as_complex(weight)
    return torch.einsum("bgixy,giox->bgoxy", xc, wc)


class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on
    the two-sphere S2 using the Spherical Harmonic Transforms in torch-harmonics, but
    supports convolutions on the periodic domain via the RealFFT2 and InverseRealFFT2
    wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        num_groups: int = 1,
        scale="auto",
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
        use_tensorly=True,
        filter_residual: bool = False,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
    ):  # pragma: no cover
        super(SpectralConvS2, self).__init__()
        if operator_type != "dhconv":
            raise NotImplementedError(
                "Only 'dhconv' operator type is currently supported."
            )
        if factorization is not None:
            raise NotImplementedError(
                "Factorizations other than None are not currently supported."
            )
        if use_tensorly:
            raise NotImplementedError(
                "Tensorly-based implementation is not currently supported."
            )
        if separable:
            raise NotImplementedError(
                "Separable convolutions are not currently supported."
            )

        if in_channels != out_channels:
            raise NotImplementedError(
                "Currently only in_channels == out_channels is supported."
            )

        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0
        self.num_groups = num_groups

        if in_channels != out_channels:
            raise NotImplementedError(
                "Currently only in_channels == out_channels is supported."
            )

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self._round_trip_residual = filter_residual or (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )
        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "Dense"  # No factorization

        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        if isinstance(self.inverse_transform, thd.DistributedInverseRealSHT):
            self.modes_lat_local = self.inverse_transform.lmax_local
            self.modes_lon_local = self.inverse_transform.mmax_local
            self.lpad_local = self.inverse_transform.lpad_local
            self.mpad_local = self.inverse_transform.mpad_local
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon
            self.lpad = 0
            self.mpad = 0

        if scale == "auto":
            scale = math.sqrt(1 / (in_channels)) * torch.ones(self.modes_lat_local, 2)
            # seemingly the first weight is not really complex, so we need to account for that
            scale[0, :] *= math.sqrt(2.0)

        weight_shape = [
            num_groups,
            in_channels // num_groups,
            out_channels // num_groups,
            self.modes_lat_local,
        ]

        assert factorization == "ComplexDense"
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))
        self.weight.is_shared_mp = ["matmul", "w"]

        if lora_rank > 0:
            self.lora_A = nn.Parameter(
                scale
                * torch.randn(
                    num_groups,
                    in_channels // num_groups,
                    lora_rank,
                    self.modes_lat_local,
                    2,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    num_groups,
                    lora_rank,
                    out_channels // num_groups,
                    self.modes_lat_local,
                    2,
                )
            )
            self.lora_alpha = lora_alpha if lora_alpha is not None else lora_rank
            self.lora_scaling = self.lora_alpha / lora_rank
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_scaling = 0.0

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels

        # rewrite old checkpoints on load
        self.register_load_state_dict_pre_hook(self._add_singleton_group_dim)

    @staticmethod
    def _add_singleton_group_dim(
        module: "SpectralConvS2",
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        key = prefix + "weight"
        if key not in state_dict:
            return

        weight = state_dict[key]

        ungrouped_shape = (
            module.in_channels,
            module.out_channels,
            module.modes_lat_local,
            2,
        )

        if weight.shape == ungrouped_shape:
            state_dict[key] = weight.view(1, *ungrouped_shape)

    def forward(self, x, timer: Timer = NullTimer()):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.float()

        with torch.amp.autocast("cuda", enabled=False):
            with timer.child("forward_transform"):
                x = self.forward_transform(x.float())
            if self._round_trip_residual:
                with timer.child("round_trip_residual"):
                    x = x.contiguous()
                    residual = self.inverse_transform(x)
                    residual = residual.to(dtype)

        B, C, H, W = x.shape
        assert C % self.num_groups == 0
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)

        if self.lora_A is not None and self.lora_B is not None:
            with timer.child("lora_update"):
                lora_update = _contract_lora_lowmem(
                    self.lora_A,
                    self.lora_B,
                    x[..., : self.modes_lat_local, : self.modes_lon_local],
                )
        else:
            lora_update = 0.0

        with timer.child("dhconv"):
            xp = torch.zeros_like(x)
            xp[..., : self.modes_lat_local, : self.modes_lon_local] = _contract_dhconv(
                x[..., : self.modes_lat_local, : self.modes_lon_local],
                self.weight,
            )
            xp = xp + self.lora_scaling * lora_update
            xp = xp.reshape(B, self.out_channels, H, W)
            x = xp.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            with timer.child("inverse_transform"):
                x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            with timer.child("add_bias"):
                x = x + self.bias

        x = x.type(dtype)

        return x, residual


class LocalConvS2(nn.Module):
    """
    S2 Convolution according to Driscoll & Healy
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        nradius=120,
        scale="auto",
        bias=False,
    ):  # pragma: no cover
        super(LocalConvS2, self).__init__()

        if scale == "auto":
            scale = 1 / (in_channels * out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nradius = nradius

        self.forward_transform = forward_transform
        self.zonal_transform = th.RealSHT(
            forward_transform.nlat,
            1,
            lmax=forward_transform.lmax,
            mmax=1,
            grid=forward_transform.grid,
        ).float()
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax
        self.output_dims = (self.inverse_transform.nlat, self.inverse_transform.nlon)

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        self.weight = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, nradius, 1)
        )

        self._contract = _contract_localconv_fwd

        if bias:
            self.bias = nn.Parameter(
                scale * torch.randn(1, out_channels, *self.output_dims)
            )

    def forward(self, x, timer: Timer = NullTimer()):  # pragma: no cover
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape

        with torch.amp.autocast("cuda", enabled=False):
            f = torch.zeros(
                (self.in_channels, self.out_channels, H, 1),
                dtype=x.dtype,
                device=x.device,
            )
            f[..., : self.nradius, :] = self.weight

            x = self.forward_transform(x)
            f = self.zonal_transform(f)[..., :, 0]

            x = torch.view_as_real(x)
            f = torch.view_as_real(f)

        x = self._contract(x, f)
        x = x.contiguous()

        x = torch.view_as_complex(x)

        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x


class SpectralAttentionS2(nn.Module):
    """
    Spherical non-linear FNO layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            w = [self.scale * torch.randn(self.embed_dim, hidden_size, 2)]
            for l in range(1, self.spectral_layers):
                w.append(self.scale * torch.randn(hidden_size, hidden_size, 2))
            self.w = nn.ParameterList(w)

            self.wout = nn.Parameter(
                self.scale * torch.randn(hidden_size, self.embed_dim, 2)
            )

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        elif operator_type == "l-dependant":
            self.mul_add_handle = compl_exp_muladd2d_fwd
            self.mul_handle = compl_exp_mul2d_fwd

            # weights
            w = [
                self.scale * torch.randn(self.modes_lat, self.embed_dim, hidden_size, 2)
            ]
            for l in range(1, self.spectral_layers):
                w.append(
                    self.scale
                    * torch.randn(self.modes_lat, hidden_size, hidden_size, 2)
                )
            self.w = nn.ParameterList(w)

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.wout = nn.Parameter(
                self.scale * torch.randn(self.modes_lat, hidden_size, self.embed_dim, 2)
            )

            self.activations = nn.ModuleList([])
            for l in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        else:
            raise ValueError("Unknown operator type")

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        if self.operator_type == "block-separable":
            x = x.permute(0, 3, 1, 2)

        xr = torch.view_as_real(x)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = torch.view_as_complex(xr)
            xr = self.activations[l](xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        x = self.mul_handle(xr, self.wout)

        x = torch.view_as_complex(x)

        if self.operator_type == "block-separable":
            x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, x, timer: Timer = NullTimer()):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.to(torch.float32)

        # FWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        x = x.contiguous()
        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x, residual


class RealSpectralAttentionS2(nn.Module):
    """
    Non-linear SFNO layer using a real-valued NN instead of a complex one
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(RealSpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)

        assert inverse_transform.lmax == self.modes_lat
        assert inverse_transform.mmax == self.modes_lon

        hidden_size = int(hidden_size_factor * self.embed_dim * 2)

        self.mul_add_handle = real_muladd2d_fwd
        self.mul_handle = real_mul2d_fwd

        # weights
        w = [self.scale * torch.randn(2 * self.embed_dim, hidden_size)]
        for l in range(1, self.spectral_layers):
            w.append(self.scale * torch.randn(hidden_size, hidden_size))
        self.w = nn.ParameterList(w)

        self.wout = nn.Parameter(
            self.scale * torch.randn(hidden_size, 2 * self.embed_dim)
        )

        if bias:
            self.b = nn.ParameterList(
                [
                    self.scale * torch.randn(hidden_size, 1, 1)
                    for _ in range(self.spectral_layers)
                ]
            )

        self.activations = nn.ModuleList([])
        for l in range(0, self.spectral_layers):
            self.activations.append(nn.ReLU())

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        xr = torch.view_as_real(x)
        xr = xr.permute(0, 1, 4, 2, 3).reshape(B, C * 2, H, W)

        for l in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[l], self.b[l])
            else:
                xr = self.mul_handle(xr, self.w[l])
            xr = self.activations[l](xr)
            xr = self.drop(xr)

        # final MLP
        xr = self.mul_handle(xr, self.wout)

        xr = xr.reshape(B, C, 2, H, W).permute(0, 1, 3, 4, 2)

        x = torch.view_as_complex(xr)

        return x

    def forward(self, x, timer: Timer = NullTimer()):  # pragma: no cover
        dtype = x.dtype
        x = x.to(torch.float32)

        # FWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.forward_transform(x)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        with torch.amp.autocast("cuda", enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x
