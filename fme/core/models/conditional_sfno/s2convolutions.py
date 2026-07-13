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

import math
import torch
import torch.nn as nn

from fme.core.benchmark.timer import NullTimer, Timer
from fme.core.distributed.distributed import Distributed

from .activations import ComplexReLU

from .contractions import (
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
)


def validate_spectral_ratio(
    spectral_ratio: float,
    channels: int,
    num_groups: int,
    *,
    channels_name: str = "embed_dim",
    num_groups_name: str = "filter_num_groups",
    filter_type: str | None = None,
    preserves_global_mean: bool = False,
    global_mean_arg_name: str = "filter_preserves_global_mean",
    local_blocks: bool = False,
) -> int:
    """Validate ``spectral_ratio`` and return the resulting spectral channel count.

    ``spectral_ratio`` controls the bottleneck width of the linear spectral
    filter: the SHT and per-mode complex weight operate on
    ``round(channels * spectral_ratio)`` channels. This helper centralizes the
    config-time validation shared by ``SpectralConvS2``, ``SFNONetConfig``, and
    the ``NoiseConditionedSFNOBuilder`` registry config so the rules live in one
    place; each caller still invokes it so its own API stays self-validating.

    Args:
        spectral_ratio: fraction of ``channels`` participating in the filter;
            must be in (0, 1].
        channels: full latent channel width (``embed_dim`` / ``in_channels``).
        num_groups: number of filter groups the spectral channels must divide.
        channels_name: name of ``channels`` to use in error messages.
        num_groups_name: name of ``num_groups`` to use in error messages.
        filter_type: if given, ``spectral_ratio < 1`` requires ``"linear"``.
        preserves_global_mean: accepted for API symmetry with the callers but
            no longer gates ``spectral_ratio``. Preserving the global mean with
            ``spectral_ratio < 1`` is supported: ``SpectralConvS2`` restores the
            input's l=0 coefficient in the full-channel grid space, outside the
            reduced-channel bottleneck (see ``SpectralConvS2.forward``).
        global_mean_arg_name: name of the global-mean flag for error messages.
        local_blocks: if True, ``spectral_ratio < 1`` is rejected.

    Returns:
        The number of spectral channels, ``round(channels * spectral_ratio)``.
    """
    if not 0.0 < spectral_ratio <= 1.0:
        raise ValueError(f"spectral_ratio must be in (0, 1], got {spectral_ratio}.")
    spectral_channels = round(channels * spectral_ratio)
    if spectral_ratio < 1.0:
        if filter_type is not None and filter_type != "linear":
            raise NotImplementedError(
                "spectral_ratio < 1 is only supported for filter_type='linear', "
                f"got filter_type='{filter_type}'."
            )
        if local_blocks:
            raise NotImplementedError(
                "spectral_ratio < 1 is not supported with local_blocks, since "
                "local (DISCO) blocks have no spectral filter to bottleneck."
            )
        if spectral_channels < 1:
            raise ValueError(
                f"spectral_ratio={spectral_ratio} with {channels_name}={channels} "
                "produces fewer than 1 spectral channel."
            )
        if spectral_channels % num_groups != 0:
            raise ValueError(
                f"spectral_ratio={spectral_ratio} with {channels_name}={channels} "
                f"yields {spectral_channels} spectral channels, which is not "
                f"divisible by {num_groups_name}={num_groups}."
            )
    return spectral_channels


def _latitude_quadrature_weights(transform) -> torch.Tensor:
    """Per-latitude quadrature weights of a transform's l=0, m=0 basis.

    For a torch-harmonics ``RealSHT`` the ``(m=0, l=0)`` row of the precomputed
    ``weights`` buffer is exactly the latitude quadrature that forms the global
    mean coefficient: ``sht(f)[..., 0, 0] == sqrt(4*pi) * <f>`` where ``<f>`` is
    the mean of ``f`` over the sphere weighted by these values (uniform over
    longitude). Returning them lets us reproduce the l=0 coefficient as a cheap
    grid-space weighted mean instead of a full spherical harmonic transform.

    For a transform without such weights (e.g. the periodic ``RealFFT2``), the
    l=0 mode is the unweighted mean, so uniform weights are returned.
    """
    weights = getattr(transform, "weights", None)
    if weights is not None:
        return weights[0, 0, :].detach().clone().float()
    return torch.ones(transform.nlat)


@torch.jit.script
def _contract_lora(
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
    tmp = torch.einsum("gxri,bgixy->bgxry", lora_A, x)
    out = torch.einsum("gxor,bgxry->bgoxy", lora_B, tmp)
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
    return torch.einsum("bgixy,gxoi->bgoxy", xc, wc)


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
        bias=False,
        filter_residual: bool = False,
        lora_rank: int = 0,
        lora_alpha: float | None = None,
        preserve_global_mean: bool = False,
        spectral_ratio: float = 1.0,
    ):  # pragma: no cover
        super(SpectralConvS2, self).__init__()

        if in_channels != out_channels:
            raise NotImplementedError(
                "Currently only in_channels == out_channels is supported."
            )

        spectral_channels = validate_spectral_ratio(
            spectral_ratio,
            in_channels,
            num_groups,
            channels_name="in_channels",
            num_groups_name="num_groups",
            preserves_global_mean=preserve_global_mean,
            global_mean_arg_name="preserve_global_mean",
        )

        assert in_channels % num_groups == 0
        assert out_channels % num_groups == 0
        self.num_groups = num_groups

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self._round_trip_residual = filter_residual or (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        dist = Distributed.get_instance()
        l_slice, _ = dist.get_local_slices((self.modes_lat, self.modes_lon))
        l_start, l_stop, _ = l_slice.indices(self.modes_lat)
        self.modes_lat_local = l_stop - l_start
        self._l_slice = l_slice
        self._preserve_global_mean = preserve_global_mean
        self._has_global_mean = l_start == 0
        # When spectral_ratio < 1 the l=0 mode lives in the reduced-channel
        # bottleneck, so it cannot be preserved channel-for-channel in the
        # spectral domain. Instead the input's per-channel global mean is
        # restored in full-channel grid space after post_proj (see forward).
        self._reduced_global_mean = preserve_global_mean and spectral_ratio < 1.0
        if self._reduced_global_mean and in_channels != out_channels:
            raise NotImplementedError(
                "preserve_global_mean with spectral_ratio < 1 requires "
                "in_channels == out_channels so the l=0 coefficient can be "
                f"swapped channel-for-channel, got in_channels={in_channels} "
                f"and out_channels={out_channels}."
            )
        if self._reduced_global_mean:
            # The grid-space global-mean swap reads per-latitude quadrature
            # weights from the transforms' ``weights`` buffer, which is only the
            # full-grid, m=0 quadrature on a single spatial rank. Under spatial
            # parallelism the buffer's latitude axis stays global (so it does
            # not match the lat-sharded grid input) and its leading axis is the
            # local-m partition (so ``[0, 0]`` is not the m=0 row off rank 0);
            # the swap would then crash under H-parallelism and silently compute
            # the wrong global mean under W-parallelism. The spectral_ratio == 1
            # path preserves the global mean correctly under spatial parallelism
            # via the l=0 coefficient on the l_start == 0 rank; matching that
            # here is left as follow-up.
            dist.require_no_spatial_parallelism(
                "preserve_global_mean with spectral_ratio < 1 does not support "
                "spatial parallelism; the grid-space global-mean swap needs the "
                "single-rank, full-grid m=0 quadrature weights."
            )
            # Latitude quadrature weights for the input (forward) grid and the
            # output (inverse) grid, uniform over longitude. inverse_transform
            # is an InverseRealSHT and carries no forward quadrature, so build a
            # matching RealSHT to read the output grid's weights.
            out_transform = dist.get_sht(
                inverse_transform.nlat,
                inverse_transform.nlon,
                lmax=inverse_transform.lmax,
                mmax=inverse_transform.mmax,
                grid=inverse_transform.grid,
            )
            self.register_buffer(
                "_gm_lat_weights_in",
                _latitude_quadrature_weights(forward_transform).reshape(1, 1, -1, 1),
                persistent=False,
            )
            self.register_buffer(
                "_gm_lat_weights_out",
                _latitude_quadrature_weights(out_transform).reshape(1, 1, -1, 1),
                persistent=False,
            )

        # When spectral_ratio < 1, the SHT and per-mode complex weight operate
        # on a reduced spectral_channels = round(in_channels * spectral_ratio)
        # latent width via real Conv1x1 projections before forward_transform
        # and after inverse_transform. By commutativity of channel-wise linear
        # maps with spatial transforms, this is equivalent to factoring each
        # per-mode C x C complex weight as Q W'_l P with shared P, Q across l.
        self.spectral_ratio = spectral_ratio
        self.spectral_channels = spectral_channels
        if spectral_ratio < 1.0:
            self.pre_proj: nn.Conv2d | None = nn.Conv2d(
                in_channels, spectral_channels, kernel_size=1, bias=False
            )
            self.post_proj: nn.Conv2d | None = nn.Conv2d(
                spectral_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.pre_proj = None
            self.post_proj = None

        scale = math.sqrt(1 / (spectral_channels)) * torch.ones(self.modes_lat, 1, 1, 2)
        # seemingly the first weight is not really complex, so we need to account for that
        scale[0, :] *= math.sqrt(2.0)

        weight_shape = [
            num_groups,
            self.modes_lat,
            spectral_channels // num_groups,
            spectral_channels // num_groups,
        ]

        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, 2))

        self.lora_rank = lora_rank
        if lora_rank > 0:
            self.lora_A = nn.Parameter(
                scale
                * torch.randn(
                    num_groups,
                    self.modes_lat,
                    lora_rank,
                    spectral_channels // num_groups,
                    2,
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    num_groups,
                    self.modes_lat,
                    spectral_channels // num_groups,
                    lora_rank,
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
        self.register_load_state_dict_pre_hook(self._reorder_weight_dims)

    @staticmethod
    def _reorder_weight_dims(
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

        # check if the weight is in the old shape (group, in_channels, out_channels, nlat, 2)
        if weight.ndim == 5 and weight.shape == (
            module.num_groups,
            module.spectral_channels // module.num_groups,
            module.spectral_channels // module.num_groups,
            module.modes_lat,
            2,
        ):
            # reorder to (group, nlat, out_channels // group, in_channels // group, 2)
            weight = weight.permute(0, 3, 2, 1, 4)
            state_dict[key] = weight

        lora_A = state_dict.get(prefix + "lora_A", None)
        if (
            lora_A is not None
            and lora_A.ndim == 5
            and lora_A.shape
            == (
                module.num_groups,
                module.spectral_channels // module.num_groups,
                module.lora_rank,
                module.modes_lat,
                2,
            )
        ):
            lora_A = lora_A.permute(0, 3, 2, 1, 4)
            state_dict[prefix + "lora_A"] = lora_A

        lora_B = state_dict.get(prefix + "lora_B", None)
        if (
            lora_B is not None
            and lora_B.ndim == 5
            and lora_B.shape
            == (
                module.num_groups,
                module.lora_rank,
                module.spectral_channels // module.num_groups,
                module.modes_lat,
                2,
            )
        ):
            lora_B = lora_B.permute(0, 3, 2, 1, 4)
            state_dict[prefix + "lora_B"] = lora_B

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
            module.spectral_channels,
            module.spectral_channels,
            module.modes_lat,
            2,
        )

        if weight.shape == ungrouped_shape:
            state_dict[key] = weight.view(1, *ungrouped_shape)

    def _swap_global_mean(
        self, output: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        """Set ``output``'s per-channel l=0 (global-mean) coefficient to ``source``'s.

        Adds a spatially-constant, per-channel shift so that, under the
        transform's spherical quadrature, ``output``'s area-weighted mean equals
        ``source``'s. Because the inverse SHT of an l=0-only field is a constant
        equal to that weighted mean, this is exactly equivalent to copying the
        l=0 coefficient, but is done in the full-channel grid space rather than
        inside the reduced-channel spectral bottleneck. Autograd-safe. Only
        reached on a single spatial rank (see the spatial-parallelism guard in
        ``__init__``); ``Distributed.weighted_mean`` then reduces over that one
        rank.

        All operands are aligned to ``output.device`` before combining. The
        transforms these buffers come from can be built on different devices
        (``get_real_sht`` moves to ``get_device``, the internally-built
        ``out_transform`` does not), and ``source`` is the pre-transform input,
        which may sit on the host while the transformed ``output`` is on device.
        The moves are no-ops once the whole module and its inputs share a device.
        """
        dist = Distributed.get_instance()
        device = output.device
        source_mean = dist.weighted_mean(
            source.to(device),
            self._gm_lat_weights_in.to(device),
            dim=(-2, -1),
            keepdim=True,
        )
        output_mean = dist.weighted_mean(
            output, self._gm_lat_weights_out.to(device), dim=(-2, -1), keepdim=True
        )
        return output + (source_mean - output_mean)

    def forward(self, x, timer: Timer = NullTimer()):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.float()
        # Full-channel input in grid space, kept for the reduced-bottleneck
        # global-mean swap below (before any pre_proj down-projection).
        global_mean_source = x if self._reduced_global_mean else None

        with torch.amp.autocast("cuda", enabled=False):
            if self.pre_proj is not None:
                with timer.child("pre_proj"):
                    x = self.pre_proj(x)
            with timer.child("forward_transform"):
                x = self.forward_transform(x.float())
            if self._round_trip_residual:
                with timer.child("round_trip_residual"):
                    x = x.contiguous()
                    residual = self.inverse_transform(x)
                    if self.post_proj is not None:
                        residual = self.post_proj(residual)
                    residual = residual.to(dtype)

        B, C, H, W = x.shape
        assert C % self.num_groups == 0
        x = x.reshape(B, self.num_groups, C // self.num_groups, H, W)

        # Slice global weights to the local spectral partition (lat only).
        weight_local = self.weight[:, self._l_slice]
        if self.lora_A is not None and self.lora_B is not None:
            with timer.child("lora_update"):
                lora_update = _contract_lora(
                    self.lora_A,
                    self.lora_B,
                    x[..., : self.modes_lat, : self.modes_lon],
                )
        else:
            lora_update = 0.0

        with timer.child("dhconv"):
            xp = torch.zeros_like(x)
            xp[..., : self.modes_lat_local, : self.modes_lon] = _contract_dhconv(
                x[..., : self.modes_lat_local, : self.modes_lon],
                weight_local,
            )
            xp = xp + self.lora_scaling * lora_update
            if (
                self._preserve_global_mean
                and not self._reduced_global_mean
                and self._has_global_mean
            ):
                xp = torch.cat([x[..., :1, :], xp[..., 1:, :]], dim=-2)
            xp = xp.reshape(B, self.spectral_channels, H, W)
            x = xp.contiguous()

        with torch.amp.autocast("cuda", enabled=False):
            with timer.child("inverse_transform"):
                x = self.inverse_transform(x)
            if self.post_proj is not None:
                with timer.child("post_proj"):
                    x = self.post_proj(x)
            if global_mean_source is not None:
                with timer.child("preserve_global_mean"):
                    x = self._swap_global_mean(x, global_mean_source)

        if hasattr(self, "bias"):
            with timer.child("add_bias"):
                x = x + self.bias

        x = x.type(dtype)

        return x, residual


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
