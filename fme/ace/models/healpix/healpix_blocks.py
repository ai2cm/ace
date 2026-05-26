# flake8: noqa
# Copied from https://github.com/NVIDIA/modulus/commit/89a6091bd21edce7be4e0539cbd91507004faf08
# Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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
from typing import Literal, Optional, Sequence, Tuple, Union, cast

import torch as th
import torch.nn as nn

from .healpix_activations import CappedGELUConfig
from .healpix_layers import HEALPixLayer


def _healpix_layer_kwargs(
    enable_nhwc: bool,
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid",
    nside: Optional[int] = None,
) -> dict:
    """
    Build keyword arguments passed to ``HEALPixLayer``.

    Args:
        enable_nhwc: Use channels-last memory format.
        hpx_padding_mode: HEALPix padding backend.
        nside: Native face height/width; included in the dict only when not ``None``.
    """
    out: dict = {"enable_nhwc": enable_nhwc, "hpx_padding_mode": hpx_padding_mode}
    if nside is not None:
        out["nside"] = nside
    return out


# --- Configuration dataclasses ---


@dataclasses.dataclass
class DownsamplingBlockConfig:
    """
    Configuration for the downsampling block (pooling or dealiased strided blur).

    Parameters:
        block_type: One of ``"MaxPool"``, ``"AvgPool"``, or ``"DealiasedDownsample"``.
        pooling: Pooling size for pool blocks.
        stride: Spatial stride for ``DealiasedDownsample`` (power of two).
        enable_nhwc: Use channels-last memory format.
        hpx_padding_mode: HEALPix padding backend passed to child modules.
        nside: Native face height/width for HEALPix padding.
        in_channels: Input channels for ``DealiasedDownsample`` (set by encoder before build).
        resample_filter: 1D filter weights for dealiased blur stages.
    """

    block_type: Literal["MaxPool", "AvgPool", "DealiasedDownsample"]
    pooling: int = 2
    enable_nhwc: bool = False
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Optional[int] = None
    in_channels: Optional[int] = None
    resample_filter: Sequence[float] = dataclasses.field(
        default_factory=lambda: [1.0, 2.0, 1.0]
    )
    stride: int = 2

    def downsample_spatial_factor(self) -> int:
        if self.block_type in ("MaxPool", "AvgPool"):
            return self.pooling
        if self.block_type == "DealiasedDownsample":
            return self.stride
        raise ValueError(f"Unsupported block type: {self.block_type}")

    def build(self) -> nn.Module:
        if self.block_type == "MaxPool":
            return MaxPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "AvgPool":
            return AvgPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "DealiasedDownsample":
            if self.in_channels is None:
                raise ValueError(
                    "DealiasedDownsample requires in_channels "
                    "(set by UNetEncoder before build)"
                )
            return DealiasedDownsample(
                in_channels=self.in_channels,
                resample_filter=self.resample_filter,
                stride=self.stride,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class UpsamplingBlockConfig:
    """
    Configuration for HEALPix upsampling (transpose conv, interpolate+conv, pure
    ``nn.Upsample``, etc.).

    ``block_type`` ``"Interpolate"`` uses ``stride`` as ``nn.Upsample`` ``scale_factor``
    and ``upsample_mode`` as the interpolation mode.

    Parameters:
        block_type: Upsampling implementation to build.
        in_channels: Input channel count for conv-based upsamplers.
        out_channels: Output channel count for conv-based upsamplers.
        stride: Upsampling scale factor (also used as ``nn.Upsample`` scale when applicable).
        kernel_size: Convolution kernel size for ``SmoothedInterpolateConv``.
        dilation: Convolution dilation for ``SmoothedInterpolateConv``.
        upsample_mode: Interpolation mode for smoothed / pure interpolate paths.
        activation: Optional ``CappedGELUConfig`` for transpose-conv upsampling.
        enable_nhwc: Use channels-last memory format.
        hpx_padding_mode: HEALPix padding backend passed to child modules.
        nside: Native face height/width for HEALPix padding at upsample input.
        nside_after: Face height/width after upsampling (``SmoothedInterpolateConv`` only).
        align_corners: Passed to ``nn.Upsample`` when ``block_type`` is ``"Interpolate"``.
        scale_factor: Alias for ``stride`` when set in config.
        mode: Alias for ``upsample_mode`` when set in config.
    """

    block_type: Literal[
        "TransposedConvUpsample",
        "SmoothedInterpolateConv",
        "Interpolate",
    ]
    in_channels: int = 3
    out_channels: int = 1
    stride: int = 2
    kernel_size: int = 3
    dilation: int = 1
    upsample_mode: str = "nearest"
    activation: Optional[CappedGELUConfig] = None
    enable_nhwc: bool = False
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Optional[int] = None
    nside_after: Optional[int] = None
    align_corners: bool = False
    scale_factor: Optional[int] = None
    mode: Optional[str] = None

    def __post_init__(self) -> None:
        if self.scale_factor is not None:
            self.stride = self.scale_factor
        if self.mode is not None:
            self.upsample_mode = self.mode

    def build(self) -> nn.Module:
        if self.block_type == "TransposedConvUpsample":
            return TransposedConvUpsample(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                upsampling=self.stride,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "SmoothedInterpolateConv":
            return SmoothedInterpolateConv(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                scale_factor=self.stride,
                mode=self.upsample_mode,
                activation=self.activation.build() if self.activation else None,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
                nside_after=self.nside_after,
            )
        if self.block_type == "Interpolate":
            if self.align_corners is False:
                return nn.Upsample(
                    scale_factor=self.stride,
                    mode=self.upsample_mode,
                )
            return nn.Upsample(
                scale_factor=self.stride,
                mode=self.upsample_mode,
                align_corners=self.align_corners,
            )
        raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class ConvBlockConfig:
    """
    Configuration for convolutional residual / ConvNeXt style blocks (no spatial resample).

    Parameters:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        dilation: Convolution dilation.
        n_layers: Number of repeated layers (for multi-block types).
        upscale_factor: Channel upscale factor inside ConvNeXt blocks.
        latent_channels: Latent channel width; defaults to ``max(in_channels, out_channels)``.
        activation: Optional ``CappedGELUConfig`` between layers.
        enable_nhwc: Use channels-last memory format.
        hpx_padding_mode: HEALPix padding backend passed to child modules.
        nside: Native face height/width for HEALPix padding.
        block_type: Which block implementation to build.
    """

    in_channels: int = 3
    out_channels: int = 1
    kernel_size: int = 3
    dilation: int = 1
    n_layers: int = 1
    upscale_factor: int = 4
    latent_channels: Optional[int] = None
    activation: Optional[CappedGELUConfig] = None
    enable_nhwc: bool = False
    hpx_padding_mode: Literal["earth2grid", "karlbauer", "isolatitude"] = "earth2grid"
    nside: Optional[int] = None
    block_type: Literal[
        "BasicConvBlock",
        "ConvNeXtBlock",
        "SymmetricConvNeXtBlock",
        "Multi_SymmetricConvNeXtBlock",
    ] = "BasicConvBlock"

    def build(self) -> nn.Module:
        if self.block_type == "BasicConvBlock":
            return BasicConvBlock(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                n_layers=self.n_layers,
                latent_channels=self.latent_channels,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "ConvNeXtBlock":
            if self.latent_channels is None:
                self.latent_channels = 1
            return ConvNeXtBlock(
                in_channels=self.in_channels,
                latent_channels=cast(int, self.latent_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                upscale_factor=self.upscale_factor,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "SymmetricConvNeXtBlock":
            if self.latent_channels is None:
                self.latent_channels = 1
            return SymmetricConvNeXtBlock(
                in_channels=self.in_channels,
                latent_channels=cast(int, self.latent_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                upscale_factor=self.upscale_factor,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        if self.block_type == "Multi_SymmetricConvNeXtBlock":
            if self.latent_channels is None:
                self.latent_channels = 1
            return Multi_SymmetricConvNeXtBlock(
                in_channels=self.in_channels,
                latent_channels=cast(int, self.latent_channels),
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                upscale_factor=self.upscale_factor,
                n_layers=self.n_layers,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                hpx_padding_mode=self.hpx_padding_mode,
                nside=self.nside,
            )
        raise ValueError(f"Unsupported block type: {self.block_type}")


# --- Downsampling modules ---


class MaxPool(nn.Module):
    """Wrapper for applying Max Pooling with HEALPix or other tensor data."""

    def __init__(
        self,
        pooling: int = 2,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Args:
            pooling: ``MaxPool2d`` kernel size (and stride).
            enable_nhwc: Use channels-last memory format.
            hpx_padding_mode: HEALPix padding backend passed to ``HEALPixLayer``.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        self.maxpool = HEALPixLayer(
            layer=nn.MaxPool2d,
            kernel_size=pooling,
            **_healpix_layer_kwargs(enable_nhwc, hpx_padding_mode, nside),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N * 12, C, H, W]``.

        Returns:
            Pooled tensor with halved spatial size per pooling factor.
        """
        return self.maxpool(x)


class AvgPool(nn.Module):
    """Wrapper for applying Average Pooling with HEALPix or other tensor data."""

    def __init__(
        self,
        pooling: int = 2,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Args:
            pooling: ``AvgPool2d`` kernel size (and stride).
            enable_nhwc: Use channels-last memory format.
            hpx_padding_mode: HEALPix padding backend passed to ``HEALPixLayer``.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        self.avgpool = HEALPixLayer(
            layer=nn.AvgPool2d,
            kernel_size=pooling,
            **_healpix_layer_kwargs(enable_nhwc, hpx_padding_mode, nside),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N * 12, C, H, W]``.

        Returns:
            Pooled tensor with halved spatial size per pooling factor.
        """
        return self.avgpool(x)


class DealiasBlurConv2d(nn.Module):
    """Depthwise blur with fixed kernel using functional conv2d."""

    @staticmethod
    def _normalized_depthwise_blur_weights(
        resample_filter: Sequence[float], in_channels: int
    ) -> th.Tensor:
        f = th.as_tensor(list(resample_filter), dtype=th.float32)
        if f.ndim != 1:
            raise ValueError("resample_filter must be 1D")
        m = int(f.numel())
        f2d = f[:, None] * f[None, :]
        f2d = f2d / f2d.sum()
        return f2d.unsqueeze(0).unsqueeze(0).expand(in_channels, 1, m, m).clone()

    def __init__(
        self,
        in_channels: int,
        stride: int = 1,
        resample_filter: Sequence[float] | None = None,
        **kwargs,
    ):
        """
        Args:
            in_channels: Number of input channels (depthwise groups).
            stride: Stride of the depthwise blur convolution.
            resample_filter: 1D separable filter weights used to build the 2D kernel.
            **kwargs: Accepted for API compatibility; not used.
        """
        super().__init__()
        if resample_filter is None:
            resample_filter = [1.0, 2.0, 1.0]
        filt = tuple(float(x) for x in resample_filter)
        if len(filt) < 1:
            raise ValueError("resample_filter must be non-empty")
        if sum(filt) == 0:
            raise ValueError("resample_filter must not sum to zero")

        self.in_channels = in_channels
        self.stride = stride
        self.register_buffer(
            "weight",
            self._normalized_depthwise_blur_weights(filt, in_channels),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N, C, H, W]``.

        Returns:
            Depthwise-blurred tensor with optional strided downsampling.
        """
        return th.nn.functional.conv2d(
            x,
            self.weight.to(device=x.device, dtype=x.dtype),
            bias=None,
            stride=self.stride,
            padding=0,
            groups=self.in_channels,
        )


class DealiasedDownsample(nn.Module):
    """De-aliased downsampling via fixed depthwise blur stages (stride power of 2)."""

    def __init__(
        self,
        in_channels: int = 3,
        resample_filter: Sequence[float] | None = None,
        stride: int = 2,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            resample_filter: 1D filter weights for each blur stage.
            stride: Total downsampling factor (must be a power of two).
            enable_nhwc: Use channels-last memory format.
            hpx_padding_mode: HEALPix padding backend passed to ``HEALPixLayer``.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        if resample_filter is None:
            resample_filter = [1.0, 2.0, 1.0]
        filt = tuple(float(x) for x in resample_filter)
        m = len(filt)
        if m < 1:
            raise ValueError("resample_filter must be non-empty")
        if sum(filt) == 0:
            raise ValueError("resample_filter must not sum to zero")
        if stride < 1 or (math.log2(stride) % 1) != 0:
            raise ValueError("stride must be a positive power of 2")

        n_layers = int(math.log2(stride))
        pool_layers = []
        healpix_kwargs = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside,
        )
        for _ in range(n_layers):
            pool_layers.append(
                HEALPixLayer(
                    layer=DealiasBlurConv2d,
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=m,
                    stride=2,
                    padding=0,
                    groups=in_channels,
                    bias=False,
                    dilation=1,
                    resample_filter=filt,
                    **healpix_kwargs,
                )
            )

        self.pool = nn.Sequential(*pool_layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N * 12, C, H, W]``.

        Returns:
            Dealiased downsampled tensor.
        """
        return self.pool(x)


# --- Upsampling modules ---


class TransposedConvUpsample(nn.Module):
    """Wrapper for upsampling with a transposed convolution using HEALPix or other tensor data.

    This class wraps the `nn.ConvTranspose2d` class to handle tensor data with
    HEALPix or other geometry layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        upsampling: int = 2,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            upsampling: Stride size that will be used for upsampling.
            activation: ModuleConfig for the activation function used in upsampling.
            enable_nhwc: Enable nhwc format, passed to wrapper.
            hpx_padding_mode: HEALPix padding backend passed to wrapper.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        upsampler = []
        # Upsample transpose conv
        upsampler.append(
            HEALPixLayer(
                layer=nn.ConvTranspose2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=upsampling,
                stride=upsampling,
                padding=0,
                **_healpix_layer_kwargs(
                    enable_nhwc,
                    hpx_padding_mode,
                    nside,
                ),
            )
        )
        if activation is not None:
            upsampler.append(activation.build())
        self.upsampler = nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the TransposedConvUpsample layer.

        Args:
            x: The values to upsample.

        Returns:
            th.Tensor: The upsampled values.
        """
        return self.upsampler(x)


class SmoothedInterpolate(nn.Module):
    """Interpolate then apply four-point smoother (zonally uniform signals)."""

    def __init__(
        self,
        in_channels: int = 3,
        scale_factor: int = 2,
        mode: str = "nearest",
        trim_size: int = 0,
    ):
        """
        Args:
            in_channels: Number of channels for the depthwise smoother.
            scale_factor: Interpolation scale factor.
            mode: Interpolation mode passed to ``F.interpolate``.
            trim_size: Border pixels to crop after smoothing (removes edge artifacts).
        """
        super().__init__()

        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.mode = mode
        self.trim_size = trim_size
        self.interp = th.nn.functional.interpolate

        smoother_kernel = th.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        smoother_kernel = smoother_kernel.unsqueeze(0).unsqueeze(0)
        smoother_kernel = smoother_kernel.repeat((in_channels, 1, 1, 1))
        self.register_buffer("smoother_kernel", smoother_kernel)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N, C, H, W]``.

        Returns:
            Upsampled and smoothed tensor, optionally trimmed.
        """
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode)

        x = (
            th.nn.functional.conv2d(
                x,
                self.smoother_kernel,
                padding=0,
                groups=self.in_channels,
            )
            / 4
        )

        if self.trim_size > 0:
            x = x[
                ...,
                self.trim_size : -self.trim_size,
                self.trim_size : -self.trim_size,
            ]

        return x


class SmoothedInterpolateConv(nn.Module):
    """Interpolate with seam padding, smoothing, then Conv2d on HEALPix data."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        kernel_size: int = 3,
        dilation: int = 1,
        scale_factor: int = 2,
        mode: str = "nearest",
        activation: Optional[nn.Module] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
        nside_after: Optional[int] = None,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size after interpolation.
            dilation: Convolution dilation (must be 1 for HEALPix resize).
            scale_factor: Interpolation scale factor.
            mode: Interpolation mode for the smoothed upsample step.
            activation: Optional activation module appended after the conv.
            enable_nhwc: Use channels-last memory format.
            hpx_padding_mode: HEALPix padding backend passed to ``HEALPixLayer``.
            nside: Face height/width before upsampling (isolatitude gather indices).
            nside_after: Face height/width after upsampling for the conv step; required
                when ``nside`` is set and ``hpx_padding_mode`` is ``"isolatitude"``.
        """
        super().__init__()
        if dilation > 1:
            raise ValueError(
                f"dilation > 1 is not supported for HEALPix resize convolutions, got {dilation}"
            )
        if nside is not None and nside_after is None:
            if hpx_padding_mode == "isolatitude":
                raise ValueError(
                    "SmoothedInterpolateConv requires nside_after when nside is set "
                    'and hpx_padding_mode="isolatitude"'
                )
            nside_after = nside
        if (
            nside is not None
            and nside_after is not None
            and nside_after != nside * scale_factor
        ):
            raise ValueError(
                f"nside_after ({nside_after}) must equal nside ({nside}) * "
                f"scale_factor ({scale_factor})"
            )

        trim_size = 1
        healpix_kwargs = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside,
        )
        healpix_kwargs_after = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside_after,
        )

        block = [
            HEALPixLayer(
                layer=SmoothedInterpolate,
                in_channels=in_channels,
                scale_factor=scale_factor,
                mode=mode,
                trim_size=trim_size,
                **healpix_kwargs,
            ),
            HEALPixLayer(
                layer=nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs_after,
            ),
        ]

        if activation is not None:
            block.append(activation)
        self.block = nn.Sequential(*block)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x: Input tensor ``[N * 12, C, H, W]``.

        Returns:
            Upsampled convolved tensor.
        """
        return self.block(x)


# --- Convolution stack modules ---


class BasicConvBlock(nn.Module):
    """Convolution block consisting of n subsequent convolutions and activations."""

    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        kernel_size=3,
        dilation=1,
        n_layers=1,
        latent_channels=None,
        activation=None,
        enable_nhwc=False,
        hpx_padding_mode="earth2grid",
        nside=None,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            kernel_size: Size of the convolutional kernel.
            dilation: Spacing between kernel points, passed to nn.Conv2d.
            n_layers: Number of convolutional layers.
            latent_channels: Number of latent channels.
            activation: ModuleConfig for activation function to use.
            enable_nhwc: Enable nhwc format, passed to wrapper.
            hpx_padding_mode: HEALPix padding backend passed to wrapper.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        if latent_channels is None:
            latent_channels = max(in_channels, out_channels)
        convblock = []
        for n in range(n_layers):
            convblock.append(
                HEALPixLayer(
                    layer=th.nn.Conv2d,
                    in_channels=in_channels if n == 0 else latent_channels,
                    out_channels=out_channels if n == n_layers - 1 else latent_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    **_healpix_layer_kwargs(
                        enable_nhwc,
                        hpx_padding_mode,
                        nside,
                    ),
                )
            )
            if activation is not None:
                convblock.append(activation.build())
        self.convblock = nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the BasicConvBlock.

        Args:
            x: Inputs to the forward pass.

        Returns:
            th.Tensor: Result of the forward pass.
        """
        return self.convblock(x)


class ConvNeXtBlock(nn.Module):
    """A modified ConvNeXt network block as described in the paper
    "A ConvNet for the 21st Century" (https://arxiv.org/pdf/2201.03545.pdf).

    This block consists of a series of convolutional layers with optional activation functions,
    and a residual connection.

    Parameters:
        skip_module: A module to align the input and output channels for the residual connection.
        convblock: A sequential container of convolutional layers with optional activation functions.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Initializes a ConvNeXtBlock instance with specified parameters.

        Args:
            in_channels: Number of input channels.
            latent_channels: Number of latent channels used in the block.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernels.
            dilation: Dilation rate for convolutions.
            upscale_factor: Factor by which to upscale the number of latent channels.
            activation: Configuration for the activation function used between layers.
            enable_nhwc: Whether to enable NHWC format.
            hpx_padding_mode: HEALPix padding backend passed to wrapper.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        healpix_kwargs = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside,
        )

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                **healpix_kwargs,
            )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        # 3x3 convolution maintaining increased channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        # Linear postprocessing
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                **healpix_kwargs,
            )
        )
        self.convblock = nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the ConvNeXtBlock.

        Args:
            x: Input tensor.

        Returns:
            The result of the forward pass.
        """
        return self.skip_module(x) + self.convblock(x)


class DoubleConvNeXtBlock(nn.Module):
    """A variant of the ConvNeXt block that includes two sequential ConvNeXt blocks within a single module.

    Parameters:
        skip_module1: A module to align the input and intermediate channels for the first residual connection.
        skip_module2: A module to align the intermediate and output channels for the second residual connection.
        convblock1: A sequential container of convolutional layers for the first ConvNeXt block.
        convblock2: A sequential container of convolutional layers for the second ConvNeXt block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        latent_channels: int = 1,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Initializes a DoubleConvNeXtBlock instance with specified parameters.

        Args:
            in_channels: Number of input channels (default is 3).
            out_channels: Number of output channels (default is 1).
            kernel_size: Size of the convolutional kernels (default is 3).
            dilation: Dilation rate for convolutions (default is 1).
            upscale_factor: Factor by which to upscale the number of latent channels (default is 4).
            latent_channels: Number of latent channels used in the block (default is 1).
            activation: Configuration for the activation function used between layers (default is None).
            enable_nhwc: Whether to enable NHWC format (default is False).
            hpx_padding_mode: HEALPix padding backend passed to wrapper.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        healpix_kwargs = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside,
        )

        if in_channels == int(latent_channels):
            self.skip_module1 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module1 = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=1,
                **healpix_kwargs,
            )
        if out_channels == int(latent_channels):
            self.skip_module2 = (
                lambda x: x
            )  # Identity-function required in forward pass
        else:
            self.skip_module2 = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=out_channels,
                kernel_size=1,
                **healpix_kwargs,
            )

        # 1st ConvNeXt block, the output of this one remains internal
        convblock1 = []
        # 3x3 convolution establishing latent channels channels
        convblock1.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock1.append(activation.build())
        # 1x1 convolution establishing increased channels
        convblock1.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock1.append(activation.build())
        # 1x1 convolution returning to latent channels
        convblock1.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock1.append(activation.build())
        self.convblock1 = nn.Sequential(*convblock1)

        # 2nd ConNeXt block, takes the output of the first convnext block
        convblock2 = []
        # 3x3 convolution establishing latent channels channels
        convblock2.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock2.append(activation.build())
        # 1x1 convolution establishing increased channels
        convblock2.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock2.append(activation.build())
        # 1x1 convolution reducing to output channels
        convblock2.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=out_channels,
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock2.append(activation.build())
        self.convblock2 = nn.Sequential(*convblock2)

    def forward(self, x):
        """Forward pass of the DoubleConvNextBlock
        Args:
            x: inputs to the forward pass
        Returns:
            result of the forward pass
        """
        # internal convnext result
        x1 = self.skip_module1(x) + self.convblock1(x)
        # return second convnext result
        return self.skip_module2(x1) + self.convblock2(x1)


class SymmetricConvNeXtBlock(nn.Module):
    """A symmetric variant of the ConvNeXt block, with convolutional layers mirrored
    around a central axis for symmetric feature extraction.

    Parameters:
        skip_module1: A module to align the input and intermediate channels for the first residual connection.
        skip_module2: A module to align the intermediate and output channels for the second residual connection.
        convblock1: A sequential container of convolutional layers for the symmetric ConvNeXt block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Initializes a SymmetricConvNeXtBlock instance with specified parameters.

        Args:
            in_channels: Number of input channels (default is 3).
            out_channels: Number of output channels (default is 1).
            kernel_size: Size of the convolutional kernels (default is 3).
            dilation: Dilation rate for convolutions (default is 1).
            upscale_factor: Upscale factor.
            latent_channels: Number of latent channels used in the block (default is 1).
            activation: Configuration for the activation function used between layers (default is None).
            enable_nhwc: Whether to enable NHWC format (default is False).
            hpx_padding_mode: HEALPix padding backend passed to wrapper.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        healpix_kwargs = _healpix_layer_kwargs(
            enable_nhwc,
            hpx_padding_mode,
            nside,
        )
        if in_channels == int(latent_channels):
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                **healpix_kwargs,
            )

        # 1st ConvNeXt block, the output of this one remains internal
        convblock = []
        # 3x3 convolution establishing latent channels channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        # 1x1 convolution establishing increased channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        # 1x1 convolution returning to latent channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels),
                kernel_size=1,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        # 3x3 convolution from latent channels to latent channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels),
                out_channels=out_channels,  # int(latent_channels),
                kernel_size=kernel_size,
                dilation=dilation,
                **healpix_kwargs,
            )
        )
        if activation is not None:
            convblock.append(activation.build())
        self.convblock = nn.Sequential(*convblock)

    def forward(self, x):
        """Forward pass of the SymmetricConvNextBlock
        Args:
            x: inputs to the forward pass
        Returns:
            result of the forward pass
        """
        # residual connection with reshaped inpute and output of conv block
        return self.skip_module(x) + self.convblock(x)


class Multi_SymmetricConvNeXtBlock(nn.Module):
    """Serial wrapper of ``SymmetricConvNeXtBlock`` repeated ``n_layers`` times."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        dilation: int = 1,
        upscale_factor: int = 4,
        n_layers: int = 1,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: Optional[int] = None,
    ):
        """
        Args:
            in_channels: Number of input channels (first block only).
            latent_channels: Latent channel width inside each symmetric block.
            out_channels: Number of output channels for every block.
            kernel_size: Convolution kernel size.
            dilation: Convolution dilation.
            upscale_factor: Channel upscale factor inside each block.
            n_layers: Number of stacked ``SymmetricConvNeXtBlock`` modules.
            activation: Optional ``CappedGELUConfig`` between layers.
            enable_nhwc: Use channels-last memory format.
            hpx_padding_mode: HEALPix padding backend passed to child blocks.
            nside: Native face height/width for HEALPix padding.
        """
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            curr_in_channels = in_channels if i == 0 else out_channels
            self.blocks.append(
                SymmetricConvNeXtBlock(
                    in_channels=curr_in_channels,
                    latent_channels=latent_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    upscale_factor=upscale_factor,
                    activation=activation,
                    enable_nhwc=enable_nhwc,
                    hpx_padding_mode=hpx_padding_mode,
                    nside=nside,
                )
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor.

        Returns:
            Output after ``n_layers`` symmetric ConvNeXt blocks.
        """
        out = x
        for block in self.blocks:
            out = block(out)
        return out


# --- Utilities ---


class Interpolate(nn.Module):
    """Helper class for interpolation.

    This class handles interpolation, storing scale factor and mode for
    `nn.functional.interpolate`.
    """

    def __init__(self, scale_factor: Union[int, Tuple], mode: str = "nearest"):
        """
        Args:
            scale_factor: Multiplier for spatial size, passed to `nn.functional.interpolate`.
            mode: Interpolation mode used for upsampling, passed to `nn.functional.interpolate`.
        """
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, inputs):
        """Forward pass of the Interpolate layer.

        Args:
            inputs: Inputs to interpolate.

        Returns:
            th.Tensor: The interpolated values.
        """
        return self.interp(inputs, scale_factor=self.scale_factor, mode=self.mode)
