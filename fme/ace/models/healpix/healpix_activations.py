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
from typing import Literal

import torch as th
import torch.nn as nn

from .healpix_layers import HEALPixLayer

# DOWNSAMPLING BLOCKS


class MaxPool(nn.Module):
    """Wrapper for applying Max Pooling with HEALPix or other tensor data.

    This class wraps the `nn.MaxPool2d` class to handle tensor data with
    HEALPix or other geometry layers.
    """

    def __init__(
        self,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Args:
            pooling (int, optional): Pooling kernel size passed to geometry layer.
            enable_nhwc (bool, optional): Enable nhwc format, passed to wrapper.
            enable_healpixpad (bool, optional): If HEALPixPadding should be enabled, passed to wrapper.
        """
        super().__init__()
        self.maxpool = HEALPixLayer(
            layer=nn.MaxPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the MaxPool.

        Args:
            x: The values to MaxPool.

        Returns:
            The MaxPooled values.
        """
        return self.maxpool(x)


class AvgPool(nn.Module):
    """Wrapper for applying Average Pooling with HEALPix or other tensor data.

    This class wraps the `nn.AvgPool2d` class to handle tensor data with
    HEALPix or other geometry layers.
    """

    def __init__(
        self,
        pooling: int = 2,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Args:
            pooling (int, optional): Pooling kernel size passed to geometry layer.
            enable_nhwc (bool, optional): Enable nhwc format, passed to wrapper.
            enable_healpixpad (bool, optional): If HEALPixPadding should be enabled, passed to wrapper.
        """
        super().__init__()
        self.avgpool = HEALPixLayer(
            layer=nn.AvgPool2d,
            kernel_size=pooling,
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Forward pass of the AvgPool layer.

        Args:
            x: The values to average.

        Returns:
            The averaged values.
        """
        return self.avgpool(x)


@dataclasses.dataclass
class DownsamplingBlockConfig:
    """
    Configuration for the downsampling block.
    Generally, either a pooling block or a striding conv block.

    Parameters:
        block_type: Type of recurrent block, either "MaxPool" or "AvgPool"
        pooling: Pooling size
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.

    """

    block_type: Literal["MaxPool", "AvgPool"]
    pooling: int = 2
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the recurrent block model.

        Returns:
            Recurrent block.
        """
        if self.block_type == "MaxPool":
            return MaxPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )

        elif self.block_type == "AvgPool":
            return AvgPool(
                pooling=self.pooling,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class UpsamplingBlockConfig:
    """
    Configuration for the upsampling block.
    Should be compatible with torch.nn.Upsample block configs.
    Attributes:
        block_type (str) – the upsampling algorithm: one of 'nearest', 'linear', 'bilinear',
            'bicubic' and 'trilinear'. Default: 'nearest'.
        scale_factor (float or Tuple[float]) - multiplier for spatial size. Has to match input size
            if it is a tuple.
        align_corners (bool, optional) – if True, the corner pixels of the input and output tensors are aligned,
            and thus preserving the values at those pixels. Only has effect when mode is 'linear', 'bilinear',
            'bicubic', or 'trilinear'. Default: False
    """

    scale_factor: int
    block_type: Literal["nearest", "linear", "bilinear", "bicubic", "trilinear"] = (
        "bilinear"
    )
    align_corners: bool = False

    def build(self) -> nn.Module:
        """
        Builds the upsampling block model.
        Returns:
            Upsampling block.
        """
        if self.align_corners == False:
            return nn.Upsample(
                scale_factor=self.scale_factor,
                mode=self.block_type,
            )  # Interpolate will complain about the align_corners option being passed at all, bad API design IMO
        else:
            return nn.Upsample(
                scale_factor=self.scale_factor,
                mode=self.block_type,
                align_corners=self.align_corners,
            )


@dataclasses.dataclass
class CappedGELUConfig:
    """
    Configuration for the CappedGELU activation function.

    Parameters:
        cap_value: Cap value for the GELU function, default is 10.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
    """

    cap_value: int = 10
    enable_nhwc: bool = False
    enable_healpixpad: bool = False

    def build(self) -> nn.Module:
        """
        Builds the CappedGELU activation function.

        Returns:
            CappedGELU activation function.
        """
        return CappedGELU(cap_value=self.cap_value)


class CappedGELU(nn.Module):
    """
    Implements a GELU with capped maximum value.

    Example
    -------
    >>> capped_gelu_func = modulus.models.layers.CappedGELU()
    >>> input = th.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_gelu_func(input)
    tensor([[-0.0455, -0.1587],
            [ 0.0000,  0.8413],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Args:
            cap_value: Maximum that values will be capped at
            **kwargs: Keyword arguments to be passed to the `th.nn.GELU` function
        """

        super().__init__()
        self.add_module("gelu", th.nn.GELU(**kwargs))
        self.register_buffer("cap", th.tensor(cap_value, dtype=th.float32))

    def forward(self, inputs):
        x = self.gelu(inputs)
        # Convert cap to a scalar value for clamping (ignores grad)
        cap_value = self.cap.item()
        x = th.clamp(x, max=cap_value)
        return x
