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

import abc
import dataclasses
from typing import Literal, Optional, Tuple, Union, cast

import torch as th
import torch.nn as nn

from .healpix_activations import CappedGELUConfig, UpsamplingBlockConfig
from .healpix_layers import HEALPixLayer

# RECURRENT BLOCKS


@dataclasses.dataclass
class RecurrentBlockConfig:
    """
    Configuration for the recurrent block.

    Parameters:
        in_channels: Number of input channels, default is 3.
        kernel_size: Size of the kernel, default is 1.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
        block_type: Type of recurrent block, either "ConvGRUBlock" or "ConvLSTMBlock",
        default is "ConvGRUBlock".
    """

    in_channels: int = 3
    kernel_size: int = 1
    enable_nhwc: bool = False
    enable_healpixpad: bool = False
    block_type: Literal["ConvGRUBlock", "ConvLSTMBlock"] = "ConvGRUBlock"

    def build(self) -> nn.Module:
        """
        Builds the recurrent block model.

        Returns:
            Recurrent block.
        """
        if self.block_type == "ConvGRUBlock":
            return ConvGRUBlock(
                in_channels=self.in_channels,
                kernel_size=self.kernel_size,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "ConvLSTMBlock":
            return ConvLSTMBlock(
                in_channels=self.in_channels,
                kernel_size=self.kernel_size,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")


@dataclasses.dataclass
class ConvBlockConfig:
    """
    Configuration for the convolutional block.

    Parameters:
        in_channels: Number of input channels, default is 3.
        out_channels: Number of output channels, default is 1.
        kernel_size: Size of the kernel, default is 3.
        dilation: Dilation rate, default is 1.
        n_layers: Number of layers, default is 1.
        upsampling: Upsampling factor for TransposedConvUpsample, default is 2.
        upscale_factor: Upscale factor for ConvNeXtBlock and SymmetricConvNeXtBlock,
            default is 4.
        latent_channels: Number of latent channels, default is None.
        activation: Activation configuration, default is None.
        enable_nhwc: Flag to enable NHWC data format, default is False.
        enable_healpixpad: Flag to enable HEALPix padding, default is False.
        block_type: Type of block, default is "BasicConvBlock".
    """

    in_channels: int = 3
    out_channels: int = 1
    kernel_size: int = 3
    dilation: int = 1
    n_layers: int = 1
    stride: int = 2
    upscale_factor: int = 4
    latent_channels: Optional[int] = None
    upsampling: Optional[UpsamplingBlockConfig] = None
    activation: Optional[CappedGELUConfig] = None
    enable_nhwc: bool = False
    enable_healpixpad: bool = False
    block_type: Literal[
        "BasicConvBlock",
        "ConvNeXtBlock",
        "SymmetricConvNeXtBlock",
        "ConvThenUpsample",
        "TransposedConvUpsample",
    ] = "BasicConvBlock"

    def build(self) -> nn.Module:
        """
        Builds the convolutional block model.

        Returns:
            Convolutional block model.
        """
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
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "ConvNeXtBlock":
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
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "SymmetricConvNeXtBlock":
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
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "ConvThenUpsample":
            return ConvThenUpsample(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=self.stride,
                upsampling=self.upsampling,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        elif self.block_type == "TransposedConvUpsample":
            return TransposedConvUpsample(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                upsampling=self.stride,
                activation=self.activation,
                enable_nhwc=self.enable_nhwc,
                enable_healpixpad=self.enable_healpixpad,
            )
        else:
            raise ValueError(f"Unsupported block type: {self.block_type}")


class ConvGRUBlock(nn.Module):
    """Class that implements a Convolutional GRU.

    Code modified from:
    https://github.com/happyjin/ConvGRU-pytorch/blob/master/convGRU.py
    """

    def __init__(
        self,
        in_channels=3,
        kernel_size=1,
        enable_nhwc=False,
        enable_healpixpad=False,
    ):
        """
        Args:
            in_channels: The number of input channels.
            kernel_size: Size of the convolutional kernel.
            enable_nhwc: Enable nhwc format, passed to wrapper.
            enable_healpixpad: If HEALPixPadding should be enabled, passed to wrapper.
        """
        super().__init__()

        self.channels = in_channels
        self.conv_gates = HEALPixLayer(
            layer=th.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=2 * self.channels,  # for update_gate, reset_gate respectively
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.conv_can = HEALPixLayer(
            layer=th.nn.Conv2d,
            in_channels=in_channels + self.channels,
            out_channels=self.channels,  # for candidate neural memory
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.h = th.zeros(1, 1, 1, 1)

    def forward(self, inputs):
        """Forward pass of the ConvGRUBlock.

        Args:
            inputs: Input to the forward pass.

        Returns:
            th.Tensor: Result of the forward pass.
        """
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
        combined = th.cat([inputs, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = th.split(combined_conv, self.channels, dim=1)
        reset_gate = th.sigmoid(gamma)
        update_gate = th.sigmoid(beta)

        combined = th.cat([inputs, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = th.tanh(cc_cnm)

        h_next = (1 - update_gate) * self.h + update_gate * cnm
        self.h = h_next

        return inputs + h_next

    def reset(self):
        """Reset the update gates."""
        self.h = th.zeros_like(self.h)


class ConvLSTMBlock(nn.Module):
    """Convolutional LSTM block."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        latent_channels: int = 1,
        kernel_size: int = 3,
        downscale_factor: int = 4,
        upscale_factor: int = 4,
        n_layers: int = 1,
        latent_conv_size: int = 3,  # Add latent_conv_size parameter
        dilation: int = 1,
        activation: nn.Module = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            latent_channels: Number of latent channels.
            kernel_size: Size of the convolutional kernel.
            downscale_factor: Downscale factor.
            upscale_factor: Upscale factor.
            n_layers: Number of layers.
            latent_conv_size: Size of latent convolution.
            dilation: Spacing between kernel points.
            activation: Activation function.
            enable_nhwc: Enable nhwc format.
            enable_healpixpad: If HEALPixPadding should be enabled.
        """
        super().__init__()
        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        # Skip connection for output
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=in_channels,  # out channels describes the space of the output of conv here; but we have the output of LSTM which is the input layer size
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        # Convolution block
        convblock = []
        # 3x3 convolution increasing channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels
                * 2,  # accounts for the h layer, which is concatenated before convolution runs
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)
        # 3x3 convolution maintaining increased channels
        convblock.append(
            HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=int(latent_channels * upscale_factor),
                out_channels=int(latent_channels * upscale_factor),
                kernel_size=kernel_size,
                dilation=dilation,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            convblock.append(activation)

        # Now for the LSTM bit
        self.channels = in_channels
        self.lstm_gates = HEALPixLayer(
            layer=th.nn.Conv2d,
            in_channels=latent_channels * upscale_factor,
            out_channels=self.channels
            * 4,  # for input_gate, forget_gate, cell_gate, output_gate respectively (LSTM)
            kernel_size=kernel_size,
            padding="same",
            enable_nhwc=enable_nhwc,
            enable_healpixpad=enable_healpixpad,
        )
        self.h = th.zeros(1, 1, 1, 1)
        self.c = th.zeros(1, 1, 1, 1)
        self.convblock = nn.Sequential(*convblock)

    def forward(self, inputs):
        """Forward pass of the ConvLSTMBlock.

        Args:
            x: Inputs to the forward pass.

        Returns:
            th.Tensor: Result of the forward pass.
        """
        if inputs.shape != self.h.shape:
            self.h = th.zeros_like(inputs)
            self.c = th.zeros_like(inputs)

        combined = th.cat([inputs, self.h], dim=1)
        conv_outputs = self.convblock(combined)

        lstm_gates = self.lstm_gates(conv_outputs)

        # Split the combined_conv into input_gate, forget_gate, cell_gate, output_gate
        i, f, c_hat, o = th.split(lstm_gates, self.channels, dim=1)
        input_gate = th.sigmoid(i)
        forget_gate = th.sigmoid(f)
        cell_gate = th.tanh(c_hat)
        output_gate = th.sigmoid(o)

        self.c = forget_gate * self.c + input_gate * cell_gate
        self.h = output_gate * th.tanh(self.c)

        skip_connection = self.skip_module(inputs)
        return skip_connection + self.h

    def reset(self):
        self.h = th.zeros_like(self.h)
        self.c = th.zeros_like(self.c)


# CONV BLOCKS


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
        enable_healpixpad=False,
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
            enable_healpixpad:: If HEALPixPadding should be enabled, passed to wrapper.
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
                    enable_nhwc=enable_nhwc,
                    enable_healpixpad=enable_healpixpad,
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
        enable_healpixpad: bool = False,
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
            enable_healpixpad: Whether to enable HEALPixPadding.
        """
        super().__init__()

        # Instantiate 1x1 conv to increase/decrease channel depth if necessary
        if in_channels == out_channels:
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
        enable_healpixpad: bool = False,
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
            enable_healpixpad: Whether to enable HEALPixPadding (default is False).
        """
        super().__init__()

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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
        enable_healpixpad: bool = False,
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
            enable_healpixpad: Whether to enable HEALPixPadding (default is False).
        """
        if in_channels == int(latent_channels):
            self.skip_module = lambda x: x  # Identity-function required in forward pass
        else:
            self.skip_module = HEALPixLayer(
                layer=th.nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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


class ConvThenUpsample(nn.Module):
    """Wrapper for upsampling and then applying a convolution using HEALPix or other tensor data.
    Allows more control over the type of upsampling (smooth with bilinear or pixelated
    with nearest-neighbor) and feature extraction.
    This class wraps the `nn.Upsample` and `nn.Conv2d` classes to replace ConvTranspose2d and handle tensor data with
    HEALPix or other geometry layers.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        stride: int = 2,
        upsampling: Optional[UpsamplingBlockConfig] = None,
        activation: Optional[CappedGELUConfig] = None,
        enable_nhwc: bool = False,
        enable_healpixpad: bool = False,
    ):
        super().__init__()
        upsampler = []
        if upsampling is not None:
            upsampler.append(upsampling.build())
        # Upsample transpose conv
        upsampler.append(
            HEALPixLayer(
                layer=nn.Conv2d,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=stride,
                stride=stride,
                padding=0,
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
            )
        )
        if activation is not None:
            upsampler.append(activation.build())
        self.upsampler = nn.Sequential(*upsampler)

    def forward(self, x):
        """Forward pass of the ConvThenUpsample layer.
        Args:
            x: The values to upsample.
        Returns:
            th.Tensor: The upsampled values.
        """
        return self.upsampler(x)


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
        enable_healpixpad: bool = False,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of output channels.
            upsampling: Stride size that will be used for upsampling.
            activation: ModuleConfig for the activation function used in upsampling.
            enable_nhwc: Enable nhwc format, passed to wrapper.
            enable_healpixpad: If HEALPixPadding should be enabled, passed to wrapper.
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
                enable_nhwc=enable_nhwc,
                enable_healpixpad=enable_healpixpad,
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


# Helpers


class Interpolate(nn.Module):
    """Helper class for interpolation.

    This class handles interpolation, storing scale factor and mode for
    `nn.functional.interpolate`.
    """

    def __init__(self, scale_factor: Union[int, Tuple], mode: str = "nearest"):
        """
        Args:
            scale_factor: Multiplier for spatial size, passed to `nn.functional.interpolate`.
            mode, : Interpolation mode used for upsampling, passed to `nn.functional.interpolate`.
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
