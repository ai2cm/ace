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

"""Activation helpers for HEALPix convolution stacks."""

import dataclasses

import torch
import torch.nn as nn


@dataclasses.dataclass
class CappedGELUConfig:
    """
    Configuration for the CappedGELU activation function.

    Parameters:
        cap_value: Cap value for the GELU function, default is 10.
    """

    cap_value: int = 10

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
    >>> input = torch.Tensor([[-2,-1],[0,1],[2,3]])
    >>> capped_gelu_func(input)
    tensor([[-0.0455, -0.1587],
            [ 0.0000,  0.8413],
            [ 1.0000,  1.0000]])

    """

    def __init__(self, cap_value=1.0, **kwargs):
        """
        Args:
            cap_value: Maximum that values will be capped at
            **kwargs: Keyword arguments to be passed to the `torch.nn.GELU` function
        """

        super().__init__()
        self.add_module("gelu", torch.nn.GELU(**kwargs))
        self.register_buffer("cap", torch.tensor(cap_value, dtype=torch.float32))

    def forward(self, inputs):
        """
        Args:
            inputs: Input tensor passed through capped GELU.

        Returns:
            Tensor with GELU applied and values clamped to ``cap_value``.
        """
        x = self.gelu(inputs)
        # Convert cap to a scalar value for clamping (ignores grad)
        cap_value = self.cap.item()
        x = torch.clamp(x, max=cap_value)
        return x
