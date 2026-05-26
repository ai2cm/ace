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

"""
HEALPix convolution / interpolation wrapper.

Builds a small ``Sequential`` that optionally prepends a HEALPix-aware padding module,
then the user-supplied base layer (e.g. ``Conv2d``). Inputs are face tensors with
12 HEALPix faces; see ``healpix_paddings`` for face ordering and padding modes.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch as th

logger = logging.getLogger(__name__)

from .healpix_paddings import (
    HEALPixFoldFaces,
    HEALPixPadding,
    HEALPixPaddingIsolatitude,
    HEALPixPaddingv2,
    HEALPixUnfoldFaces,
    build_isolatitude_gather_index,
    have_earth2grid,
    isolatitude_pad_folded,
    make_hpx_padding_layer,
)


class HEALPixLayer(th.nn.Module):
    """
    Apply a base ``torch.nn.Module`` on data laid out as HEALPix faces.

    Expected layout: folded ``[N * 12, C, H, W]``. When computed edge padding is
    positive, a HEALPix-aware padding module is prepended, then the base layer.
    """

    def __init__(
        self,
        layer,
        hpx_padding_mode: Literal[
            "earth2grid", "karlbauer", "isolatitude"
        ] = "earth2grid",
        nside: int | None = None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        layer : type or torch.nn.Module
            Layer class (e.g. ``torch.nn.Conv2d``) or module; must match the
            detection logic for convolution vs interpolation vs other.
        hpx_padding_mode : Literal["earth2grid", "karlbauer", "isolatitude"], optional
            Which padding implementation to use (default ``"earth2grid"``):
            - ``"earth2grid"`` — ``earth2grid.healpix.pad`` (default).
            - ``"karlbauer"`` — Karlbauer et al. (2024) face stitching, same result as earth2grid but slower.
            - ``"isolatitude"`` — alternate padding scheme which preserves isolatitude signals.
        nside : int or None, optional
            Native resolution of each HEALPix face (height = width). Required when
            ``hpx_padding_mode=="isolatitude"``.
        **kwargs
            Forwarded to ``layer`` after removing ``enable_nhwc`` (e.g. ``in_channels``,
            ``out_channels``, ``kernel_size``, ``dilation``, ``enable_nhwc``). If ``nside``
            appears here (e.g. Hydra), it is consumed and overrides the corresponding
            argument.
        """
        super().__init__()
        layers = []

        if "nside" in kwargs:
            _ns = kwargs.pop("nside")
            nside = int(_ns) if _ns is not None else None

        if "enable_nhwc" in kwargs:
            enable_nhwc = kwargs["enable_nhwc"]
            del kwargs["enable_nhwc"]
        else:
            enable_nhwc = False

        kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
        dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
        padding = ((kernel_size - 1) // 2) * dilation

        # Define a HEALPixPadding layer if padding is necessary
        if padding > 0:
            # Disable native padding for conv layers
            if layer.__bases__[0] is th.nn.modules.conv._ConvNd:
                kwargs["padding"] = 0
            padding_layer = make_hpx_padding_layer(
                padding=padding,
                hpx_padding_mode=hpx_padding_mode,
                enable_nhwc=enable_nhwc,
                nside=nside,
            )
            layers.append(padding_layer)

        layers.append(layer(**kwargs))
        self.layers = th.nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=th.channels_last)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Run padding (if configured) and the wrapped layer.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B*F, C, H, W).

        Returns
        -------
        torch.Tensor
            Output of the composed ``Sequential`` of shape (B*F, C', H', W').
        """
        return self.layers(x)


__all__ = [
    "HEALPixFoldFaces",
    "HEALPixLayer",
    "HEALPixPadding",
    "HEALPixPaddingIsolatitude",
    "HEALPixPaddingv2",
    "HEALPixUnfoldFaces",
    "build_isolatitude_gather_index",
    "have_earth2grid",
    "isolatitude_pad_folded",
    "make_hpx_padding_layer",
]
