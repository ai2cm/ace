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
This file contains padding and convolution classes to perform according operations on the twelve faces of the HEALPix.


         HEALPix                              Face order                 3D array representation
                                                                            -----------------
--------------------------               //\\  //\\  //\\  //\\             |   |   |   |   |
|| 0  |  1  |  2  |  3  ||              //  \\//  \\//  \\//  \\            |0  |1  |2  |3  |
|\\  //\\  //\\  //\\  //|             /\\0 //\\1 //\\2 //\\3 //            -----------------
| \\//  \\//  \\//  \\// |            // \\//  \\//  \\//  \\//             |   |   |   |   |
|4//\\5 //\\6 //\\7 //\\4|            \\4//\\5 //\\6 //\\7 //\\             |4  |5  |6  |7  |
|//  \\//  \\//  \\//  \\|             \\/  \\//  \\//  \\//  \\            -----------------
|| 8  |  9  |  10 |  11  |              \\8 //\\9 //\\10//\\11//            |   |   |   |   |
--------------------------               \\//  \\//  \\//  \\//             |8  |9  |10 |11 |
                                                                            -----------------
                                    "\\" are top and bottom, whereas
                                    "//" are left and right borders


Details on the HEALPix can be found at https://iopscience.iop.org/article/10.1086/427976

"""

import logging
import sys

import torch as th
import torch.nn as nn


class HEALPixFoldFaces(nn.Module):
    """Class that folds the faces of a HealPIX tensor"""

    def __init__(self, enable_nhwc: bool = False):
        """
        Args:
            enable_nhwc: Use nhwc format instead of nchw format
        """
        super().__init__()
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Forward pass that folds a HEALPix tensor
        [B, F, C, H, W] -> [B*F, C, H, W]

        Args:
            tensor: The tensor to fold

        Returns:
            th.Tensor: the folded tensor

        """
        N, F, C, H, W = tensor.shape
        tensor = th.reshape(tensor, shape=(N * F, C, H, W))

        if self.enable_nhwc:
            tensor = tensor.to(memory_format=th.channels_last)

        return tensor


class HEALPixUnfoldFaces(nn.Module):
    """Class that unfolds the faces of a HealPIX tensor"""

    def __init__(self, num_faces=12, enable_nhwc=False):
        """
        Args:
            num_faces: The number of faces on the grid, default 12
            enable_nhwc: If nhwc format is being used, default False
        """
        super().__init__()
        self.num_faces = num_faces
        self.enable_nhwc = enable_nhwc

    def forward(self, tensor: th.Tensor) -> th.Tensor:
        """
        Forward pass that unfolds a HEALPix tensor
        [B*F, C, H, W] -> [B, F, C, H, W]

        Args:
            tensor: the tensor to unfold

        Returns:
            The unfolded tensor

        """
        NF, C, H, W = tensor.shape
        N = int(NF / self.num_faces)
        tensor = th.reshape(tensor, shape=(N, self.num_faces, C, H, W))

        return tensor


class HEALPixPadding(nn.Module):
    """
    Padding layer for data on a HEALPix sphere. The requirements for using this layer are as follows:
    - The last three dimensions are (face=12, height, width)
    - The first four indices in the faces dimension [0, 1, 2, 3] are the faces on the northern hemisphere
    - The second four indices in the faces dimension [4, 5, 6, 7] are the faces on the equator
    - The last four indices in the faces dimension [8, 9, 10, 11] are the faces on the southern hemisphere

    Orientation and arrangement of the HEALPix faces are outlined above.
    """

    def __init__(self, padding: int, enable_nhwc: bool = False):
        """
        Args:
            padding: The padding size
            enable_nhwc: If nhwc format is being used, default False
        """
        super().__init__()
        self.p = padding
        self.d = [-2, -1]
        self.enable_nhwc = enable_nhwc
        if not isinstance(padding, int) or padding < 1:
            raise ValueError(
                f"invalid value for 'padding', expected int > 0 but got {padding}"
            )

        self.fold = HEALPixFoldFaces(enable_nhwc=self.enable_nhwc)
        self.unfold = HEALPixUnfoldFaces(num_faces=12, enable_nhwc=self.enable_nhwc)

    def forward(self, data: th.Tensor) -> th.Tensor:
        """
        Pad each face consistently with its according neighbors in the HEALPix (see ordering and neighborhoods above).
        Assumes the Tensor is folded

        Args:
        data: The input tensor of shape [..., F, H, W] where each face is to be padded in its HPX context

        Returns:
            The padded tensor where each face's height and width are increased by 2*p
        """
        try:
            th.cuda.nvtx.range_push("HEALPixPadding:forward")
        except RuntimeError:
            # Don't bother with NVTX profiling if it's not available
            pass

        # unfold faces from batch dim
        data = self.unfold(data)

        # Extract the twelve faces (as views of the original tensors)
        f00, f01, f02, f03, f04, f05, f06, f07, f08, f09, f10, f11 = [
            th.squeeze(x, dim=1)
            for x in th.split(tensor=data, split_size_or_sections=1, dim=1)
        ]

        # Assemble the four padded faces on the northern hemisphere
        p00 = self.pn(
            c=f00, t=f01, tl=f02, lft=f03, bl=f03, b=f04, br=f08, rgt=f05, tr=f01
        )
        p01 = self.pn(
            c=f01, t=f02, tl=f03, lft=f00, bl=f00, b=f05, br=f09, rgt=f06, tr=f02
        )
        p02 = self.pn(
            c=f02, t=f03, tl=f00, lft=f01, bl=f01, b=f06, br=f10, rgt=f07, tr=f03
        )
        p03 = self.pn(
            c=f03, t=f00, tl=f01, lft=f02, bl=f02, b=f07, br=f11, rgt=f04, tr=f00
        )

        # Assemble the four padded faces on the equator
        p04 = self.pe(
            c=f04,
            t=f00,
            tl=self.tl(f00, f03),
            lft=f03,
            bl=f07,
            b=f11,
            br=self.br(f11, f08),
            rgt=f08,
            tr=f05,
        )
        p05 = self.pe(
            c=f05,
            t=f01,
            tl=self.tl(f01, f00),
            lft=f00,
            bl=f04,
            b=f08,
            br=self.br(f08, f09),
            rgt=f09,
            tr=f06,
        )
        p06 = self.pe(
            c=f06,
            t=f02,
            tl=self.tl(f02, f01),
            lft=f01,
            bl=f05,
            b=f09,
            br=self.br(f09, f10),
            rgt=f10,
            tr=f07,
        )
        p07 = self.pe(
            c=f07,
            t=f03,
            tl=self.tl(f03, f02),
            lft=f02,
            bl=f06,
            b=f10,
            br=self.br(f10, f11),
            rgt=f11,
            tr=f04,
        )

        # Assemble the four padded faces on the southern hemisphere
        p08 = self.ps(
            c=f08, t=f05, tl=f00, lft=f04, bl=f11, b=f11, br=f10, rgt=f09, tr=f09
        )
        p09 = self.ps(
            c=f09, t=f06, tl=f01, lft=f05, bl=f08, b=f08, br=f11, rgt=f10, tr=f10
        )
        p10 = self.ps(
            c=f10, t=f07, tl=f02, lft=f06, bl=f09, b=f09, br=f08, rgt=f11, tr=f11
        )
        p11 = self.ps(
            c=f11, t=f04, tl=f03, lft=f07, bl=f10, b=f10, br=f09, rgt=f08, tr=f08
        )

        res = th.stack(
            (p00, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11), dim=1
        )

        # fold faces into batch dim
        res = self.fold(res)

        try:
            th.cuda.nvtx.range_pop()
        except RuntimeError:
            # Don't bother with NVTX profiling if it's not available
            pass

        return res

    def pn(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to a northern hemisphere face c under consideration of its given neighbors.

        Args:
            c: The central face and tensor that is subject for padding
            t: The top neighboring face tensor
            tl: The top left neighboring face tensor
            lft: The left neighboring face tensor
            bl: The bottom left neighboring face tensor
            b: The bottom neighboring face tensor
            br: The bottom right neighboring face tensor
            rgt: The right neighboring face tensor
            tr: The top right neighboring face  tensor

        Returns:
            The padded tensor p
        """
        p = self.p  # Padding size
        d = self.d  # Dimensions for rotations

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t.rot90(1, d)[..., -p:, :], c, b[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat(
            (
                tl.rot90(2, d)[..., -p:, -p:],
                lft.rot90(-1, d)[..., -p:],
                bl[..., :p, -p:],
            ),
            dim=-2,
        )
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def pe(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to an equatorial face c under consideration of its given neighbors.

        Args:
            c: The central face and tensor that is subject for padding
            t: The top neighboring face tensor
            tl: The top left neighboring face tensor
            lft: The left neighboring face tensor
            bl: The bottom left neighboring face tensor
            b: The bottom neighboring face tensor
            br: The bottom right neighboring face tensor
            rgt: The right neighboring face tensor
            tr: The top right neighboring face  tensor

        Returns
        -------
        th.Tensor:
            The padded tensor p
        """
        p = self.p  # Padding size

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t[..., -p:, :], c, b[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat((tr[..., -p:, :p], rgt[..., :p], br[..., :p, :p]), dim=-2)

        return th.cat((left, c, right), dim=-1)

    def ps(
        self,
        c: th.Tensor,
        t: th.Tensor,
        tl: th.Tensor,
        lft: th.Tensor,
        bl: th.Tensor,
        b: th.Tensor,
        br: th.Tensor,
        rgt: th.Tensor,
        tr: th.Tensor,
    ) -> th.Tensor:
        """
        Applies padding to a southern hemisphere face c under consideration of its given neighbors.

        Args:
            c: The central face and tensor that is subject for padding
            t: The top neighboring face tensor
            tl: The top left neighboring face tensor
            lft: The left neighboring face tensor
            bl: The bottom left neighboring face tensor
            b: The bottom neighboring face tensor
            br: The bottom right neighboring face tensor
            rgt: The right neighboring face tensor
            tr: The top right neighboring face  tensor

        Returns:
            The padded tensor p
        """
        p = self.p  # Padding size
        d = self.d  # Dimensions for rotations

        # Start with top and bottom to extend the height of the c tensor
        c = th.cat((t[..., -p:, :], c, b.rot90(1, d)[..., :p, :]), dim=-2)

        # Construct the left and right pads including the corner faces
        left = th.cat((tl[..., -p:, -p:], lft[..., -p:], bl[..., :p, -p:]), dim=-2)
        right = th.cat(
            (tr[..., -p:, :p], rgt.rot90(-1, d)[..., :p], br.rot90(2, d)[..., :p, :p]),
            dim=-2,
        )

        return th.cat((left, c, right), dim=-1)

    def tl(self, top: th.Tensor, lft: th.Tensor) -> th.Tensor:
        """
        Assembles the top left corner of a center face in the cases where no according top left face is defined on the
        HPX.

        Args:
            top: The face above the center face
            lft: The face left of the center face

        Returns:
            The assembled top left corner (only the sub-part that is required for padding)
        """
        ret = th.zeros_like(top)[..., : self.p, : self.p]  # super ugly but super fast

        # Bottom left point
        ret[..., -1, -1] = 0.5 * top[..., -1, 0] + 0.5 * lft[..., 0, -1]

        # Remaining points
        for i in range(1, self.p):
            ret[..., -i - 1, -i:] = top[
                ..., -i - 1, :i
            ]  # Filling top right above main diagonal
            ret[..., -i:, -i - 1] = lft[
                ..., :i, -i - 1
            ]  # Filling bottom left below main diagonal
            ret[..., -i - 1, -i - 1] = (
                0.5 * top[..., -i - 1, 0] + 0.5 * lft[..., 0, -i - 1]
            )  # Diagonal

        return ret

    def br(self, b: th.Tensor, r: th.Tensor) -> th.Tensor:
        """
        Assembles the bottom right corner of a center face in the cases where no according bottom right face is defined
        on the HPX.

        Args:
            b: The face below the center face
            r: The face right of the center face

        Returns:
            The assembled bottom right corner (only the sub-part that is required for padding)
        """
        ret = th.zeros_like(b)[..., : self.p, : self.p]

        # Top left point
        ret[..., 0, 0] = 0.5 * b[..., 0, -1] + 0.5 * r[..., -1, 0]

        # Remaining points
        for i in range(1, self.p):
            ret[..., :i, i] = r[..., -i:, i]  # Filling top right above main diagonal
            ret[..., i, :i] = b[..., i, -i:]  # Filling bottom left below main diagonal
            ret[..., i, i] = 0.5 * b[..., i, -1] + 0.5 * r[..., -1, i]  # Diagonal

        return ret


class HEALPixLayer(nn.Module):
    """Pytorch module for applying any base torch Module on a HEALPix tensor. Expects all input/output tensors to have a
    shape [..., 12, H, W], where 12 is the dimension of the faces.
    """

    def __init__(self, layer, **kwargs):
        """
        Args:
            layer: Any torch layer function, e.g., nn.Conv2d
            kwargs: The arguments that are passed to the torch layer function, e.g., kernel_size
        """
        super().__init__()
        layers = []

        if "enable_nhwc" in kwargs:
            enable_nhwc = kwargs["enable_nhwc"]
            del kwargs["enable_nhwc"]
        else:
            enable_nhwc = False

        if "enable_healpixpad" in kwargs and kwargs["enable_healpixpad"]:
            raise NotImplementedError(
                "HEALPixPaddingv2 is not available in this environment"
            )

        if "enable_healpixpad" in kwargs:
            del kwargs["enable_healpixpad"]

        if not isinstance(layer, type) or not issubclass(layer, th.nn.Module):
            raise TypeError(
                f"Expected a subclass of torch.nn.Module, got {type(layer).__name__}"
            )
        # Define a HEALPixPadding layer if the given layer is a convolution layer
        if layer.__bases__[0] is nn.modules.conv._ConvNd and kwargs["kernel_size"] > 1:
            kwargs["padding"] = 0  # Disable native padding
            kernel_size = 3 if "kernel_size" not in kwargs else kwargs["kernel_size"]
            dilation = 1 if "dilation" not in kwargs else kwargs["dilation"]
            padding = ((kernel_size - 1) // 2) * dilation
            layers.append(HEALPixPadding(padding=padding, enable_nhwc=enable_nhwc))

        layers.append(layer(**kwargs))
        self.layers = nn.Sequential(*layers)

        if enable_nhwc:
            self.layers = self.layers.to(memory_format=th.channels_last)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Performs the forward pass using the defined layer function and the given data.

        :param x: The input tensor of shape [..., F=12, H, W]
        :return: The output tensor of this HEALPix layer
        """
        res = self.layers(x)
        return res
