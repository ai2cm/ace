# BSD 3-Clause License
#
# Copyright (c) 2022, FourCastNet authors
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The code was authored by the following people:
#
# Jaideep Pathak - NVIDIA Corporation
# Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
# Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
# Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory
# Ashesh Chattopadhyay - Rice University
# Morteza Mardani - NVIDIA Corporation
# Thorsten Kurth - NVIDIA Corporation
# David Hall - NVIDIA Corporation
# Zongyi Li - California Institute of Technology, NVIDIA Corporation
# Kamyar Azizzadenesheli - Purdue University
# Pedram Hassanzadeh - Rice University
# Karthik Kashinath - NVIDIA Corporation
# Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal


class PeriodicPad2d(nn.Module):
    """
    pad longitudinal (left-right) circular
    and pad latitude (top-bottom) with zeros
    """

    def __init__(self, pad_width):
        super(PeriodicPad2d, self).__init__()
        self.pad_width = pad_width

    def forward(self, x):
        # pad left and right circular
        out = F.pad(x, (self.pad_width, self.pad_width, 0, 0), mode="circular")
        # pad top and bottom zeros
        out = F.pad(
            out, (0, 0, self.pad_width, self.pad_width), mode="constant", value=0
        )
        return out


def reshape_fields(
    img,
    input_or_target: Literal["input", "target"],
    normalization,
    means,
    stds,
    normalize=True,
):
    # Takes in np array of size (n_history+1, c, h, w) and returns
    # torch tensor of size ((n_channels*(n_history+1), w, h)

    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1]  # this will either be N_in_channels or N_out_channels

    # Note: this is the only place normalization happens right now!
    # TODO: move normalization from data loading into training logic
    if normalize:
        if normalization == "minmax":
            raise Exception("minmax not supported. Use zscore")
        elif normalization == "zscore":
            img -= means
            img /= stds

    if input_or_target == "input":
        n_steps = n_history + 1
    elif input_or_target == "target":
        n_steps = 1
    img = np.reshape(img, (n_channels * n_steps, img_shape_x, img_shape_y))

    return torch.as_tensor(img)
