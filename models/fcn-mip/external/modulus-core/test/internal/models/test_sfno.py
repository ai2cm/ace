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

import torch
import pytest

from modulus.internal.models.sfno.sfnonet import FourierNeuralOperatorNet

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent.parent / "models"))
import common


class Params:
    def __init__(self):
        self.img_crop_shape_x = 32
        self.img_crop_shape_y = 64
        self.scale_factor = 16
        self.in_chans = 2
        self.out_chans = 2
        self.embed_dim = 32
        self.num_layers = 2


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_forward(device):
    """Test SFNO forward pass"""
    torch.manual_seed(0)
    # Construct FC model
    params = Params()
    model = FourierNeuralOperatorNet(
        img_size=(params.img_crop_shape_x, params.img_crop_shape_y),
        scale_factor=params.scale_factor,
        in_chans=params.in_chans,
        out_chans=params.out_chans,
        embed_dim=params.embed_dim,
        num_layers=params.num_layers,
    ).to(device)

    bsize = 8
    invar = torch.randn(
        bsize, params.in_chans, params.img_crop_shape_x, params.img_crop_shape_y
    ).to(device)
    assert common.validate_forward_accuracy(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_params_forward(device):
    """Test SFNO forward pass"""
    torch.manual_seed(0)
    # Construct FC model
    params = Params()
    model = FourierNeuralOperatorNet(params).to(device)

    bsize = 8
    invar = torch.randn(
        bsize, params.in_chans, params.img_crop_shape_x, params.img_crop_shape_y
    ).to(device)
    assert common.validate_forward_accuracy(model, (invar,))
