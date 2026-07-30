"""
This file is vendorized from physicsnemo/physicsnemo/models/diffusion/preconditioning.py which you can find here:
https://github.com/NVIDIA/physicsnemo/blob/327d9928abc17983ad7aa3df94da9566c197c468/physicsnemo/models/diffusion/preconditioning.py
"""

# fmt: off
# flake8: noqa

# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
Preconditioning schemes used in the paper"Elucidating the Design Space of
Diffusion-Based Generative Models".
"""

import torch
from fme.core.device import get_device


def _is_autocast_enabled():
    device = get_device()
    if device.type == "cuda":
        return torch.is_autocast_enabled()
    elif device.type == "cpu":
        return torch.is_autocast_cpu_enabled()
    else:
        return False


class EDMPrecond(torch.nn.Module):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM)

    Parameters
    ----------
    model: torch.nn.Module
        The underlying neural network model that predicts denoised images from noisy
        inputs. Expected signature: `model(x, sigma, class_labels=None)`
    label_dim : int
        Number of class labels, 0 = unconditional, by default 0.
    use_fp16 : bool
        Execute the underlying model at FP16 precision?, by default False.
    sigma_data : float
        Expected standard deviation of the training data, by default 0.5.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self,
        model,
        label_dim=0,
        sigma_data=0.5,
    ):
        super().__init__()
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.model = model
        self._dtype = torch.float32

    def forward(
        self,
        x,
        condition,
        sigma,
        class_labels=None,
        force_fp32=False,
    ):
        with torch.amp.autocast(device_type=get_device().type, enabled=False):
            x = x.to(torch.float32)
            sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
            class_labels = (
                None
                if self.label_dim == 0
                else torch.zeros([1, self.label_dim], device=x.device)
                if class_labels is None
                else class_labels.to(torch.float32).reshape(-1, self.label_dim)
            )

            c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
            c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
            c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
            c_noise = sigma.log() / 4

            arg = c_in * x

            if condition is not None:
                arg = torch.cat([arg, condition], dim=1)

        F_x = self.model(
            arg.to(self._dtype),
            c_noise.flatten(),
            class_labels=class_labels,
        )

        if (F_x.dtype != self._dtype) and not _is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {self._dtype}, but got {F_x.dtype} instead."
            )
        with torch.amp.autocast(device_type=get_device().type, enabled=False):
            D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
