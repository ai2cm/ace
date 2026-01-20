"""
This file is vendorized from physicsnemo/physicsnemo/utis/generative/stochastic_sampler.py which you can find here:
https://github.com/NVIDIA/physicsnemo/blob/327d9928abc17983ad7aa3df94da9566c197c468/physicsnemo/utils/generative/stochastic_sampler.py
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


from typing import Callable

import torch
from torch import Tensor


def stochastic_sampler(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    randn_like: Callable[[Tensor], Tensor] = torch.randn_like,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 80.0,
    rho: float = 7.0,
    S_churn: float = 0.0,
    S_min: float = 0.0,
    S_max: float = float("inf"),
    S_noise: float = 1.0,
) -> Tensor:
    """
    Proposed EDM sampler (Algorithm 2) with minor changes to enable
    super-resolution and patch-based diffusion.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model that generates denoised images from noisy
        inputs.
        Expected signature: `net(x, x_lr, t_hat)`,
        where:
            x (torch.Tensor): Noisy input of shape (batch_size, C_out, H, W)
            x_lr (torch.Tensor): Conditioning input of shape (batch_size, C_cond, H, W)
            t_hat (torch.Tensor): Noise level of shape (batch_size, 1, 1, 1) or scalar
        Returns:
            torch.Tensor: Denoised prediction of shape (batch_size, C_out, H, W)
    latents : Tensor
        The latent variables (e.g., noise) used as the initial input for the
        sampler. Has shape (batch_size, C_out, img_shape_y, img_shape_x).
    img_lr : Tensor
        Low-resolution input image for conditioning the super-resolution
        process. Must have shape (batch_size, C_lr, img_lr_ shape_y,
        img_lr_shape_x).
    randn_like : Callable[[Tensor], Tensor]
        Function to generate random noise with the same shape as the input
        tensor.
        By default torch.randn_like.
    num_steps : int
        Number of time steps for the sampler. By default 18.
    sigma_min : float
        Minimum noise level. By default 0.002.
    sigma_max : float
        Maximum noise level. By default 800.
    rho : float
        Exponent used in the time step discretization. By default 7.
    S_churn : float
        Churn parameter controlling the level of noise added in each step. By
        default 0.
    S_min : float
        Minimum time step for applying churn. By default 0.
    S_max : float
        Maximum time step for applying churn. By default float("inf").
    S_noise : float
        Noise scaling factor applied during the churn step. By default 1.

    Returns
    -------
    Tensor
        The final denoised image produced by the sampler. Same shape as
        `latents`: (batch_size, C_out, img_shape_y, img_shape_x).

    """

    # img_lr and latents must also have the same batch_size, otherwise mismatch
    # when processed by the network
    if img_lr.shape[0] != latents.shape[0]:
        raise ValueError(
            f"img_lr and latents must have the same batch size, but found "
            f"{img_lr.shape[0]} vs {latents.shape[0]}."
        )

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [t_steps, torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    x_lr = img_lr

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    latent_steps = [x_next.to(latents.dtype)]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next
        # Increase noise temporarily.
        gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur

        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step. Perform patching operation on score tensor if patch-based
        # generation is used denoised = net(x_hat, t_hat,
        # ).to(torch.float64)

        x_hat_batch = (x_hat).to(
            latents.device
        )
        x_lr = x_lr.to(latents.device)
        denoised = net(
            x_hat_batch,
            x_lr,
            t_hat,
        ).to(torch.float64)

        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            x_next_batch = (x_next).to(
                latents.device
            )
            denoised = net(
                x_next_batch,
                x_lr,
                t_next,
            ).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        latent_steps.append(x_next.to(latents.dtype))
    return x_next.to(latents.dtype), latent_steps
