"""
This file is vendorized from edm/training/networks.py which you can find here:
https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/networks.py
"""

# fmt: off
# flake8: noqa

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

#----------------------------------------------------------------------------
# Improved preconditioning proposed in the paper "Elucidating the Design
# Space of Diffusion-Based Generative Models" (EDM).

import torch


class EDMPrecond(torch.nn.Module):
    def __init__(self,
        model,                              # The underlying neural network model. Add by <gideond@allenai.org>
        label_dim       = 0,                # Number of class labels, 0 = unconditional.
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
    ):
        super().__init__()
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, latent, conditioning, sigma, class_labels=None, force_fp32=False):
        latent = latent.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = None if self.label_dim == 0 else torch.zeros([1, self.label_dim], device=latent.device) if class_labels is None else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and latent.device.type == 'cuda') else torch.float32

        # Check if autocast is enabled (e.g., via UNetDiffusionModule with use_amp=True)
        # If so, allow bfloat16 as a valid dtype for input
        autocast_enabled = torch.is_autocast_enabled()
        if autocast_enabled and latent.device.type == 'cuda':
            # When autocast with bfloat16 is enabled, use bfloat16 for input
            # This allows autocast to work efficiently without unnecessary dtype conversions
            model_input_dtype = torch.bfloat16
        else:
            model_input_dtype = dtype

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        channel_dim = 1
        input_ = torch.concat(((c_in.to(latent.device) * latent), conditioning.to(latent.device)), dim=channel_dim)

        F_x = self.model(input_.to(model_input_dtype), c_noise.flatten().to(input_.device), class_labels=class_labels)
        # Allow bfloat16 when autocast is enabled (e.g., via UNetDiffusionModule with use_amp=True)
        assert F_x.dtype in (dtype, torch.bfloat16), (
            f"Expected F_x.dtype to be {dtype} or bfloat16 (when autocast enabled), "
            f"but got {F_x.dtype}"
        )
        # matches how UNetDiffusionModule concatenates (latent, inputs)
        n_latent_channels = F_x.shape[1]
        D_x = c_skip * latent[:, :n_latent_channels, ...] + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
