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

import numpy as np
from typing import Optional

from . import logging_utils

logging_utils.config_logger()
import torch

import fme


def unlog_tp(x, eps=1e-5):
    #    return np.exp(x + np.log(eps)) - eps
    return eps * (np.exp(x) - 1)


def unlog_tp_torch(x, eps=1e-5):
    #    return torch.exp(x + torch.log(eps)) - eps
    return eps * (torch.exp(x) - 1)


def mean(x, axis=None):
    # spatial mean
    y = np.sum(x, axis) / np.size(x, axis)
    return y


def lat_np(j, num_lat):
    return 90 - j * 180 / (num_lat - 1)


def weighted_acc(pred, target, weighted=True):
    # takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    #    pred -= mean(pred)
    #    target -= mean(target)
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = (
        np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
        if weighted
        else 1
    )
    r = (weight * pred * target).sum() / np.sqrt(
        (weight * pred * pred).sum() * (weight * target * target).sum()
    )
    return r


def weighted_acc_masked(pred, target, weighted=True, maskarray=1):
    # takes in shape [1, num_lat, num_long]
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)

    num_lat = np.shape(pred)[1]
    pred -= mean(pred)
    target -= mean(target)
    s = np.sum(np.cos(np.pi / 180 * lat(np.arange(0, num_lat), num_lat)))
    weight = (
        np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
        if weighted
        else 1
    )
    r = (maskarray * weight * pred * target).sum() / np.sqrt(
        (maskarray * weight * pred * pred).sum()
        * (maskarray * weight * target * target).sum()
    )
    return r


def weighted_rmse(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    # takes in arrays of size [1, h, w]  and returns latitude-weighted rmse
    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi / 180 * lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(
        latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1
    )
    return np.sqrt(
        1
        / num_lat
        * 1
        / num_long
        * np.sum(np.dot(weight.T, (pred[0] - target[0]) ** 2))
    )


def latitude_weighting_factor(j, num_lat, s):
    return num_lat * np.cos(np.pi / 180.0 * lat_np(j, num_lat)) / s


def top_quantiles_error(pred, target):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) == 2:
        target = np.expand_dims(target, 0)
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1.0 - np.logspace(-qlim, -qcut, num=qs)
    P_tar = np.quantile(target, q=qtile, axis=(1, 2))
    P_pred = np.quantile(pred, q=qtile, axis=(1, 2))
    return np.mean(P_pred - P_tar, axis=0)


# torch version for rmse comp
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90.0 - j * 180.0 / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor_torch(
    j: torch.Tensor, num_lat: int, s: torch.Tensor
) -> torch.Tensor:
    return num_lat * torch.cos(3.1416 / 180.0 * lat(j, num_lat)) / s


@torch.jit.script
def weighted_rmse_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]
    # and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    result = torch.sqrt(torch.mean(weight * (pred - target) ** 2.0, dim=(-1, -2)))
    return result


@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_global_mean_channels(pred: torch.Tensor) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted global
    # mean for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    result = torch.mean(weight * pred, dim=(-1, -2))
    return result


@torch.jit.script
def weighted_global_mean_gradient_magnitude_channels(
    pred: torch.Tensor,
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted global
    # mean of spatial gradient magnitude
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    # workaround for https://github.com/pytorch/pytorch/issues/67919
    if pred.shape[0] == 1:
        workaround = True
        pred_ = torch.cat((pred, pred), dim=0)
    else:
        workaround = False
        pred_ = pred
    gradient_x, gradient_y = torch.gradient(pred_, dim=(-1, -2))
    if workaround:
        gradient_x, gradient_y = gradient_x[0], gradient_y[0]
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    result = torch.mean(weight * gradient_magnitude, dim=(-1, -2))
    return result


@torch.jit.script
def weighted_global_mean_gradient_magnitude(pred: torch.Tensor) -> torch.Tensor:
    result = weighted_global_mean_gradient_magnitude_channels(pred)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_masked_torch_channels(
    pred: torch.Tensor, target: torch.Tensor, maskarray: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    result = torch.sum(maskarray * weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(maskarray * weight * pred * pred, dim=(-1, -2))
        * torch.sum(maskarray * weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    # takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    # num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416 / 180.0 * lat(lat_t, num_lat)))
    weight = torch.reshape(
        latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1)
    )
    result = torch.sum(weight * pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(weight * pred * pred, dim=(-1, -2))
        * torch.sum(weight * target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def unweighted_acc_torch_channels(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    result = torch.sum(pred * target, dim=(-1, -2)) / torch.sqrt(
        torch.sum(pred * pred, dim=(-1, -2)) * torch.sum(target * target, dim=(-1, -2))
    )
    return result


@torch.jit.script
def unweighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = unweighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 100
    qlim = 3
    qcut = 0.1
    n, c, h, w = pred.size()
    qtile = 1.0 - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device)
    P_tar = torch.quantile(target.view(n, c, h * w), q=qtile, dim=-1)
    P_pred = torch.quantile(pred.view(n, c, h * w), q=qtile, dim=-1)
    return torch.mean(P_pred - P_tar, dim=0)


def compute_time_rmse(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the time-averaged RMSE for each variable in the sequence.

    Args:
        truth: truth tensor of shape (time, variable, lat, lon)
        predicted: predicted tensor of shape (time, variable, lat, lon)
        weights: torch.Tensor of shape (lat, lon) or None. If None, do
            area-weighting inferred from truth tensor's shape.

    Returns:
        Shape (variable,) tensor. The RMSE between the time-mean of
        the two input tensors.
    """
    time_dim, variable_dim, lat_dim, lon_dim = 0, 1, 2, 3
    lat, lon = truth.shape[lat_dim], truth.shape[lon_dim]

    if weights is None:
        weights = fme.spherical_area_weights(lat, lon)

    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)

    ret = fme.root_mean_squared_error(
        truth_time_mean.cpu(), predicted_time_mean.cpu(), weights=weights, dim=(1, 2)
    )
    assert ret.shape == (
        truth.shape[variable_dim],
    ), "Expected one time RMSE per variable."
    return ret


def compute_time_and_global_mean_bias(
    truth: torch.Tensor,
    predicted: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the global- and time-mean bias for each variable in the sequence.

    Args:
        truth: truth tensor of shape (time, variable, lat, lon)
        predicted: predicted tensor of shape (time, variable, lat, lon)
        weights: torch.Tensor of shape (lat, lon) or None. If None, use
            area-weighting inferred from truth tensor's shape.

    Returns:
        Shape (variable,) tensor. The global- and time-mean bias between the
        predicted and truth tensors.
    """
    time_dim, variable_dim, lat_dim, lon_dim = 0, 1, 2, 3
    lat_size, lon_size = truth.shape[lat_dim], truth.shape[lon_dim]

    if weights is None:
        weights = fme.spherical_area_weights(lat_size, lon_size)

    truth_time_mean = truth.mean(dim=time_dim)
    predicted_time_mean = predicted.mean(dim=time_dim)
    result = fme.weighted_mean(
        predicted_time_mean - truth_time_mean, weights=weights, dim=(1, 2)
    )

    assert result.shape == (
        truth.shape[variable_dim],
    ), "Expected one time-mean global bias per variable."

    return result
