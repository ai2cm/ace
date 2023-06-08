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

import os
import sys
from typing import Mapping
import numpy as np
import argparse

import netCDF4

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import torch
import logging
from fme.fcn_training.utils import logging_utils

from fme.fcn_training.utils.data_loader_fv3gfs import load_series_data
from fme.core import SingleModuleStepper
import xarray as xr
from fme.core.wandb import WandB

import fme
from fme.fcn_training.train_config import TrainConfig
import dacite
import yaml

DECORRELATION_TIME = 36

wandb = WandB.get_instance()


def load_stepper(checkpoint_file: str) -> SingleModuleStepper:
    checkpoint = torch.load(checkpoint_file, map_location=fme.get_device())
    stepper = SingleModuleStepper.from_state(
        checkpoint["stepper"], load_optimizer=False
    )
    return stepper


def setup(config: TrainConfig):
    best_checkpoint_path = config.best_checkpoint_path
    logging.info(f"Loading trained model checkpoint from {best_checkpoint_path}")

    # load the validation data
    logging.info("Loading inference data")
    ds = netCDF4.MFDataset(os.path.join(config.validation_data.data_path, "*.nc"))
    data = load_series_data(
        idx=0,
        n_steps=config.prediction_length + 1,
        ds=ds,
        names=list(set(config.stepper.in_names).union(config.stepper.out_names)),
    )

    checkpoint_file = config.best_checkpoint_path
    stepper = load_stepper(checkpoint_file)

    return data, stepper


def autoregressive_inference(
    ic,
    valid_data_full: Mapping[str, torch.Tensor],
    stepper: SingleModuleStepper,
    log_on_each_unroll_step: bool,
    prediction_length: int,
):
    ic = int(ic)
    prediction_length = int(prediction_length)

    valid_data = {
        name: tensor[ic : ic + prediction_length + 1].unsqueeze(0)
        for name, tensor in valid_data_full.items()
    }

    # autoregressive inference
    logging.info("Begin autoregressive inference")

    lat_dim, lon_dim = 2, 3
    example_name = list(stepper.data_shapes.keys())[0]
    example_shape = valid_data[example_name].shape
    lat_size, lon_size = example_shape[lat_dim], example_shape[lon_dim]
    area_weights = fme.spherical_area_weights(
        lat_size, lon_size, device=fme.get_device()
    )

    rmse = {}
    global_mean_pred = {}
    global_mean_target = {}
    gradient_magnitude_pred = {}
    gradient_magnitude_target = {}
    inference_logs = {}
    snapshot_lead_steps = [20, 40]
    with torch.no_grad():
        _, gen_data, _, _ = stepper.run_on_batch(
            data=valid_data, train=False, n_forward_steps=prediction_length
        )
        for i in range(0, prediction_length + 1):
            for name in gen_data:
                pred = gen_data[name][0:1, i : i + 1, :]
                tar = valid_data[name][0:1, i : i + 1, :].to(fme.get_device())
                rmse[(name, i)] = (
                    fme.root_mean_squared_error(
                        tar, pred, weights=area_weights, dim=[-2, -1]
                    )
                    .cpu()
                    .numpy()
                )
                global_mean_pred[(name, i)] = (
                    fme.weighted_mean(pred, weights=area_weights, dim=[-2, -1])
                    .cpu()
                    .numpy()
                )
                global_mean_target[(name, i)] = (
                    fme.weighted_mean(tar, weights=area_weights, dim=[-2, -1])
                    .cpu()
                    .numpy()
                )
                gradient_magnitude_pred[(name, i)] = (
                    fme.weighted_mean_gradient_magnitude(
                        pred, weights=area_weights, dim=[-2, -1]
                    )
                    .cpu()
                    .numpy()
                )
                gradient_magnitude_target[(name, i)] = (
                    fme.weighted_mean_gradient_magnitude(
                        tar, weights=area_weights, dim=[-2, -1]
                    )
                    .cpu()
                    .numpy()
                )

            logging.info(
                "Predicted timestep {} of {}. {} RMS Error: {}".format(
                    i, prediction_length, name, rmse[(name, i)]
                )
            )
            rmse_metrics = {f"rmse/ic{ic}/{name}": rmse[(name, i)] for name in gen_data}
            mean_pred_metrics = {
                f"global_mean_prediction/ic{ic}/{name}": global_mean_pred[  # noqa: E501
                    (name, i)
                ]
                for name in gen_data
            }
            mean_target_metrics = {
                f"global_mean_target/ic{ic}/{name}": global_mean_target[(name, i)]
                for name in gen_data
            }
            grad_mag_pred_metrics = {
                f"global_mean_gradient_magnitude_prediction/ic{ic}/{name}": gradient_magnitude_pred[  # noqa: E501
                    (name, i)
                ]
                for name in gen_data
            }
            grad_mag_target_metrics = {
                f"global_mean_gradient_magnitude_target/ic{ic}/{name}": gradient_magnitude_target[  # noqa: E501
                    (name, i)
                ]
                for name in gen_data
            }
            if log_on_each_unroll_step:
                wandb.log(
                    {
                        **rmse_metrics,
                        **mean_pred_metrics,
                        **mean_target_metrics,
                        **grad_mag_pred_metrics,
                        **grad_mag_target_metrics,
                    }
                )
            if i in snapshot_lead_steps:
                for name in gen_data:
                    rmse_metric_name = f"rmse_{i}-lead-step/ic{ic}/{name}"
                    inference_logs[rmse_metric_name] = rmse[(name, i)]
                    inference_logs[
                        f"global_mean_prediction_{i}-lead-step/ic{ic}/{name}"
                    ] = global_mean_pred[(name, i)]
                    inference_logs[
                        f"global_mean_target_{i}-lead-step/ic{ic}/{name}"
                    ] = global_mean_target[(name, i)]
                    inference_logs[
                        f"global_mean_gradient_magnitude_prediction_{i}-lead-step/ic{ic}/{name}"  # noqa: E501
                    ] = gradient_magnitude_pred[(name, i)]
                    inference_logs[
                        f"global_mean_gradient_magnitude_target_{i}-lead-step/ic{ic}/{name}"  # noqa: E501
                    ] = gradient_magnitude_target[(name, i)]

        for name in gen_data:
            time_rmse = fme.rmse_of_time_mean(
                valid_data[name][0, 1:].to(fme.get_device()),
                gen_data[name][0, 1:],
                area_weights,
            ).cpu()
            global_time_mean_bias = fme.time_and_global_mean_bias(
                valid_data[name][0, 1:].to(fme.get_device()),
                gen_data[name][0, 1:],
                area_weights,
            ).cpu()
            inference_logs[f"rmse_of_time_mean/ic{ic}/{name}"] = time_rmse
            inference_logs[
                f"global_and_time_mean_bias/ic{ic}/{name}"
            ] = global_time_mean_bias

    return (
        valid_data,
        gen_data,
        inference_logs,
    )


def main(
    yaml_config: str,
    vis: bool,
):
    with open(yaml_config, "r") as f:
        data = yaml.safe_load(f)
        with open(os.path.join(data["experiment_dir"], "config.yaml"), "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    train_config = dacite.from_dict(
        data_class=TrainConfig,
        data=data,
        config=dacite.Config(strict=True),
    )
    train_config.configure_logging(log_filename="inference_out.log")
    train_config.configure_wandb()

    torch.backends.cudnn.benchmark = True

    if not os.path.isdir(train_config.experiment_dir):
        os.makedirs(train_config.experiment_dir)

    logging_utils.log_versions()
    logging_utils.log_beaker_url()

    n_ics = 1
    ics = [0]

    logging.info("Inference for {} initial conditions".format(n_ics))
    autoregressive_inference_filetag = ""

    if vis:
        autoregressive_inference_filetag += "_vis"
    # get data and models
    valid_data_full, stepper = setup(train_config)

    # initialize lists for image sequences and RMSE/ACC
    valid_data_per_ic = []
    gen_data_per_ic = []
    # run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i + 1, n_ics))
        valid_data, gen_data, _ = autoregressive_inference(
            ic=ic,
            valid_data_full=valid_data_full,
            stepper=stepper,
            log_on_each_unroll_step=True,
            prediction_length=train_config.prediction_length,  # type: ignore
        )
        valid_data_per_ic.append(valid_data)
        gen_data_per_ic.append(gen_data)

    data_vars = {}
    for name in gen_data_per_ic[0].keys():
        valid = torch.cat([valid_data_per_ic[i][name] for i in range(n_ics)], dim=0)
        gen = torch.cat([gen_data_per_ic[i][name] for i in range(n_ics)], dim=0)
        data = torch.stack([valid, gen.cpu()], dim=0)
        data_vars[name] = xr.DataArray(
            data.numpy(),
            dims=["source", "initial_condition", "time", "lat", "lon"],
            coords={
                "source": ["truth", "prediction"],
                "initial_condition": ics,
            },
        )
    ds = xr.Dataset(data_vars)

    prediction_length = len(ds.time)
    img_shape_x = len(ds.lat)

    # save predictions and loss
    filename = os.path.join(
        train_config.experiment_dir,
        "autoregressive_predictions" + autoregressive_inference_filetag + ".nc",
    )
    if vis:
        logging.info(f"Saving files at {filename}")
        ds.to_netcdf(filename)
        if train_config.logging.log_to_wandb:  # type: ignore
            gap = np.zeros((prediction_length, 1, img_shape_x, 10))
            source_valid = 0
            source_gen = 1
            for name in data_vars:
                # wandb.Video requires 4D array, hence adding singleton channel dim
                channel_video_data = np.concatenate(
                    (
                        np.expand_dims(data_vars[name][source_gen, 0, :], axis=1),
                        gap,
                        np.expand_dims(data_vars[name][source_valid, 0, :], axis=1),
                    ),
                    axis=-1,
                )
                # rescale appropriately given that wandb.Video casts
                # data to np.uint8
                # use 'real' data for determining max/min scaling bounds.
                # 'pred' data may saturate bounds, so clip at 0 and 255.
                data_min = data_vars[name][source_valid, 0, :].values.min()
                data_max = data_vars[name][source_valid, 0, :].values.max()
                channel_video_data = (
                    255 * (channel_video_data - data_min) / (data_max - data_min)
                )
                channel_video_data = np.minimum(channel_video_data, 255)
                channel_video_data = np.maximum(channel_video_data, 0)
                wandb_video = wandb.Video(
                    channel_video_data,
                    caption=(
                        "Autoregressive (left) prediction and"
                        f"(right) target for {name}"
                    ),
                )
                wandb.log({f"video/{name}": wandb_video})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument(
        "--vis", action="store_true", help="Whether to store netCDF output"
    )

    args = parser.parse_args()

    main(
        yaml_config=args.yaml_config,
        vis=args.vis,
    )
