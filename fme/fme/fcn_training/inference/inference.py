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
from typing import Optional, Mapping
from fme.fcn_training.utils.data_requirements import DataRequirements
import numpy as np
import argparse

from networks.geometric_v1.sfnonet import FourierNeuralOperatorBuilder
from fourcastnet.networks.afnonet import AFNONetBuilder
from fme.fcn_training.registry import ModuleBuilder
import netCDF4

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import torch
from collections import OrderedDict
import logging
from fme.fcn_training.utils import logging_utils

logging_utils.config_logger()
from fme.fcn_training.utils.YParams import YParams
from fme.fcn_training.utils.data_loader_fv3gfs import load_series_data
import wandb
from fme.fcn_training.stepper import SingleModuleStepper
from fme.fcn_training.utils.darcy_loss import LpLoss
from fme.core.normalizer import get_normalizer
import xarray as xr

import fme

DECORRELATION_TIME = 36


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return x + noise


def load_model(model, checkpoint_file, device=None):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    kwargs = dict(map_location=torch.device("cpu")) if device == "cpu" else {}
    checkpoint = torch.load(checkpoint_fname, **kwargs)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint["stepper"]["module"].items():
            if key != "ged":
                if key.startswith("module."):
                    # model was stored using ddp which prepends 'module.' if training
                    # with multiple GPUs
                    name = str(key[7:])
                else:
                    name = key
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:  # noqa: E722
        model.load_state_dict(checkpoint["stepper"]["module"])
    model.eval()
    return model


def module_builder(params) -> ModuleBuilder:
    # TODO: remove duplication with TrainerParams.new when we have a better
    #       system of configuration classes/files
    if params.nettype == "FourierNeuralOperatorNet":
        model_params = {}
        for param_name in [
            "spectral_transform",
            "filter_type",
            "scale_factor",
            "embed_dim",
            "num_layers",
            "num_blocks",
            "hard_thresholding_fraction",
            "normalization_layer",
            "mlp_mode",
            "big_skip",
            "compression",
            "rank",
            "complex_network",
            "complex_activation",
            "spectral_layers",
            "laplace_weighting",
            "checkpointing",
        ]:
            if param_name in params.__dict__:
                model_params[param_name] = params.__dict__[param_name]
        builder = FourierNeuralOperatorBuilder(**model_params)
    elif params.nettype == "afno":
        model_params = {}
        for param_name in [
            "patch_size",
            "embed_dim",
            "num_blocks",
        ]:
            if param_name in params.__dict__:
                model_params[param_name] = params.__dict__[param_name]
        builder = AFNONetBuilder(**model_params)
    else:
        raise ValueError("Unknown nettype: " + str(params.nettype))
    return builder


def setup(params):
    device = fme.get_device()
    if params.log_to_screen:
        best_checkpoint_path = params["best_checkpoint_path"]
        logging.info(f"Loading trained model checkpoint from {best_checkpoint_path}")

    params.log_on_each_unroll_step_inference = True

    # load the validation data
    if params.log_to_screen:
        logging.info("Loading inference data")
    ds = netCDF4.MFDataset(os.path.join(params.valid_data_path, "*.nc"))
    data = load_series_data(
        idx=0,
        n_steps=params.prediction_length + 1,
        ds=ds,
        names=list(set(params.in_names).union(params.out_names)),
    )

    builder = module_builder(params)

    shapes = {k: v.shape for k, v in data.items()}
    data_requirements = DataRequirements(
        names=list(set(params.in_names).union(params.out_names)),
        in_names=params.in_names,
        out_names=params.out_names,
        n_timesteps=params.prediction_length + 1,
    )
    normalizer = get_normalizer(
        global_means_path=params.global_means_path,
        global_stds_path=params.global_stds_path,
        names=data_requirements.names,
    )
    stepper = SingleModuleStepper(
        builder=builder,
        data_shapes=shapes,
        normalizer=normalizer,
        in_names=params.in_names,
        out_names=params.out_names,
        loss_obj=LpLoss(),
    )

    checkpoint_file = params["best_checkpoint_path"]
    stepper.module = load_model(stepper.module, checkpoint_file, device).to(device)

    return data, stepper


def autoregressive_inference(
    params,
    ic,
    valid_data_full: Mapping[str, torch.Tensor],
    stepper: SingleModuleStepper,
):
    ic = int(ic)
    # initialize global variables
    prediction_length = int(params.prediction_length)
    out_names = params.out_names

    valid_data = {
        name: tensor[ic : ic + prediction_length + 1].unsqueeze(0)
        for name, tensor in valid_data_full.items()
    }

    # autoregressive inference
    if params.log_to_screen:
        logging.info("Begin autoregressive inference")

    lat_dim, lon_dim = 2, 3
    example_shape = valid_data[out_names[0]].shape
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

            if params.log_to_screen:
                logging.info(
                    "Predicted timestep {} of {}. {} RMS Error: {}".format(
                        i, prediction_length, name, rmse[(name, i)]
                    )
                )
            if params.log_to_wandb:
                rmse_metrics = {
                    f"rmse/ic{ic}/{name}": rmse[(name, i)] for name in out_names
                }
                mean_pred_metrics = {
                    f"global_mean_prediction/ic{ic}/{name}": global_mean_pred[  # noqa: E501
                        (name, i)
                    ]
                    for name in out_names
                }
                mean_target_metrics = {
                    f"global_mean_target/ic{ic}/{name}": global_mean_target[(name, i)]
                    for name in out_names
                }
                grad_mag_pred_metrics = {
                    f"global_mean_gradient_magnitude_prediction/ic{ic}/{name}": gradient_magnitude_pred[  # noqa: E501
                        (name, i)
                    ]
                    for name in out_names
                }
                grad_mag_target_metrics = {
                    f"global_mean_gradient_magnitude_target/ic{ic}/{name}": gradient_magnitude_target[  # noqa: E501
                        (name, i)
                    ]
                    for name in out_names
                }
                if params.log_on_each_unroll_step_inference:
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
                    for name in out_names:
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
    run_num: str,
    yaml_config: str,
    config: str,
    use_daily_climatology: bool,
    vis: bool,
    override_dir: Optional[str],
    weights: Optional[str],
):
    params = YParams(os.path.abspath(yaml_config), config)
    params["world_size"] = 1
    params["use_daily_climatology"] = use_daily_climatology

    torch.backends.cudnn.benchmark = True

    # Set up directory
    if override_dir is not None:
        assert (
            weights is not None
        ), "Must set --weights argument if using --override_dir"
        expDir = override_dir
    else:
        assert (
            weights is None
        ), "Cannot use --weights argument without also using --override_dir"
        expDir = os.path.join(params.exp_dir, config, str(run_num))  # type: ignore

    if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params["experiment_dir"] = os.path.abspath(expDir)
    params["best_checkpoint_path"] = (
        weights
        if override_dir is not None
        else os.path.join(expDir, "training_checkpoints/best_ckpt.tar")
    )
    params["resuming"] = False
    params["local_rank"] = 0

    logging_utils.log_to_file(
        logger_name=None, log_filename=os.path.join(expDir, "inference_out.log")
    )
    logging_utils.log_versions()
    params.log()

    if params.log_to_wandb:  # type: ignore
        wandb.init(config=params, project="fourcastnet-era5", entity="ai2cm")
        logging_utils.log_beaker_url()

    if params["n_initial_conditions"] != 1:
        raise NotImplementedError(
            "Currently only supports inference for a single initial condition"
        )

    n_ics = 1
    ics = [0]

    logging.info("Inference for {} initial conditions".format(n_ics))
    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:  # noqa: E722
        autoregressive_inference_filetag = ""

    if vis:
        autoregressive_inference_filetag += "_vis"
    # get data and models
    valid_data_full, stepper = setup(params)

    # initialize lists for image sequences and RMSE/ACC
    valid_data_per_ic = []
    gen_data_per_ic = []
    # run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i + 1, n_ics))
        valid_data, gen_data, _ = autoregressive_inference(
            params, ic, valid_data_full, stepper
        )
        valid_data_per_ic.append(valid_data)
        gen_data_per_ic.append(gen_data)

    data_vars = {}
    for name in valid_data_per_ic[0].keys():
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
        params["experiment_dir"],
        "autoregressive_predictions" + autoregressive_inference_filetag + ".nc",
    )
    if vis:
        if params.log_to_screen:  # type: ignore
            logging.info(f"Saving files at {filename}")
        ds.to_netcdf(filename)
        if params.log_to_wandb:  # type: ignore
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
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="full_field", type=str)
    parser.add_argument("--use_daily_climatology", action="store_true")
    parser.add_argument(
        "--vis", action="store_true", help="Whether to store netCDF output"
    )
    parser.add_argument(
        "--override_dir",
        default=None,
        type=str,
        help="Path to store inference outputs; must also set --weights arg",
    )
    parser.add_argument(
        "--weights",
        default=None,
        type=str,
        help="Path to model weights, for use with override_dir option",
    )

    args = parser.parse_args()

    main(
        run_num=args.run_num,
        yaml_config=args.yaml_config,
        config=args.config,
        vis=args.vis,
        use_daily_climatology=args.use_daily_climatology,
        override_dir=args.override_dir,
        weights=args.weights,
    )
