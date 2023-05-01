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
from typing import Any, List, Optional
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import h5py
import torch
import torch.distributed as dist
from collections import OrderedDict
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import (
    compute_time_rmse,
    weighted_rmse_torch_channels,
    weighted_acc_torch_channels,
    unweighted_acc_torch_channels,
    weighted_global_mean_channels,
    weighted_global_mean_gradient_magnitude_channels,
)

logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
import wandb
from datetime import datetime

import fme
from fme.fcn_training import NET_REGISTRY

DECORRELATION_TIME = 36


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return x + noise


def load_model(model, params, checkpoint_file, device=None):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    kwargs = dict(map_location=torch.device("cpu")) if device == "cpu" else {}
    checkpoint = torch.load(checkpoint_fname, **kwargs)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint["model_state"].items():
            name = key[7:]
            if name != "ged":
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
    except:  # noqa: E722
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    _, valid_dataset = get_data_loader(
        params, params.inf_data_path, dist.is_initialized(), train=False
    )
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    params.img_crop_shape_x = img_shape_x  # needed by FourierNeuralOperatorNet
    params.img_crop_shape_y = img_shape_y  # needed by FourierNeuralOperatorNet
    if params.log_to_screen:
        best_checkpoint_path = params["best_checkpoint_path"]
        logging.info(f"Loading trained model checkpoint from {best_checkpoint_path}")

    n_in_channels = valid_dataset.n_in_channels
    n_out_channels = valid_dataset.n_out_channels
    params.in_names = valid_dataset.in_names
    params.out_names = valid_dataset.out_names

    params["N_in_channels"] = n_in_channels
    params["N_out_channels"] = n_out_channels
    params.means = valid_dataset.out_means[0]
    params.stds = valid_dataset.out_stds[0]
    params.time_means = valid_dataset.out_time_means[0]

    params.log_on_each_unroll_step_inference = True

    # load the model
    model = NET_REGISTRY[params.nettype](params).to(device)

    checkpoint_file = params["best_checkpoint_path"]
    model = load_model(model, params, checkpoint_file, device)
    model = model.to(device)

    # load the validation data
    if params.log_to_screen:
        logging.info("Loading inference data")
    valid_data_full = valid_dataset.data_array

    return valid_data_full, model


def autoregressive_inference(params, ic, valid_data_full, model):
    ic = int(ic)
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    dt = int(params.dt)
    prediction_length = int(params.prediction_length / dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    n_in_channels = params.N_in_channels
    n_out_channels = params.N_out_channels
    out_names = params.out_names
    means = params.means
    stds = params.stds
    time_means = params.time_means

    # initialize memory for image sequences and RMSE/ACC
    metric_shape = (prediction_length, n_out_channels)
    valid_loss = torch.zeros(metric_shape).to(device, dtype=torch.float)
    acc = torch.zeros(metric_shape).to(device, dtype=torch.float)
    global_mean_pred = torch.zeros(metric_shape).to(device, dtype=torch.float)
    global_mean_target = torch.zeros(metric_shape).to(device, dtype=torch.float)
    gradient_magnitude_pred = torch.zeros(metric_shape).to(device, dtype=torch.float)
    gradient_magnitude_target = torch.zeros(metric_shape).to(device, dtype=torch.float)

    acc_unweighted = torch.zeros(metric_shape).to(device, dtype=torch.float)

    output_shape = (prediction_length, n_out_channels, img_shape_x, img_shape_y)
    seq_real = torch.zeros(output_shape).to(device, dtype=torch.float)
    seq_pred = torch.zeros(output_shape).to(device, dtype=torch.float)

    valid_data = valid_data_full[
        ic : (ic + prediction_length * dt + n_history * dt) : dt
    ]  # extract valid data from first year
    if valid_data.shape[2] > 720:
        # might be necessary for ERA5 data
        valid_data = valid_data[:, :, 0:720]

    # standardize
    valid_data = (valid_data - means) / stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    # load time means
    if not params.use_daily_climatology:
        m = torch.as_tensor((time_means - means) / stds)[:, 0:img_shape_x]
        m = torch.unsqueeze(m, 0)
    else:
        # use daily clim like weyn et al. (different from rasp)
        dc_path = params.dc_path
        with h5py.File(dc_path, "r") as f:
            dc = f["time_means_daily"][
                ic : ic + prediction_length * dt : dt
            ]  # 1460,21,721,1440
        m = torch.as_tensor(
            (dc[:, params.out_channels, 0:img_shape_x, :] - means) / stds
        )

    m = m.to(device, dtype=torch.float)

    std = torch.as_tensor(stds[:, 0, 0]).to(device, dtype=torch.float)
    mean_ = torch.as_tensor(means[:, 0, 0]).to(device, dtype=torch.float)

    # autoregressive inference
    if params.log_to_screen:
        logging.info("Begin autoregressive inference")

    with torch.no_grad():
        for i in range(valid_data.shape[0]):
            if i == 0:  # start of sequence
                first = valid_data[0 : n_history + 1]
                future = valid_data[n_history + 1]
                for h in range(n_history + 1):
                    # extract history from 1st
                    indices = slice(h * n_in_channels, (h + 1) * n_in_channels)
                    seq_real[h] = first[indices][0:n_out_channels]
                    seq_pred[h] = seq_real[h]
                if params.perturb:
                    first = gaussian_perturb(
                        first, level=params.n_level, device=device
                    )  # perturb the ic
                future_pred = model(first)
            else:
                if i < prediction_length - 1:
                    future = valid_data[n_history + i + 1]
                future_pred = model(future_pred)  # autoregressive step

            if i < prediction_length - 1:  # not on the last step
                seq_pred[n_history + i + 1] = future_pred
                seq_real[n_history + i + 1] = future
                history_stack = seq_pred[i + 1 : i + 2 + n_history]

            future_pred = history_stack

            # Compute metrics
            if params.use_daily_climatology:
                clim = m[i : i + 1]
            else:
                clim = m

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar = torch.unsqueeze(seq_real[i], 0)
            valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std
            acc[i] = weighted_acc_torch_channels(pred - clim, tar - clim)
            acc_unweighted[i] = unweighted_acc_torch_channels(pred - clim, tar - clim)
            global_mean_pred[i] = weighted_global_mean_channels(pred) * std + mean_
            global_mean_target[i] = weighted_global_mean_channels(tar) * std + mean_
            gradient_magnitude_pred[i] = (
                weighted_global_mean_gradient_magnitude_channels(pred) * std
            )
            gradient_magnitude_target[i] = (
                weighted_global_mean_gradient_magnitude_channels(tar) * std
            )

            if params.log_to_screen:
                logging.info(
                    "Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}".format(
                        i, prediction_length, out_names[0], valid_loss[i, 0], acc[i, 0]
                    )
                )
            if params.log_to_wandb:
                rmse_metrics = {
                    f"rmse/ic{ic}/channel{c}-{name}": valid_loss[i, c]
                    for c, name in enumerate(out_names)
                }
                acc_metrics = {
                    f"acc/ic{ic}/channel{c}-{name}": acc[i, c]
                    for c, name in enumerate(out_names)
                }
                mean_pred_metrics = {
                    f"global_mean_prediction/ic{ic}/channel{c}-{name}": global_mean_pred[  # noqa: E501
                        i, c
                    ]
                    for c, name in enumerate(out_names)
                }
                mean_target_metrics = {
                    f"global_mean_target/ic{ic}/channel{c}-{name}": global_mean_target[
                        i, c
                    ]
                    for c, name in enumerate(out_names)
                }
                grad_mag_pred_metrics = {
                    f"global_mean_gradient_magnitude_prediction/ic{ic}/channel{c}-{name}": gradient_magnitude_pred[  # noqa: E501
                        i, c
                    ]
                    for c, name in enumerate(out_names)
                }
                grad_mag_target_metrics = {
                    f"global_mean_gradient_magnitude_target/ic{ic}/channel{c}-{name}": gradient_magnitude_target[  # noqa: E501
                        i, c
                    ]
                    for c, name in enumerate(out_names)
                }
                if params.log_on_each_unroll_step_inference:
                    wandb.log(
                        {
                            **rmse_metrics,
                            **acc_metrics,
                            **mean_pred_metrics,
                            **mean_target_metrics,
                            **grad_mag_pred_metrics,
                            **grad_mag_target_metrics,
                        }
                    )

    # populate inference logs, if the caller decides to log to wandb. Otherwise,
    # leave it as an empty dict.
    inference_logs = {}
    if params.log_to_wandb:
        # inspect snapshot times at 5-days and 10-days.
        snapshot_timesteps = [(24 // 6 * k, f"{k}-days") for k in [5, 10]]

        # TODO(gideond) move these names to a higher-level to avoid potential bugs
        metric_names = [
            "rmse",
            "acc",
            "global_mean_prediction",
            "global_mean_target",
            "global_mean_gradient_magnitude_prediction",
            "global_mean_gradient_magnitude_target",
        ]
        # All metrics has shape [metric_type, timestep, channel]
        all_metrics = [
            valid_loss,
            acc,
            global_mean_pred,
            global_mean_target,
            gradient_magnitude_pred,
            gradient_magnitude_target,
        ]
        all_metrics = np.array([m.cpu().numpy() for m in all_metrics])
        inference_logs = {}
        for t, time_name in snapshot_timesteps:
            if params.log_to_screen:
                logging.info(f"Logging metrics at {time_name}")
            for i in range(len(metric_names)):
                for j in range(len(out_names)):
                    name = (
                        f"{metric_names[i]}_{time_name}/"
                        f"ic{ic}/channel{j}-{out_names[j]}"
                    )
                    try:
                        assert (
                            name not in inference_logs
                        ), "Duplicate name in inference logs"
                        inference_logs[name] = all_metrics[i, t, j]
                    except IndexError:
                        logging.error(f"Failed to label {name}")

    lat_dim, lon_dim = 2, 3
    lat_size, lon_size = seq_real.shape[lat_dim], seq_real.shape[lon_dim]
    weights = fme.spherical_area_weights(lat_size, lon_size)
    time_rmse = compute_time_rmse(seq_real, seq_pred, weights=weights)
    time_rmse *= std.cpu()
    for i in range(len(out_names)):
        tag = f"ic{ic}/channel{i}-{out_names[i]}"
        inference_logs[f"rmse_of_time_mean/{tag}"] = time_rmse[i].item()

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()

    return (
        np.expand_dims(seq_real[n_history:], 0),
        np.expand_dims(seq_pred[n_history:], 0),
        np.expand_dims(valid_loss, 0),
        np.expand_dims(acc, 0),
        np.expand_dims(acc_unweighted, 0),
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
    params["global_batch_size"] = params.batch_size

    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.cuda.set_device(device)
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
        expDir = os.path.join(params.exp_dir, config, str(run_num))

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

    if params.log_to_wandb:
        wandb.init(config=params, project="fourcastnet-era5", entity="ai2cm")
        logging_utils.log_beaker_url()

    n_ics = params["n_initial_conditions"]

    if params["ics_type"] == "default":
        n_samples_per_year = 1336
        num_samples = n_samples_per_year - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        if vis:  # visualization for just the first ic (or any ic)
            ics = [0]
        n_ics = len(ics)
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        if (
            params.perturb
        ):  # for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch / 6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24 * day_of_year + hour_of_day
                ics.append(int(hours_since_jan_01_epoch / 6))
        n_ics = len(ics)

    logging.info("Inference for {} initial conditions".format(n_ics))
    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:  # noqa: E722
        autoregressive_inference_filetag = ""

    if vis:
        autoregressive_inference_filetag += "_vis"
    # get data and models
    valid_data_full, model = setup(params)

    # initialize lists for image sequences and RMSE/ACC
    valid_loss: List[Any] = []
    acc_unweighted = []
    acc = []
    seq_pred = []
    seq_real = []

    # run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i + 1, n_ics))
        sr, sp, vl, a, au, _ = autoregressive_inference(
            params, ic, valid_data_full, model
        )

        if i == 0 or len(valid_loss) == 0:
            seq_real = sr
            seq_pred = sp
            valid_loss = vl
            acc = a
            acc_unweighted = au
        else:
            valid_loss = np.concatenate((valid_loss, vl), 0)
            acc = np.concatenate((acc, a), 0)
            acc_unweighted = np.concatenate((acc_unweighted, au), 0)

    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]
    out_names = params.out_names

    # save predictions and loss
    filename = os.path.join(
        params["experiment_dir"],
        "autoregressive_predictions" + autoregressive_inference_filetag + ".h5",
    )
    if params.log_to_screen:
        logging.info(f"Saving files at {filename}")
    with h5py.File(filename, "a") as f:
        if vis:
            inference_data_shape = (
                n_ics,
                prediction_length,
                n_out_channels,
                img_shape_x,
                img_shape_y,
            )
            f.create_dataset(
                "ground_truth",
                data=seq_real,
                shape=inference_data_shape,
                dtype=np.float32,
            )
            f.create_dataset(
                "predicted",
                data=seq_pred,
                shape=inference_data_shape,
                dtype=np.float32,
            )

            if params.log_to_wandb:
                gap = np.zeros((prediction_length, n_out_channels, img_shape_x, 10))
                video_data = np.concatenate((seq_pred[0], gap, seq_real[0]), axis=-1)
                for c in range(n_out_channels):
                    # wandb.Video requires 4D array, hence keeping
                    # singleton channel dim
                    channel_video_data = video_data[:, [c], :, :]
                    # rescale appropriately given that wandb.Video casts
                    # data to np.uint8
                    # use 'real' data for determining max/min scaling bounds.
                    # 'pred' data may saturate bounds, so clip at 0 and 255.
                    data_min = seq_real[0][:, c, :, :].min()
                    data_max = seq_real[0][:, c, :, :].max()
                    channel_video_data = (
                        255 * (channel_video_data - data_min) / (data_max - data_min)
                    )
                    channel_video_data = np.minimum(channel_video_data, 255)
                    channel_video_data = np.maximum(channel_video_data, 0)
                    wandb_video = wandb.Video(
                        channel_video_data,
                        caption=(
                            "Autoregressive (left) prediction and"
                            f"(right) target for channel {c}"
                        ),
                    )
                    wandb.log({f"video/channel{c}-{out_names[c]}": wandb_video})

        metric_shape = (n_ics, prediction_length, n_out_channels)
        f.create_dataset("rmse", data=valid_loss, shape=metric_shape, dtype=np.float32)
        f.create_dataset("acc", data=acc, shape=metric_shape, dtype=np.float32)
        f.create_dataset(
            "acc_unweighted", data=acc_unweighted, shape=metric_shape, dtype=np.float32
        )

        f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default="00", type=str)
    parser.add_argument("--yaml_config", default="./config/AFNO.yaml", type=str)
    parser.add_argument("--config", default="full_field", type=str)
    parser.add_argument("--use_daily_climatology", action="store_true")
    parser.add_argument("--vis", action="store_true")
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
