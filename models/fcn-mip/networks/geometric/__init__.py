# flake8: noqa
import sys, os

import torch
import numpy as np

# load helper modules from FCN
from networks.YParams import YParams
from networks.geometric.models import get_model as get_model_from_params
from fcn_mip.schema import Grid
import pathlib

config_folder = pathlib.Path(__file__).parent / "config"



def get_model(grid, config_key="gfno_26ch_sc3_layers8_tt64", config_file="recovered_from_wandb.yaml"):

    yaml_config = config_folder / config_file 
    params = YParams(os.path.abspath(yaml_config), config_key)

    params["epsilon_factor"] = 0
    params["world_size"] = 1
    params["prediction_length"] = 500

    world_rank = 0

    torch.backends.cudnn.benchmark = True

    # point it to the training data
    params["n_future"] = 0

    # statistics and other info
    params[
        "min_path"
    ] = "/home/bbonev/Projects/fourcastnet/climate_fno/CWO-data/34Vars/stats/mins.npy"
    params[
        "max_path"
    ] = "/home/bbonev/Projects/fourcastnet/climate_fno/CWO-data/34Vars/stats/maxs.npy"
    params[
        "time_means_path"
    ] = "/home/bbonev/Projects/fourcastnet/climate_fno/CWO-data/34Vars/stats/time_means.npy"
    params[
        "global_means_path"
    ] = "/home/bbonev/Projects/fourcastnet/climate_fno/CWO-data/34Vars/stats/global_means.npy"
    params[
        "global_stds_path"
    ] = "/home/bbonev/Projects/fourcastnet/climate_fno/CWO-data/34Vars/stats/global_stds.npy"

    # changes that happen in load model
    # TODO refactor to shared place
    nlat = {Grid.grid_720x1440: 720, Grid.grid_721x1440: 721}[grid]

    class valid_dataset:
        img_shape_x: int = nlat
        img_shape_y: int = 1440
        crop_size_x: int = nlat
        crop_size_y: int = 1440


    # initialize global variables
    # exp_dir = params['experiment_dir']
    dt = int(params.dt)
    prediction_length = int(params.prediction_length / dt)
    n_history = params.n_history
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    crop_size_x = valid_dataset.crop_size_x
    crop_size_y = valid_dataset.crop_size_y
    params.crop_size_x = crop_size_x
    params.crop_size_y = crop_size_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    params.N_in_channels = len(in_channels)
    params.N_out_channels = len(out_channels)


    # load model
    return get_model_from_params(params)
