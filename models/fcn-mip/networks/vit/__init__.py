# flake8: noqa
import sys, os

import torch
import numpy as np

# load helper modules from FCN
from networks.YParams import YParams
from networks.vit.vit import VIT
import pathlib

config_folder = pathlib.Path(__file__).parent


def get_model(config_key="afno_26ch_v_finetune", config_file="AFNO2.yaml"):

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

    class valid_dataset:
        img_shape_x: int = 720
        img_shape_y: int = 1440
        crop_size_x: int = 720
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
    return VIT(params)
