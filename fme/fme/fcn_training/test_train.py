# Author(s): Gideon Dresdner <gideond@allenai.org>

import h5py
import numpy as np
import subprocess
import tempfile

def _get_test_yaml_file(train_data_path, 
                        valid_data_path, 
                        inf_data_path, 
                        results_dir,
                        time_means_path,
                        global_means_path,
                        global_stds_path,
                        prediction_length,
                        num_channels=2,
                        config_name="unit_test"):

    channels = list(range(num_channels))

    string = f"""
     {config_name}: &{config_name}
       loss: 'l2'
       lr: 5E-4
       scheduler: 'CosineAnnealingLR'
       num_data_workers: 4
       dt: 1 # how many timesteps ahead the model will predict
       n_history: 0 #how many previous timesteps to consider
       prediction_type: 'iterative'
       prediction_length: {prediction_length} #applicable only if prediction_type == 'iterative'
       n_initial_conditions: 1 #applicable only if prediction_type == 'iterative'
       ics_type: "default"
       save_raw_forecasts: !!bool True
       save_channel: !!bool False
       masked_acc: !!bool False
       maskpath: None
       perturb: !!bool False
       add_grid: !!bool False
       N_grid_channels: 0
       gridtype: 'sinusoidal' #options 'sinusoidal' or 'linear'
       roll: !!bool False
       max_epochs: 1
       batch_size: 2

       #afno hyperparams
       num_blocks: 8
       nettype: 'afno'
       patch_size: 8
       width: 56
       modes: 32
       #options default, residual
       target: 'default'
       in_channels: {channels}
       out_channels: {channels}   #must be same as in_channels if prediction_type == 'iterative'
       normalization: 'zscore' #options zscore (minmax not supported)
       train_data_path: '{train_data_path}'
       valid_data_path: '{valid_data_path}'
       inf_data_path: '{inf_data_path}'
       exp_dir: '{results_dir}'
       time_means_path:   '{time_means_path}'
       global_means_path: '{global_means_path}'
       global_stds_path:  '{global_stds_path}'

       orography: !!bool False
       orography_path: None

       log_to_screen: !!bool True
       log_to_wandb: !!bool False
       save_checkpoint: !!bool True

       enable_nhwc: !!bool False
       optimizer_type: 'FusedAdam'
       crop_size_x: None
       crop_size_y: None

       two_step_training: !!bool False
       plot_animations: !!bool False

       add_noise: !!bool False
       noise_std: 0
    """

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        f.write(string)
        return f.name

def _save_to_tmpfile(data, dir, filetype='h5'):
    suffices = {'h5': '.h5', 'npy': '.npy'}
    if filetype not in suffices:
        raise ValueError(f'Unknown save format {filetype}')

    with tempfile.NamedTemporaryFile(dir=dir, mode='w', delete=False, suffix=suffices[filetype]) as f:
        if filetype == 'h5':
            with h5py.File(f.name, 'w') as hf:
                hf.create_dataset('fields', data=data)
        elif filetype == 'npy':
            np.save(f.name, data)
        else:
            raise ValueError(f'Unknown save format {filetype}')
        return f.name


def test_train_runs_era5():
    """Make sure that training runs without errors."""

    # TODO(gideond) parameterize
    seed = 0
    np.random.seed(seed)
    num_time_steps, num_channels, height, width = 2, 2, 720, 40

    with tempfile.TemporaryDirectory() as train_dir, \
         tempfile.TemporaryDirectory() as valid_dir, \
         tempfile.TemporaryDirectory() as stats_dir, \
         tempfile.TemporaryDirectory() as results_dir:
        _ = _save_to_tmpfile(np.random.randn(
            num_time_steps, num_channels, height + 1, width), dir=train_dir, filetype='h5')
        _ = _save_to_tmpfile(np.random.randn(
            num_time_steps, num_channels, height + 1, width), dir=valid_dir, filetype='h5')
        time_means = _save_to_tmpfile(np.random.randn(
            1, num_channels + 1, height, width), dir=valid_dir, filetype='npy')
        global_means = _save_to_tmpfile(np.random.randn(
            1, num_channels + 1, height, width), dir=stats_dir, filetype='npy')
        global_stds = _save_to_tmpfile(abs(np.random.randn(
            1, num_channels + 1, height, width)), dir=stats_dir, filetype='npy')

        yaml_config = _get_test_yaml_file(
            train_dir, valid_dir, valid_dir, 
            results_dir, time_means, global_means, global_stds, 
            prediction_length=num_time_steps, 
            num_channels=num_channels)

        train_process = subprocess.run(['python', 'train.py', '--yaml_config', yaml_config, '--config', 'unit_test'])
        train_process.check_returncode()