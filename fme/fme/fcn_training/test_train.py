# Author(s): Gideon Dresdner <gideond@allenai.org>

import netCDF4
import numpy as np
import pathlib
import pytest
import subprocess
import tempfile
from fme.fcn_training.train import main as train_main
from fme.fcn_training.train import _restore_checkpoint
from fme.fcn_training.inference.inference import main as inference_main
import unittest.mock

REPOSITORY_PATH = pathlib.PurePath(__file__).parent.parent.parent.parent


def _get_test_yaml_file(
    train_data_path,
    valid_data_path,
    inf_data_path,
    results_dir,
    time_means_path,
    global_means_path,
    global_stds_path,
    prediction_length,
    variable_names,
    config_name="unit_test",
    nettype="afno",
):
    string = f"""
     {config_name}: &{config_name}
       data_type: 'FV3GFS'
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
       perturb: !!bool False
       N_grid_channels: 0
       max_epochs: 1
       batch_size: 2

       #afno hyperparams
       num_blocks: 8
       nettype: '{nettype}'
       patch_size: 8
       embed_dim: 8
       width: 56
       modes: 32
       in_names: {variable_names}
       out_names: {variable_names}   #must be same as in_channels if prediction_type == 'iterative'
       normalization: 'zscore' #options zscore (minmax not supported)
       train_data_path: '{train_data_path}'
       valid_data_path: '{valid_data_path}'
       inf_data_path: '{inf_data_path}'
       exp_dir: '{results_dir}'
       time_means_path:   '{time_means_path}'
       global_means_path: '{global_means_path}'
       global_stds_path:  '{global_stds_path}'

       log_to_screen: !!bool True
       log_to_wandb: !!bool False
       save_checkpoint: !!bool True

       enable_nhwc: !!bool False
       optimizer_type: 'Adam'

       plot_animations: !!bool False

       normalize: !!bool True
       compression: tt
    """  # noqa: E501

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        f.write(string)
        return f.name


def _save_netcdf(filename, dim_sizes, variable_names):
    """Save netCDF with random data and given dims and variables."""
    ds = netCDF4.Dataset(filename, "w", format="NETCDF4_CLASSIC")
    for dim_name, size in dim_sizes.items():
        dim_size = None if dim_name == "time" else size
        ds.createDimension(dim_name, dim_size)
        ds.createVariable(dim_name, np.float32, (dim_name,))
        ds[dim_name][:] = np.arange(size)
    for name in variable_names:
        ds.createVariable(name, np.float32, list(dim_sizes))
        ds[name][:] = np.random.randn(*list(dim_sizes.values()))
    ds.close()


@pytest.mark.parametrize("nettype", ["afno", "FourierNeuralOperatorNet"])
def test_train_and_inference_runs(tmp_path, nettype, debug=False):
    """Make sure that training and inference run without errors

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        debug: option for developers to allow use of pdb.
    """
    seed = 0
    np.random.seed(seed)
    config_name = "unit_test"
    variable_names = ["foo", "bar"]
    data_dim_sizes = {"time": 3, "grid_yt": 16, "grid_xt": 32}
    stats_dim_sizes = {}
    time_mean_dim_sizes = {k: data_dim_sizes[k] for k in ["grid_yt", "grid_xt"]}

    data_dir = tmp_path / "data"
    stats_dir = tmp_path / "stats"
    results_dir = tmp_path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    _save_netcdf(data_dir / "data.nc", data_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-timemean.nc", time_mean_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-mean.nc", stats_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-stddev.nc", stats_dim_sizes, variable_names)

    yaml_config = _get_test_yaml_file(
        data_dir,
        data_dir,
        data_dir,
        results_dir,
        stats_dir / "stats-timemean.nc",
        stats_dir / "stats-mean.nc",
        stats_dir / "stats-stddev.nc",
        prediction_length=2,
        variable_names=variable_names,
        nettype=nettype,
    )

    if debug:
        # using pdb requires calling main functions directly
        train_main(
            run_num="00",
            yaml_config=yaml_config,
            config=config_name,
            enable_automatic_mixed_precision=False,
        )

        # use --vis flag because this is how the script is called in the
        # run-train-and-inference.sh script. This option saves dataset/video of output.
        inference_main(
            run_num="00",
            yaml_config=yaml_config,
            config=config_name,
            use_daily_climatology=False,
            vis=True,
            override_dir=None,
            weights=None,
        )
    else:
        # in regular testing, call the actual submission script used for batch jobs
        script_path = (
            REPOSITORY_PATH
            / "fme"
            / "fme"
            / "fcn_training"
            / "run-train-and-inference.sh"
        )
        train_and_inference_process = subprocess.run(
            [
                script_path,
                yaml_config,
                config_name,
                "1",
            ]
        )
        train_and_inference_process.check_returncode()


@pytest.mark.parametrize("nettype", ["FourierNeuralOperatorNet"])
def test_resume(tmp_path, nettype):
    """Make sure the training is resumed from a checkpoint when restarted."""
    seed = 0
    np.random.seed(seed)
    config_name = "unit_test"
    variable_names = ["foo", "bar"]
    data_dim_sizes = {"time": 3, "grid_yt": 16, "grid_xt": 32}
    stats_dim_sizes = {}
    time_mean_dim_sizes = {k: data_dim_sizes[k] for k in ["grid_yt", "grid_xt"]}

    data_dir = tmp_path / "data"
    stats_dir = tmp_path / "stats"
    results_dir = tmp_path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    _save_netcdf(data_dir / "data.nc", data_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-timemean.nc", time_mean_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-mean.nc", stats_dim_sizes, variable_names)
    _save_netcdf(stats_dir / "stats-stddev.nc", stats_dim_sizes, variable_names)

    yaml_config = _get_test_yaml_file(
        data_dir,
        data_dir,
        data_dir,
        results_dir,
        stats_dir / "stats-timemean.nc",
        stats_dir / "stats-mean.nc",
        stats_dir / "stats-stddev.nc",
        prediction_length=2,
        variable_names=variable_names,
        nettype=nettype,
    )

    mock = unittest.mock.MagicMock(side_effect=_restore_checkpoint)
    with unittest.mock.patch("fme.fcn_training.train._restore_checkpoint", new=mock):
        train_main(
            run_num="00",
            yaml_config=yaml_config,
            config=config_name,
            enable_automatic_mixed_precision=False,
        )
        assert not mock.called
        train_main(
            run_num="00",
            yaml_config=yaml_config,
            config=config_name,
            enable_automatic_mixed_precision=False,
        )
    mock.assert_called()
