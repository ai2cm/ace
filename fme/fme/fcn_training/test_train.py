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
JOB_SUBMISSION_SCRIPT_PATH = (
    REPOSITORY_PATH / "fme" / "fme" / "fcn_training" / "run-train-and-inference.sh"
)


def _get_test_yaml_file(
    train_data_path,
    valid_data_path,
    results_dir,
    global_means_path,
    global_stds_path,
    in_variable_names,
    out_variable_names,
    mask_name,
    nettype="afno",
):
    string = f"""
train_data:
  data_path: '{train_data_path}'
  data_type: "FV3GFS"
  batch_size: 2
  num_data_workers: 1
  dt: 1
validation_data:
  data_path: '{valid_data_path}'
  data_type: "FV3GFS"
  batch_size: 2
  num_data_workers: 1
  dt: 1
stepper:
  in_names: {in_variable_names}
  out_names: {out_variable_names}
  optimization:
    optimizer_type: "Adam"
    lr: 0.001
    enable_automatic_mixed_precision: true
    scheduler:
        type: CosineAnnealingLR
        kwargs:
          T_max: 1
  normalization:
    global_means_path: '{global_means_path}'
    global_stds_path: '{global_stds_path}'
  builder:
    type: {nettype}
    config:
      num_blocks: 2
      embed_dim: 12
  prescriber:
    prescribed_name: {in_variable_names[0]}
    mask_name: {mask_name}
    mask_value: 0
prediction_length: 2
max_epochs: 1
save_checkpoint: true
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: false
  project: fme
  entity: ai2cm
experiment_dir: {results_dir}
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


def _setup(path, nettype):
    seed = 0
    np.random.seed(seed)
    in_variable_names = ["foo", "bar", "baz"]
    out_variable_names = ["foo", "bar"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names)) + [mask_name]
    data_dim_sizes = {"time": 3, "grid_yt": 16, "grid_xt": 32}
    stats_dim_sizes = {}

    data_dir = path / "data"
    stats_dir = path / "stats"
    results_dir = path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    _save_netcdf(data_dir / "data.nc", data_dim_sizes, all_variable_names)
    _save_netcdf(stats_dir / "stats-mean.nc", stats_dim_sizes, all_variable_names)
    _save_netcdf(stats_dir / "stats-stddev.nc", stats_dim_sizes, all_variable_names)

    yaml_config_filename = _get_test_yaml_file(
        train_data_path=data_dir,
        valid_data_path=data_dir,
        results_dir=results_dir,
        global_means_path=stats_dir / "stats-mean.nc",
        global_stds_path=stats_dir / "stats-stddev.nc",
        in_variable_names=in_variable_names,
        out_variable_names=out_variable_names,
        mask_name=mask_name,
        nettype=nettype,
    )
    return yaml_config_filename


@pytest.mark.parametrize(
    "nettype", ["SphericalFourierNeuralOperatorNet", "FourierNeuralOperatorNet", "afno"]
)
def test_train_and_inference_inline(tmp_path, nettype):
    """Make sure that training and inference run without errors

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        debug: option for developers to allow use of pdb.
    """
    yaml_config = _setup(tmp_path, nettype)

    # using pdb requires calling main functions directly
    train_main(
        yaml_config=yaml_config,
    )

    # use --vis flag because this is how the script is called in the
    # run-train-and-inference.sh script. This option saves dataset/video of output.
    inference_main(
        yaml_config=yaml_config,
        vis=True,
    )


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_train_and_inference_script(tmp_path, nettype, skip_slow: bool):
    """Make sure that training and inference run without errors

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        debug: option for developers to allow use of pdb.
    """
    if skip_slow:
        # script is slow as everything is re-imported when it runs
        pytest.skip("Skipping slow tests")
    yaml_config = _setup(tmp_path, nettype)

    train_and_inference_process = subprocess.run(
        [
            JOB_SUBMISSION_SCRIPT_PATH,
            yaml_config,
            "1",
        ]
    )
    train_and_inference_process.check_returncode()


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume(tmp_path, nettype):
    """Make sure the training is resumed from a checkpoint when restarted."""
    yaml_config = _setup(tmp_path, nettype)

    mock = unittest.mock.MagicMock(side_effect=_restore_checkpoint)
    with unittest.mock.patch("fme.fcn_training.train._restore_checkpoint", new=mock):
        train_main(
            yaml_config=yaml_config,
        )
        assert not mock.called
        train_main(
            yaml_config=yaml_config,
        )
    mock.assert_called()


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume_two_workers(tmp_path, nettype, skip_slow: bool):
    """Make sure the training is resumed from a checkpoint when restarted, using
    torchrun with NPROC_PER_NODE set to 2."""
    if skip_slow:
        # script is slow as everything is re-imported when it runs
        pytest.skip("Skipping slow tests")
    yaml_config = _setup(tmp_path, nettype)
    subprocess_args = [
        JOB_SUBMISSION_SCRIPT_PATH,
        yaml_config,
        "2",  # this makes the training run on two GPUs
    ]
    initial_process = subprocess.run(subprocess_args)
    initial_process.check_returncode()
    resume_process = subprocess.run(subprocess_args)
    resume_process.check_returncode()
