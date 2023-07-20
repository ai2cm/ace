from typing import Dict
import numpy as np
import pathlib
import pytest
import subprocess
import tempfile
from fme.fcn_training.train import main as train_main
from fme.fcn_training.train import _restore_checkpoint
from fme.fcn_training.inference.inference import main as inference_main
import unittest.mock
import xarray as xr

REPOSITORY_PATH = pathlib.PurePath(__file__).parent.parent.parent.parent
JOB_SUBMISSION_SCRIPT_PATH = (
    REPOSITORY_PATH / "fme" / "fme" / "fcn_training" / "run-train-and-inference.sh"
)


def _get_test_yaml_files(
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
    train_string = f"""
train_data:
  data_path: '{train_data_path}'
  data_type: "FV3GFS"
  batch_size: 2
  num_data_workers: 1
validation_data:
  data_path: '{valid_data_path}'
  data_type: "FV3GFS"
  batch_size: 2
  num_data_workers: 1
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
inference_n_forward_steps: 2
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
    inference_string = f"""
experiment_dir: {results_dir}
n_forward_steps: 2
checkpoint_path: {results_dir}/training_checkpoints/best_ckpt.tar
log_video: true
save_prediction_files: true
logging:
  log_to_screen: true
  log_to_wandb: false
  log_to_file: false
  project: fme
  entity: ai2cm
validation_data:
  data_path: '{valid_data_path}'
  data_type: "FV3GFS"
  batch_size: 1
  num_data_workers: 1
  n_samples: 3
    """  # noqa: E501

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(train_string)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as f_inference:
        f_inference.write(inference_string)

    return f_train.name, f_inference.name


def _save_netcdf(
    filename, dim_sizes, variable_names, coords_override: Dict[str, xr.DataArray]
):
    data_vars = {}
    for name in variable_names:
        data = np.random.randn(*list(dim_sizes.values()))
        if len(dim_sizes) > 0:
            data = data.astype(np.float32)
        data_vars[name] = xr.DataArray(
            data, dims=list(dim_sizes), attrs={"units": "m", "long_name": name}
        )

    coords = {
        dim_name: xr.DataArray(
            np.arange(size, dtype=np.float32),
            dims=(dim_name,),
        )
        if dim_name not in coords_override
        else coords_override[dim_name]
        for dim_name, size in dim_sizes.items()
    }

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    ds.to_netcdf(filename, unlimited_dims=["time"], format="NETCDF4_CLASSIC")


def _setup(path, nettype):
    seed = 0
    np.random.seed(seed)
    in_variable_names = ["foo", "bar", "baz"]
    out_variable_names = ["foo", "bar"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names)) + [mask_name]
    data_dim_sizes = {"time": 20, "grid_yt": 16, "grid_xt": 32}
    grid_yt = np.linspace(-89.5, 89.5, data_dim_sizes["grid_yt"])
    stats_dim_sizes = {}

    data_dir = path / "data"
    stats_dir = path / "stats"
    results_dir = path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    _save_netcdf(
        data_dir / "data.nc",
        data_dim_sizes,
        all_variable_names,
        coords_override={"grid_yt": grid_yt},
    )
    _save_netcdf(
        stats_dir / "stats-mean.nc",
        stats_dim_sizes,
        all_variable_names,
        coords_override={},
    )
    _save_netcdf(
        stats_dir / "stats-stddev.nc",
        stats_dim_sizes,
        all_variable_names,
        coords_override={},
    )

    train_config_filename, inference_config_filename = _get_test_yaml_files(
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
    return train_config_filename, inference_config_filename


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
    train_config, inference_config = _setup(tmp_path, nettype)

    # using pdb requires calling main functions directly
    train_main(
        yaml_config=train_config,
    )
    # inference should not require stats files
    (tmp_path / "stats" / "stats-mean.nc").unlink()
    (tmp_path / "stats" / "stats-stddev.nc").unlink()
    inference_main(
        yaml_config=inference_config,
    )
    netcdf_output_path = tmp_path / "output" / "autoregressive_predictions.nc"
    assert netcdf_output_path.exists()
    ds = xr.open_dataset(netcdf_output_path)
    assert np.sum(np.isnan(ds["foo"].values)) == 0
    assert np.sum(np.isnan(ds["bar"].values)) == 0
    assert np.sum(np.isnan(ds["baz"].sel(source="target").values)) == 0
    assert np.all(np.isnan(ds["baz"].sel(source="prediction").values))


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
    train_config, inference_config = _setup(tmp_path, nettype)

    train_and_inference_process = subprocess.run(
        [
            JOB_SUBMISSION_SCRIPT_PATH,
            train_config,
            inference_config,
            "1",
        ]
    )
    train_and_inference_process.check_returncode()


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume(tmp_path, nettype):
    """Make sure the training is resumed from a checkpoint when restarted."""
    train_config, inference_config = _setup(tmp_path, nettype)

    mock = unittest.mock.MagicMock(side_effect=_restore_checkpoint)
    with unittest.mock.patch("fme.fcn_training.train._restore_checkpoint", new=mock):
        train_main(
            yaml_config=train_config,
        )
        assert not mock.called
        train_main(
            yaml_config=train_config,
        )
    mock.assert_called()


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume_two_workers(tmp_path, nettype, skip_slow: bool):
    """Make sure the training is resumed from a checkpoint when restarted, using
    torchrun with NPROC_PER_NODE set to 2."""
    if skip_slow:
        # script is slow as everything is re-imported when it runs
        pytest.skip("Skipping slow tests")
    train_config, inference_config = _setup(tmp_path, nettype)
    subprocess_args = [
        JOB_SUBMISSION_SCRIPT_PATH,
        train_config,
        inference_config,
        "2",  # this makes the training run on two GPUs
    ]
    initial_process = subprocess.run(subprocess_args)
    initial_process.check_returncode()
    resume_process = subprocess.run(subprocess_args)
    resume_process.check_returncode()
