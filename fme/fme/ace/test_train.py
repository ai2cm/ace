import pathlib
import subprocess
import tempfile
import unittest.mock

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.inference.evaluator import main as inference_evaluator_main
from fme.ace.train.train import _restore_checkpoint, count_parameters
from fme.ace.train.train import main as train_main
from fme.ace.train.train_config import epoch_checkpoint_enabled
from fme.core.data_loading.config import Slice
from fme.core.testing import (
    DimSizes,
    MonthlyReferenceData,
    save_2d_netcdf,
    save_scalar_netcdf,
)
from fme.core.testing.wandb import mock_wandb

REPOSITORY_PATH = pathlib.PurePath(__file__).parent.parent.parent.parent
JOB_SUBMISSION_SCRIPT_PATH = (
    REPOSITORY_PATH / "fme" / "fme" / "ace" / "run-train-and-inference.sh"
)


def _get_test_yaml_files(
    *,
    train_data_path,
    valid_data_path,
    monthly_data_filename,
    results_dir,
    global_means_path,
    global_stds_path,
    in_variable_names,
    out_variable_names,
    mask_name,
    n_forward_steps=2,
    nettype="SphericalFourierNeuralOperatorNet",
    stepper_checkpoint_file=None,
    log_to_wandb=False,
    max_epochs=1,
    segment_epochs=1,
    inference_forward_steps=2,
):
    new_stepper_config = f"""
  in_names: {in_variable_names}
  out_names: {out_variable_names}
  normalization:
    global_means_path: '{global_means_path}'
    global_stds_path: '{global_stds_path}'
  residual_normalization:
    global_means_path: '{global_means_path}'
    global_stds_path: '{global_stds_path}'
  loss:
    global_mean_type: "LpLoss"
  builder:
    type: {nettype}
    config:
      num_layers: 2
      embed_dim: 12
  ocean:
    surface_temperature_name: {in_variable_names[0]}
    ocean_fraction_name: {mask_name}
"""
    existing_stepper_config = f"""
  checkpoint_file: {stepper_checkpoint_file}
"""

    if stepper_checkpoint_file:
        stepper_config = existing_stepper_config
    else:
        stepper_config = new_stepper_config

    train_string = f"""
train_loader:
  dataset:
    - data_path: '{train_data_path}'
  batch_size: 2
  num_data_workers: 1
validation_loader:
  dataset:
    - data_path: '{valid_data_path}'
  batch_size: 2
  num_data_workers: 1
optimization:
  optimizer_type: "Adam"
  lr: 0.001
  enable_automatic_mixed_precision: true
  scheduler:
      type: CosineAnnealingLR
      kwargs:
        T_max: 1
stepper:
{stepper_config}
inference:
  aggregator:
    monthly_reference_data: {monthly_data_filename}
  loader:
    dataset:
        data_path: '{valid_data_path}'
    start_indices:
      first: 0
      n_initial_conditions: 2
      interval: 1
  n_forward_steps: {inference_forward_steps}
  forward_steps_in_memory: 2
n_forward_steps: {n_forward_steps}
max_epochs: {max_epochs}
segment_epochs: {segment_epochs}
save_checkpoint: true
logging:
  log_to_screen: true
  log_to_wandb: {str(log_to_wandb).lower()}
  log_to_file: false
  project: fme
  entity: ai2cm
experiment_dir: {results_dir}
    """  # noqa: E501
    inference_string = f"""
experiment_dir: {results_dir}
n_forward_steps: 6
forward_steps_in_memory: 2
checkpoint_path: {results_dir}/training_checkpoints/best_ckpt.tar
data_writer:
  save_prediction_files: true
aggregator:
  log_video: true
logging:
  log_to_screen: true
  log_to_wandb: {str(log_to_wandb).lower()}
  log_to_file: false
  project: fme
  entity: ai2cm
loader:
  dataset:
    data_path: '{valid_data_path}'
  start_indices:
    first: 0
    n_initial_conditions: 2
    interval: 1
    """  # noqa: E501

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(train_string)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as f_inference:
        f_inference.write(inference_string)

    return f_train.name, f_inference.name


def _setup(
    path,
    nettype,
    log_to_wandb=False,
    max_epochs=1,
    segment_epochs=1,
    n_time=10,
    timestep_days=5,
    inference_forward_steps=2,
):
    if not path.exists():
        path.mkdir()
    seed = 0
    np.random.seed(seed)
    in_variable_names = ["foo", "bar", "baz"]
    out_variable_names = ["foo", "bar"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names))

    dim_sizes = DimSizes(
        n_time=n_time,
        n_lat=16,
        n_lon=32,
        nz_interface=7,
    )

    data_dir = path / "data"
    stats_dir = path / "stats"
    results_dir = path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    save_2d_netcdf(
        data_dir / "data.nc",
        dim_sizes,
        variable_names=all_variable_names + [mask_name],
        timestep_days=1,
    )
    save_scalar_netcdf(
        stats_dir / "stats-mean.nc",
        variable_names=all_variable_names,
    )
    save_scalar_netcdf(
        stats_dir / "stats-stddev.nc",
        variable_names=all_variable_names,
    )

    monthly_reference_data = MonthlyReferenceData(
        path=data_dir,
        names=out_variable_names,
        dim_sizes=DimSizes(
            n_time=10 * 12,
            n_lat=16,
            n_lon=32,
            nz_interface=1,
        ),
        n_ensemble=3,
    )

    train_config_filename, inference_config_filename = _get_test_yaml_files(
        train_data_path=data_dir,
        valid_data_path=data_dir,
        monthly_data_filename=monthly_reference_data.data_filename,
        results_dir=results_dir,
        global_means_path=stats_dir / "stats-mean.nc",
        global_stds_path=stats_dir / "stats-stddev.nc",
        in_variable_names=in_variable_names,
        out_variable_names=out_variable_names,
        mask_name=mask_name,
        nettype=nettype,
        log_to_wandb=log_to_wandb,
        max_epochs=max_epochs,
        segment_epochs=segment_epochs,
        inference_forward_steps=inference_forward_steps,
    )
    return train_config_filename, inference_config_filename


@pytest.mark.parametrize(
    "nettype", ["SphericalFourierNeuralOperatorNet", "SFNO-v0.1.0"]
)
def test_train_and_inference_inline(tmp_path, nettype):
    """Make sure that training and inference run without errors

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        debug: option for developers to allow use of pdb.
    """
    # need multi-year to cover annual aggregator
    train_config, inference_config = _setup(
        tmp_path,
        nettype,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
    )
    # using pdb requires calling main functions directly
    train_main(
        yaml_config=train_config,
    )
    # inference should not require stats files
    (tmp_path / "stats" / "stats-mean.nc").unlink()
    (tmp_path / "stats" / "stats-stddev.nc").unlink()
    inference_logs = inference_evaluator_main(yaml_config=inference_config)
    assert len(inference_logs) == 7  # 6 forward steps + 1 initial state
    prediction_output_path = tmp_path / "output" / "autoregressive_predictions.nc"
    assert prediction_output_path.exists()
    best_checkpoint_path = (
        tmp_path / "output" / "training_checkpoints" / "best_ckpt.tar"
    )
    assert best_checkpoint_path.exists()
    best_inference_checkpoint_path = (
        tmp_path / "output" / "training_checkpoints" / "best_inference_ckpt.tar"
    )
    assert best_inference_checkpoint_path.exists()
    ds_prediction = xr.open_dataset(prediction_output_path)
    assert np.sum(np.isnan(ds_prediction["foo"].values)) == 0
    assert np.sum(np.isnan(ds_prediction["bar"].values)) == 0
    ds_target = xr.open_dataset(tmp_path / "output" / "autoregressive_target.nc")
    assert np.sum(np.isnan(ds_target["baz"].values)) == 0


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume(tmp_path, nettype):
    """Make sure the training is resumed from a checkpoint when restarted."""

    mock = unittest.mock.MagicMock(side_effect=_restore_checkpoint)
    with unittest.mock.patch("fme.ace.train.train._restore_checkpoint", new=mock):
        train_config, _ = _setup(
            tmp_path, nettype, log_to_wandb=True, max_epochs=2, segment_epochs=1
        )
        with mock_wandb() as wandb:
            train_main(
                yaml_config=train_config,
            )
        assert (
            min([val["epoch"] for val in wandb.get_logs().values() if "epoch" in val])
            == 0
        )
        assert (
            max([val["epoch"] for val in wandb.get_logs().values() if "epoch" in val])
            == 0
        )
        assert not mock.called
        with mock_wandb() as wandb:
            train_main(
                yaml_config=train_config,
            )
        mock.assert_called()
        assert (
            min([val["epoch"] for val in wandb.get_logs().values() if "epoch" in val])
            == 1
        )
        assert (
            max([val["epoch"] for val in wandb.get_logs().values() if "epoch" in val])
            == 1
        )


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_resume_two_workers(tmp_path, nettype, skip_slow: bool, tmpdir: pathlib.Path):
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
    initial_process = subprocess.run(subprocess_args, cwd=tmpdir)
    initial_process.check_returncode()
    resume_subprocess_args = [
        "torchrun",
        "--nproc_per_node",
        "2",
        "-m",
        "fme.ace.train",
        train_config,
    ]
    resume_process = subprocess.run(resume_subprocess_args, cwd=tmpdir)
    resume_process.check_returncode()


def _create_fine_tuning_config(path_to_train_config_yaml: str, path_to_checkpoint: str):
    # TODO(gideond) rename to "overwrite" or something of that nature
    with open(path_to_train_config_yaml, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        config_data["stepper"] = {"checkpoint_path": path_to_checkpoint}
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as new_config_file:
            new_config_file.write(yaml.dump(config_data))

    return new_config_file.name


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_fine_tuning(tmp_path, nettype):
    """Check that fine tuning config runs without errors."""
    train_config, _ = _setup(tmp_path, nettype)

    train_main(
        yaml_config=train_config,
    )

    results_dir = tmp_path / "output"
    ckpt = f"{results_dir}/training_checkpoints/best_ckpt.tar"

    fine_tuning_config = _create_fine_tuning_config(train_config, ckpt)

    train_main(yaml_config=fine_tuning_config)


def _create_copy_weights_after_batch_config(
    path_to_train_config_yaml: str, path_to_checkpoint: str, experiment_dir: str
):
    # TODO(gideond) rename to "overwrite" or something of that nature
    with open(path_to_train_config_yaml, "r") as config_file:
        config_data = yaml.safe_load(config_file)
        config_data["stepper"]["parameter_init"] = {"weights_path": path_to_checkpoint}
        config_data["copy_weights_after_batch"] = {"include": ["*"], "exclude": []}
        config_data["experiment_dir"] = experiment_dir
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".yaml"
        ) as new_config_file:
            new_config_file.write(yaml.dump(config_data))

    return new_config_file.name


@pytest.mark.parametrize("nettype", ["SphericalFourierNeuralOperatorNet"])
def test_copy_weights_after_batch(tmp_path, nettype, skip_slow: bool):
    """Check that fine tuning config using copy_weights_after_batch
    runs without errors."""
    if skip_slow:
        pytest.skip("Skipping slow tests")

    train_config, _ = _setup(tmp_path, nettype)

    train_main(
        yaml_config=train_config,
    )

    results_dir = tmp_path / "output"
    ckpt = f"{results_dir}/training_checkpoints/best_ckpt.tar"

    fine_tuning_config = _create_copy_weights_after_batch_config(
        train_config, ckpt, experiment_dir=str(tmp_path / "fine_tuning_dir")
    )
    train_main(yaml_config=fine_tuning_config)


@pytest.mark.parametrize(
    "checkpoint_save_epochs,expected_save_epochs",
    [(None, []), (Slice(start=-2), [2, 3]), (Slice(step=2), [0, 2])],
)
def test_epoch_checkpoint_enabled(checkpoint_save_epochs, expected_save_epochs):
    max_epochs = 4
    for i in range(max_epochs):
        if i in expected_save_epochs:
            assert epoch_checkpoint_enabled(i, max_epochs, checkpoint_save_epochs)
        else:
            assert not epoch_checkpoint_enabled(i, max_epochs, checkpoint_save_epochs)


@pytest.mark.parametrize(
    "module_list,expected_num_parameters",
    [
        (torch.nn.ModuleList([torch.nn.Linear(10, 5), torch.nn.Linear(5, 2)]), 67),
        (torch.nn.ModuleList([]), 0),
    ],
)
def test_count_parameters(module_list, expected_num_parameters):
    num_parameters = count_parameters(module_list)
    assert num_parameters == expected_num_parameters
