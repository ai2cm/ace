import copy
import pathlib
import tempfile

import numpy as np
import pytest
import torch

from fme.ace.testing import (
    DimSizes,
    MonthlyReferenceData,
    save_nd_netcdf,
    save_scalar_netcdf,
)
from fme.core.coordinates import HorizontalCoordinates, LatLonCoordinates
from fme.core.generics.trainer import count_parameters
from fme.core.testing.wandb import mock_wandb
from fme.diffusion.train import main as train_main

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
    n_forward_steps=1,
    nettype="ConditionalSFNO",
    log_to_wandb=False,
    max_epochs=1,
    segment_epochs=1,
    inference_forward_steps=2,
):
    config_str = """
      num_layers: 2
      embed_dim: 12"""
    spatial_dimensions_str = "latlon"

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
    type: "MSE"
  builder:
    type: {nettype}
    config: {config_str}
  ocean:
    surface_temperature_name: {in_variable_names[0]}
    ocean_fraction_name: {mask_name}
  n_sigma_embedding_channels: 6
"""

    stepper_config = new_stepper_config

    train_string = f"""
train_loader:
  dataset:
    - data_path: '{train_data_path}'
      spatial_dimensions: {spatial_dimensions_str}
  batch_size: 2
  num_data_workers: 0
validation_loader:
  dataset:
    - data_path: '{valid_data_path}'
      spatial_dimensions: {spatial_dimensions_str}
  batch_size: 2
  num_data_workers: 0
optimization:
  optimizer_type: "Adam"
  lr: 0.001
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
      spatial_dimensions: {spatial_dimensions_str}
    start_indices:
      first: 0
      n_initial_conditions: 2
      interval: 1
  n_forward_steps: {inference_forward_steps}
  forward_steps_in_memory: 4
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
    spatial_dimensions: {spatial_dimensions_str}
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


def get_sizes(
    spatial_dims: HorizontalCoordinates = LatLonCoordinates(
        lon=torch.Tensor(np.arange(32)),
        lat=torch.Tensor(np.arange(16)),
        loaded_lat_name="lat",
        loaded_lon_name="lon",
    ),
    n_time=3,
    nz_interface=3,
) -> DimSizes:
    return DimSizes(
        n_time=n_time,
        horizontal=copy.deepcopy(spatial_dims.loaded_sizes),
        nz_interface=nz_interface,
    )


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
    in_variable_names = [
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "baz",
    ]
    out_variable_names = ["PRESsfc", "specific_total_water_0", "specific_total_water_1"]
    mask_name = "mask"
    all_variable_names = list(set(in_variable_names + out_variable_names))

    dim_sizes = get_sizes(n_time=n_time)

    data_dir = path / "data"
    stats_dir = path / "stats"
    results_dir = path / "output"
    data_dir.mkdir()
    stats_dir.mkdir()
    results_dir.mkdir()
    save_nd_netcdf(
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

    monthly_dim_sizes = get_sizes(n_time=10 * 12, nz_interface=1)
    monthly_reference_data = MonthlyReferenceData(
        path=data_dir,
        names=out_variable_names,
        dim_sizes=monthly_dim_sizes,
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
    "nettype",
    [
        "ConditionalSFNO",
    ],
)
def test_train_inline(tmp_path, nettype, very_fast_only: bool):
    """Make sure that training runs without errors.

    Args:
        tmp_path: pytext fixture for temporary workspace.
        nettype: parameter indicating model architecture to use.
        very_fast_only: parameter indicating whether to skip slow tests.
    """
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    # need multi-year to cover annual aggregator
    train_config, _ = _setup(
        tmp_path,
        nettype,
        log_to_wandb=True,
        timestep_days=20,
        n_time=int(366 * 3 / 20 + 1),
        inference_forward_steps=int(366 * 3 / 20 / 2 - 1) * 2,  # must be even
    )
    # using pdb requires calling main functions directly
    with mock_wandb() as wandb:
        train_main(
            yaml_config=train_config,
        )
        wandb_logs = wandb.get_logs()

        for log in wandb_logs:
            # ensure inference time series is not logged
            assert "inference/mean/forecast_step" not in log


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
