import os
import tempfile
from typing import List

import numpy as np
import pytest

from fme.core.testing.wandb import mock_wandb

from .data_loading.test_data_loader import create_coupled_data_on_disk
from .train.train import main as train_main

_TRAIN_CONFIG_TEMPLATE = """
experiment_dir: {experiment_dir}
save_checkpoint: true
max_epochs: {max_epochs}
n_coupled_steps: {n_coupled_steps}
logging:
  log_to_screen: true
  log_to_wandb: true
  log_to_file: false
  project: project
  entity: entity
train_loader:
  batch_size: 2
  num_data_workers: 0
  dataset:
    - ocean:
        data_path: {ocean_data_path}
        subset:
            start_time: '1970-01-01'
      atmosphere:
        data_path: {atmosphere_data_path}
        subset:
            start_time: '1970-01-01'
validation_loader:
  batch_size: 2
  num_data_workers: 0
  dataset:
    - ocean:
        data_path: {ocean_data_path}
        subset:
            start_time: '1970-01-01'
      atmosphere:
        data_path: {atmosphere_data_path}
        subset:
            start_time: '1970-01-01'
inference:
  loader:
    dataset:
      ocean:
        data_path: {ocean_data_path}
        subset:
            start_time: '1970-01-01'
      atmosphere:
        data_path: {atmosphere_data_path}
        subset:
            start_time: '1970-01-01'
    start_indices:
      times:
        - '1970-01-01T00:00:00'
        - '1970-01-03T00:00:00'
  n_coupled_steps: {inference_n_coupled_steps}
  aggregator:
    log_zonal_mean_images: {log_zonal_mean_images}
optimization:
  enable_automatic_mixed_precision: false
  lr: 0.0001
  optimizer_type: Adam
stepper:
  sst_name: {ocean_sfc_temp_name}
  sst_mask_name: {ocean_sfc_mask_name}
  ocean:
    timedelta: 2D
    stepper:
      builder:
        type: SphericalFourierNeuralOperatorNet
        config:
          num_layers: 2
          embed_dim: 12
      loss:
        type: MSE
      normalization:
        global_means_path: {global_means_path}
        global_stds_path: {global_stds_path}
      in_names: {ocean_in_names}
      out_names: {ocean_out_names}
  atmosphere:
    timedelta: 1D
    stepper:
      builder:
        type: SphericalFourierNeuralOperatorNet
        config:
          num_layers: 2
          embed_dim: 12
      loss:
        type: MSE
      normalization:
        global_means_path: {global_means_path}
        global_stds_path: {global_stds_path}
      in_names: {atmos_in_names}
      out_names: {atmos_out_names}
      ocean:
        surface_temperature_name: {atmos_sfc_temp_name}
        ocean_fraction_name: {ocean_frac_name}
"""


def _write_test_yaml_files(
    tmp_path,
    mock_coupled_data,
    ocean_in_names: List[str],
    ocean_out_names: List[str],
    atmos_in_names: List[str],
    atmos_out_names: List[str],
    ocean_sfc_temp_name: str,
    ocean_sfc_mask_name: str,
    atmos_sfc_temp_name: str,
    ocean_frac_name: str,
    n_coupled_steps: int = 1,
    max_epochs: int = 1,
    inference_n_coupled_steps: int = 2,
    log_zonal_mean_images: bool = False,
):
    train_config = _TRAIN_CONFIG_TEMPLATE.format(
        experiment_dir=tmp_path / "exper",
        max_epochs=max_epochs,
        n_coupled_steps=n_coupled_steps,
        ocean_data_path=mock_coupled_data.ocean_dir,
        atmosphere_data_path=mock_coupled_data.atmosphere_dir,
        global_means_path=os.path.join(mock_coupled_data.means_path),
        global_stds_path=os.path.join(mock_coupled_data.stds_path),
        inference_n_coupled_steps=inference_n_coupled_steps,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        ocean_sfc_temp_name=ocean_sfc_temp_name,
        ocean_sfc_mask_name=ocean_sfc_mask_name,
        atmos_sfc_temp_name=atmos_sfc_temp_name,
        ocean_frac_name=ocean_frac_name,
        log_zonal_mean_images=str(log_zonal_mean_images).lower(),
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(train_config)

    return f_train.name


@pytest.mark.parametrize(
    "log_zonal_mean_images",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason=(
                    "There is an unresolved bug when logging "
                    "zonal mean images during coupled inference"
                )
            ),
        ),
    ],
)
def test_train_and_inference_inline(tmp_path, very_fast_only, log_zonal_mean_images):
    # TODO: call the standalone inference entrypoint in this test, when
    # implemented in a downstream PR
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    data_dir = tmp_path / "coupled_data"
    data_dir.mkdir()
    ocean_names = ["o_exog", "o_prog", "o_sfc", "o_mask"]
    atmos_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]

    outnames = ["o_prog", "o_sfc", "a_diag", "a_prog", "a_sfc"]

    n_forward_times_ocean = 3
    n_forward_times_atmos = 6
    mock_coupled_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=True,
    )

    ocean_in_names = ["o_exog", "o_prog", "o_sfc", "o_mask"]
    ocean_out_names = ["o_prog", "o_sfc"]
    atmos_in_names = ["a_exog", "a_diag", "a_prog", "a_sfc", "constant_mask"]
    atmos_out_names = ["a_diag", "a_prog", "a_sfc"]

    train_config_fname = _write_test_yaml_files(
        tmp_path,
        mock_coupled_data,
        ocean_in_names,
        ocean_out_names,
        atmos_in_names,
        atmos_out_names,
        ocean_sfc_temp_name="o_sfc",
        ocean_sfc_mask_name="o_mask",
        atmos_sfc_temp_name="a_sfc",
        ocean_frac_name="constant_mask",
        log_zonal_mean_images=log_zonal_mean_images,
    )

    with mock_wandb() as wandb:  # noqa
        train_main(yaml_config=train_config_fname)
        wandb_logs = wandb.get_logs()

    for log in wandb_logs:
        # ensure inference time series is not logged
        assert "inference/mean/forecast_step" not in log

    batch_logs = wandb_logs[0]
    assert "batch_loss" in batch_logs
    assert "batch_loss/ocean" in batch_logs

    epoch_logs = wandb_logs[1]
    assert "train/mean/loss" in epoch_logs
    assert "val/mean/loss" in epoch_logs
    assert "inference/time_mean_norm/rmse/channel_mean" in epoch_logs
    assert "inference/time_mean_norm/rmse/ocean_channel_mean" in epoch_logs
    assert "inference/time_mean_norm/rmse/atmosphere_channel_mean" in epoch_logs
    assert np.isclose(
        epoch_logs["inference/time_mean_norm/rmse/channel_mean"],
        (
            0.4 * epoch_logs["inference/time_mean_norm/rmse/ocean_channel_mean"]
            + 0.6 * epoch_logs["inference/time_mean_norm/rmse/atmosphere_channel_mean"]
        ),
    )
    for name in outnames:
        assert f"inference/time_mean/rmse/{name}" in epoch_logs

    # check that inference map captions includes expected units
    for map_name in ["val/mean_map/image-error", "inference/time_mean/bias_map"]:
        for name in ["o_prog", "a_diag"]:
            wandb_img = epoch_logs[f"{map_name}/{name}"]
            captions = wandb_img.captions([wandb_img])
            assert len(captions) > 0
            assert name in captions[0]
            assert "[unknown_units]" not in captions[0]
            assert "[m]" in captions[0]  # set by _save_netcdf
