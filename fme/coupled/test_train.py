import os
import tempfile
from typing import List

import numpy as np
import pytest
import xarray as xr

from fme.core.testing.wandb import mock_wandb

from .data_loading.test_data_loader import create_coupled_data_on_disk
from .inference.evaluator import main as inference_evaluator_main
from .train.train import main as train_main

_TRAIN_CONFIG_TEMPLATE = """
experiment_dir: {experiment_dir}
save_checkpoint: true
save_per_epoch_diagnostics: {save_per_epoch_diagnostics}
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
        fill_nans:
            method: constant
            value: 0
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
        fill_nans:
            method: constant
            value: 0
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
        fill_nans:
            method: constant
            value: 0
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
  ocean_fraction_prediction:
    sea_ice_fraction_name: {sea_ice_frac_name}
    land_fraction_name: {land_frac_name}
  ocean:
    timedelta: 2D
    loss_contributions:
      weight: {loss_ocean_weight}
    stepper:
      builder:
        type: Samudra
        config:
            ch_width: [8, 16]
            dilation: [2, 4]
            n_layers: [1, 1]
            norm: batch
            norm_kwargs:
                track_running_stats: false
      loss:
        type: MSE
      normalization:
        global_means_path: {global_means_path}
        global_stds_path: {global_stds_path}
      corrector:
        type: "ocean_corrector"
        config:
          sea_ice_fraction_correction:
            sea_ice_fraction_name: {sea_ice_frac_name}
            land_fraction_name: {land_frac_name}
      next_step_forcing_names: {ocean_next_step_forcing_names}
      in_names: {ocean_in_names}
      out_names: {ocean_out_names}
  atmosphere:
    timedelta: 1D
    loss_contributions:
      n_steps: {loss_atmos_n_steps}
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
      corrector:
        type: "atmosphere_corrector"
        config:
          conserve_dry_air: true
"""

_INFERENCE_CONFIG_TEMPLATE = """
experiment_dir: {experiment_dir}
n_coupled_steps: {n_coupled_steps}
coupled_steps_in_memory: {coupled_steps_in_memory}
checkpoint_path: {checkpoint_path}
data_writer:
  ocean:
    save_prediction_files: true
  atmosphere:
    save_prediction_files: true
aggregator:
  log_video: true
logging:
  log_to_screen: true
  log_to_wandb: true
  log_to_file: false
  project: project
  entity: entity
loader:
  dataset:
    ocean:
      data_path: {ocean_data_path}
      subset:
          start_time: '1970-01-01'
      fill_nans:
          method: constant
          value: 0
    atmosphere:
      data_path: {atmosphere_data_path}
      subset:
          start_time: '1970-01-01'
  start_indices:
    times:
      - '1970-01-01T00:00:00'
      - '1970-01-03T00:00:00'
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
    sea_ice_frac_name: str,
    land_frac_name: str,
    atmos_sfc_temp_name: str,
    ocean_frac_name: str,
    n_coupled_steps: int = 1,
    max_epochs: int = 1,
    inline_inference_n_coupled_steps: int = 3,
    log_zonal_mean_images: bool = False,
    inference_n_coupled_steps: int = 6,
    coupled_steps_in_memory: int = 2,
    save_per_epoch_diagnostics: bool = True,
    loss_atmos_n_steps: int = 1000,  # large number ~= inf
    loss_ocean_weight: float = 1.0,
):
    exper_dir = tmp_path / "results"
    ocean_next_step_forcing_names = list(
        set(atmos_out_names).intersection(ocean_in_names)
    )
    train_config = _TRAIN_CONFIG_TEMPLATE.format(
        experiment_dir=exper_dir,
        max_epochs=max_epochs,
        n_coupled_steps=n_coupled_steps,
        ocean_data_path=mock_coupled_data.ocean_dir,
        atmosphere_data_path=mock_coupled_data.atmosphere_dir,
        global_means_path=os.path.join(mock_coupled_data.means_path),
        global_stds_path=os.path.join(mock_coupled_data.stds_path),
        inference_n_coupled_steps=inline_inference_n_coupled_steps,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        ocean_next_step_forcing_names=ocean_next_step_forcing_names,
        ocean_sfc_temp_name=ocean_sfc_temp_name,
        ocean_sfc_mask_name=ocean_sfc_mask_name,
        sea_ice_frac_name=sea_ice_frac_name,
        land_frac_name=land_frac_name,
        atmos_sfc_temp_name=atmos_sfc_temp_name,
        ocean_frac_name=ocean_frac_name,
        log_zonal_mean_images=str(log_zonal_mean_images).lower(),
        save_per_epoch_diagnostics=str(save_per_epoch_diagnostics).lower(),
        loss_atmos_n_steps=loss_atmos_n_steps,
        loss_ocean_weight=loss_ocean_weight,
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(train_config)

    inference_config = _INFERENCE_CONFIG_TEMPLATE.format(
        experiment_dir=exper_dir,
        checkpoint_path=exper_dir / "training_checkpoints/best_ckpt.tar",
        n_coupled_steps=inference_n_coupled_steps,
        coupled_steps_in_memory=coupled_steps_in_memory,
        ocean_data_path=mock_coupled_data.ocean_dir,
        atmosphere_data_path=mock_coupled_data.atmosphere_dir,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as f_inference:
        f_inference.write(inference_config)

    return f_train.name, f_inference.name


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
def test_train_and_inference(tmp_path, log_zonal_mean_images, very_fast_only: bool):
    """Ensure that coupled training and standalone inference run without errors."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    data_dir = tmp_path / "coupled_data"
    data_dir.mkdir()

    # variable names for the ocean data on disk
    ocean_names = [
        "thetao_0",
        "thetao_1",
        "sst",
        "mask_0",
        "mask_1",
        "sea_ice_fraction",
    ]
    # variable names for the atmos data on disk
    atmos_names = [
        "DLWRFsfc",
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
        "ocean_fraction",
        "land_fraction",
    ]

    n_forward_times_ocean = 8
    n_forward_times_atmos = 16
    mock_coupled_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=True,
        n_levels_ocean=2,
        n_levels_atmosphere=2,
    )

    ocean_in_names = [
        "thetao_0",
        "thetao_1",
        "sst",
        "mask_0",
        "mask_1",
        "land_fraction",
        "sea_ice_fraction",
    ]
    ocean_out_names = ["thetao_0", "thetao_1", "sst", "sea_ice_fraction"]
    ocean_derived_names = ["ocean_heat_content"]
    atmos_in_names = [
        "DLWRFsfc",
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
        "ocean_fraction",
        "land_fraction",
        "sea_ice_fraction",
    ]
    atmos_out_names = [
        "DLWRFsfc",
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
    ]
    atmos_derived_names = [
        "surface_pressure_due_to_dry_air",
        "surface_pressure_due_to_dry_air_absolute_tendency",
        "total_water_path",
    ]
    all_out_names = (
        ocean_out_names + ocean_derived_names + atmos_out_names + atmos_derived_names
    )

    train_config_fname, inference_config_fname = _write_test_yaml_files(
        tmp_path,
        mock_coupled_data,
        ocean_in_names,
        ocean_out_names,
        atmos_in_names,
        atmos_out_names,
        ocean_sfc_temp_name="sst",
        ocean_sfc_mask_name="mask_0",
        sea_ice_frac_name="sea_ice_fraction",
        land_frac_name="land_fraction",
        atmos_sfc_temp_name="surface_temperature",
        ocean_frac_name="ocean_fraction",
        log_zonal_mean_images=log_zonal_mean_images,
        n_coupled_steps=2,
        max_epochs=1,
        inline_inference_n_coupled_steps=3,
        inference_n_coupled_steps=6,
        coupled_steps_in_memory=2,
        loss_atmos_n_steps=3,
    )

    with mock_wandb() as wandb:
        train_main(yaml_config=train_config_fname)
        train_logs = wandb.get_logs()

    assert len(train_logs) == 4  # initialization + 3 batches

    for log in train_logs:
        # ensure inference time series is not logged
        assert "inference/mean/forecast_step" not in log

    batch_logs = train_logs[0]
    assert "batch_loss" in batch_logs
    assert "batch_loss/ocean" in batch_logs
    assert "batch_loss/atmosphere" in batch_logs
    # NOTE: step numbers start at 0
    for i in range(2):
        assert f"batch_loss/ocean_step_{i}" in batch_logs
    # only 2 ocean steps configured
    assert f"batch_loss/ocean_step_2" not in batch_logs
    for i in range(3):
        assert f"batch_loss/atmosphere_step_{i}" in batch_logs
    # atmos loss contributions config with n_steps = 3
    assert f"batch_loss/atmosphere_step_3" not in batch_logs

    epoch_logs = train_logs[-1]
    assert "train/mean/loss" in epoch_logs
    assert "val/mean/loss" in epoch_logs
    assert "val/mean/loss/ocean" in epoch_logs
    assert "val/mean/loss/atmosphere" in epoch_logs
    assert "inference/time_mean_norm/rmse/channel_mean" in epoch_logs
    assert "inference/time_mean_norm/rmse/ocean_channel_mean" in epoch_logs
    assert "inference/time_mean_norm/rmse/atmosphere_channel_mean" in epoch_logs
    ocean_weight = len(ocean_out_names + ocean_derived_names) / len(all_out_names)
    assert np.isclose(
        epoch_logs["inference/time_mean_norm/rmse/channel_mean"],
        (
            ocean_weight
            * epoch_logs["inference/time_mean_norm/rmse/ocean_channel_mean"]
            + (1 - ocean_weight)
            * epoch_logs["inference/time_mean_norm/rmse/atmosphere_channel_mean"]
        ),
    )
    for name in all_out_names:
        assert f"inference/time_mean/rmse/{name}" in epoch_logs

    # check that inference map captions includes expected units
    for map_name in ["val/mean_map/image-error", "inference/time_mean/bias_map"]:
        name = "thetao_0"
        wandb_img = epoch_logs[f"{map_name}/{name}"]
        captions = wandb_img.captions([wandb_img])
        assert len(captions) > 0
        assert name in captions[0]
        assert "[unknown_units]" not in captions[0]
        assert "[m]" in captions[0]  # set by _save_netcdf

    for domain in ("ocean", "atmosphere"):
        validation_output_dir = (
            tmp_path / "results" / "output" / "val" / domain / "epoch_0001"
        )
        assert validation_output_dir.exists()
        for diagnostic in ("mean", "snapshot", "mean_map"):
            diagnostic_output = validation_output_dir / f"{diagnostic}_diagnostics.nc"
            assert diagnostic_output.exists()
            ds = xr.open_dataset(diagnostic_output, decode_timedelta=False)
            assert len(ds) > 0

    for domain in ("ocean", "atmosphere"):
        inline_inference_output_dir = (
            tmp_path / "results" / "output" / "inference" / domain / "epoch_0001"
        )
        assert inline_inference_output_dir.exists()
        for diagnostic in (
            "time_mean",
            "time_mean_norm",
        ):
            diagnostic_output = (
                inline_inference_output_dir / f"{diagnostic}_diagnostics.nc"
            )
            assert diagnostic_output.exists()
            ds = xr.open_dataset(diagnostic_output, decode_timedelta=False)
            assert len(ds) > 0

    best_checkpoint_path = (
        tmp_path / "results" / "training_checkpoints" / "best_ckpt.tar"
    )
    best_inference_checkpoint_path = (
        tmp_path / "results" / "training_checkpoints" / "best_inference_ckpt.tar"
    )
    assert best_checkpoint_path.exists()
    assert best_inference_checkpoint_path.exists()

    with mock_wandb() as wandb:
        inference_evaluator_main(yaml_config=inference_config_fname)
        inference_logs = wandb.get_logs()

    n_inner_steps = n_forward_times_atmos // n_forward_times_ocean
    n_ic_timesteps = 1
    n_forward_steps = 6 * n_inner_steps
    n_summary_steps = 1
    assert len(inference_logs) == n_ic_timesteps + n_forward_steps + n_summary_steps

    for i, log in enumerate(inference_logs[:-1]):
        assert "inference/mean/forecast_step" in log
        assert log["inference/mean/forecast_step"] == i
        if i % n_inner_steps == 0:
            assert "inference/mean/weighted_bias/thetao_0" in log
            assert "inference/mean/weighted_bias/PRESsfc" in log
        if i % n_inner_steps == 1:
            assert "inference/mean/weighted_bias/thetao_0" not in log
            assert "inference/mean/weighted_bias/PRESsfc" in log

    assert "inference/time_mean_norm/rmse/channel_mean" in inference_logs[-1]
    assert "inference/time_mean_norm/rmse/ocean_channel_mean" in inference_logs[-1]
    assert "inference/time_mean_norm/rmse/atmosphere_channel_mean" in inference_logs[-1]

    ocean_output_path = tmp_path / "results" / "ocean" / "autoregressive_predictions.nc"
    assert ocean_output_path.exists()

    atmosphere_output_path = (
        tmp_path / "results" / "atmosphere" / "autoregressive_predictions.nc"
    )
    assert atmosphere_output_path.exists()

    ds_ocean = xr.open_dataset(ocean_output_path, decode_timedelta=False)
    assert ds_ocean["time"].size == 6  # configured inference coupled steps
    assert (
        ds_ocean["sample"].size == 2
    )  # 2 initial conditions in _INFERENCE_CONFIG_TEMPLATE
    assert np.sum(np.isnan(ds_ocean["sst"].values)) == 0
    assert np.sum(np.isnan(ds_ocean["thetao_0"].values)) == 0
    ds_atmos = xr.open_dataset(atmosphere_output_path, decode_timedelta=False)
    assert ds_atmos["time"].size == 6 * n_inner_steps
    assert ds_atmos["sample"].size == 2
    assert np.sum(np.isnan(ds_atmos["surface_temperature"].values)) == 0
    assert np.sum(np.isnan(ds_atmos["PRESsfc"].values)) == 0
    assert np.sum(np.isnan(ds_atmos["DLWRFsfc"].values)) == 0
