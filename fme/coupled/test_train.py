import os
import tempfile
import unittest.mock

import numpy as np
import pytest
import torch
import xarray as xr

from fme.core.generics.trainer import _restore_checkpoint
from fme.core.rand import set_seed, use_cpu_randn
from fme.core.testing.wandb import mock_wandb

from .data_loading.test_data_loader import create_coupled_data_on_disk
from .inference.evaluator import main as inference_evaluator_main
from .train.train import main as train_main

_TRAIN_CONFIG_TEMPLATE = """
experiment_dir: {experiment_dir}
seed: 0
save_checkpoint: true
save_per_epoch_diagnostics: {save_per_epoch_diagnostics}
max_epochs: {max_epochs}
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
    ocean:
      data_path: {ocean_data_path}
      subset:
          start_time: '1970-01-01'
    atmosphere:
      data_path: {atmosphere_data_path}
      subset:
          start_time: '1970-01-01'
validation:
- loader:
    batch_size: 2
    num_data_workers: 0
    dataset:
      ocean:
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
      atmosphere:
        data_path: {atmosphere_data_path}
    start_indices:
      times:
        - '1970-01-01T00:00:00'
        - '1970-01-03T00:00:00'
  n_coupled_steps: {inference_n_coupled_steps}
  aggregator:
    log_zonal_mean_images: True
optimization:
  enable_automatic_mixed_precision: false
  lr: 0.00001
  optimizer_type: Adam
stepper_training:
  n_coupled_steps: {n_coupled_steps}
  ocean:
    loss:
      type: {loss_type}
      kwargs: {loss_kwargs}
    loss_weight: {loss_ocean_weight}
  atmosphere:
    loss:
      type: {loss_type}
      kwargs: {loss_kwargs}
    n_steps: {loss_atmos_n_steps}
stepper:
  sst_name: {ocean_sfc_temp_name}
  ocean_fraction_prediction:
    sea_ice_fraction_name: {sea_ice_frac_name}
    land_fraction_name: {land_frac_name}
    sea_ice_fraction_name_in_atmosphere: {atmos_sea_ice_frac_name}
  ocean:
    timedelta: 2D
    stepper:
      input_masking:
        mask_value: 0
        fill_value: 0.0
      step:
        type: single_module
        config:
          builder:
            type: Samudra
            config:
                ch_width: [8, 16]
                dilation: [2, 4]
                n_layers: [1, 1]
                norm: batch
                norm_kwargs:
                    track_running_stats: false
          normalization:
            network:
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
    stepper:
      step:
        type: single_module
        config:
          builder:
            type: {atmos_network_type}
            config:
              num_layers: 2
              embed_dim: 12
          normalization:
            network:
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
    atmosphere:
      data_path: {atmosphere_data_path}
  start_indices:
    times:
      - '1970-01-01T00:00:00'
      - '1970-01-03T00:00:00'
"""


def _write_test_yaml_files(
    tmp_path,
    mock_coupled_data,
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmos_in_names: list[str],
    atmos_out_names: list[str],
    ocean_sfc_temp_name: str,
    sea_ice_frac_name: str,
    atmos_sea_ice_frac_name: str,
    land_frac_name: str,
    atmos_sfc_temp_name: str,
    ocean_frac_name: str,
    n_coupled_steps: int = 1,
    max_epochs: int = 1,
    inline_inference_n_coupled_steps: int = 3,
    inference_n_coupled_steps: int = 6,
    coupled_steps_in_memory: int = 2,
    save_per_epoch_diagnostics: bool = True,
    loss_atmos_n_steps: int = 3,
    loss_ocean_weight: float = 1.0,
    crps_training: bool = False,
):
    exper_dir = tmp_path / "results"
    ocean_next_step_forcing_names = list(
        set(atmos_out_names).intersection(ocean_in_names)
    )

    # Configure atmosphere network and loss based on crps_training
    if crps_training:
        atmos_network_type = "NoiseConditionedSFNO"
        loss_type = "EnsembleLoss"
        loss_kwargs = "{'crps_weight': 1.0, 'energy_score_weight': 0.0}"
    else:
        atmos_network_type = "SphericalFourierNeuralOperatorNet"
        loss_type = "MSE"
        loss_kwargs = "{}"

    train_config = _TRAIN_CONFIG_TEMPLATE.format(
        experiment_dir=exper_dir,
        max_epochs=max_epochs,
        n_coupled_steps=n_coupled_steps,
        ocean_data_path=mock_coupled_data.ocean.data_dir,
        atmosphere_data_path=mock_coupled_data.atmosphere.data_dir,
        global_means_path=os.path.join(mock_coupled_data.ocean.means_path),
        global_stds_path=os.path.join(mock_coupled_data.ocean.stds_path),
        inference_n_coupled_steps=inline_inference_n_coupled_steps,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        ocean_next_step_forcing_names=ocean_next_step_forcing_names,
        ocean_sfc_temp_name=ocean_sfc_temp_name,
        sea_ice_frac_name=sea_ice_frac_name,
        atmos_sea_ice_frac_name=atmos_sea_ice_frac_name,
        land_frac_name=land_frac_name,
        atmos_sfc_temp_name=atmos_sfc_temp_name,
        ocean_frac_name=ocean_frac_name,
        save_per_epoch_diagnostics=str(save_per_epoch_diagnostics).lower(),
        loss_atmos_n_steps=loss_atmos_n_steps,
        loss_ocean_weight=loss_ocean_weight,
        atmos_network_type=atmos_network_type,
        loss_type=loss_type,
        loss_kwargs=loss_kwargs,
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f_train:
        f_train.write(train_config)

    inference_config = _INFERENCE_CONFIG_TEMPLATE.format(
        experiment_dir=exper_dir,
        checkpoint_path=exper_dir / "training_checkpoints/best_ckpt.tar",
        n_coupled_steps=inference_n_coupled_steps,
        coupled_steps_in_memory=coupled_steps_in_memory,
        ocean_data_path=mock_coupled_data.ocean.data_dir,
        atmosphere_data_path=mock_coupled_data.atmosphere.data_dir,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as f_inference:
        f_inference.write(inference_config)

    return f_train.name, f_inference.name


@pytest.mark.parametrize(
    "loss_atmos_n_steps, crps_training",
    [
        (2, False),
        (0, False),
        (3, True),  # CRPS training with EnsembleLoss
    ],
)
@pytest.mark.medium_duration
def test_train_and_inference(tmp_path, loss_atmos_n_steps, crps_training: bool):
    """Ensure that coupled training and standalone inference run without errors."""
    set_seed(42 + loss_atmos_n_steps)

    data_dir = tmp_path / "coupled_data"
    data_dir.mkdir()

    # variable names for the ocean data on disk
    ocean_names = [
        "thetao_0",
        "thetao_1",
        "sst",
        "mask_2d",
        "mask_0",
        "mask_1",
        "ocean_sea_ice_fraction",
        "land_fraction",
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
        "sea_ice_fraction",
    ]

    n_forward_times_ocean = 8
    n_forward_times_atmos = 16
    mock_coupled_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=1,
        n_levels_ocean=2,
        n_levels_atmosphere=2,
    )

    ocean_in_names = [
        "thetao_0",
        "thetao_1",
        "sst",
        "mask_2d",
        "mask_0",
        "mask_1",
        "land_fraction",
        "ocean_sea_ice_fraction",
    ]
    ocean_out_names = ["thetao_0", "thetao_1", "sst", "ocean_sea_ice_fraction"]
    ocean_derived_names = [
        "ocean_heat_content",
        "ocean_heat_content_tendency",
        "sea_ice_fraction",
    ]
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
    all_out_normed_names = ocean_out_names + atmos_out_names

    train_config_fname, inference_config_fname = _write_test_yaml_files(
        tmp_path,
        mock_coupled_data,
        ocean_in_names,
        ocean_out_names,
        atmos_in_names,
        atmos_out_names,
        ocean_sfc_temp_name="sst",
        sea_ice_frac_name="ocean_sea_ice_fraction",
        atmos_sea_ice_frac_name="sea_ice_fraction",
        land_frac_name="land_fraction",
        atmos_sfc_temp_name="surface_temperature",
        ocean_frac_name="ocean_fraction",
        n_coupled_steps=2,
        max_epochs=1,
        inline_inference_n_coupled_steps=3,
        inference_n_coupled_steps=6,
        coupled_steps_in_memory=2,
        loss_atmos_n_steps=loss_atmos_n_steps,
        crps_training=crps_training,
    )

    with use_cpu_randn(), mock_wandb() as wandb:
        train_main(yaml_config=train_config_fname)
        train_logs = wandb.get_logs()

    assert len(train_logs) == 4  # initialization + 3 batches

    for log in train_logs:
        # ensure inference time series is not logged
        assert "inference/mean/forecast_step" not in log

    batch_logs = train_logs[0]
    assert "batch_loss" in batch_logs
    assert "batch_loss/ocean" in batch_logs
    # NOTE: step numbers start at 0
    for i in range(2):
        assert f"batch_loss/ocean_step_{i}" in batch_logs
    # only 2 ocean steps configured
    assert f"batch_loss/ocean_step_2" not in batch_logs

    # atmos loss contributions
    assert "batch_loss/atmosphere" in batch_logs
    if loss_atmos_n_steps > 0:
        for i in range(loss_atmos_n_steps):
            assert f"batch_loss/atmosphere_step_{i}" in batch_logs
        # atmos loss contributions config with n_steps = 3
        assert f"batch_loss/atmosphere_step_3" not in batch_logs
    else:
        torch.testing.assert_close(
            batch_logs["batch_loss/atmosphere"],
            torch.tensor(0.0, device=batch_logs["batch_loss/atmosphere"].device),
        )
        for i in range(loss_atmos_n_steps):
            assert f"batch_loss/atmosphere_step_{i}" not in batch_logs

    epoch_logs = train_logs[-1]
    assert "train/mean/loss" in epoch_logs
    assert "val/mean/loss" in epoch_logs
    assert "val/mean/loss/ocean" in epoch_logs
    # atmos loss contributions
    assert "val/mean/loss/atmosphere" in epoch_logs
    np.testing.assert_allclose(
        epoch_logs["val/mean/loss"],
        epoch_logs["val/mean/loss/atmosphere"] + epoch_logs["val/mean/loss/ocean"],
        atol=1e-6,
    )
    if loss_atmos_n_steps == 0:
        np.testing.assert_allclose(epoch_logs["val/mean/loss/atmosphere"], 0.0)

    for name in all_out_names:
        assert f"inference/time_mean/rmse/{name}" in epoch_logs
        if name in all_out_normed_names:
            assert f"val/mean_norm/weighted_rmse/{name}" in epoch_logs
            assert f"inference/time_mean_norm/rmse/{name}" in epoch_logs
        else:
            assert f"val/mean_norm/weighted_rmse/{name}" not in epoch_logs
            assert f"inference/time_mean_norm/rmse/{name}" not in epoch_logs

    ocean_weight = len(ocean_out_names + ocean_derived_names) / len(all_out_names)
    atol, rtol = 1e-6, 1e-6
    for prefix in ["inference/time_mean_norm/rmse", "val/mean_norm/weighted_rmse"]:
        assert f"{prefix}/ocean_channel_mean" in epoch_logs
        ocean_channel_mean = sum(
            [epoch_logs[f"{prefix}/{name}"] for name in ocean_out_names]
        ) / len(ocean_out_names)
        np.testing.assert_allclose(
            epoch_logs[f"{prefix}/ocean_channel_mean"],
            ocean_channel_mean,
            rtol=rtol,
            atol=atol,
        )

        assert f"{prefix}/atmosphere_channel_mean" in epoch_logs
        atmos_channel_mean = sum(
            [epoch_logs[f"{prefix}/{name}"] for name in atmos_out_names]
        ) / len(atmos_out_names)
        np.testing.assert_allclose(
            epoch_logs[f"{prefix}/atmosphere_channel_mean"],
            atmos_channel_mean,
            rtol=rtol,
            atol=atol,
        )

        if "time_mean_norm" in prefix:
            assert f"{prefix}/channel_mean" in epoch_logs
            np.testing.assert_allclose(
                epoch_logs[f"{prefix}/channel_mean"],
                (
                    ocean_weight * epoch_logs[f"{prefix}/ocean_channel_mean"]
                    + (1 - ocean_weight)
                    * epoch_logs[f"{prefix}/atmosphere_channel_mean"]
                ),
                rtol=rtol,
                atol=atol,
            )

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

    with use_cpu_randn(), mock_wandb() as wandb:
        inference_evaluator_main(yaml_config=inference_config_fname)
        inference_logs = wandb.get_logs()

    n_inner_steps = n_forward_times_atmos // n_forward_times_ocean
    n_ic_timesteps = 1
    n_forward_steps = 6 * n_inner_steps
    n_summary_steps = 1
    assert len(inference_logs) == n_ic_timesteps + n_forward_steps + n_summary_steps

    for name in atmos_out_names + ocean_out_names:
        np.testing.assert_allclose(
            inference_logs[0][f"inference/mean/weighted_mean_target/{name}"],
            inference_logs[0][f"inference/mean/weighted_mean_gen/{name}"],
        )

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

    ds_ocean = xr.open_dataset(ocean_output_path, decode_timedelta=False)
    assert ds_ocean["time"].size == 6  # configured inference coupled steps
    assert (
        ds_ocean["sample"].size == 2
    )  # 2 initial conditions in _INFERENCE_CONFIG_TEMPLATE
    for name in ds_ocean.data_vars:
        # outputs should have some non-null values
        assert not np.isnan(ds_ocean[name].values).all()
        # outputs should have some masked regions
        assert np.isnan(ds_ocean[name].values).any()

    # check that ocean derived variables are in the output
    for name in ocean_derived_names:
        assert name in ds_ocean.data_vars
    atmosphere_output_path = (
        tmp_path / "results" / "atmosphere" / "autoregressive_predictions.nc"
    )
    atmosphere_target_path = (
        tmp_path / "results" / "atmosphere" / "autoregressive_target.nc"
    )
    assert atmosphere_output_path.exists()
    assert atmosphere_target_path.exists()
    ds_atmos = xr.open_dataset(atmosphere_output_path, decode_timedelta=False)
    ds_atmos_tar = xr.open_dataset(atmosphere_target_path, decode_timedelta=False)
    for ds in [ds_atmos, ds_atmos_tar]:
        assert ds["time"].size == 6 * n_inner_steps
        assert ds["sample"].size == 2
    area_weights = np.cos(np.deg2rad(ds_atmos["lat"]))
    for name in atmos_out_names:
        assert name in ds_atmos.data_vars
        assert name in ds_atmos_tar.data_vars
        da = ds_atmos[name]
        da_tar = ds_atmos_tar[name]
        assert not np.isnan(da.values).any()
        assert not np.isnan(da_tar.values).any()
        wm_tar = da_tar.isel(time=0).weighted(area_weights).mean().item()
        wm_gen = (
            da.isel(time=0).weighted(area_weights).mean(["lat", "lon"]).mean().item()
        )
        np.testing.assert_allclose(
            wm_tar,
            inference_logs[1][f"inference/mean/weighted_mean_target/{name}"],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            wm_gen,
            inference_logs[1][f"inference/mean/weighted_mean_gen/{name}"],
            atol=1e-4,
        )


_PER_COMPONENT_OPTIMIZATION_YAML = """optimization:
  atmosphere:
    optimizer_type: AdamW
    lr: 0.0001
    scheduler: {type: StepLR, kwargs: {step_size: 1, gamma: 0.5}}
  ocean:
    optimizer_type: Adam
    lr: 0.001
    max_grad_norm: 1.0
    scheduler: {type: StepLR, kwargs: {step_size: 1, gamma: 0.1}}
"""


def _per_component_train_config(
    tmp_path,
    experiment_name: str,
    max_epochs: int,
    segment_epochs: int | None = None,
    optimization_yaml: str = _PER_COMPONENT_OPTIMIZATION_YAML,
):
    data_dir = tmp_path / f"coupled_data_{experiment_name}"
    data_dir.mkdir()
    ocean_names = [
        "thetao_0",
        "thetao_1",
        "sst",
        "mask_2d",
        "mask_0",
        "mask_1",
        "ocean_sea_ice_fraction",
        "land_fraction",
    ]
    atmos_names = [
        "DLWRFsfc",
        "PRESsfc",
        "specific_total_water_0",
        "specific_total_water_1",
        "surface_temperature",
        "ocean_fraction",
        "land_fraction",
        "sea_ice_fraction",
    ]
    mock_coupled_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=8,
        n_forward_times_atmosphere=16,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=1,
        n_levels_ocean=2,
        n_levels_atmosphere=2,
    )
    experiment_path = tmp_path / experiment_name
    experiment_path.mkdir(exist_ok=True)
    train_config_fname, _ = _write_test_yaml_files(
        experiment_path,
        mock_coupled_data,
        ocean_in_names=ocean_names,
        ocean_out_names=["thetao_0", "thetao_1", "sst", "ocean_sea_ice_fraction"],
        atmos_in_names=atmos_names,
        atmos_out_names=[
            "DLWRFsfc",
            "PRESsfc",
            "specific_total_water_0",
            "specific_total_water_1",
            "surface_temperature",
        ],
        ocean_sfc_temp_name="sst",
        sea_ice_frac_name="ocean_sea_ice_fraction",
        atmos_sea_ice_frac_name="sea_ice_fraction",
        land_frac_name="land_fraction",
        atmos_sfc_temp_name="surface_temperature",
        ocean_frac_name="ocean_fraction",
        n_coupled_steps=1,
        max_epochs=max_epochs,
        loss_atmos_n_steps=2,
    )
    with open(train_config_fname) as f:
        text = f.read()
    single_form = (
        "optimization:\n  enable_automatic_mixed_precision: false\n"
        "  lr: 0.00001\n  optimizer_type: Adam\n"
    )
    assert single_form in text
    with open(train_config_fname, "w") as f:
        f.write(
            text.replace(single_form, optimization_yaml)
            + "log_train_every_n_batches: 1\n"
            + (f"segment_epochs: {segment_epochs}\n" if segment_epochs else "")
        )
    return train_config_fname


def _epoch_component_lrs(logs) -> list[tuple[float, float]]:
    return [
        (log["lr/atmosphere"], log["lr/ocean"])
        for log in logs
        if "epoch" in log and "lr/atmosphere" in log
    ]


@pytest.mark.medium_duration
def test_train_per_component_optimization(tmp_path):
    set_seed(0)
    uninterrupted = _per_component_train_config(tmp_path, "uninterrupted", 2)
    with use_cpu_randn(), mock_wandb() as wandb:
        train_main(yaml_config=uninterrupted)
        uninterrupted_logs = wandb.get_logs()
    expected = [(1e-4, 1e-3), (0.5e-4, 1e-4)]
    assert _epoch_component_lrs(uninterrupted_logs) == pytest.approx(expected)
    batch_logs = [
        log
        for log in uninterrupted_logs
        if "training_samples_per_second_on_rank_0" in log
    ]
    assert len(batch_logs) > 0
    assert all("lr/atmosphere" in log and "lr/ocean" in log for log in batch_logs)

    resumed_config = _per_component_train_config(
        tmp_path, "resumed", 2, segment_epochs=1
    )
    with use_cpu_randn(), mock_wandb() as wandb:
        train_main(yaml_config=resumed_config)
        first_logs = wandb.get_logs()
        wandb_id = wandb.get_id()
        last_step = len(first_logs) - 1
    checkpoint = torch.load(
        tmp_path / "resumed" / "results" / "training_checkpoints" / "ckpt.tar",
        weights_only=False,
    )
    assert set(checkpoint["optimization"]) == {"components", "gscaler_state_dict"}
    assert set(checkpoint["optimization"]["components"]) == {"atmosphere", "ocean"}
    with (
        unittest.mock.patch(
            "fme.core.generics.trainer._restore_checkpoint",
            new=unittest.mock.MagicMock(side_effect=_restore_checkpoint),
        ) as restore,
        use_cpu_randn(),
        mock_wandb() as wandb,
    ):
        wandb.set_id(wandb_id)
        wandb.set_last_received_step(last_step)
        train_main(yaml_config=resumed_config)
        second_logs = wandb.get_logs()
    restore.assert_called()
    assert [
        log["epoch"] for log in first_logs if "lr/atmosphere" in log and "epoch" in log
    ] == [1]
    assert [
        log["epoch"] for log in second_logs if "lr/atmosphere" in log and "epoch" in log
    ] == [2]
    resumed = _epoch_component_lrs(first_logs) + _epoch_component_lrs(second_logs)
    assert resumed == pytest.approx(expected)


_OCEAN_LR = 1e-3
_ATMOSPHERE_LR = 1e-5
_GROWTH_ITERS = 16


def _scheduler_yaml(kind: str, step_each_iteration: bool) -> str:
    if kind == "decaying":
        scheduler = "{type: ExponentialLR, kwargs: {gamma: 0.5}"
    elif kind == "growing":
        scheduler = (
            "{type: LinearLR, kwargs: {start_factor: 0.25, end_factor: 1.0, "
            f"total_iters: {_GROWTH_ITERS}}}"
        )
    else:
        return "{}"
    return f"{scheduler}, step_each_iteration: {str(step_each_iteration).lower()}}}"


def _expected_lr(kind: str, n_steps: int) -> float:
    if kind == "decaying":
        return _OCEAN_LR * 0.5**n_steps
    if kind == "growing":
        growth = min(n_steps, _GROWTH_ITERS) / _GROWTH_ITERS
        return _ATMOSPHERE_LR * (0.25 + 0.75 * growth)
    return _ATMOSPHERE_LR


@pytest.mark.medium_duration
@pytest.mark.parametrize(
    "atmosphere, atmosphere_each_iteration, ocean_each_iteration",
    [
        ("constant", False, False),
        ("growing", True, False),
        ("growing", False, True),
    ],
    ids=[
        "atmos_constant-ocean_epoch",
        "atmos_growing_iteration-ocean_epoch",
        "atmos_growing_epoch-ocean_iteration",
    ],
)
def test_train_per_component_lr_schedules(
    tmp_path,
    atmosphere: str,
    atmosphere_each_iteration: bool,
    ocean_each_iteration: bool,
):
    optimization_yaml = (
        "optimization:\n"
        "  atmosphere:\n"
        f"    lr: {_ATMOSPHERE_LR:.6f}\n"
        f"    scheduler: {_scheduler_yaml(atmosphere, atmosphere_each_iteration)}\n"
        "  ocean:\n"
        f"    lr: {_OCEAN_LR:.6f}\n"
        f"    scheduler: {_scheduler_yaml('decaying', ocean_each_iteration)}\n"
    )
    max_epochs = 2
    set_seed(0)
    config = _per_component_train_config(
        tmp_path, "schedules", max_epochs, optimization_yaml=optimization_yaml
    )
    with use_cpu_randn(), mock_wandb() as wandb:
        train_main(yaml_config=config)
        logs = wandb.get_logs()
    epoch_end_steps = [
        step for step, log in enumerate(logs) if "epoch" in log and "lr/ocean" in log
    ]
    n_batches = epoch_end_steps[0]
    assert n_batches > 1
    assert epoch_end_steps == [n_batches * (e + 1) for e in range(max_epochs)]
    kinds = {"atmosphere": atmosphere, "ocean": "decaying"}
    each_iteration = {
        "atmosphere": atmosphere_each_iteration,
        "ocean": ocean_each_iteration,
    }
    # Batch logs come after that batch's per-iteration scheduler step; the
    # per-epoch scheduler steps after the epoch's last log.
    for step in range(1, n_batches * max_epochs + 1):
        epoch = (step - 1) // n_batches
        for key, kind in kinds.items():
            n_steps = step if each_iteration[key] else epoch
            assert logs[step][f"lr/{key}"] == pytest.approx(
                _expected_lr(kind, n_steps)
            ), (key, step)
        assert logs[step]["lr"] == logs[step]["lr/atmosphere"]
