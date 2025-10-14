import dataclasses
import os
import pathlib

import numpy as np
import pytest
import xarray as xr
import yaml

from fme.ace.data_loading.inference import InferenceInitialConditionIndices
from fme.ace.inference.data_writer.main import DataWriterConfig
from fme.ace.inference.inference import ForcingDataLoaderConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.testing import mock_wandb
from fme.coupled.data_loading.inference import CoupledForcingDataLoaderConfig
from fme.coupled.data_loading.test_data_loader import create_coupled_data_on_disk
from fme.coupled.inference.data_writer import CoupledDataWriterConfig
from fme.coupled.inference.inference import (
    ComponentInitialConditionConfig,
    CoupledInitialConditionConfig,
    InferenceConfig,
    main,
)
from fme.coupled.inference.test_evaluator import save_coupled_stepper
from fme.coupled.test_stepper import CoupledDatasetInfoBuilder


def _setup(
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmos_in_names: list[str],
    atmos_out_names: list[str],
    tmp_path: pathlib.Path,
    n_coupled_steps: int,
    coupled_steps_in_memory: int,
    n_initial_conditions: int,
    empty_ocean_forcing: bool = False,
):
    all_ocean_names = set(ocean_in_names + ocean_out_names)
    all_atmos_names = set(atmos_in_names + atmos_out_names)

    # variables with larger ocean timestep
    ocean_names = list(all_ocean_names - set(atmos_out_names))
    # variables with smaller atmosphere timestep
    atmos_names = list(all_atmos_names - set(ocean_out_names))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # create_coupled_data_on_disk already accounts for one initial condition
    atmos_steps_per_ocean_step = 2
    n_extra_initial_conditions = n_initial_conditions - 1
    n_forward_times_ocean = n_coupled_steps + n_extra_initial_conditions
    n_forward_times_atmos = n_forward_times_ocean * atmos_steps_per_ocean_step
    mock_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=False,
        n_levels_ocean=1,
        n_levels_atmosphere=1,
    )
    dataset_info = CoupledDatasetInfoBuilder(
        vcoord=mock_data.vcoord,
        hcoord=mock_data.hcoord,
        ocean_timestep=mock_data.ocean.timestep,
        atmos_timestep=mock_data.atmosphere.timestep,
        ocean_mask_provider=mock_data.ocean.mask_provider,
        atmos_mask_provider=mock_data.atmosphere.mask_provider,
    ).dataset_info
    checkpoint_path = save_coupled_stepper(
        tmp_path,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        dataset_info=dataset_info,
        save_standalone_component_checkpoints=True,
        ocean_timedelta=mock_data.ocean.timedelta,
        atmosphere_timedelta=mock_data.atmosphere.timedelta,
    )
    if empty_ocean_forcing:
        forcing_loader = CoupledForcingDataLoaderConfig(
            ocean=None,
            atmosphere=ForcingDataLoaderConfig(
                dataset=mock_data.dataset_config.atmosphere, num_data_workers=0
            ),
        )
    else:
        ocean_config = mock_data.dataset_config.ocean
        assert ocean_config is not None
        forcing_loader = CoupledForcingDataLoaderConfig(
            ocean=ForcingDataLoaderConfig(dataset=ocean_config, num_data_workers=0),
            atmosphere=ForcingDataLoaderConfig(
                dataset=mock_data.dataset_config.atmosphere, num_data_workers=0
            ),
        )

    config = InferenceConfig(
        experiment_dir=str(tmp_path),
        n_coupled_steps=n_coupled_steps,
        checkpoint_path=checkpoint_path,
        logging=LoggingConfig(log_to_screen=True, log_to_file=False, log_to_wandb=True),
        initial_condition=CoupledInitialConditionConfig(
            ocean=ComponentInitialConditionConfig(
                os.path.join(mock_data.ocean.data_dir, "data.nc"),
            ),
            atmosphere=ComponentInitialConditionConfig(
                os.path.join(mock_data.atmosphere.data_dir, "data.nc"),
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=n_initial_conditions
            ),
        ),
        forcing_loader=forcing_loader,
        data_writer=CoupledDataWriterConfig(
            ocean=DataWriterConfig(
                save_prediction_files=True,
                save_monthly_files=True,
            ),
            atmosphere=DataWriterConfig(
                save_prediction_files=True,
                save_monthly_files=True,
            ),
        ),
        coupled_steps_in_memory=coupled_steps_in_memory,
    )
    return config, mock_data, atmos_steps_per_ocean_step


@pytest.mark.parametrize(
    ("n_coupled_steps,coupled_steps_in_memory,n_initial_conditions"),
    [
        (2, 2, 3),
        (4, 1, 1),
        (3, 1, 2),
    ],
)
def test_inference(
    tmp_path: pathlib.Path,
    n_coupled_steps: int,
    coupled_steps_in_memory: int,
    n_initial_conditions: int,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
    ocean_out_names = ["o_prog", "sst", "o_diag"]
    atmos_in_names = ["a_prog", "surface_temperature", "forcing_var", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

    config, mock_data, atmos_steps_per_ocean_step = _setup(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        tmp_path=tmp_path,
        n_coupled_steps=n_coupled_steps,
        coupled_steps_in_memory=coupled_steps_in_memory,
        n_initial_conditions=n_initial_conditions,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(yaml_config=str(config_filename))
        wandb_logs = wandb.get_logs()

    for i, log in enumerate(wandb_logs):
        for metric, val in log.items():
            # check that time series metrics match
            if "inference/mean" in metric:
                if i > 0:
                    assert metric in wandb_logs[i]
                    if np.isnan(val):
                        assert np.isnan(wandb_logs[i][metric])
                    else:
                        assert wandb_logs[i][metric] == val
                elif not np.isnan(val):  # for IC only valid data is reported to wandb
                    assert metric in wandb_logs[i]
                    assert wandb_logs[i][metric] == val

    ds_atmosphere = xr.open_dataset(
        tmp_path / "atmosphere/autoregressive_predictions.nc", decode_timedelta=False
    )
    for var in atmos_out_names:
        assert (
            var in ds_atmosphere.data_vars
        ), f"Variable {var} not found in atmosphere dataset"
        assert ds_atmosphere[var].sizes == {
            "time": n_coupled_steps * atmos_steps_per_ocean_step,
            "sample": n_initial_conditions,
            "lat": mock_data.img_shape[0],
            "lon": mock_data.img_shape[1],
        }
    ds_ocean = xr.open_dataset(
        tmp_path / "ocean/autoregressive_predictions.nc", decode_timedelta=False
    )
    for var in ocean_out_names:
        assert var in ds_ocean.data_vars, f"Variable {var} not found in ocean dataset"
        assert ds_ocean[var].sizes == {
            "time": n_coupled_steps,
            "sample": n_initial_conditions,
            "lat": mock_data.img_shape[0],
            "lon": mock_data.img_shape[1],
        }


def test_inference_with_empty_ocean_forcing(
    tmp_path: pathlib.Path,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    n_coupled_steps = 2
    coupled_steps_in_memory = 2
    n_initial_conditions = 3
    ocean_in_names = ["o_prog", "sst", "a_diag"]
    ocean_out_names = ["o_prog", "sst", "o_diag"]
    atmos_in_names = ["a_prog", "surface_temperature", "forcing_var", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

    config, mock_data, atmos_steps_per_ocean_step = _setup(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        tmp_path=tmp_path,
        n_coupled_steps=n_coupled_steps,
        coupled_steps_in_memory=coupled_steps_in_memory,
        n_initial_conditions=n_initial_conditions,
        empty_ocean_forcing=True,
    )
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(yaml_config=str(config_filename))
