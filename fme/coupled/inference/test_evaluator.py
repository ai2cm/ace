import dataclasses
import inspect
import pathlib
import shutil

import pytest
import torch
import xarray as xr
import yaml

from fme.ace.inference.data_writer.main import DataWriterConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.testing import mock_wandb
from fme.coupled.data_loading.config import CoupledDatasetWithOptionalOceanConfig
from fme.coupled.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.coupled.data_loading.test_data_loader import create_coupled_data_on_disk
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.inference.data_writer import CoupledDataWriterConfig
from fme.coupled.inference.evaluator import (
    InferenceEvaluatorConfig,
    StandaloneComponentCheckpointsConfig,
    StandaloneComponentConfig,
    main,
)
from fme.coupled.stepper import CoupledStepperConfig
from fme.coupled.test_stepper import CoupledDatasetInfoBuilder, get_stepper_config

DIR = pathlib.Path(__file__).parent


def test_standalone_checkpoints_config_init_args():
    ignore_args = ["parameter_init"]
    stepper_config_init_args = set(
        inspect.signature(CoupledStepperConfig.__init__).parameters.keys()
    ).difference(ignore_args)
    init_args = set(
        inspect.signature(
            StandaloneComponentCheckpointsConfig.__init__
        ).parameters.keys()
    )
    assert init_args == stepper_config_init_args, (
        "StandaloneComponentCheckpointsConfig should have the same init args as "
        "CoupledStepperConfig. Were new args added?"
    )


def save_coupled_stepper(
    base_dir: pathlib.Path,
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmos_in_names: list[str],
    atmos_out_names: list[str],
    dataset_info: CoupledDatasetInfo,
    sst_name_in_ocean_data: str = "sst",
    sfc_temp_name_in_atmosphere_data: str = "surface_temperature",
    ocean_fraction_name: str = "ocean_fraction",
    save_standalone_component_checkpoints: bool = False,
    ocean_timedelta: str = "2D",
    atmosphere_timedelta: str = "1D",
) -> str | StandaloneComponentCheckpointsConfig:
    config = get_stepper_config(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmosphere_in_names=atmos_in_names,
        atmosphere_out_names=atmos_out_names,
        sst_name_in_ocean_data=sst_name_in_ocean_data,
        sfc_temp_name_in_atmosphere_data=sfc_temp_name_in_atmosphere_data,
        ocean_fraction_name=ocean_fraction_name,
        ocean_timedelta=ocean_timedelta,
        atmosphere_timedelta=atmosphere_timedelta,
    )
    if save_standalone_component_checkpoints:
        ocean_stepper = config.ocean.stepper.get_stepper(dataset_info.ocean)
        atmos_stepper = config.atmosphere.stepper.get_stepper(dataset_info.atmosphere)
        ocean_path = base_dir / "ocean.pt"
        atmos_path = base_dir / "atmos.pt"
        torch.save({"stepper": ocean_stepper.get_state()}, ocean_path)
        torch.save({"stepper": atmos_stepper.get_state()}, atmos_path)
        return StandaloneComponentCheckpointsConfig(
            ocean=StandaloneComponentConfig(
                timedelta=ocean_timedelta,
                path=str(ocean_path),
            ),
            atmosphere=StandaloneComponentConfig(
                timedelta=atmosphere_timedelta,
                path=str(atmos_path),
            ),
            sst_name=sst_name_in_ocean_data,
        )
    coupled_stepper = config.get_stepper(dataset_info)
    coupled_path = base_dir / "coupled.pt"
    torch.save({"stepper": coupled_stepper.get_state()}, coupled_path)
    return str(coupled_path)


def inference_helper(
    tmp_path: pathlib.Path,
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmos_in_names: list[str],
    atmos_out_names: list[str],
    n_coupled_steps: int,
    coupled_steps_in_memory: int,
    n_initial_conditions: int,
    checkpoint_path: str | StandaloneComponentCheckpointsConfig,
):
    """
    Reusable helper for running coupled inference tests.

    Creates mock data, runs inference using the provided checkpoint, and
    performs assertions on the output.

    Note: Mock data is created with a 2:1 ratio of atmosphere to ocean timesteps,
    resulting in ocean_timedelta="2D" and atmosphere_timedelta="1D".
    """
    all_ocean_names = set(ocean_in_names + ocean_out_names)
    all_atmos_names = set(atmos_in_names + atmos_out_names)

    # variables with larger ocean timestep
    ocean_names = list(all_ocean_names - set(atmos_out_names))
    # variables with smaller atmosphere timestep
    atmos_names = list(all_atmos_names - set(ocean_out_names))

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # create_coupled_data_on_disk already accounts for one initial condition
    n_extra_initial_conditions = n_initial_conditions - 1
    n_forward_times_ocean = n_coupled_steps + n_extra_initial_conditions
    n_forward_times_atmos = n_forward_times_ocean * 2
    mock_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=1,
        n_levels_ocean=1,
        n_levels_atmosphere=1,
    )

    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_coupled_steps=n_coupled_steps,
        checkpoint_path=checkpoint_path,
        logging=LoggingConfig(log_to_screen=True, log_to_file=False, log_to_wandb=True),
        loader=InferenceDataLoaderConfig(
            dataset=CoupledDatasetWithOptionalOceanConfig(
                ocean=XarrayDataConfig(data_path=mock_data.ocean.data_dir),
                atmosphere=XarrayDataConfig(
                    data_path=mock_data.atmosphere.data_dir,
                ),
            ),
            start_indices=InferenceInitialConditionIndices(
                first=0, n_initial_conditions=n_initial_conditions, interval=1
            ),
        ),
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
    config_filename = tmp_path / "config.yaml"
    with open(config_filename, "w") as f:
        yaml.dump(dataclasses.asdict(config), f)

    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(yaml_config=str(config_filename))
        wandb_logs = wandb.get_logs()

    # n_coupled_steps + initial_condition + summaries
    assert len(wandb_logs) == n_coupled_steps * 2 + 2

    ocean_output_path = tmp_path / "ocean" / "autoregressive_predictions.nc"
    atmos_output_path = tmp_path / "atmosphere" / "autoregressive_predictions.nc"
    ds_ocean = xr.open_dataset(ocean_output_path, decode_timedelta=False)
    ds_atmos = xr.open_dataset(atmos_output_path, decode_timedelta=False)

    # vertical coordinates are not written out
    assert "lat" in ds_ocean.coords
    assert "lon" in ds_ocean.coords
    assert "lat" in ds_atmos.coords
    assert "lon" in ds_atmos.coords
    assert (
        "idepth_0" not in ds_ocean.coords
    ), "TODO: update this assertion now that vertical coords are written"
    assert "idepth_0" not in ds_atmos.coords
    assert (
        "ak_0" not in ds_atmos.coords
    ), "TODO: update this assertion now that vertical coords are written"
    assert "ak_0" not in ds_ocean.coords


def _create_dataset_info_for_stepper(
    ocean_in_names: list[str],
    ocean_out_names: list[str],
    atmos_in_names: list[str],
    atmos_out_names: list[str],
    n_coupled_steps: int,
    n_initial_conditions: int,
    data_dir: pathlib.Path,
):
    """
    Create mock data and return dataset_info needed for stepper creation.

    Uses the same parameters as inference_helper to ensure compatibility.
    """
    all_ocean_names = set(ocean_in_names + ocean_out_names)
    all_atmos_names = set(atmos_in_names + atmos_out_names)

    # variables with larger ocean timestep
    ocean_names = list(all_ocean_names - set(atmos_out_names))
    # variables with smaller atmosphere timestep
    atmos_names = list(all_atmos_names - set(ocean_out_names))

    data_dir.mkdir(exist_ok=True)
    n_extra_initial_conditions = n_initial_conditions - 1
    n_forward_times_ocean = n_coupled_steps + n_extra_initial_conditions
    n_forward_times_atmos = n_forward_times_ocean * 2
    mock_data = create_coupled_data_on_disk(
        data_dir,
        n_forward_times_ocean=n_forward_times_ocean,
        n_forward_times_atmosphere=n_forward_times_atmos,
        ocean_names=ocean_names,
        atmosphere_names=atmos_names,
        atmosphere_start_time_offset_from_ocean=True,
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
    return dataset_info, mock_data


@pytest.mark.parametrize(
    (
        "n_coupled_steps,"
        "coupled_steps_in_memory,"
        "n_initial_conditions,"
        "save_standalone_component_checkpoints,"
    ),
    [(2, 1, 2, True), (4, 2, 1, False), (2, 2, 1, False)],
)
def test_evaluator_inference(
    tmp_path: pathlib.Path,
    n_coupled_steps: int,
    coupled_steps_in_memory: int,
    n_initial_conditions: int,
    save_standalone_component_checkpoints: bool,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
    ocean_out_names = ["o_prog", "sst", "o_diag"]
    atmos_in_names = ["a_prog", "surface_temperature", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

    # Create mock data for stepper creation
    stepper_data_dir = tmp_path / "stepper_data"
    dataset_info, mock_data = _create_dataset_info_for_stepper(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        n_coupled_steps=n_coupled_steps,
        n_initial_conditions=n_initial_conditions,
        data_dir=stepper_data_dir,
    )
    checkpoint_path = save_coupled_stepper(
        tmp_path,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        dataset_info=dataset_info,
        save_standalone_component_checkpoints=save_standalone_component_checkpoints,
        ocean_timedelta=mock_data.ocean.timedelta,
        atmosphere_timedelta=mock_data.atmosphere.timedelta,
    )

    # Run inference using the helper
    inference_helper(
        tmp_path=tmp_path,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        n_coupled_steps=n_coupled_steps,
        coupled_steps_in_memory=coupled_steps_in_memory,
        n_initial_conditions=n_initial_conditions,
        checkpoint_path=checkpoint_path,
    )


def test_inference_backwards_compatibility(tmp_path: pathlib.Path):
    """
    Inference test using a serialized model from an earlier commit, to ensure
    earlier models can be used with the updated inference code.
    """
    ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
    ocean_out_names = ["o_prog", "sst", "o_diag"]
    atmos_in_names = ["a_prog", "surface_temperature", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

    stepper_path = DIR / "coupled_stepper_test_data"
    n_coupled_steps = 2
    n_initial_conditions = 1

    if not stepper_path.exists():
        # This helps us to re-generate data if the stepper changes.
        # To re-generate, just delete the data and run the test (it will fail).
        stepper_data_dir = tmp_path / "stepper_data"
        dataset_info, mock_data = _create_dataset_info_for_stepper(
            ocean_in_names=ocean_in_names,
            ocean_out_names=ocean_out_names,
            atmos_in_names=atmos_in_names,
            atmos_out_names=atmos_out_names,
            n_coupled_steps=n_coupled_steps,
            n_initial_conditions=n_initial_conditions,
            data_dir=stepper_data_dir,
        )
        checkpoint_path = save_coupled_stepper(
            tmp_path,
            ocean_in_names=ocean_in_names,
            ocean_out_names=ocean_out_names,
            atmos_in_names=atmos_in_names,
            atmos_out_names=atmos_out_names,
            dataset_info=dataset_info,
            save_standalone_component_checkpoints=False,
            ocean_timedelta=mock_data.ocean.timedelta,
            atmosphere_timedelta=mock_data.atmosphere.timedelta,
        )
        assert isinstance(checkpoint_path, str)
        shutil.copy(checkpoint_path, stepper_path)
        assert False, "coupled_stepper_test_data did not exist, it has been created"

    inference_helper(
        tmp_path=tmp_path,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        n_coupled_steps=n_coupled_steps,
        coupled_steps_in_memory=1,
        n_initial_conditions=n_initial_conditions,
        checkpoint_path=str(stepper_path),
    )
