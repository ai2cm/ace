import dataclasses
import inspect
import os
import pathlib
import shutil
from typing import cast

import pytest
import torch
import xarray as xr
import yaml

from fme.ace.inference.data_writer.main import DataWriterConfig
from fme.ace.stepper import Stepper, StepperOverrideConfig
from fme.core.corrector.atmosphere import EnergyBudgetConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.testing import mock_wandb
from fme.coupled.data_loading.config import CoupledDatasetWithOptionalOceanConfig
from fme.coupled.data_loading.inference import (
    InferenceDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.coupled.data_loading.test_data_loader import (
    MockCoupledData,
    create_coupled_data_on_disk,
)
from fme.coupled.dataset_info import CoupledDatasetInfo
from fme.coupled.inference.data_writer import CoupledDataWriterConfig
from fme.coupled.inference.evaluator import (
    InferenceEvaluatorConfig,
    StandaloneComponentCheckpointsConfig,
    StandaloneComponentConfig,
    load_stepper,
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


def _get_corrector_total_energy_budget_correction_from_stepper(stepper: Stepper):
    from fme.core.step.multi_call import MultiCallStep

    if isinstance(stepper._step_obj, MultiCallStep):
        config = stepper._step_obj._wrapped_step.config
    else:
        config = stepper._step_obj.config
    config = cast(SingleModuleStepConfig, config)
    corrector = config.corrector
    if hasattr(corrector, "total_energy_budget_correction"):
        return corrector.total_energy_budget_correction
    return corrector.config.get("total_energy_budget_correction")


def test_stepper_override_total_energy_budget_correction_to_none(
    tmp_path: pathlib.Path, very_fast_only: bool
):
    """StepperOverrideConfig correctly overrides total_energy_budget_correction from
    EnergyBudgetConfig to None (no energy corrector) for the atmosphere component."""
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")
    ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
    ocean_out_names = ["o_prog", "sst", "o_diag"]
    atmos_in_names = ["a_prog", "surface_temperature", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]
    n_coupled_steps = 2
    n_initial_conditions = 1

    stepper_data_dir = tmp_path / "stepper_data"
    dataset_info, _ = _create_dataset_info_for_stepper(
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        n_coupled_steps=n_coupled_steps,
        n_initial_conditions=n_initial_conditions,
        data_dir=stepper_data_dir,
    )
    # Save coupled stepper with atmosphere total_energy_budget_correction set
    default_energy_config = EnergyBudgetConfig("constant_temperature", 0.0)
    checkpoint_path = save_coupled_stepper(
        tmp_path,
        ocean_in_names=ocean_in_names,
        ocean_out_names=ocean_out_names,
        atmos_in_names=atmos_in_names,
        atmos_out_names=atmos_out_names,
        dataset_info=dataset_info,
        atmosphere_total_energy_budget_correction=default_energy_config,
    )
    assert isinstance(checkpoint_path, str)

    # Load without override: atmosphere should still have the energy corrector
    stepper_no_override = load_stepper(checkpoint_path)
    assert (
        _get_corrector_total_energy_budget_correction_from_stepper(
            stepper_no_override.atmosphere
        )
        is not None
    )

    # Load with override: atmosphere should have total_energy_budget_correction=None
    stepper_with_override = load_stepper(
        checkpoint_path,
        stepper_override_ocean=None,
        stepper_override_atmosphere=StepperOverrideConfig(
            total_energy_budget_correction=None
        ),
    )
    assert (
        _get_corrector_total_energy_budget_correction_from_stepper(
            stepper_with_override.atmosphere
        )
        is None
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
    atmosphere_total_energy_budget_correction: EnergyBudgetConfig | None = None,
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
        atmosphere_total_energy_budget_correction=atmosphere_total_energy_budget_correction,
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
    use_prediction_data: bool = False,
    expected_derived_names: list[str] | None = None,
    mock_data: MockCoupledData | None = None,
):
    """
    Reusable helper for running coupled inference tests.

    Creates mock data, runs inference using the provided checkpoint, and
    performs assertions on the output.

    Args:
        mock_data: If provided, reuse this data for inference instead of
            creating new mock data. This ensures the inference data shares the
            same mask as the stepper checkpoint, which is necessary for derived
            variable metrics to be non-NaN.

    Note: Mock data is created with a 2:1 ratio of atmosphere to ocean timesteps,
    resulting in ocean_timedelta="2D" and atmosphere_timedelta="1D".
    """
    if expected_derived_names is None:
        expected_derived_names = []

    if mock_data is None:
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
    inference_data_config = InferenceDataLoaderConfig(
        dataset=CoupledDatasetWithOptionalOceanConfig(
            ocean=XarrayDataConfig(data_path=mock_data.ocean.data_dir),
            atmosphere=XarrayDataConfig(
                data_path=mock_data.atmosphere.data_dir,
            ),
        ),
        start_indices=InferenceInitialConditionIndices(
            first=0, n_initial_conditions=n_initial_conditions, interval=1
        ),
    )
    if use_prediction_data:
        prediction_data = inference_data_config
    else:
        prediction_data = None
    config = InferenceEvaluatorConfig(
        experiment_dir=str(tmp_path),
        n_coupled_steps=n_coupled_steps,
        checkpoint_path=checkpoint_path,
        logging=LoggingConfig(log_to_screen=True, log_to_file=False, log_to_wandb=True),
        loader=inference_data_config,
        prediction_loader=prediction_data,
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

    all_out_names = ocean_out_names + atmos_out_names + expected_derived_names
    ds_out_names = set(ds_ocean.data_vars).union(ds_atmos.data_vars)

    for name in expected_derived_names:
        assert name in ds_out_names

    if use_prediction_data:
        assert not os.path.exists(tmp_path / "atmosphere/restart.nc")
        assert not os.path.exists(tmp_path / "atmosphere/initial_condition.nc")
        assert not os.path.exists(tmp_path / "ocean/restart.nc")
        assert not os.path.exists(tmp_path / "ocean/initial_condition.nc")
        for var in all_out_names:
            rmse_key = f"inference/mean/weighted_rmse/{var}"
            bias_key = f"inference/mean/weighted_bias/{var}"
            logs_with_rmse = [log for log in wandb_logs if rmse_key in log]
            logs_with_bias = [log for log in wandb_logs if bias_key in log]
            for log in logs_with_rmse:
                assert log[rmse_key] == 0.0
            for log in logs_with_bias:
                assert log[bias_key] == 0.0
        for var in expected_derived_names:
            rmse_key = f"inference/mean/weighted_rmse/{var}"
            logs_with_rmse = [log for log in wandb_logs if rmse_key in log]
            assert len(logs_with_rmse) > 0, f"No RMSE logged for {var}"


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
    "n_coupled_steps,coupled_steps_in_memory,n_initial_conditions,"
    "save_standalone_component_checkpoints,use_prediction_data",
    [
        (2, 1, 2, True, False),
        (4, 2, 1, False, False),
        (2, 2, 1, False, False),
        (2, 1, 2, False, True),
    ],
)
def test_evaluator_inference(
    tmp_path: pathlib.Path,
    n_coupled_steps: int,
    coupled_steps_in_memory: int,
    n_initial_conditions: int,
    save_standalone_component_checkpoints: bool,
    use_prediction_data: bool,
    very_fast_only: bool,
):
    if very_fast_only:
        pytest.skip("Skipping non-fast tests")

    ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag", "thetao_0"]
    ocean_out_names = ["o_prog", "sst", "o_diag", "thetao_0"]
    atmos_in_names = ["a_prog", "surface_temperature", "ocean_fraction"]
    atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]
    expected_derived_names = ["ocean_heat_content"]

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

    # Run inference using the helper, reusing stepper mock data so the
    # inference data shares the same mask as the checkpoint's depth coordinate.
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
        use_prediction_data=use_prediction_data,
        expected_derived_names=expected_derived_names,
        mock_data=mock_data,
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

    mock_data: MockCoupledData | None = None
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
        assert mock_data is not None
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
        mock_data=mock_data,
    )
