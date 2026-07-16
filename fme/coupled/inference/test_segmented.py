"""Tests for the segmented coupled inference entrypoint."""

import dataclasses
import os
import pathlib
import tempfile
import unittest.mock

import cftime
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.inference.data_writer import PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.data_writer.time_coarsen import MonthlyCoarsenConfig
from fme.ace.inference.data_writer.zarr import ZarrWriterConfig
from fme.ace.inference.inference import ForcingDataLoaderConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.random_state import RandomState
from fme.core.stepper_state import StepperState
from fme.coupled.data_loading.batch_data import CoupledPrognosticState
from fme.coupled.data_loading.inference import CoupledForcingDataLoaderConfig
from fme.coupled.inference.data_writer import (
    ATMOSPHERE_OUTPUT_DIR_NAME,
    OCEAN_OUTPUT_DIR_NAME,
    CoupledPairedDataWriter,
)
from fme.coupled.inference.inference import (
    ComponentInitialConditionConfig,
    CoupledInitialConditionConfig,
    InferenceConfig,
    main,
    run_segmented_inference,
)
from fme.coupled.inference.test_inference import _setup


def _get_mock_config(experiment_dir: str) -> InferenceConfig:
    return InferenceConfig(
        experiment_dir=experiment_dir,
        n_coupled_steps=3,
        checkpoint_path="mock_checkpoint",
        logging=LoggingConfig(
            log_to_screen=True, log_to_file=False, log_to_wandb=False
        ),
        initial_condition=CoupledInitialConditionConfig(
            ocean=ComponentInitialConditionConfig(path="mock_ocean_ic"),
            atmosphere=ComponentInitialConditionConfig(path="mock_atmosphere_ic"),
        ),
        forcing_loader=CoupledForcingDataLoaderConfig(
            atmosphere=ForcingDataLoaderConfig(
                dataset=XarrayDataConfig(data_path="mock_forcing")
            ),
        ),
    )


def _restart_paths(segment_dir: str) -> tuple[str, str]:
    return (
        os.path.join(segment_dir, OCEAN_OUTPUT_DIR_NAME, "restart.nc"),
        os.path.join(segment_dir, ATMOSPHERE_OUTPUT_DIR_NAME, "restart.nc"),
    )


def _run_inference_from_config_mock(config: InferenceConfig, segment_context=None):
    for restart_path in _restart_paths(config.experiment_dir):
        os.makedirs(os.path.dirname(restart_path), exist_ok=True)
        with open(restart_path, "w") as f:
            f.write("mock restart file")
    with open(os.path.join(config.experiment_dir, "wandb_name_env_var"), "w") as f:
        f.write(os.environ.get("WANDB_NAME", ""))
    with open(os.path.join(config.experiment_dir, "ic_paths"), "w") as f:
        f.write(
            f"{config.initial_condition.ocean.path}\n"
            f"{config.initial_condition.atmosphere.path}"
        )


def test_run_segmented_inference(tmp_path, monkeypatch):
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    config = _get_mock_config(str(tmp_path))

    with unittest.mock.patch(
        "fme.coupled.inference.inference.run_inference_from_config", new=mock
    ):
        # run a single segment
        monkeypatch.setenv("WANDB_NAME", "run_name")
        run_segmented_inference(config, 1)
        segment_dir = os.path.join(config.experiment_dir, "segment_0000")
        for restart_path in _restart_paths(segment_dir):
            assert os.path.exists(restart_path)
        assert mock.call_count == 1
        with open(os.path.join(segment_dir, "wandb_name_env_var")) as f:
            assert f.read() == "run_name-segment_0000"

        # rerun the same segment and ensure run_inference_from_config isn't
        # called again
        run_segmented_inference(config, 1)
        assert mock.call_count == 1

        # extend to three segments; exactly two more segments are run, and each
        # segment after the first initializes from the previous segment's
        # restart files
        monkeypatch.setenv("WANDB_NAME", "run_name")
        run_segmented_inference(config, 3)
        assert mock.call_count == 3
        for i in range(3):
            segment_dir = os.path.join(config.experiment_dir, f"segment_{i:04d}")
            for restart_path in _restart_paths(segment_dir):
                assert os.path.exists(restart_path)
        for i in range(1, 3):
            segment_dir = os.path.join(config.experiment_dir, f"segment_{i:04d}")
            with open(os.path.join(segment_dir, "wandb_name_env_var")) as f:
                assert f.read() == f"run_name-segment_{i:04d}"
            previous_segment_dir = os.path.join(
                config.experiment_dir, f"segment_{i - 1:04d}"
            )
            with open(os.path.join(segment_dir, "ic_paths")) as f:
                assert f.read().splitlines() == list(
                    _restart_paths(previous_segment_dir)
                )


def test_run_segmented_inference_reruns_partial_segment(tmp_path):
    """A segment with only one of its two restart files is incomplete and must
    be re-run."""
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    config = _get_mock_config(str(tmp_path))
    segment_dir = os.path.join(config.experiment_dir, "segment_0000")
    ocean_restart_path, _ = _restart_paths(segment_dir)
    os.makedirs(os.path.dirname(ocean_restart_path))
    with open(ocean_restart_path, "w") as f:
        f.write("mock restart file")

    with unittest.mock.patch(
        "fme.coupled.inference.inference.run_inference_from_config", new=mock
    ):
        run_segmented_inference(config, 1)
    assert mock.call_count == 1


def test_segmented_inference_rejects_ensemble(tmp_path):
    config = _get_mock_config(str(tmp_path))
    config.n_ensemble_per_ic = 3
    with pytest.raises(ValueError, match="n_ensemble_per_ic"):
        run_segmented_inference(config, 3)


def _paired_writer(path: pathlib.Path) -> PairedDataWriter:
    os.makedirs(path, exist_ok=True)
    return PairedDataWriter(
        writers=[],
        path=str(path),
        variable_metadata={},
        coords={},
        dataset_metadata=DatasetMetadata(),
    )


def _coupled_prognostic_state(
    n_samples: int = 2,
    atmosphere_time_offset: int = 0,
) -> CoupledPrognosticState:
    def _component(name: str, hour: int) -> PrognosticState:
        time = xr.DataArray(
            [[cftime.DatetimeProlepticGregorian(2000, 1, 2, hour)]] * n_samples,
            dims=["sample", "time"],
        )
        return PrognosticState(
            BatchData.new_on_cpu(
                data={name: torch.rand(n_samples, 1, 4, 8)},
                time=time,
                stepper_state=StepperState(random_state=RandomState.from_seed(0)),
            )
        )

    return CoupledPrognosticState(
        ocean_data=_component("o_prog", 0),
        atmosphere_data=_component("a_prog", atmosphere_time_offset),
    )


def _write_coupled_restart(
    tmp_path: pathlib.Path, state: CoupledPrognosticState
) -> tuple[str, str]:
    writer = CoupledPairedDataWriter(
        ocean_writer=_paired_writer(tmp_path / OCEAN_OUTPUT_DIR_NAME),
        atmosphere_writer=_paired_writer(tmp_path / ATMOSPHERE_OUTPUT_DIR_NAME),
    )
    writer.write(state, "restart.nc")
    return _restart_paths(str(tmp_path))


def test_restart_files_as_initial_condition(tmp_path):
    """Paired restart files have no sample coordinate; the coupled initial
    condition config must read them positionally and restore the embedded
    stepper state."""
    state = _coupled_prognostic_state()
    ocean_restart_path, atmosphere_restart_path = _write_coupled_restart(
        tmp_path, state
    )

    config = CoupledInitialConditionConfig(
        ocean=ComponentInitialConditionConfig(path=ocean_restart_path),
        atmosphere=ComponentInitialConditionConfig(path=atmosphere_restart_path),
    )
    restored = config.get_initial_condition(
        ocean_prognostic_names=["o_prog"],
        atmosphere_prognostic_names=["a_prog"],
        n_ensemble_per_ic=1,
    )

    for restored_component, written_component, name in [
        (restored.ocean_data, state.ocean_data, "o_prog"),
        (restored.atmosphere_data, state.atmosphere_data, "a_prog"),
    ]:
        restored_batch = restored_component.as_batch_data()
        written_batch = written_component.as_batch_data()
        torch.testing.assert_close(
            restored_batch.data[name], written_batch.data[name], rtol=0, atol=0
        )
        assert (restored_batch.time.values == written_batch.time.values).all()
        assert restored_batch.stepper_state is not None
        assert restored_batch.stepper_state.random_state is not None


def test_restart_files_as_initial_condition_rejects_start_indices(tmp_path):
    from fme.ace.data_loading.inference import ExplicitIndices

    ocean_restart_path, atmosphere_restart_path = _write_coupled_restart(
        tmp_path, _coupled_prognostic_state()
    )
    config = CoupledInitialConditionConfig(
        ocean=ComponentInitialConditionConfig(path=ocean_restart_path),
        atmosphere=ComponentInitialConditionConfig(path=atmosphere_restart_path),
        start_indices=ExplicitIndices([0]),
    )
    with pytest.raises(ValueError, match="start_indices"):
        config.get_initial_condition(
            ocean_prognostic_names=["o_prog"],
            atmosphere_prognostic_names=["a_prog"],
            n_ensemble_per_ic=1,
        )


def test_restart_files_as_initial_condition_rejects_mismatched_times(tmp_path):
    ocean_restart_path, atmosphere_restart_path = _write_coupled_restart(
        tmp_path, _coupled_prognostic_state(atmosphere_time_offset=6)
    )
    config = CoupledInitialConditionConfig(
        ocean=ComponentInitialConditionConfig(path=ocean_restart_path),
        atmosphere=ComponentInitialConditionConfig(path=atmosphere_restart_path),
    )
    with pytest.raises(ValueError, match="different times"):
        config.get_initial_condition(
            ocean_prognostic_names=["o_prog"],
            atmosphere_prognostic_names=["a_prog"],
            n_ensemble_per_ic=1,
        )


@pytest.mark.medium_duration
def test_inference_segmented_entrypoint():
    """Two segments of N coupled steps reproduce a single 2N-step run exactly,
    and completed segments are skipped on re-invocation."""
    # tempfile instead of the tmp_path fixture, since the latter causes issues
    # with checking the last modified time of files produced by the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        n_coupled_steps = 2
        ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
        ocean_out_names = ["o_prog", "sst", "o_diag"]
        atmos_in_names = [
            "a_prog",
            "surface_temperature",
            "forcing_var",
            "ocean_fraction",
        ]
        atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

        # provide enough forcing data for both segments
        config, mock_data, atmos_steps_per_ocean_step = _setup(
            ocean_in_names=ocean_in_names,
            ocean_out_names=ocean_out_names,
            atmos_in_names=atmos_in_names,
            atmos_out_names=atmos_out_names,
            tmp_path=tmp_path,
            n_coupled_steps=2 * n_coupled_steps,
            coupled_steps_in_memory=1,
            n_initial_conditions=1,
        )
        segmented_dir = tmp_path / "segmented_run"
        config.n_coupled_steps = n_coupled_steps
        config.experiment_dir = str(segmented_dir)
        config.logging = LoggingConfig(
            log_to_screen=True, log_to_file=False, log_to_wandb=False
        )
        config_filename = tmp_path / "config.yaml"
        with open(config_filename, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)

        # both segments run in a single invocation
        main(yaml_config=str(config_filename), segments=2)

        # re-invoke and ensure completed segments are not re-run
        prediction_filenames = [
            os.path.join(
                segmented_dir,
                f"segment_{segment:04d}",
                OCEAN_OUTPUT_DIR_NAME,
                "autoregressive_predictions.nc",
            )
            for segment in range(2)
        ]
        mtimes = [os.path.getmtime(filename) for filename in prediction_filenames]
        main(yaml_config=str(config_filename), segments=2)
        for filename, mtime in zip(prediction_filenames, mtimes):
            assert os.path.getmtime(filename) == pytest.approx(mtime)

        # a non-segmented run over the same total duration
        single_dir = tmp_path / "single_run"
        config.n_coupled_steps = 2 * n_coupled_steps
        config.experiment_dir = str(single_dir)
        with open(config_filename, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        main(yaml_config=str(config_filename))

        # the second segment must match the second half of the single run
        for component_dir, n_component_steps in [
            (OCEAN_OUTPUT_DIR_NAME, n_coupled_steps),
            (ATMOSPHERE_OUTPUT_DIR_NAME, n_coupled_steps * atmos_steps_per_ocean_step),
        ]:
            ds_segment_1 = xr.open_dataset(
                segmented_dir
                / "segment_0001"
                / component_dir
                / "autoregressive_predictions.nc",
                decode_timedelta=False,
            )
            ds_single = xr.open_dataset(
                single_dir / component_dir / "autoregressive_predictions.nc",
                decode_timedelta=False,
            )
            assert ds_single.sizes["time"] == 2 * n_component_steps
            # drop time coordinates, which differ by construction
            # (per-segment initial time)
            ds_segment_1 = ds_segment_1.drop_vars(["init_time", "time"])
            ds_single = ds_single.drop_vars(["init_time", "time"])
            xr.testing.assert_equal(
                ds_segment_1, ds_single.isel(time=slice(n_component_steps, None))
            )


@pytest.mark.medium_duration
def test_inference_segmented_entrypoint_zarr_outputs():
    """Zarr outputs configured via `files` accumulate into single whole-run
    stores shared by all segments, identical to those of an unsegmented run.
    Monthly zarr outputs accumulate across the segment boundary via the
    per-segment snapshots."""
    from fme.ace.inference.data_writer.main import DataWriterConfig
    from fme.coupled.inference.data_writer import CoupledDataWriterConfig

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        n_coupled_steps = 2
        ocean_in_names = ["o_prog", "sst", "mask_0", "a_diag"]
        ocean_out_names = ["o_prog", "sst", "o_diag"]
        atmos_in_names = [
            "a_prog",
            "surface_temperature",
            "forcing_var",
            "ocean_fraction",
        ]
        atmos_out_names = ["a_prog", "surface_temperature", "a_diag"]

        config, _, _ = _setup(
            ocean_in_names=ocean_in_names,
            ocean_out_names=ocean_out_names,
            atmos_in_names=atmos_in_names,
            atmos_out_names=atmos_out_names,
            tmp_path=tmp_path,
            n_coupled_steps=2 * n_coupled_steps,
            coupled_steps_in_memory=1,
            n_initial_conditions=1,
        )
        component_writer_config = DataWriterConfig(
            save_prediction_files=False,
            save_monthly_files=False,
            files=[
                FileWriterConfig(label="output", format=ZarrWriterConfig()),
                FileWriterConfig(
                    label="monthly",
                    format=ZarrWriterConfig(),
                    time_coarsen=MonthlyCoarsenConfig(),
                ),
            ],
        )
        config.data_writer = CoupledDataWriterConfig(
            ocean=component_writer_config,
            atmosphere=component_writer_config,
        )
        config.logging = LoggingConfig(
            log_to_screen=True, log_to_file=False, log_to_wandb=False
        )

        segmented_dir = tmp_path / "segmented_run"
        config.n_coupled_steps = n_coupled_steps
        config.experiment_dir = str(segmented_dir)
        config_filename = tmp_path / "config.yaml"
        with open(config_filename, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        main(yaml_config=str(config_filename), segments=2)

        single_dir = tmp_path / "single_run"
        config.n_coupled_steps = 2 * n_coupled_steps
        config.experiment_dir = str(single_dir)
        with open(config_filename, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        main(yaml_config=str(config_filename))

        for component_dir in [OCEAN_OUTPUT_DIR_NAME, ATMOSPHERE_OUTPUT_DIR_NAME]:
            for store in [
                "output_predictions.zarr",
                "output_target.zarr",
                "monthly_predictions.zarr",
                "monthly_target.zarr",
            ]:
                ds_segmented = xr.open_zarr(
                    str(segmented_dir / component_dir / store),
                    decode_timedelta=False,
                ).load()
                ds_single = xr.open_zarr(
                    str(single_dir / component_dir / store),
                    decode_timedelta=False,
                ).load()
                # the creation timestamps differ by construction
                ds_segmented.attrs.pop("history.created", None)
                ds_single.attrs.pop("history.created", None)
                xr.testing.assert_identical(ds_segmented, ds_single)
        # the monthly snapshots ride the segment directories
        assert (
            segmented_dir
            / "segment_0000"
            / OCEAN_OUTPUT_DIR_NAME
            / "monthly_snapshots"
            / "monthly_predictions.nc"
        ).exists()
