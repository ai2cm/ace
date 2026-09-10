"""Tests for the segmented coupled inference entrypoint."""

import contextlib
import dataclasses
import datetime
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
from fme.ace.data_loading.inference import ExplicitIndices
from fme.ace.inference.data_writer import PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.inference import ForcingDataLoaderConfig
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.random_state import RandomState
from fme.core.stepper_state import StepperState
from fme.core.testing import mock_wandb
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


# Fixed anchor used by the unit tests below: the mock config carries a
# placeholder checkpoint/IC, so we stub out the (expensive, stepper-loading)
# _get_initialization_time_and_timestep with a known start time and coupled
# timestep. With n_coupled_steps=3 and a 2-day timestep, segments start 6 days
# apart.
_MOCK_INITIALIZATION_TIME = cftime.DatetimeProlepticGregorian(1970, 1, 1)
_MOCK_TIMESTEP = datetime.timedelta(days=2)
_MOCK_SEGMENT_LABELS = [
    "segment_19700101T00",
    "segment_19700107T00",
    "segment_19700113T00",
]


@contextlib.contextmanager
def _mock_segmented_dependencies():
    """Mock the two functions run_segmented_inference calls into: the per-segment
    inference run, and the one-time initialization-time/timestep lookup that
    otherwise loads a real stepper and initial condition."""
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    with (
        unittest.mock.patch(
            "fme.coupled.inference.inference.run_inference_from_config", new=mock
        ),
        unittest.mock.patch(
            "fme.coupled.inference.inference._get_initialization_time_and_timestep",
            return_value=(_MOCK_INITIALIZATION_TIME, _MOCK_TIMESTEP),
        ),
    ):
        yield mock


def _restart_paths(segment_dir: str) -> tuple[str, str]:
    return (
        os.path.join(segment_dir, OCEAN_OUTPUT_DIR_NAME, "restart.nc"),
        os.path.join(segment_dir, ATMOSPHERE_OUTPUT_DIR_NAME, "restart.nc"),
    )


def _run_inference_from_config_mock(config: InferenceConfig):
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
    config = _get_mock_config(str(tmp_path))

    with _mock_segmented_dependencies() as mock:
        # run a single segment
        monkeypatch.setenv("WANDB_NAME", "run_name")
        run_segmented_inference(config, 1)
        segment_dir = os.path.join(config.experiment_dir, _MOCK_SEGMENT_LABELS[0])
        for restart_path in _restart_paths(segment_dir):
            assert os.path.exists(restart_path)
        assert mock.call_count == 1
        with open(os.path.join(segment_dir, "wandb_name_env_var")) as f:
            assert f.read() == f"run_name-{_MOCK_SEGMENT_LABELS[0]}"

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
        for label in _MOCK_SEGMENT_LABELS:
            segment_dir = os.path.join(config.experiment_dir, label)
            for restart_path in _restart_paths(segment_dir):
                assert os.path.exists(restart_path)
        for i in range(1, 3):
            segment_dir = os.path.join(config.experiment_dir, _MOCK_SEGMENT_LABELS[i])
            with open(os.path.join(segment_dir, "wandb_name_env_var")) as f:
                assert f.read() == f"run_name-{_MOCK_SEGMENT_LABELS[i]}"
            previous_segment_dir = os.path.join(
                config.experiment_dir, _MOCK_SEGMENT_LABELS[i - 1]
            )
            with open(os.path.join(segment_dir, "ic_paths")) as f:
                assert f.read().splitlines() == list(
                    _restart_paths(previous_segment_dir)
                )


def test_run_segmented_inference_reruns_partial_segment(tmp_path):
    """A segment with only one of its two restart files is incomplete and must
    be re-run."""
    config = _get_mock_config(str(tmp_path))
    segment_dir = os.path.join(config.experiment_dir, _MOCK_SEGMENT_LABELS[0])
    ocean_restart_path, _ = _restart_paths(segment_dir)
    os.makedirs(os.path.dirname(ocean_restart_path))
    with open(ocean_restart_path, "w") as f:
        f.write("mock restart file")

    with _mock_segmented_dependencies() as mock:
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


def test_restart_files_as_initial_condition_rejects_mismatched_sample_counts(
    tmp_path,
):
    """Positionally-aligned restarts must have matching sample counts, since
    there is no coordinate to align them by."""
    ocean_restart_path, _ = _write_coupled_restart(
        tmp_path / "two_samples", _coupled_prognostic_state(n_samples=2)
    )
    _, atmosphere_restart_path = _write_coupled_restart(
        tmp_path / "three_samples", _coupled_prognostic_state(n_samples=3)
    )
    config = CoupledInitialConditionConfig(
        ocean=ComponentInitialConditionConfig(path=ocean_restart_path),
        atmosphere=ComponentInitialConditionConfig(path=atmosphere_restart_path),
    )
    with pytest.raises(ValueError, match="different numbers of"):
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


def _segmented_config_factory(
    tmp_path: pathlib.Path,
    total_coupled_steps: int,
    *,
    log_to_wandb: bool = False,
):
    """Write coupled forcing data and a checkpoint under ``tmp_path``, with enough
    forcing for ``total_coupled_steps`` coupled steps so that a run of that length
    can be split into segments.

    Returns ``(make_config, atmos_steps_per_ocean_step)``, where
    ``make_config(experiment_dir, n_coupled_steps)`` writes a config yaml and
    returns its path.
    """
    config, _, atmos_steps_per_ocean_step = _setup(
        ocean_in_names=["o_prog", "sst", "mask_0", "a_diag"],
        ocean_out_names=["o_prog", "sst", "o_diag"],
        atmos_in_names=[
            "a_prog",
            "surface_temperature",
            "forcing_var",
            "ocean_fraction",
        ],
        atmos_out_names=["a_prog", "surface_temperature", "a_diag"],
        tmp_path=tmp_path,
        n_coupled_steps=total_coupled_steps,
        coupled_steps_in_memory=1,
        n_initial_conditions=1,
    )
    config.logging = LoggingConfig(
        log_to_screen=True, log_to_file=False, log_to_wandb=log_to_wandb
    )

    def make_config(experiment_dir: pathlib.Path, n_coupled_steps: int) -> str:
        config.n_coupled_steps = n_coupled_steps
        config.experiment_dir = str(experiment_dir)
        # one yaml per experiment dir, so a previously written config stays valid
        config_filename = str(tmp_path / f"config-{experiment_dir.name}.yaml")
        with open(config_filename, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        return config_filename

    return make_config, atmos_steps_per_ocean_step


@pytest.mark.medium_duration
def test_inference_segmented_entrypoint():
    """Two segments of N coupled steps reproduce a single 2N-step run exactly,
    and completed segments are skipped on re-invocation."""
    # tempfile instead of the tmp_path fixture, since the latter causes issues
    # with checking the last modified time of files produced by the test
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        n_coupled_steps = 2
        make_config, atmos_steps_per_ocean_step = _segmented_config_factory(
            tmp_path, total_coupled_steps=2 * n_coupled_steps
        )
        segmented_dir = tmp_path / "segmented_run"
        config_filename = make_config(segmented_dir, n_coupled_steps)

        # both segments run in a single invocation
        main(yaml_config=config_filename, segments=2)

        # re-invoke and ensure completed segments are not re-run. Ocean data
        # starts at 1970-01-01 with a 2-day ocean timestep, so with
        # n_coupled_steps=2 the segments start 4 days apart.
        segment_labels = ["segment_19700101T00", "segment_19700105T00"]
        prediction_filenames = [
            os.path.join(
                segmented_dir,
                label,
                OCEAN_OUTPUT_DIR_NAME,
                "autoregressive_predictions.nc",
            )
            for label in segment_labels
        ]
        mtimes = [os.path.getmtime(filename) for filename in prediction_filenames]
        main(yaml_config=config_filename, segments=2)
        for filename, mtime in zip(prediction_filenames, mtimes):
            assert os.path.getmtime(filename) == pytest.approx(mtime)

        # a non-segmented run over the same total duration
        single_dir = tmp_path / "single_run"
        main(yaml_config=make_config(single_dir, 2 * n_coupled_steps))

        # the second segment must match the second half of the single run
        for component_dir, n_component_steps in [
            (OCEAN_OUTPUT_DIR_NAME, n_coupled_steps),
            (ATMOSPHERE_OUTPUT_DIR_NAME, n_coupled_steps * atmos_steps_per_ocean_step),
        ]:
            ds_segment_1 = xr.open_dataset(
                segmented_dir
                / segment_labels[1]
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
def test_segmented_inference_wandb_run_per_segment(monkeypatch):
    """Each segment gets its own wandb run named ``<base>-<start time>``, matching
    the fme.ace segmented driver (issue #471). Sharing one run across segments
    would collide their step counters and silently ignore the per-segment names,
    since wandb reads WANDB_NAME only when the first run starts.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        make_config, _ = _segmented_config_factory(
            tmp_path, total_coupled_steps=4, log_to_wandb=True
        )
        config_filename = make_config(tmp_path / "segmented_run", 2)
        monkeypatch.setenv("WANDB_NAME", "myrun")
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=True)
            main(yaml_config=config_filename, segments=2)
            assert [run["name"] for run in wandb.runs] == [
                "myrun-segment_19700101T00",
                "myrun-segment_19700105T00",
            ]
            assert len({run["id"] for run in wandb.runs}) == 2  # distinct runs
