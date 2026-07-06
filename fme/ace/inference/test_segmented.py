"""Tests for segmented inference entrypoint."""

import dataclasses
import datetime
import os
import pathlib
import tempfile
import unittest.mock

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import fme
from fme.ace.aggregator.inference import InferenceAggregatorConfig
from fme.ace.data_loading.batch_data import BatchData, PrognosticState
from fme.ace.data_loading.inference import ForcingDataLoaderConfig, TimestampList
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.inference import (
    InitialConditionConfig,
    main,
    run_segmented_inference,
)
from fme.ace.registry import ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.stepper import StepperConfig
from fme.ace.testing import DimSizes, FV3GFSData
from fme.core.coordinates import (
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.corrector.state import CorrectorState
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NetworkAndLossNormalizationConfig, NormalizationConfig
from fme.core.random_state import RandomState
from fme.core.step.single_module import SingleModuleStepConfig
from fme.core.step.step import StepSelector
from fme.core.stepper_state import StepperState
from fme.core.testing import mock_wandb

TIMESTEP = datetime.timedelta(hours=6)


class PlusOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


def save_stepper(
    path: pathlib.Path,
    in_names: list[str],
    out_names: list[str],
    mean: float,
    std: float,
    data_shape: list[int],
    timestep: datetime.timedelta = TIMESTEP,
):
    all_names = list(set(in_names).union(out_names))
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="prebuilt", config={"module": PlusOne()}
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={name: mean for name in all_names},
                            stds={name: std for name in all_names},
                        ),
                    ),
                ),
            ),
        ),
    )
    horizontal_coordinate = LatLonCoordinates(
        lat=torch.zeros(data_shape[-2]), lon=torch.zeros(data_shape[-1])
    )
    vertical_coordinate = HybridSigmaPressureCoordinate(
        ak=torch.arange(7), bk=torch.arange(7)
    )
    stepper = config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=horizontal_coordinate,
            vertical_coordinate=vertical_coordinate,
            timestep=timestep,
        ),
    )
    torch.save({"stepper": stepper.get_state()}, path)


def test_inference_segmented_entrypoint():
    # we use tempfile here instead of pytest tmp_path fixture, because the latter causes
    # issues with checking last modified time of files produced by the test.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = pathlib.Path(tmp_dir)
        forward_steps_in_memory = 2
        in_names = ["prog", "forcing_var"]
        out_names = ["prog", "diagnostic_var"]
        stepper_path = tmp_path / "stepper"
        horizontal = [DimSize("lat", 16), DimSize("lon", 32)]

        dim_sizes = DimSizes(
            n_time=18,
            horizontal=horizontal,
            nz_interface=4,
        )
        save_stepper(
            stepper_path,
            in_names=in_names,
            out_names=out_names,
            mean=0.0,
            std=1.0,
            data_shape=dim_sizes.shape_nd,
        )
        data = FV3GFSData(
            path=tmp_path,
            names=["forcing_var"],
            dim_sizes=dim_sizes,
            timestep_days=0.25,
        )
        initial_condition = xr.Dataset(
            {
                "prog": xr.DataArray(
                    np.random.rand(2, 16, 32).astype(np.float32),
                    dims=["sample", "lat", "lon"],
                )
            }
        )

        initial_condition_path = tmp_path / "init_data" / "ic.nc"
        initial_condition_path.parent.mkdir()
        initial_condition["time"] = xr.DataArray(
            [
                cftime.DatetimeProlepticGregorian(2000, 1, 1, 6),
                cftime.DatetimeProlepticGregorian(2000, 1, 1, 18),
            ],
            dims=["sample"],
        )
        initial_condition.to_netcdf(initial_condition_path, mode="w")
        forcing_loader = ForcingDataLoaderConfig(
            dataset=data.inference_data_loader_config.dataset,
            num_data_workers=0,
        )

        run_dir = tmp_path / "segmented_run"
        config = fme.ace.InferenceConfig(
            experiment_dir=str(run_dir),
            n_forward_steps=3,
            forward_steps_in_memory=forward_steps_in_memory,
            checkpoint_path=str(stepper_path),
            logging=LoggingConfig(
                log_to_screen=True, log_to_file=False, log_to_wandb=False
            ),
            initial_condition=InitialConditionConfig(
                path=str(initial_condition_path),
                start_indices=TimestampList(["2000-01-01T06:00:00"]),
            ),
            forcing_loader=forcing_loader,
            data_writer=DataWriterConfig(
                save_prediction_files=False,
                save_monthly_files=False,
                files=[FileWriterConfig("autoregressive")],
            ),
            allow_incompatible_dataset=True,  # stepper checkpoint has arbitrary info  # noqa: E501
        )

        # run one segment of 3 steps
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        main(config_path, 1)

        # run another segment of 3 steps, and ensure first segment is not being re-run
        filename = os.path.join(
            run_dir, "segment_0000", "autoregressive_predictions.nc"
        )
        before_second_segment_mtime = os.path.getmtime(filename)
        main(config_path, 2)
        after_second_segment_mtime = os.path.getmtime(filename)
        assert before_second_segment_mtime == pytest.approx(after_second_segment_mtime)

        # do a non-segmented run of 6 steps
        config.n_forward_steps = 6
        config.experiment_dir = str(tmp_path / "non_segmented_run")
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        main(config_path)

        # assert each segment generated output of correct duration
        ds_two_segments_0 = xr.open_dataset(
            run_dir / "segment_0000" / "autoregressive_predictions.nc",
            decode_timedelta=False,
        )
        ds_two_segments_1 = xr.open_dataset(
            run_dir / "segment_0001" / "autoregressive_predictions.nc",
            decode_timedelta=False,
        )
        assert len(ds_two_segments_0.time) == len(ds_two_segments_1.time)

        # Ensure the second half of the 6-step run matches the second segment of the
        # 3-step run. Before comparing, drop init_time and time coordinates, since
        # we don't expect these to match.
        ds_one_segment = xr.open_dataset(
            tmp_path / "non_segmented_run" / "autoregressive_predictions.nc",
            decode_timedelta=False,
        )
        ds_two_segments_1 = ds_two_segments_1.drop_vars(["init_time", "time"])
        ds_one_segment = ds_one_segment.drop_vars(["init_time", "time"])
        xr.testing.assert_equal(
            ds_two_segments_1, ds_one_segment.isel(time=slice(3, None))
        )


def _run_inference_from_config_mock(config: fme.ace.InferenceConfig):
    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "restart.nc"), "w") as f:
        f.write("mock restart file")
    with open(os.path.join(config.experiment_dir, "wandb_name_env_var"), "w") as f:
        f.write(os.environ.get("WANDB_NAME", ""))


def _get_mock_config(experiment_dir: str) -> fme.ace.InferenceConfig:
    return fme.ace.InferenceConfig(
        experiment_dir=experiment_dir,
        n_forward_steps=3,
        checkpoint_path="mock_checkpoint",
        logging=LoggingConfig(
            log_to_screen=True, log_to_file=False, log_to_wandb=False
        ),
        initial_condition=InitialConditionConfig(path="mock_ic"),
        forcing_loader=ForcingDataLoaderConfig(
            dataset=XarrayDataConfig(data_path="mock_forcing")
        ),
        data_writer=DataWriterConfig(
            save_prediction_files=False, save_monthly_files=False
        ),
    )


def test_run_segmented_inference(tmp_path, monkeypatch):
    WRITTEN_WANDB_NAME_FILENAME = "wandb_name_env_var"
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    config = _get_mock_config(str(tmp_path))

    with unittest.mock.patch(
        "fme.ace.inference.inference.run_inference_from_config", new=mock
    ):
        # run a single segment
        monkeypatch.setenv("WANDB_NAME", "run_name")
        run_segmented_inference(config, 1)
        segment_dir = os.path.join(config.experiment_dir, "segment_0000")
        expected_restart_path = os.path.join(segment_dir, "restart.nc")
        assert os.path.exists(expected_restart_path)
        assert mock.call_count == 1
        with open(os.path.join(segment_dir, WRITTEN_WANDB_NAME_FILENAME)) as f:
            assert f.read() == "run_name-segment_0000"

        # rerun the same segment and ensure run_inference_from_config isn't called again
        run_segmented_inference(config, 1)
        assert os.path.exists(expected_restart_path)
        assert mock.call_count == 1

        # extend to three segments and ensure exactly three run_inference_from_config
        # calls have been made
        monkeypatch.setenv("WANDB_NAME", "run_name")
        run_segmented_inference(config, 3)
        for i in range(3):
            segment_dir = os.path.join(config.experiment_dir, f"segment_{i:04d}")
            expected_restart_path = os.path.join(segment_dir, "restart.nc")
            assert os.path.exists(expected_restart_path)
            with open(os.path.join(segment_dir, WRITTEN_WANDB_NAME_FILENAME)) as f:
                assert f.read() == f"run_name-segment_{i:04d}"
        assert mock.call_count == 3


def save_noise_conditioned_stepper(
    path: pathlib.Path,
    in_names: list[str],
    out_names: list[str],
    data_shape: list[int],
    timestep: datetime.timedelta = TIMESTEP,
):
    """Save a minimal NoiseConditionedSFNO stepper whose noise actually affects the
    output, so a stochastic rollout is only reproducible across a restart if the
    random state is threaded through the sidecar."""
    all_names = list(set(in_names).union(out_names))
    config = StepperConfig(
        step=StepSelector(
            type="single_module",
            config=dataclasses.asdict(
                SingleModuleStepConfig(
                    builder=ModuleSelector(
                        type="NoiseConditionedSFNO",
                        config=dataclasses.asdict(
                            NoiseConditionedSFNOBuilder(
                                embed_dim=4,
                                noise_embed_dim=4,
                                noise_type="gaussian",
                                num_layers=2,
                                pos_embed=False,
                                filter_type="linear",
                                filter_num_groups=1,
                            )
                        ),
                    ),
                    in_names=in_names,
                    out_names=out_names,
                    normalization=NetworkAndLossNormalizationConfig(
                        network=NormalizationConfig(
                            means={name: 0.0 for name in all_names},
                            stds={name: 1.0 for name in all_names},
                        ),
                    ),
                ),
            ),
        ),
    )
    stepper = config.get_stepper(
        dataset_info=DatasetInfo(
            horizontal_coordinates=LatLonCoordinates(
                lat=torch.zeros(data_shape[-2]), lon=torch.zeros(data_shape[-1])
            ),
            vertical_coordinate=HybridSigmaPressureCoordinate(
                ak=torch.arange(7), bk=torch.arange(7)
            ),
            timestep=timestep,
        ),
    )
    with torch.no_grad():
        for name, param in stepper._step_obj.modules.named_parameters():
            if "W_scale_2d" in name or "W_bias_2d" in name:
                param.fill_(0.1)
    torch.save({"stepper": stepper.get_state()}, path)


@pytest.mark.slow
def test_segmented_stochastic_inference_matches_single_run(tmp_path):
    """The acceptance test: a seeded stochastic rollout split into two segments is
    bitwise-identical to the same rollout run in one segment.

    This only holds if the random state (advancing noise generator) is serialized
    at the segment boundary and restored for the next segment, which is the point
    of the restart sidecar. With a deterministic stepper this would pass
    vacuously; the NoiseConditionedSFNO here has active noise (asserted below via
    seed sensitivity), so the cross-restart match is a real reproducibility
    result."""
    forward_steps_in_memory = 2
    in_names = ["prog", "forcing_var"]
    out_names = ["prog", "diagnostic_var"]
    stepper_path = tmp_path / "stepper"
    horizontal = [DimSize("lat", 8), DimSize("lon", 16)]
    dim_sizes = DimSizes(n_time=13, horizontal=horizontal, nz_interface=4)
    save_noise_conditioned_stepper(
        stepper_path,
        in_names=in_names,
        out_names=out_names,
        data_shape=dim_sizes.shape_nd,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=["forcing_var"],
        dim_sizes=dim_sizes,
        timestep_days=0.25,
    )
    initial_condition = xr.Dataset(
        {
            "prog": xr.DataArray(
                np.random.rand(1, 8, 16).astype(np.float32),
                dims=["sample", "lat", "lon"],
            )
        }
    )
    ic_path = tmp_path / "init_data" / "ic.nc"
    ic_path.parent.mkdir()
    initial_condition["time"] = xr.DataArray(
        [cftime.DatetimeProlepticGregorian(2000, 1, 1, 6)],
        dims=["sample"],
    )
    initial_condition.to_netcdf(ic_path, mode="w")

    def make_config(experiment_dir: str, n_forward_steps: int, seed: int | None):
        return fme.ace.InferenceConfig(
            experiment_dir=experiment_dir,
            n_forward_steps=n_forward_steps,
            forward_steps_in_memory=forward_steps_in_memory,
            checkpoint_path=str(stepper_path),
            logging=LoggingConfig(
                log_to_screen=False, log_to_file=False, log_to_wandb=False
            ),
            initial_condition=InitialConditionConfig(
                path=str(ic_path),
                start_indices=TimestampList(["2000-01-01T06:00:00"]),
            ),
            forcing_loader=ForcingDataLoaderConfig(
                dataset=data.inference_data_loader_config.dataset,
                num_data_workers=0,
            ),
            data_writer=DataWriterConfig(
                save_prediction_files=False,
                save_monthly_files=False,
                files=[FileWriterConfig("autoregressive")],
            ),
            aggregator=InferenceAggregatorConfig(log_global_mean_time_series=False),
            allow_incompatible_dataset=True,
            seed=seed,
        )

    def run(config: fme.ace.InferenceConfig, segments: int | None):
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        with mock_wandb() as wandb:
            wandb.configure(log_to_wandb=False)
            main(config_path, segments)

    # Two segments of 3 steps each, seeded.
    two_seg_dir = tmp_path / "two_segments"
    run(make_config(str(two_seg_dir), n_forward_steps=3, seed=0), segments=2)

    # The same rollout as a single 6-step run, seeded identically.
    single_dir = tmp_path / "single_run"
    run(make_config(str(single_dir), n_forward_steps=6, seed=0), segments=None)

    # The second segment (steps 4-6) must match the single run's steps 4-6. Drop
    # the time coordinates, which differ by construction (per-segment init_time).
    ds_segment_1 = xr.open_dataset(
        two_seg_dir / "segment_0001" / "autoregressive_predictions.nc",
        decode_timedelta=False,
    ).drop_vars(["init_time", "time"])
    ds_single = xr.open_dataset(
        single_dir / "autoregressive_predictions.nc",
        decode_timedelta=False,
    ).drop_vars(["init_time", "time"])
    xr.testing.assert_equal(ds_segment_1, ds_single.isel(time=slice(3, None)))

    # Non-vacuousness: a different seed changes the noise, so the segment-1 output
    # differs. This confirms the match above is a genuine reproduction of an
    # active stochastic sequence, not a deterministic artifact.
    two_seg_seed1_dir = tmp_path / "two_segments_seed1"
    run(make_config(str(two_seg_seed1_dir), n_forward_steps=3, seed=1), segments=2)
    ds_segment_1_seed1 = xr.open_dataset(
        two_seg_seed1_dir / "segment_0001" / "autoregressive_predictions.nc",
        decode_timedelta=False,
    ).drop_vars(["init_time", "time"])
    assert not np.allclose(
        ds_segment_1["prog"].values, ds_segment_1_seed1["prog"].values
    )


def _paired_writer(tmp_path: pathlib.Path) -> PairedDataWriter:
    """A minimal PairedDataWriter for exercising the stepper-state sidecar."""
    return PairedDataWriter(
        writers=[],
        path=str(tmp_path),
        variable_metadata={},
        coords={},
        dataset_metadata=DatasetMetadata(),
    )


def _prognostic_state_with(stepper_state: StepperState | None) -> PrognosticState:
    batch = BatchData.new_on_cpu(
        data={"prog": torch.zeros(1, 1, 4, 8)},
        time=xr.DataArray(
            [[cftime.DatetimeProlepticGregorian(2000, 1, 1)]],
            dims=["sample", "time"],
        ),
        stepper_state=stepper_state,
    )
    return PrognosticState(batch)


def test_stepper_state_sidecar_none_is_backcompat_noop(tmp_path):
    """A restart with no stepper state writes no sidecar, and a config that does
    not point at one restores nothing (stepper_state stays None) - the pre-feature
    behavior for restart files written before the sidecar existed."""
    writer = _paired_writer(tmp_path)
    writer.write_stepper_state(_prognostic_state_with(None), "restart_stepper_state.pt")
    assert not (tmp_path / "restart_stepper_state.pt").exists()

    # A config that names no sidecar restores nothing.
    config = InitialConditionConfig(path=str(tmp_path / "ic.nc"))
    assert config.stepper_state_path is None
    with pytest.raises(ValueError, match="not set"):
        config.get_stepper_state()

    # A set-but-missing sidecar is a user error and raises clearly.
    missing = InitialConditionConfig(
        path=str(tmp_path / "ic.nc"),
        stepper_state_path=str(tmp_path / "does_not_exist.pt"),
    )
    with pytest.raises(FileNotFoundError, match="does not exist"):
        missing.get_stepper_state()


def test_stepper_state_sidecar_corrector_and_random_continuity(tmp_path):
    """The sidecar written by the writer and loaded via InitialConditionConfig
    preserves the corrector's pinned global_dry_air_mass exactly and continues the
    random generator's draw sequence."""
    mass = torch.randn(1, 1, 1)
    random_state = RandomState.from_seed(11)
    stepper_state = StepperState(
        corrector_state=CorrectorState(global_dry_air_mass=mass.clone()),
        random_state=RandomState.from_seed(11),
    )
    writer = _paired_writer(tmp_path)
    writer.write_stepper_state(
        _prognostic_state_with(stepper_state), "restart_stepper_state.pt"
    )

    config = InitialConditionConfig(
        path=str(tmp_path / "ic.nc"),
        stepper_state_path=str(tmp_path / "restart_stepper_state.pt"),
    )
    restored = config.get_stepper_state()

    assert restored.corrector_state is not None
    assert restored.corrector_state.global_dry_air_mass is not None
    torch.testing.assert_close(
        restored.corrector_state.global_dry_air_mass, mass, rtol=0, atol=0
    )
    assert restored.random_state is not None
    torch.testing.assert_close(
        torch.randn(4, generator=random_state.generator),
        torch.randn(4, generator=restored.random_state.generator),
        rtol=0,
        atol=0,
    )


def test_segmented_inference_rejects_ensemble(tmp_path):
    # Ensemble inference is unsupported with segmented inference: a segment's
    # restart already carries the broadcasted ensemble as its sample dimension,
    # so later segments cannot re-broadcast it consistently. The config should be
    # rejected up front rather than silently re-interpreted.
    config = _get_mock_config(str(tmp_path))
    config.n_ensemble_per_ic = 3
    with pytest.raises(ValueError, match="n_ensemble_per_ic"):
        run_segmented_inference(config, 3)
