"""Tests for segmented inference entrypoint."""

import dataclasses
import datetime
import os
import pathlib
import unittest.mock

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

import fme
from fme.ace.aggregator.inference import InferenceAggregatorConfig
from fme.ace.data_loading.batch_data import _RESERVED_PREFIX, BatchData, PrognosticState
from fme.ace.data_loading.inference import (
    ExplicitIndices,
    ForcingDataLoaderConfig,
    TimestampList,
)
from fme.ace.inference.data_writer import DataWriterConfig, PairedDataWriter
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.inference import (
    InitialConditionConfig,
    get_initial_condition,
    main,
    run_segmented_inference,
)
from fme.ace.registry import ModuleSelector
from fme.ace.registry.stochastic_sfno import NoiseConditionedSFNOBuilder
from fme.ace.requirements import InitialConditionRequirements
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
from fme.core.labels import BatchLabels
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


def _segmented_config_factory(tmp_path: pathlib.Path, *, log_to_wandb: bool):
    """Write a PlusOne stepper, forcing data and initial condition into tmp_path,
    returning a ``make_config(experiment_dir, n_forward_steps)`` builder."""
    stepper_path = tmp_path / "stepper"
    dim_sizes = DimSizes(
        n_time=18, horizontal=[DimSize("lat", 16), DimSize("lon", 32)], nz_interface=4
    )
    save_stepper(
        stepper_path,
        in_names=["prog", "forcing_var"],
        out_names=["prog", "diagnostic_var"],
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
    )
    data = FV3GFSData(
        path=tmp_path, names=["forcing_var"], dim_sizes=dim_sizes, timestep_days=0.25
    )
    ic = xr.Dataset(
        {
            "prog": xr.DataArray(
                np.random.rand(1, 16, 32).astype(np.float32),
                dims=["sample", "lat", "lon"],
            )
        }
    )
    ic_path = tmp_path / "init_data" / "ic.nc"
    ic_path.parent.mkdir()
    ic["time"] = xr.DataArray(
        [cftime.DatetimeProlepticGregorian(2000, 1, 1, 6)], dims=["sample"]
    )
    ic.to_netcdf(ic_path, mode="w")

    def make_config(experiment_dir, n_forward_steps: int) -> str:
        config = fme.ace.InferenceConfig(
            experiment_dir=str(experiment_dir),
            n_forward_steps=n_forward_steps,
            forward_steps_in_memory=2,
            checkpoint_path=str(stepper_path),
            logging=LoggingConfig(
                log_to_screen=True, log_to_file=False, log_to_wandb=log_to_wandb
            ),
            initial_condition=InitialConditionConfig(
                path=str(ic_path),
                start_indices=TimestampList(["2000-01-01T06:00:00"]),
            ),
            forcing_loader=ForcingDataLoaderConfig(
                dataset=data.inference_data_loader_config.dataset, num_data_workers=0
            ),
            data_writer=DataWriterConfig(
                save_prediction_files=False,
                save_monthly_files=False,
                files=[FileWriterConfig("autoregressive")],
            ),
            allow_incompatible_dataset=True,
        )
        config_path = str(tmp_path / "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(dataclasses.asdict(config), f)
        return config_path

    return make_config


def test_inference_segmented_entrypoint(tmp_path, monkeypatch):
    """End-to-end: a segmented run reproduces the equivalent single run, and each
    segment gets its own wandb run named ``<base>-<start time>`` (issue #471)."""
    make_config = _segmented_config_factory(tmp_path, log_to_wandb=True)
    run_dir = tmp_path / "segmented_run"
    single_dir = tmp_path / "non_segmented_run"
    monkeypatch.setenv("WANDB_NAME", "myrun")
    with mock_wandb() as wandb:
        wandb.configure(log_to_wandb=True)
        main(make_config(run_dir, 3), 2)
        assert [run["name"] for run in wandb.runs] == [
            "myrun-20000101T06",
            "myrun-20000102T00",
        ]
        assert len({run["id"] for run in wandb.runs}) == 2  # distinct runs
        # a single 6-step run should reproduce the two 3-step segments
        main(make_config(single_dir, 6))

    def _predictions(path):
        return xr.open_dataset(
            path / "autoregressive_predictions.nc", decode_timedelta=False
        ).drop_vars(["init_time", "time"])  # per-segment init_time differs

    xr.testing.assert_equal(
        _predictions(run_dir / "20000102T00"),
        _predictions(single_dir).isel(time=slice(3, None)),
    )


def _run_inference_from_config_mock(config: fme.ace.InferenceConfig):
    if not os.path.exists(config.experiment_dir):
        os.makedirs(config.experiment_dir)
    with open(os.path.join(config.experiment_dir, "restart.nc"), "w") as f:
        f.write("mock restart file")


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


def _mock_config_with_real_checkpoint_and_ic(
    tmp_path: pathlib.Path, timestep: datetime.timedelta = TIMESTEP
) -> fme.ace.InferenceConfig:
    """A mock config (see _get_mock_config) with a real stepper checkpoint and
    initial condition standing in for the placeholder ones, since
    _get_initialization_time_and_timestep loads both to compute each segment's
    timestamped directory label (anchored to the first, or only, ensemble
    member's start time)."""
    stepper_path = tmp_path / "stepper"
    save_stepper(
        stepper_path,
        in_names=["prog"],
        out_names=["prog"],
        mean=0.0,
        std=1.0,
        data_shape=[4, 8],
        timestep=timestep,
    )
    ic_path = tmp_path / "ic.nc"
    ic = xr.Dataset(
        {
            "prog": xr.DataArray(
                np.zeros((1, 4, 8), dtype=np.float32), dims=["sample", "lat", "lon"]
            )
        }
    )
    ic["time"] = xr.DataArray(
        [cftime.DatetimeProlepticGregorian(2000, 1, 1, 6)], dims=["sample"]
    )
    ic.to_netcdf(ic_path)

    config = _get_mock_config(str(tmp_path))
    config.checkpoint_path = str(stepper_path)
    config.initial_condition = InitialConditionConfig(path=str(ic_path))
    return config


def test_run_segmented_inference(tmp_path):
    """The loop runs missing segments and skips those whose restart already
    exists, without re-running completed segments."""
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    config = _mock_config_with_real_checkpoint_and_ic(tmp_path)

    # n_forward_steps=3, TIMESTEP=6h: segment start times step by 18h.
    segment_labels = [
        "20000101T06",
        "20000102T00",
        "20000102T18",
    ]

    with unittest.mock.patch(
        "fme.ace.inference.inference.run_inference_from_config", new=mock
    ):
        run_segmented_inference(config, 1)
        segment_dir = os.path.join(config.experiment_dir, segment_labels[0])
        assert os.path.exists(os.path.join(segment_dir, "restart.nc"))
        assert mock.call_count == 1

        # rerunning the same segment does not call run_inference_from_config again
        run_segmented_inference(config, 1)
        assert mock.call_count == 1

        # extending to three segments runs exactly the two missing segments
        run_segmented_inference(config, 3)
        for label in segment_labels:
            segment_dir = os.path.join(config.experiment_dir, label)
            assert os.path.exists(os.path.join(segment_dir, "restart.nc"))
        assert mock.call_count == 3


def test_run_segmented_inference_custom_segment_label_format(tmp_path):
    mock = unittest.mock.MagicMock(side_effect=_run_inference_from_config_mock)
    config = _mock_config_with_real_checkpoint_and_ic(
        tmp_path, timestep=datetime.timedelta(minutes=15)
    )

    # n_forward_steps=3, timestep=15min: segments start 45min apart, which the
    # default "%Y%m%dT%H" format would fold onto the same "T06" label.
    with unittest.mock.patch(
        "fme.ace.inference.inference.run_inference_from_config", new=mock
    ):
        run_segmented_inference(config, 2, segment_label_format="%Y%m%dT%H%M")
        for label in ["20000101T0600", "20000101T0645"]:
            segment_dir = os.path.join(config.experiment_dir, label)
            assert os.path.exists(os.path.join(segment_dir, "restart.nc"))
        assert mock.call_count == 2


def save_noise_conditioned_stepper(
    path: pathlib.Path,
    in_names: list[str],
    out_names: list[str],
    data_shape: list[int],
    timestep: datetime.timedelta = TIMESTEP,
):
    """Save a minimal NoiseConditionedSFNO stepper whose noise actually affects the
    output, so a stochastic rollout is only reproducible across a restart if the
    random state is threaded through the restart stepper state file."""
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
    of the restart stepper state file. With a deterministic stepper this would pass
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
        two_seg_dir / "20000102T00" / "autoregressive_predictions.nc",
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
        two_seg_seed1_dir / "20000102T00" / "autoregressive_predictions.nc",
        decode_timedelta=False,
    ).drop_vars(["init_time", "time"])
    assert not np.allclose(
        ds_segment_1["prog"].values, ds_segment_1_seed1["prog"].values
    )


def _paired_writer(tmp_path: pathlib.Path) -> PairedDataWriter:
    """A minimal PairedDataWriter for exercising the full-state restart netCDF."""
    return PairedDataWriter(
        writers=[],
        path=str(tmp_path),
        variable_metadata={},
        coords={},
        dataset_metadata=DatasetMetadata(),
    )


def _prognostic_state(
    n_samples: int = 1,
    stepper_state: StepperState | None = None,
    labels: BatchLabels | None = None,
    data_mask: dict[str, torch.Tensor] | None = None,
    lat: int = 4,
    lon: int = 8,
) -> PrognosticState:
    time = xr.DataArray(
        [[cftime.DatetimeProlepticGregorian(2000, 1, 1)]] * n_samples,
        dims=["sample", "time"],
    )
    batch = BatchData.new_on_cpu(
        data={"prog": torch.zeros(n_samples, 1, lat, lon)},
        time=time,
        stepper_state=stepper_state,
        labels=labels,
        data_mask=data_mask,
    )
    return PrognosticState(batch)


def _write_restart(tmp_path: pathlib.Path, state: PrognosticState) -> pathlib.Path:
    _paired_writer(tmp_path).write(state, "restart.nc")
    return tmp_path / "restart.nc"


def test_plain_restart_netcdf_is_backcompat(tmp_path):
    """A deterministic run's restart embeds no reserved variables (byte-clean),
    and reading a plain netCDF (no schema marker) restores stepper_state=None with
    labels from config - legacy restart resume unchanged."""
    restart = _write_restart(tmp_path, _prognostic_state(n_samples=2))

    ds = xr.open_dataset(restart, decode_timedelta=False)
    assert not BatchData.dataset_has_embedded_state(ds)
    assert not any(str(v).startswith(_RESERVED_PREFIX) for v in ds.variables)
    # The prognostic variable is still a normal, readable xarray variable.
    assert "prog" in ds and ds["prog"].shape == (2, 4, 8)

    ic = get_initial_condition(
        InitialConditionConfig(path=str(restart)).get_dataset(),
        InitialConditionRequirements(prognostic_names=["prog"], labels=["from_config"]),
    )
    restored = ic.as_batch_data()
    assert restored.stepper_state is None
    assert restored.labels is not None and restored.labels.names == ["from_config"]


def test_full_state_restart_roundtrip(tmp_path):
    """Writing a PrognosticState carrying stepper_state (+labels +data_mask) to a
    restart netCDF and reading it back via InitialConditionConfig restores every
    field: the corrector tensor exactly, the generator continues its stream, and
    labels/data_mask round-trip. The embedded labels win over the config labels."""
    n = 2
    mass = torch.randn(n, 1, 1)
    stored_random = RandomState.from_seed(11)
    torch.randn(3, generator=stored_random.generator)  # advance past the raw seed
    stepper_state = StepperState(
        corrector_state=CorrectorState(global_dry_air_mass=mass.clone()),
        random_state=stored_random,
    )
    labels = BatchLabels(torch.ones(n, 2), names=["a", "b"])
    data_mask = {"prog": torch.tensor([True, False])}
    restart = _write_restart(
        tmp_path,
        _prognostic_state(
            n_samples=n,
            stepper_state=stepper_state,
            labels=labels,
            data_mask=data_mask,
        ),
    )

    # Inspectability: the full-state restart still opens as a normal Dataset.
    ds = xr.open_dataset(restart, decode_timedelta=False)
    assert BatchData.dataset_has_embedded_state(ds)
    assert ds["prog"].shape == (n, 4, 8)

    ic = get_initial_condition(
        InitialConditionConfig(path=str(restart)).get_dataset(),
        # labels match the embedded labels (validated, not re-set)
        InitialConditionRequirements(prognostic_names=["prog"], labels=["a", "b"]),
    )
    restored = ic.as_batch_data()

    ss = restored.stepper_state
    assert ss is not None
    assert ss.corrector_state is not None
    assert ss.corrector_state.global_dry_air_mass is not None
    # get_initial_condition returns the state on CPU; InferenceGriddedData
    # moves it to the compute device.
    torch.testing.assert_close(
        ss.corrector_state.global_dry_air_mass, mass, rtol=0, atol=0
    )
    assert ss.random_state is not None
    # get_state() is non-consuming, so stored_random still sits at the point it
    # was serialized; the restored generator must continue the same stream.
    torch.testing.assert_close(
        torch.randn(4, generator=stored_random.generator),
        torch.randn(4, generator=ss.random_state.generator),
        rtol=0,
        atol=0,
    )
    assert restored.labels is not None
    assert restored.labels.names == ["a", "b"]  # embedded labels preserved
    torch.testing.assert_close(restored.labels.tensor, torch.ones(n, 2))
    assert restored.data_mask is not None
    torch.testing.assert_close(restored.data_mask["prog"], torch.tensor([True, False]))


def test_full_state_restart_start_indices(tmp_path):
    """start_indices subselection on a full-state restart subsets the per-sample
    corrector state in step with the prognostic variables while leaving the shared
    generator state untouched."""
    n = 3
    mass = torch.arange(n, dtype=torch.float32).reshape(n, 1, 1)
    stored_random = RandomState.from_seed(0)
    stepper_state = StepperState(
        corrector_state=CorrectorState(global_dry_air_mass=mass.clone()),
        random_state=RandomState.from_seed(0),
    )
    restart = _write_restart(
        tmp_path, _prognostic_state(n_samples=n, stepper_state=stepper_state)
    )

    config = InitialConditionConfig(
        path=str(restart), start_indices=ExplicitIndices([1])
    )
    ic = get_initial_condition(
        config.get_dataset(), InitialConditionRequirements(prognostic_names=["prog"])
    )
    restored = ic.as_batch_data()

    ss = restored.stepper_state
    assert ss is not None and ss.corrector_state is not None
    # Only sample index 1 was selected, so the corrector state is mass[[1]].
    assert ss.corrector_state.global_dry_air_mass is not None
    torch.testing.assert_close(
        ss.corrector_state.global_dry_air_mass, mass[[1]], rtol=0, atol=0
    )
    # The generator (no sample dim) is untouched by subselection.
    assert ss.random_state is not None
    torch.testing.assert_close(
        torch.randn(4, generator=stored_random.generator),
        torch.randn(4, generator=ss.random_state.generator),
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
