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
from fme.ace.data_loading.inference import ForcingDataLoaderConfig, TimestampList
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.inference import (
    InitialConditionConfig,
    main,
    run_segmented_inference,
)
from fme.ace.registry import ModuleSelector
from fme.ace.stepper import SingleModuleStepperConfig
from fme.ace.testing import DimSizes, FV3GFSData
from fme.core.coordinates import (
    DimSize,
    HybridSigmaPressureCoordinate,
    LatLonCoordinates,
)
from fme.core.dataset.config import XarrayDataConfig
from fme.core.dataset_info import DatasetInfo
from fme.core.logging_utils import LoggingConfig
from fme.core.normalizer import NormalizationConfig

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
    config = SingleModuleStepperConfig(
        builder=ModuleSelector(type="prebuilt", config={"module": PlusOne()}),
        in_names=in_names,
        out_names=out_names,
        normalization=NormalizationConfig(
            means={name: mean for name in all_names},
            stds={name: std for name in all_names},
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
            [cftime.datetime(2000, 1, 1, 6), cftime.datetime(2000, 1, 1, 18)],
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
            data_writer=DataWriterConfig(save_prediction_files=True),
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
