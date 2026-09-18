"""Subprocess integration tests for multi-GPU inference.

Each test launches ``torchrun --nproc-per-node 2`` on CPU (FME_FORCE_CPU=1,
gloo backend) with a tiny plus-one stepper and synthetic data, then compares
the distributed output against a serial in-process reference run.  Marked
``slow`` so they stay out of the very-fast CI job.

These tests do NOT use ``@pytest.mark.parallel`` because they are not designed
to be run under ``torchrun -m pytest`` (they spawn their own child processes).
"""

import dataclasses
import datetime
import os
import pathlib
import subprocess
import sys

import cftime
import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from fme.ace.inference.evaluator import (
    InferenceEvaluatorConfig,
    run_evaluator_from_config,
)
from fme.ace.data_loading.inference import (
    ForcingDataLoaderConfig,
    InferenceInitialConditionIndices,
)
from fme.ace.inference.data_writer import DataWriterConfig
from fme.ace.inference.test_evaluator import save_plus_one_stepper
from fme.ace.inference.test_inference import save_stepper
from fme.ace.testing import DimSize, DimSizes, FV3GFSData
from fme.core.dataset.xarray import XarrayDataConfig
from fme.core.logging_utils import LoggingConfig
from fme.core.testing import mock_wandb

TIMESTEP = datetime.timedelta(hours=6)
N_LAT = 4
N_LON = 8
NZ_INTERFACE = 2
N_IC = 2  # must be divisible by 2 (our nproc)


def _no_writers_config() -> DataWriterConfig:
    """DataWriterConfig with all per-timestep writers disabled."""
    return DataWriterConfig(
        save_prediction_files=False,
        save_monthly_files=False,
        save_step_diagnostics=False,
        files=None,
    )


def _make_evaluator_test_data(
    tmp_path: pathlib.Path,
    n_forward_steps: int = 2,
):
    """Create synthetic data and a plus-one stepper for the evaluator path."""
    in_names = ["var"]
    out_names = ["var"]
    stepper_path = tmp_path / "stepper"
    horizontal = [DimSize("lat", N_LAT), DimSize("lon", N_LON)]
    dim_sizes = DimSizes(
        # Need enough time steps for the last IC (at index N_IC - 1)
        # to run n_forward_steps: total = (N_IC - 1) + n_forward_steps + 1.
        n_time=(N_IC - 1) + n_forward_steps + 1,
        horizontal=horizontal,
        nz_interface=NZ_INTERFACE,
    )
    data = FV3GFSData(
        path=tmp_path,
        names=in_names,
        dim_sizes=dim_sizes,
        timestep_days=TIMESTEP.total_seconds() / 86400,
        save_vertical_coordinate=False,
    )
    save_plus_one_stepper(
        stepper_path,
        in_names,
        out_names,
        mean=0.0,
        std=1.0,
        data_shape=dim_sizes.shape_nd,
    )
    return stepper_path, data


def _make_standalone_inference_inputs(
    tmp_path: pathlib.Path,
    n_forward_steps: int = 2,
):
    """Create forcing data, IC file, and stepper for the standalone inference path.

    The standalone path needs separate forcing variables (the forcing loader)
    and prognostic variables (the IC file), with an ocean config so the stepper
    has a surface_temperature_name.
    """
    from fme.core.ocean import OceanConfig

    in_names = ["prog", "sst", "forcing_var", "DSWRFtoa"]
    out_names = ["prog", "sst", "ULWRFtoa", "USWRFtoa"]
    forcing_names = ["forcing_var", "DSWRFtoa", "sst", "ocean_fraction"]
    prognostic_names = ["prog", "sst", "DSWRFtoa"]  # go in the IC file

    horizontal = [DimSize("lat", N_LAT), DimSize("lon", N_LON)]
    nz_interface = NZ_INTERFACE
    n_time = (N_IC - 1) + n_forward_steps + 1
    dim_sizes = DimSizes(
        n_time=n_time, horizontal=horizontal, nz_interface=nz_interface
    )
    data = FV3GFSData(
        path=tmp_path,
        names=forcing_names,
        dim_sizes=dim_sizes,
        timestep_days=TIMESTEP.total_seconds() / 86400,
        save_vertical_coordinate=False,
    )
    stepper_path = tmp_path / "stepper"
    save_stepper(
        stepper_path,
        in_names=in_names,
        out_names=out_names,
        mean=0.0,
        std=1.0,
        horizontal_coords=data.horizontal_coords,
        nz_interface=nz_interface,
    )

    # IC file with N_IC initial conditions.
    ic_times = xr.open_dataset(data.data_filename).time.values[:N_IC]
    ic_dict = {
        name: xr.DataArray(
            np.random.rand(N_IC, N_LAT, N_LON).astype(np.float32),
            dims=["sample", "lat", "lon"],
        )
        for name in prognostic_names
    }
    ic_dict["time"] = xr.DataArray(ic_times, dims=["time"])
    ic_path = tmp_path / "ic.nc"
    xr.Dataset(ic_dict).to_netcdf(ic_path)

    return stepper_path, data, ic_path


def _run_torchrun(config_yaml: str, module: str, extra_args: list[str] | None = None):
    """Launch ``torchrun --nproc-per-node 2`` on CPU and return the result."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc-per-node",
        "2",
        "-m",
        module,
        config_yaml,
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = {**os.environ, "FME_FORCE_CPU": "1", "WANDB_MODE": "disabled"}
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"torchrun failed (exit {result.returncode}):\n"
            f"STDOUT:\n{result.stdout[-2000:]}\n"
            f"STDERR:\n{result.stderr[-2000:]}"
        )
    return result


@pytest.mark.slow
def test_distributed_evaluator(tmp_path: pathlib.Path):
    """Evaluator under 2 GPU ranks matches a serial in-process reference."""
    n_forward_steps = 2
    stepper_path, data = _make_evaluator_test_data(tmp_path, n_forward_steps=n_forward_steps)

    # Build config with 2 ICs and all writers disabled.
    loader_config = dataclasses.replace(
        data.inference_data_loader_config,
        start_indices=InferenceInitialConditionIndices(
            n_initial_conditions=N_IC,
            first=0,
            interval=1,
        ),
    )
    serial_dir = str(tmp_path / "serial")
    config = InferenceEvaluatorConfig(
        experiment_dir=serial_dir,
        n_forward_steps=n_forward_steps,
        checkpoint_path=str(stepper_path),
        logging=LoggingConfig(log_to_screen=False, log_to_file=False, log_to_wandb=False),
        loader=loader_config,
        forward_steps_in_memory=1,
        data_writer=_no_writers_config(),
        allow_incompatible_dataset=True,
    )

    # Serial reference run.
    with mock_wandb(), torch.no_grad():
        run_evaluator_from_config(config)

    # Distributed run with a separate experiment_dir.
    dist_dir = str(tmp_path / "distributed")
    dist_config = dataclasses.replace(config, experiment_dir=dist_dir)
    config_yaml = str(tmp_path / "dist_evaluator_config.yaml")
    with open(config_yaml, "w") as f:
        yaml.dump(dataclasses.asdict(dist_config), f)

    _run_torchrun(config_yaml, "fme.ace.evaluator")

    # Compare reduced diagnostics.
    for name in ("time_mean", "reduced", "zonal_mean"):
        serial_nc = os.path.join(serial_dir, f"{name}_diagnostics.nc")
        dist_nc = os.path.join(dist_dir, f"{name}_diagnostics.nc")
        if os.path.exists(serial_nc):
            serial_ds = xr.open_dataset(serial_nc)
            assert os.path.exists(dist_nc), f"Missing {dist_nc}"
            dist_ds = xr.open_dataset(dist_nc)
            xr.testing.assert_allclose(serial_ds, dist_ds)

    # Compare restart.nc.
    serial_restart = xr.open_dataset(os.path.join(serial_dir, "restart.nc"))
    dist_restart = xr.open_dataset(os.path.join(dist_dir, "restart.nc"))
    xr.testing.assert_allclose(serial_restart, dist_restart)


@pytest.mark.slow
def test_distributed_standalone_inference(tmp_path: pathlib.Path):
    """Standalone inference under 2 ranks matches serial (restart + IC files)."""
    n_forward_steps = 2
    stepper_path, data, ic_path = _make_standalone_inference_inputs(
        tmp_path, n_forward_steps=n_forward_steps
    )

    config_dict = {
        "experiment_dir": str(tmp_path / "serial"),
        "n_forward_steps": n_forward_steps,
        "checkpoint_path": str(stepper_path),
        "logging": {"log_to_screen": False, "log_to_file": False, "log_to_wandb": False},
        "initial_condition": {"path": str(ic_path)},
        "forcing_loader": {"dataset": {"data_path": str(data.data_path)}},
        "forward_steps_in_memory": 1,
        "data_writer": dataclasses.asdict(_no_writers_config()),
    }

    # Serial reference.
    serial_yaml = str(tmp_path / "serial_config.yaml")
    with open(serial_yaml, "w") as f:
        yaml.dump(config_dict, f)

    with mock_wandb(), torch.no_grad():
        from fme.ace.inference.inference import main as inference_main

        inference_main(serial_yaml)

    # Distributed run.
    serial_dir = config_dict["experiment_dir"]
    dist_dir = str(tmp_path / "distributed")
    config_dict["experiment_dir"] = dist_dir
    dist_yaml = str(tmp_path / "dist_config.yaml")
    with open(dist_yaml, "w") as f:
        yaml.dump(config_dict, f)

    _run_torchrun(dist_yaml, "fme.ace.inference")

    # Compare restart.nc.
    serial_restart = xr.open_dataset(os.path.join(serial_dir, "restart.nc"))
    dist_restart = xr.open_dataset(os.path.join(dist_dir, "restart.nc"))
    xr.testing.assert_allclose(serial_restart, dist_restart)

    # Compare initial_condition.nc.
    serial_ic = xr.open_dataset(os.path.join(serial_dir, "initial_condition.nc"))
    dist_ic = xr.open_dataset(os.path.join(dist_dir, "initial_condition.nc"))
    xr.testing.assert_allclose(serial_ic, dist_ic)
