import pytest

from fme.ace.inference.data_writer.benchmark import BenchmarkConfig, benchmark
from fme.ace.inference.data_writer.file_writer import FileWriterConfig
from fme.ace.inference.data_writer.main import DataWriterConfig
from fme.ace.inference.data_writer.raw import NetCDFWriterConfig
from fme.ace.inference.data_writer.zarr import ZarrWriterConfig
from fme.ace.testing import DimSize, DimSizes, FV3GFSData
from fme.core import logging_utils
from fme.core.testing.wandb import mock_wandb

NAMES = ["foo", "bar"]
N_FORWARD_STEPS = 6
FORWARD_STEPS_IN_MEMORY = 3


def get_config(tmp_path, data_writer: DataWriterConfig) -> BenchmarkConfig:
    data = FV3GFSData(
        path=tmp_path,
        names=NAMES,
        dim_sizes=DimSizes(
            n_time=N_FORWARD_STEPS + 1,
            horizontal=[DimSize("grid_yt", 4), DimSize("grid_xt", 8)],
            nz_interface=2,
        ),
        timestep_days=0.25,
    )
    return BenchmarkConfig(
        experiment_dir=str(tmp_path / "output"),
        loader=data.inference_data_loader_config,
        data_writer=data_writer,
        logging=logging_utils.LoggingConfig(project="test", entity="test"),
        names=NAMES,
        n_forward_steps=N_FORWARD_STEPS,
        forward_steps_in_memory=FORWARD_STEPS_IN_MEMORY,
    )


def get_file_writer_config(format_, save_reference: bool = False) -> DataWriterConfig:
    return DataWriterConfig(
        save_prediction_files=False,
        save_monthly_files=False,
        files=[
            FileWriterConfig(
                label="predictions", save_reference=save_reference, format=format_
            )
        ],
    )


@pytest.mark.parametrize(
    "format_",
    [
        pytest.param(NetCDFWriterConfig(), id="netcdf"),
        pytest.param(ZarrWriterConfig(), id="zarr"),
    ],
)
def test_benchmark_writes_and_logs_throughput(tmp_path, format_):
    """The benchmark runs end to end and reports its write throughput."""
    config = get_config(tmp_path, data_writer=get_file_writer_config(format_))
    with mock_wandb() as wandb:
        benchmark(config)
    logs = wandb.get_logs()
    summary = logs[-1]
    n_windows = N_FORWARD_STEPS // FORWARD_STEPS_IN_MEMORY
    assert summary["total_mb_written"] > 0.0
    assert summary["write_mb_per_s"] > 0.0
    assert summary["storage_write_mb_per_s"] > 0.0
    assert summary["data_writer"] > 0.0
    assert summary["storage_write"] <= summary["data_writer"]
    assert len(logs) == n_windows
    for step_logs in logs:
        assert step_logs["seconds_per_window"] >= 0.0


def test_benchmark_rejects_reference_writers(tmp_path):
    with pytest.raises(ValueError, match="save_reference"):
        get_config(
            tmp_path,
            data_writer=get_file_writer_config(ZarrWriterConfig(), save_reference=True),
        )


def test_benchmark_rejects_step_diagnostics(tmp_path):
    """The stepper reports the diagnostics, and the benchmark runs no stepper."""
    with pytest.raises(ValueError, match="save_step_diagnostics"):
        get_config(
            tmp_path,
            data_writer=DataWriterConfig(
                save_prediction_files=False,
                save_monthly_files=False,
                save_step_diagnostics=True,
            ),
        )


def test_benchmark_rejects_legacy_netcdf_writers(tmp_path):
    """The legacy netCDF writers write reference data the benchmark cannot supply."""
    with pytest.raises(ValueError, match="save_prediction_files"):
        get_config(tmp_path, data_writer=DataWriterConfig())


def test_benchmark_errors_when_nothing_is_written(tmp_path):
    """A writer stack with no data to write reports no storage time to divide by."""
    config = get_config(
        tmp_path,
        data_writer=DataWriterConfig(
            save_prediction_files=False, save_monthly_files=False
        ),
    )
    with mock_wandb():
        with pytest.raises(RuntimeError, match="No storage write time"):
            benchmark(config)
