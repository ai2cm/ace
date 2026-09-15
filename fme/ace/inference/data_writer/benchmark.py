"""This module provides an entrypoint for benchmarking data writing performance.
It writes real data loaded from a configured dataset, timing the writers
separately from the loader that feeds them.
"""

import argparse
import dataclasses
import logging
import os
import shutil
import time
import uuid

import dacite
import torch
import yaml

from fme.ace.data_loading.batch_data import PairedData
from fme.ace.data_loading.getters import get_inference_data
from fme.ace.data_loading.gridded_data import InferenceGriddedData
from fme.ace.data_loading.inference import InferenceDataLoaderConfig
from fme.ace.inference.data_writer.dataset_metadata import DatasetMetadata
from fme.ace.inference.data_writer.main import DataWriterConfig, PairedDataWriter
from fme.ace.requirements import DataRequirements, PrognosticStateDataRequirements
from fme.core import logging_utils
from fme.core.cloud import makedirs
from fme.core.dicts import to_flat_dict
from fme.core.distributed.distributed import Distributed
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

TMPDIR = f"/tmp/ace_write_benchmark_{uuid.uuid4()}"


@dataclasses.dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking data writing.

    Parameters:
        experiment_dir: Directory to write output to. May be local or a remote
            path recognized by fsspec, such as ``gs://bucket/results``.
        loader: Parameters for the inference data loader supplying the windows
            that are written. The inference loader is required because its
            windows tile the time axis, whereas training loader windows start at
            consecutive indices and overlap, leaving no coherent time slot for
            the writers to fill.
        data_writer: Configuration for the data writers under test.
        logging: Configuration for logging.
        names: Names of the variables to load and write.
        n_forward_steps: Total number of timesteps to write.
        forward_steps_in_memory: Number of timesteps written per window.
    """

    experiment_dir: str
    loader: InferenceDataLoaderConfig
    data_writer: DataWriterConfig
    logging: logging_utils.LoggingConfig
    names: list[str]
    n_forward_steps: int
    forward_steps_in_memory: int

    def __post_init__(self):
        self.data_writer.validate_time_coarsen(
            self.forward_steps_in_memory, self.n_forward_steps
        )
        self._validate_writers()

    def _validate_writers(self):
        if (
            self.data_writer.save_prediction_files
            or self.data_writer.save_monthly_files
        ):
            raise ValueError(
                "save_prediction_files and save_monthly_files must be False. The "
                "benchmark provides no reference data, so writers of it would "
                "produce empty output. Configure writers through 'files' instead."
            )
        if self.data_writer.save_step_diagnostics:
            raise ValueError(
                "save_step_diagnostics must be False. The benchmark runs no "
                "stepper, so the diagnostics writer would produce empty output."
            )
        for file_config in self.data_writer.files or []:
            if file_config.save_reference:
                raise ValueError(
                    f"File writer {file_config.label!r} sets save_reference, but the "
                    "benchmark provides no reference data. Set it to False."
                )

    def build_data(self) -> InferenceGriddedData:
        return get_inference_data(
            config=self.loader,
            total_forward_steps=self.n_forward_steps,
            window_requirements=DataRequirements(
                names=self.names, n_timesteps=self.forward_steps_in_memory + 1
            ),
            initial_condition=PrognosticStateDataRequirements(
                names=self.names, n_timesteps=1
            ),
        )

    def build_writer(self, data: InferenceGriddedData) -> PairedDataWriter:
        return self.data_writer.build_paired(
            experiment_dir=self.experiment_dir,
            initial_condition_times=data.initial_time.to_numpy(),
            n_timesteps=self.n_forward_steps,
            timestep=data.timestep,
            variable_metadata=data.variable_metadata,
            coords=data.coords,
            dataset_metadata=DatasetMetadata.from_env(),
        )

    def configure_logging(self):
        config = to_flat_dict(dataclasses.asdict(self))
        os.makedirs(TMPDIR, exist_ok=True)
        self.logging.configure_logging(
            TMPDIR, "log.txt", config=config, resumable=False
        )


def _payload_bytes(data: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in data.values())


def benchmark(config: BenchmarkConfig):
    config.configure_logging()
    wandb = WandB.get_instance()

    with GlobalTimer():
        timer = GlobalTimer.get_instance()
        logging.info("Initializing data loader and writers.")
        with timer.context("initialization"):
            makedirs(config.experiment_dir, exist_ok=True)
            data = config.build_data()
            loader = data.loader
            writer = config.build_writer(data)

        n_windows = len(loader)
        total_bytes = 0
        bytes_per_window = 0

        logging.info(f"Starting loop to write {n_windows} windows.")
        timer.start("data_loading")
        window_start = time.time()
        for i, batch in enumerate(loader):
            timer.stop("data_loading")
            window = batch.remove_initial_condition(1)
            if i == 0:
                bytes_per_window = _payload_bytes(dict(window.data))
                logging.info(f"Each window is {bytes_per_window / 1e6:.3f} MB.")
            if i % 10 == 0:
                logging.info(f"Writing window {i}")
            with timer.context("data_writer"):
                writer.append_batch(
                    PairedData.new_on_device(
                        prediction=window.data,
                        reference={},
                        time=window.time,
                        labels=window.labels,
                        n_ensemble=window.n_ensemble,
                    )
                )
            total_bytes += _payload_bytes(dict(window.data))
            wandb.log({"seconds_per_window": time.time() - window_start}, step=i)
            window_start = time.time()
            timer.start("data_loading")
        timer.stop("data_loading")
        logging.info(f"Finished writing {n_windows} windows.")

        with timer.context("final_writer_flush"):
            writer.finalize()

        durations = timer.get_durations()
        if "storage_write" not in durations:
            raise RuntimeError(
                "No storage write time was recorded, so the write throughput "
                "cannot be separated from writer overhead. The configured writers "
                "do not time their storage calls."
            )
        total_time = durations["data_loading"] + durations["data_writer"]
        write_throughput = total_bytes / durations["data_writer"]
        logging.info(f"Write throughput achieved: {write_throughput / 1e6:.2f} MB/s")
        logging.info("Timer results:")
        timer.log_durations()
        wandb_logs = durations | {
            "total_time": total_time,
            "mb_per_window": bytes_per_window / 1e6,
            "total_mb_written": total_bytes / 1e6,
            "write_mb_per_s": write_throughput / 1e6,
            "throughput_mb_per_s": total_bytes / total_time / 1e6,
            "storage_write_mb_per_s": total_bytes / durations["storage_write"] / 1e6,
        }
        wandb.log(wandb_logs, step=n_windows - 1)
    shutil.rmtree(TMPDIR, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACE data writing.")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = dacite.from_dict(
            BenchmarkConfig, yaml.safe_load(f), config=dacite.Config(strict=True)
        )

    benchmark(config)


if __name__ == "__main__":
    with Distributed.context():
        main()
