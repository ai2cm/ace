"""This module provides an entrypoint for benchmarking data loading performance.
It uses a time.sleep() to mock ML training but allows loading real datasets.
"""

import argparse
import dataclasses
import logging
import os
import shutil
import time
import uuid

import dacite
import yaml

from fme.core import logging_utils
from fme.core.dicts import to_flat_dict
from fme.core.timing import GlobalTimer
from fme.core.wandb import WandB

from ..requirements import DataRequirements
from .config import DataLoaderConfig
from .getters import get_gridded_data

TMPDIR = f"/tmp/ace_benchmark_{uuid.uuid4()}"


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for benchmarking data loading."""

    loader: DataLoaderConfig
    logging: logging_utils.LoggingConfig
    names: list[str]
    n_timesteps: int
    train: bool = True
    sleep: float = 0.1

    def build(self):
        return get_gridded_data(
            self.loader,
            train=self.train,
            requirements=DataRequirements(self.names, self.n_timesteps),
        )

    def configure_wandb(self, env_vars: dict | None = None, **kwargs):
        config = to_flat_dict(dataclasses.asdict(self))
        # our wandb class requires "experiment_dir" to be in config
        config["experiment_dir"] = TMPDIR
        os.makedirs(TMPDIR)
        self.logging.configure_wandb(config=config, env_vars=env_vars, **kwargs)

    def configure_logging(self):
        self.logging.configure_logging("/tmp", "log.txt")


def benchmark(config: BenchmarkConfig):
    config.configure_logging()
    env_vars = logging_utils.retrieve_env_vars()
    beaker_url = logging_utils.log_beaker_url()
    config.configure_wandb(env_vars=env_vars, notes=beaker_url)
    wandb = WandB.get_instance()

    with GlobalTimer():
        logging.info("Initializing data loader.")
        timer = GlobalTimer.get_instance()
        with timer.context("initialization"):
            data = config.build()
            loader = data.loader

        logging.info("Getting data size in bytes from first batch.")
        example_batch = next(iter(loader))
        bytes_per_batch = sum(
            [x.numel() * x.element_size() for x in example_batch.data.values()]
        )
        logging.info(f"Each batch will be {bytes_per_batch / 1e6:.3f} MB.")
        if config.sleep > 0:
            # assuming no time wasted data loading, estimate max throughput
            max_throughput = bytes_per_batch / config.sleep
            logging.info(
                "Max throughput possible given batch size and configured sleep: "
                f"{max_throughput / 1e6:.2f} MB/s"
            )
        logging.info(f"Starting loop to load {len(loader)} batches.")
        timer.start("data_loading")
        seconds_per_batch = time.time()
        for i, batch in enumerate(loader):
            timer.stop()
            if i % 10 == 0:
                logging.info(f"Loaded batch {i}")
            with timer.context("sleeping"):
                time.sleep(config.sleep)
            seconds_per_batch = time.time() - seconds_per_batch
            wandb.log({"seconds_per_batch": seconds_per_batch}, step=i)
            seconds_per_batch = time.time()
            timer.start("data_loading")
        timer.stop()
        logging.info(f"Finished loading {len(loader)} batches.")
        total_time = timer.get_duration("data_loading") + timer.get_duration("sleeping")
        actual_throughput = (bytes_per_batch * len(loader)) / total_time
        logging.info(f"Actual throughput achieved: {actual_throughput / 1e6:.2f} MB/s")
        logging.info("Timer results:")
        logging.info(timer.get_durations())
        wandb_logs = timer.get_durations() | {
            "total_time": total_time,
            "throughput_mb_per_s": actual_throughput / 1e6,
            "mb_per_batch": bytes_per_batch / 1e6,
        }
        if config.sleep > 0:
            ratio = actual_throughput / max_throughput
            wandb_logs["ratio_of_actual_to_possible_throughput"] = ratio
            wandb_logs["max_possible_throughput_mb_per_s"] = max_throughput / 1e6
        wandb.log(wandb_logs, step=i)
        shutil.rmtree(TMPDIR)


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACE data loading.")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = dacite.from_dict(
            BenchmarkConfig, yaml.safe_load(f), config=dacite.Config(strict=True)
        )

    benchmark(config)


if __name__ == "__main__":
    main()
