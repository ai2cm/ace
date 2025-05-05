"""This module provides an entrypoint for benchmarking data loading performance.
It uses a time.sleep() to mock ML training but allows loading real datasets.
"""

import argparse
import dataclasses
import logging
import time

import dacite
import yaml

from fme.core.timing import GlobalTimer

from ..requirements import DataRequirements
from .config import DataLoaderConfig
from .getters import get_data_loader


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for benchmarking data loading."""

    loader: DataLoaderConfig
    names: list[str]
    n_timesteps: int
    train: bool = True
    sleep: float = 0.1

    def build(self):
        return get_data_loader(
            self.loader,
            train=self.train,
            requirements=DataRequirements(self.names, self.n_timesteps),
        )


def benchmark(config: BenchmarkConfig):
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
        logging.info(f"Starting loop to load {len(loader)} batches.")
        timer.start("data_loading")
        for i, batch in enumerate(loader):
            timer.stop()
            if i % 10 == 0:
                logging.info(f"Loaded batch {i}")
            with timer.context("sleeping"):
                time.sleep(config.sleep)
            timer.start("data_loading")
        timer.stop()
        logging.info(f"Finished loading {len(loader)} batches.")
        logging.info("Timer results:")
        logging.info(timer.get_durations())


def main():
    parser = argparse.ArgumentParser(description="Benchmark ACE data loading.")
    parser.add_argument("config", help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = dacite.from_dict(BenchmarkConfig, yaml.safe_load(f))

    logging.basicConfig(level=logging.INFO)
    benchmark(config)


if __name__ == "__main__":
    main()
