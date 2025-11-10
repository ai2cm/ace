import dataclasses
import logging
import os
import time

import xarray as xr


@dataclasses.dataclass
class OutputWriterConfig:
    n_split: int = 50
    starting_split: int = 0
    n_dask_workers: int | None = None
    max_retries_per_partition: int = 10

    def __post_init__(self):
        self._client = None

    def start_dask_client(self, debug: bool = False):
        if debug or self.n_dask_workers is None:
            return None
        logging.info(f"Using dask Client(n_workers={self.n_dask_workers})...")

        # Import dask-related things here to enable testing in environments
        # without dask.
        import dask

        dask.config.set(
            {"logging.distributed": "error"}
        )  # before Client improt to avoid a ton of dask INFO

        from dask.distributed import Client

        client = Client(n_workers=self.n_dask_workers)
        logging.info(client.dashboard_link)
        self._client = client

    def close_dask_client(self):
        if self._client is not None:
            logging.info("Closing dask client...")
            self._client.close()
            self._client = None

    def write(self, ds: xr.Dataset, output_store: str):
        import xpartition  # noqa: F401

        ds = ds.chunk(OUTER_CHUNKS)
        if self.starting_split == 0:
            if os.path.isdir(output_store):
                raise ValueError(
                    f"Output store {output_store} already exists. "
                    "Use starting_split > 0 to continue writing or "
                    "manually delete the directory to start from 0."
                )
            ds.partition.initialize_store(output_store, inner_chunks=INNER_CHUNKS)
        else:
            if not os.path.isdir(output_store):
                raise ValueError(
                    f"starting_split > 0 but output store {output_store} "
                    "hasn't yet been initialized. Use starting_split = 0?"
                )

        logging.info(f"Writing to output store: {output_store}")

        for i in range(self.starting_split, self.n_split):
            n_retries = 0
            delay = 1.0
            segment_time = time.time()
            while True:
                try:
                    logging.info(f"Writing segment {i + 1} / {self.n_split}")
                    ds.partition.write(
                        output_store,
                        self.n_split,
                        ["time"],
                        i,
                        collect_variable_writes=True,
                    )
                    segment_time = time.time() - segment_time
                    logging.info(
                        f"Appended segment {i + 1} in {segment_time:0.2f} seconds"
                    )
                    break
                except RuntimeError:
                    if n_retries > self.max_retries_per_partition:
                        raise
                    logging.info("RuntimeError, retrying...")
                    time.sleep(delay)
                    n_retries += 1
                    delay *= 1.1

        logging.info(f"Completed writing to output store: {output_store}")


INNER_CHUNKS = {"time": 1, "lon": -1, "lat": -1}
OUTER_CHUNKS = {"time": 360, "lon": -1, "lat": -1}
