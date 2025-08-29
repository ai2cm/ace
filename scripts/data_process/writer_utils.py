import dataclasses
import os

import xarray as xr
from dask.distributed import Client


@dataclasses.dataclass
class OutputWriterConfig:
    n_split: int = 50
    starting_split: int = 0
    n_dask_workers: int | None = None
    _client: Client | None = None

    def start_dask_client(self, debug: bool = False):
        if debug or self.n_dask_workers is None:
            return None
        print(f"Using dask Client(n_workers={self.n_dask_workers})...")
        client = Client(n_workers=self.n_dask_workers)
        print(client.dashboard_link)
        self._client = client

    def close_dask_client(self):
        if self._client is not None:
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
        for i in range(self.starting_split, self.n_split):
            print(f"Writing segment {i + 1} / {self.n_split}")
            ds.partition.write(
                output_store,
                self.n_split,
                ["time"],
                i,
                collect_variable_writes=True,
            )


INNER_CHUNKS = {"time": 1, "lon": -1, "lat": -1}
OUTER_CHUNKS = {"time": 360, "lon": -1, "lat": -1}
