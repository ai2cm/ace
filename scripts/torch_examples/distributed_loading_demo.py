"""
This script demonstrates that forkserver data workers inherit their rank
from the parent process.
"""

import torch

from fme.core.distributed import Distributed

_CONTEXT = None


class Dataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int, nlat: int, nlon: int):
        super().__init__()
        self.n_samples = n_samples
        self.nlat = nlat
        self.nlon = nlon

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        global _CONTEXT
        if _CONTEXT is None:
            _CONTEXT = Distributed.context()
            _CONTEXT.__enter__()
        dist = Distributed.get_instance()
        return dist.rank


if __name__ == "__main__":
    with Distributed.context():
        dataset = Dataset(n_samples=10, nlat=64, nlon=128)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=2,
            num_workers=2,
            multiprocessing_context="forkserver",
            persistent_workers=True,
        )
        dist = Distributed.get_instance()
        print("Loader created ---- rank", dist.rank)
        for batch in loader:
            print(dist.rank, batch)
        print(torch.distributed.get_world_size(), torch.distributed.get_rank())
        torch.testing.assert_close(batch, torch.full_like(batch, fill_value=dist.rank))
