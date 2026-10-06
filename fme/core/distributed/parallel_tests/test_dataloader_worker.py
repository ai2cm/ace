"""
Tests that the DataLoader worker code path reports the same sharding metadata
as a fully initialized backend.

A worker skips the process group, the CUDA device and (under spatial
parallelism) the DeviceMesh, so it derives rank and data-parallel metadata from
the launcher's environment variables and mesh arithmetic instead. Inference
dataloaders use that metadata to decide which samples belong to their rank, so
a mismatch with the real backend would silently drop or duplicate samples.

These tests need a launcher to be meaningful and skip without one.
"""

import os

import pytest
import torch.utils.data

from fme.core.device import using_srun
from fme.core.distributed import Distributed
from fme.core.distributed.model_torch_distributed import ModelTorchDistributed
from fme.core.distributed.torch_distributed import TorchDistributed


def _build_worker_backend(monkeypatch):
    """Build the configured backend as a DataLoader worker would see it."""
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: object())
    if os.environ.get("FME_DISTRIBUTED_BACKEND") == "model":
        return ModelTorchDistributed(
            h_size=int(os.environ["FME_DISTRIBUTED_H"]),
            w_size=int(os.environ["FME_DISTRIBUTED_W"]),
        )
    return TorchDistributed()


@pytest.mark.parallel
def test_worker_sharding_metadata_matches_backend(monkeypatch):
    if "RANK" not in os.environ and not using_srun():
        pytest.skip("requires torchrun or srun")
    if os.environ.get("FME_DISTRIBUTED_BACKEND") == "none":
        pytest.skip("requires a distributed backend")
    dist = Distributed.get_instance()

    worker = _build_worker_backend(monkeypatch)

    assert worker.rank == dist.rank
    assert worker.total_ranks == dist.world_size
    assert worker.data_parallel_rank == dist.data_parallel_rank
    assert worker.total_data_parallel_ranks == dist.total_data_parallel_ranks
