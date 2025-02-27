import os

import pytest
import torch
import torch.multiprocessing as mp

from fme import get_device

from .distributed import Distributed, pad_tensor_at_end, unpad_tensor_at_end


@pytest.mark.parametrize(
    ["padding", "fill_value"],
    [
        pytest.param([0, 0, 0], None, id="no_padding"),
        pytest.param([1, 1, 1], 0.0, id="padding_1"),
        pytest.param([1, 1, 1], 1.0, id="padding_1_fill_one"),
    ],
)
def test_pad_tensor_at_end(padding, fill_value):
    tensor = torch.ones(2, 3, 4)
    padded_tensor = pad_tensor_at_end(tensor, padding, fill_value)
    assert padded_tensor.size() == (2 + padding[0], 3 + padding[1], 4 + padding[2])
    for dim, pad in enumerate(padding):
        if pad > 0:
            assert torch.allclose(
                padded_tensor.select(dim=dim, index=padded_tensor.size(dim) - 1),
                torch.tensor(fill_value),
            )


@pytest.mark.parametrize(
    ["padding"],
    [
        pytest.param([0, 0, 0], id="no_padding"),
        pytest.param([1, 1, 1], id="padding_1"),
    ],
)
def test_pad_unpad_rountrip(padding):
    tensor = torch.ones(2, 3, 4, device=get_device())
    padded_tensor = pad_tensor_at_end(tensor, padding)
    unpadded_tensor = unpad_tensor_at_end(padded_tensor, padding)
    assert unpadded_tensor.size() == tensor.size()
    assert torch.allclose(unpadded_tensor, tensor)


def run_gather_test(rank, worldsize):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(worldsize)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["NCCL_SHM_DISABLE"] = "1"

    dist = Distributed()
    assert dist.is_distributed()
    assert dist.rank == rank, f"Failed with Distrubuted rank {dist.rank} and arg {rank}"

    tensor = torch.ones(2, 5, device=get_device()) * dist.rank
    gathered = dist.gather(tensor)
    if dist.rank == 0:
        assert gathered is not None, "Gathered tensor are none instead of List"
        assert len(gathered) == dist.world_size
        for i in range(dist.world_size):
            assert torch.allclose(gathered[i].cpu(), torch.ones(2, 5) * i)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires multi-GPU machine")
def test_distributed_gather():
    world_size = 2
    mp.spawn(run_gather_test, args=(world_size,), nprocs=world_size, join=True)


def test_non_distributed_gather():
    dist = Distributed()
    assert not dist.is_distributed()
    tensor = torch.ones(2, 5, device=get_device()) * 5
    gathered = dist.gather(tensor)
    assert gathered is not None, "Gathered tensor are none instead of List"
    assert len(gathered) == 1
    assert torch.allclose(gathered[0], tensor)
