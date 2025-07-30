import numpy as np
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.rand import set_seed


def test_set_seed_np_rand():
    set_seed(0)
    a = np.random.randn(10)
    set_seed(0)
    b = np.random.randn(10)
    assert np.allclose(a, b)


def test_set_seed_torch_rand():
    device = get_device()
    set_seed(0)
    a = torch.randn(10, device=device)
    set_seed(0)
    b = torch.randn(10, device=device)
    assert torch.allclose(a, b)


def test_set_distributed_shuffler():
    dist = Distributed.get_instance()
    set_seed(0)
    dataset = torch.utils.data.TensorDataset(torch.randn(10))
    sampler = dist.get_sampler(dataset, shuffle=True)
    first_results = list(sampler)
    set_seed(0)
    dist = Distributed.get_instance()
    sampler = dist.get_sampler(dataset, shuffle=True)
    second_results = list(sampler)
    assert torch.allclose(
        torch.as_tensor(first_results), torch.as_tensor(second_results)
    )


def test_set_random_sampler():
    dataset = torch.utils.data.TensorDataset(torch.randn(10))
    set_seed(0)
    sampler = torch.utils.data.RandomSampler(dataset)
    first_results = list(sampler)
    set_seed(0)
    sampler = torch.utils.data.RandomSampler(dataset)
    second_results = list(sampler)
    assert torch.allclose(
        torch.as_tensor(first_results), torch.as_tensor(second_results)
    )
