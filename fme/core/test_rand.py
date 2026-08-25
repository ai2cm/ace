import numpy as np
import torch

from fme.core.device import get_device
from fme.core.distributed import Distributed
from fme.core.rand import randn, randn_like, set_seed, use_generator


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


def _seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def test_use_generator_is_reproducible_and_advances():
    device = get_device()
    with use_generator(_seeded_generator(0)):
        a0 = randn(torch.Size([4, 3]), device=device)
        a1 = randn(torch.Size([4, 3]), device=device)
    with use_generator(_seeded_generator(0)):
        b0 = randn(torch.Size([4, 3]), device=device)
        b1 = randn(torch.Size([4, 3]), device=device)
    # Same seed reproduces the full sequence of draws...
    assert torch.equal(a0, b0)
    assert torch.equal(a1, b1)
    # ...and the generator advances between draws.
    assert not torch.equal(a0, a1)
    assert a0.device.type == device.type


def test_use_generator_differs_by_seed():
    with use_generator(_seeded_generator(0)):
        a = randn(torch.Size([8]))
    with use_generator(_seeded_generator(1)):
        b = randn(torch.Size([8]))
    assert not torch.equal(a, b)


def test_randn_like_honors_active_generator():
    x = torch.zeros(4, 3, device=get_device())
    with use_generator(_seeded_generator(0)):
        a = randn_like(x)
    with use_generator(_seeded_generator(0)):
        b = randn(torch.Size([4, 3]), device=x.device)
    assert a.shape == x.shape
    assert a.device.type == x.device.type
    # randn_like and randn agree for the same seed/shape.
    assert torch.equal(a, b)


def test_use_generator_none_is_global_rng_passthrough():
    set_seed(0)
    a = randn(torch.Size([5]))
    set_seed(0)
    with use_generator(None):
        b = randn(torch.Size([5]))
    assert torch.equal(a, b)


def test_use_generator_restores_previous_on_exit():
    outer = _seeded_generator(0)
    with use_generator(outer):
        a = randn(torch.Size([3]))
        with use_generator(_seeded_generator(1)):
            randn(torch.Size([3]))
        # After the nested block, draws continue from the outer generator as if
        # the inner block had not consumed it.
        b = randn(torch.Size([3]))
    with use_generator(_seeded_generator(0)):
        a_ref = randn(torch.Size([3]))
        b_ref = randn(torch.Size([3]))
    assert torch.equal(a, a_ref)
    assert torch.equal(b, b_ref)


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
