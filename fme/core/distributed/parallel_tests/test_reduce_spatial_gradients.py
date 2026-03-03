"""
Backend-agnostic tests for Distributed.reduce_spatial_gradients.
"""

import pytest
import torch

from fme.core import get_device
from fme.core.distributed import Distributed


def _simple_module(in_features: int = 4, out_features: int = 4) -> torch.nn.Module:
    """Return a small module with .requires_grad parameters."""
    return torch.nn.Linear(in_features, out_features, bias=True).to(get_device())


def _set_uniform_grad(module: torch.nn.Module, value: float) -> None:
    """Set every parameter's .grad to a constant tensor."""
    for p in module.parameters():
        p.grad = torch.full_like(p, value)


@pytest.mark.parallel
def test_reduce_spatial_gradients_no_op_when_no_spatial():
    """When spatial parallelism is inactive, gradients should be unchanged."""
    dist = Distributed.get_instance()
    if dist.world_size // dist.total_data_parallel_ranks > 1:
        pytest.skip("This test targets non-spatial backends only")

    module = _simple_module()
    _set_uniform_grad(module, 3.0)
    dist.reduce_spatial_gradients(module)

    for p in module.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, 3.0))


@pytest.mark.parallel
def test_reduce_spatial_gradients_sums_across_spatial_ranks():
    """After reduction, each spatial rank should hold the sum of all partial grads.

    Every spatial rank sets its grad to 1.0.  After the spatial all-reduce the
    expected value is spatial_size (= h_size * w_size), because SUM of
    spatial_size copies of 1.0 is spatial_size.
    """
    dist = Distributed.get_instance()
    spatial_size = dist.world_size // dist.total_data_parallel_ranks

    module = _simple_module()
    _set_uniform_grad(module, 1.0)
    dist.reduce_spatial_gradients(module)

    expected = float(spatial_size)
    for p in module.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, expected))


@pytest.mark.parallel
def test_reduce_spatial_gradients_with_rank_dependent_grads():
    """Each spatial rank sets grad = dist.rank % spatial_size (its spatial index).

    After the 1-D all-reduces (h then w), each rank should hold the sum of all
    spatial indices: sum_{i=0}^{spatial_size-1} i = spatial_size*(spatial_size-1)/2.
    """
    dist = Distributed.get_instance()
    spatial_size = dist.world_size // dist.total_data_parallel_ranks
    spatial_index = float(dist.rank % spatial_size)

    module = _simple_module()
    _set_uniform_grad(module, spatial_index)
    dist.reduce_spatial_gradients(module)

    expected = float(spatial_size * (spatial_size - 1) / 2)
    for p in module.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, expected))


@pytest.mark.parallel
def test_reduce_spatial_gradients_skips_params_without_grad():
    """Parameters with grad=None should remain None after reduction."""
    dist = Distributed.get_instance()
    module = _simple_module()
    # Leave .grad as None (default)
    assert all(p.grad is None for p in module.parameters())

    dist.reduce_spatial_gradients(module)

    for p in module.parameters():
        assert p.grad is None


@pytest.mark.parallel
def test_reduce_spatial_gradients_skips_frozen_params():
    """Frozen parameters (requires_grad=False) should be untouched."""
    dist = Distributed.get_instance()
    module = _simple_module()
    for p in module.parameters():
        p.requires_grad_(False)
        p.grad = torch.full_like(p, 99.0)

    dist.reduce_spatial_gradients(module)

    # Grad should be exactly as we set it — not all-reduced
    for p in module.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, 99.0))


@pytest.mark.parallel
def test_reduce_spatial_gradients_mixed_grad_states():
    """Module with a mix of graded and non-graded params."""
    dist = Distributed.get_instance()
    spatial_size = dist.world_size // dist.total_data_parallel_ranks

    module = _simple_module()
    params = list(module.parameters())
    assert len(params) >= 2, "Need at least 2 params (weight + bias)"

    # First param: has grad, should be reduced
    params[0].grad = torch.full_like(params[0], 1.0)
    # Second param: no grad, should stay None
    params[1].grad = None

    dist.reduce_spatial_gradients(module)

    torch.testing.assert_close(
        params[0].grad, torch.full_like(params[0], float(spatial_size))
    )
    assert params[1].grad is None


@pytest.mark.parallel
def test_reduce_spatial_gradients_preserves_grad_shape():
    """Gradient shape and dtype should be preserved through reduction."""
    dist = Distributed.get_instance()
    module = _simple_module(in_features=8, out_features=3)
    _set_uniform_grad(module, 2.0)

    shapes_before = {name: p.grad.shape for name, p in module.named_parameters()}
    dtypes_before = {name: p.grad.dtype for name, p in module.named_parameters()}

    dist.reduce_spatial_gradients(module)

    for name, p in module.named_parameters():
        assert p.grad.shape == shapes_before[name], f"Shape changed for {name}"
        assert p.grad.dtype == dtypes_before[name], f"Dtype changed for {name}"


@pytest.mark.parallel
def test_reduce_spatial_gradients_idempotent_for_single_spatial():
    """With spatial_size=1, calling reduce multiple times should not change grads."""
    dist = Distributed.get_instance()
    if dist.world_size // dist.total_data_parallel_ranks > 1:
        pytest.skip("This test targets non-spatial configs")

    module = _simple_module()
    _set_uniform_grad(module, 5.0)

    dist.reduce_spatial_gradients(module)
    dist.reduce_spatial_gradients(module)

    for p in module.parameters():
        torch.testing.assert_close(p.grad, torch.full_like(p, 5.0))
