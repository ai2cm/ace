import pytest
import torch

from fme.core.corrector.state import CorrectorState
from fme.core.random_state import RandomState
from fme.core.stepper_state import StepperState


def test_from_seed_is_reproducible():
    a = RandomState.from_seed(0)
    b = RandomState.from_seed(0)
    x = torch.randn(4, generator=a.generator)
    y = torch.randn(4, generator=b.generator)
    assert torch.equal(x, y)


def test_requires_cpu_generator():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    g = torch.Generator(device="cuda")
    with pytest.raises(ValueError, match="CPU torch.Generator"):
        RandomState(generator=g)


def test_device_and_ensemble_transforms_preserve_generator():
    """A single generator is shared across the batch and consumed in place, so
    the device/ensemble helpers must return the same advancing object."""
    rs = RandomState.from_seed(0)
    assert rs.to_device() is rs
    assert rs.to_cpu() is rs
    assert rs.pin_memory() is rs
    assert rs.broadcast_ensemble(4) is rs
    assert rs.sample_dim_size() is None


def test_stepper_state_threads_random_state_through_transforms():
    rs = RandomState.from_seed(0)
    state = StepperState(random_state=rs)
    # The same generator object survives each transform (not reset or copied).
    assert state.to_device().random_state is rs
    assert state.to_cpu().random_state is rs
    assert state.pin_memory().random_state is rs
    assert state.broadcast_ensemble(3).random_state is rs


def test_stepper_state_sample_dim_size_ignores_random_state():
    """random_state has no per-sample dim, so only corrector_state constrains it."""
    rs = RandomState.from_seed(0)
    assert StepperState(random_state=rs).sample_dim_size() is None
    corrector_state = CorrectorState(global_dry_air_mass=torch.zeros(5, 1, 1))
    state = StepperState(corrector_state=corrector_state, random_state=rs)
    assert state.sample_dim_size() == 5
