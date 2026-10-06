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


def test_random_state_round_trip_continues_draw_sequence():
    """A restored RandomState continues the original generator's draw sequence:
    draws after restore match draws from the original at the same point."""
    original = RandomState.from_seed(7)
    # Advance the generator so the serialized state is not just the raw seed.
    torch.randn(5, generator=original.generator)

    restored = RandomState.from_state_dict(original.to_state_dict())
    assert restored.generator.device.type == "cpu"

    expected = torch.randn(4, generator=original.generator)
    got = torch.randn(4, generator=restored.generator)
    assert torch.equal(expected, got)


def test_corrector_state_round_trip_populated():
    mass = torch.randn(3, 1, 1)
    state = CorrectorState(global_dry_air_mass=mass)
    restored = CorrectorState.from_state_dict(state.to_state_dict())
    assert restored.global_dry_air_mass is not None
    assert torch.equal(restored.global_dry_air_mass, mass)


def test_corrector_state_round_trip_empty():
    state = CorrectorState()
    assert state.to_state_dict() == {}
    restored = CorrectorState.from_state_dict(state.to_state_dict())
    assert restored.global_dry_air_mass is None


def _assert_stepper_states_equal(a: StepperState, b: StepperState) -> None:
    if a.corrector_state is None:
        assert b.corrector_state is None
    else:
        assert b.corrector_state is not None
        if a.corrector_state.global_dry_air_mass is None:
            assert b.corrector_state.global_dry_air_mass is None
        else:
            assert b.corrector_state.global_dry_air_mass is not None
            assert torch.equal(
                a.corrector_state.global_dry_air_mass,
                b.corrector_state.global_dry_air_mass,
            )
    if a.random_state is None:
        assert b.random_state is None
    else:
        assert b.random_state is not None
        # A restored generator continues the same sequence as the original.
        assert torch.equal(
            torch.randn(3, generator=a.random_state.generator),
            torch.randn(3, generator=b.random_state.generator),
        )


@pytest.mark.parametrize("has_corrector", [True, False])
@pytest.mark.parametrize("has_random", [True, False])
@pytest.mark.parametrize("empty_corrector", [True, False])
def test_stepper_state_round_trip_each_substate_combination(
    has_corrector: bool, has_random: bool, empty_corrector: bool
):
    """Every present/None combination of sub-states survives the round-trip, and
    a None sub-state is restored as None (not an empty stand-in)."""
    corrector_state: CorrectorState | None
    if not has_corrector:
        corrector_state = None
    elif empty_corrector:
        corrector_state = CorrectorState()
    else:
        corrector_state = CorrectorState(global_dry_air_mass=torch.randn(2, 1, 1))
    random_state = RandomState.from_seed(3) if has_random else None
    state = StepperState(corrector_state=corrector_state, random_state=random_state)

    restored = StepperState.from_state_dict(state.to_state_dict())
    _assert_stepper_states_equal(state, restored)


def test_empty_stepper_state_round_trip():
    """A fully-empty StepperState round-trips to another fully-empty one."""
    state = StepperState()
    assert state.to_state_dict() == {}
    restored = StepperState.from_state_dict(state.to_state_dict())
    assert restored.corrector_state is None
    assert restored.random_state is None
