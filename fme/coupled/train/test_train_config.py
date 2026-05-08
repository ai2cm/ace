import pytest

from fme.ace.stepper.time_length_probabilities import (
    TimeLengthProbabilities,
    TimeLengthProbability,
)
from fme.coupled.loss import LossContributionsConfig
from fme.coupled.typing_ import CoupledOptionalInt

from .train_config import _validate_loss_n_steps


def test_validate_loss_n_steps_passes_when_unbounded():
    _validate_loss_n_steps(
        n_coupled_steps=1,
        n_inner_steps=2,
        component_n_steps_max=CoupledOptionalInt(ocean=None, atmosphere=None),
    )


def test_validate_loss_n_steps_passes_at_limit():
    # Equality is allowed: n_steps==n_coupled_steps means losses for steps
    # 0..n_steps-1, all within range.
    _validate_loss_n_steps(
        n_coupled_steps=4,
        n_inner_steps=2,
        component_n_steps_max=CoupledOptionalInt(ocean=4, atmosphere=8),
    )


def test_validate_loss_n_steps_rejects_ocean_overshoot():
    with pytest.raises(ValueError, match=r"ocean.*exceeds n_coupled_steps"):
        _validate_loss_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=3, atmosphere=None),
        )


def test_validate_loss_n_steps_rejects_atmosphere_overshoot():
    with pytest.raises(
        ValueError,
        match=r"atmosphere.*exceeds n_coupled_steps \* n_inner_steps",
    ):
        _validate_loss_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=None, atmosphere=7),
        )


def test_validate_loss_n_steps_lists_both_components_when_both_misconfigured():
    with pytest.raises(ValueError) as exc_info:
        _validate_loss_n_steps(
            n_coupled_steps=2,
            n_inner_steps=3,
            component_n_steps_max=CoupledOptionalInt(ocean=5, atmosphere=10),
        )
    msg = str(exc_info.value)
    assert "ocean" in msg
    assert "atmosphere" in msg


def test_validate_loss_n_steps_uses_sampler_max_via_config():
    sampler = TimeLengthProbabilities(
        outcomes=[
            TimeLengthProbability(steps=1, probability=0.5),
            TimeLengthProbability(steps=5, probability=0.5),
        ]
    )
    config = LossContributionsConfig(n_steps=sampler)
    bounds = CoupledOptionalInt(ocean=config.n_steps_max, atmosphere=None)
    with pytest.raises(ValueError, match=r"ocean"):
        _validate_loss_n_steps(
            n_coupled_steps=2, n_inner_steps=2, component_n_steps_max=bounds
        )


def test_validate_loss_n_steps_does_not_short_circuit_on_null_weight():
    # A weight=0 component still has a non-None n_steps_max if a value was
    # explicitly set, and the validator surfaces the misconfiguration even
    # though the loss contribution is null. This avoids silently accepting
    # confused configs.
    null_config = LossContributionsConfig(weight=0.0, n_steps=999)
    bounds = CoupledOptionalInt(ocean=null_config.n_steps_max, atmosphere=None)
    with pytest.raises(ValueError, match=r"ocean"):
        _validate_loss_n_steps(
            n_coupled_steps=1, n_inner_steps=1, component_n_steps_max=bounds
        )
