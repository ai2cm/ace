import pytest
import torch

from fme.core.corrector.state import CorrectorState
from fme.core.random_state import RandomState
from fme.core.stepper_state import (
    GatheredStepperState,
    StepperState,
    UngatheredStateDictError,
)


def _make_random_state(seed: int) -> RandomState:
    return RandomState.from_seed(seed)


def _make_stepper_state(n_samples: int, seed: int | None = None) -> StepperState:
    corrector = CorrectorState(
        global_dry_air_mass=torch.randn(n_samples, 1, 1)
    )
    random_state = _make_random_state(seed) if seed is not None else None
    return StepperState(corrector_state=corrector, random_state=random_state)


class TestGatheredStepperState:
    def test_get_for_rank(self):
        s0 = _make_stepper_state(n_samples=2, seed=10)
        s1 = _make_stepper_state(n_samples=2, seed=20)
        gathered = GatheredStepperState(states=[s0, s1])
        assert gathered.get_for_rank(0) is s0
        assert gathered.get_for_rank(1) is s1

    def test_n_ranks(self):
        s0 = _make_stepper_state(n_samples=2, seed=10)
        s1 = _make_stepper_state(n_samples=2, seed=20)
        gathered = GatheredStepperState(states=[s0, s1])
        assert gathered.n_ranks == 2

    def test_round_trip_state_dict(self):
        rs0 = _make_random_state(10)
        rs1 = _make_random_state(20)
        torch.randn(5, generator=rs0.generator)

        s0 = StepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[1.0]]])
            ),
            random_state=rs0,
        )
        s1 = StepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[2.0]]])
            ),
            random_state=rs1,
        )
        gathered = GatheredStepperState(states=[s0, s1])

        state_dict = gathered.to_state_dict()
        assert "n_ranks" in state_dict
        assert state_dict["n_ranks"].item() == 2
        assert "rank_0.corrector_state.global_dry_air_mass" in state_dict
        assert "rank_1.corrector_state.global_dry_air_mass" in state_dict

        restored = GatheredStepperState.from_state_dict(state_dict)
        assert restored.n_ranks == 2
        for i in range(2):
            original = gathered.get_for_rank(i)
            restored_rank = restored.get_for_rank(i)
            assert original.random_state is not None
            assert restored_rank.random_state is not None
            assert torch.equal(
                original.random_state.generator.get_state(),
                restored_rank.random_state.generator.get_state(),
            )
            assert original.corrector_state is not None
            assert restored_rank.corrector_state is not None
            torch.testing.assert_close(
                original.corrector_state.global_dry_air_mass,
                restored_rank.corrector_state.global_dry_air_mass,
            )

    def test_from_state_dict_raises_on_ungathered(self):
        stepper = _make_stepper_state(n_samples=2, seed=42)
        state_dict = stepper.to_state_dict()
        with pytest.raises(UngatheredStateDictError):
            GatheredStepperState.from_state_dict(state_dict)

    def test_per_sample_state_keys_is_empty(self):
        """Per-rank variables are not per-sample in the gathered sense."""
        s0 = _make_stepper_state(n_samples=2, seed=10)
        s1 = _make_stepper_state(n_samples=2, seed=20)
        gathered = GatheredStepperState(states=[s0, s1])
        assert gathered.per_sample_state_keys() == set()

    def test_to_cpu(self):
        s0 = _make_stepper_state(n_samples=1, seed=10)
        gathered = GatheredStepperState(states=[s0])
        cpu_gathered = gathered.to_cpu()
        result = cpu_gathered.get_for_rank(0)
        assert result.corrector_state is not None
        assert result.corrector_state.global_dry_air_mass.device.type == "cpu"

    def test_no_random_state(self):
        s0 = _make_stepper_state(n_samples=2, seed=None)
        s1 = _make_stepper_state(n_samples=2, seed=None)
        gathered = GatheredStepperState(states=[s0, s1])
        assert gathered.n_ranks == 2
        assert gathered.get_for_rank(0).random_state is None
        assert gathered.get_for_rank(1).random_state is None
