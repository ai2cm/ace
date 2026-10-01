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
    def test_get_for_rank_slices_corrector(self):
        gathered = GatheredStepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.arange(4).reshape(4, 1, 1).float()
            ),
            per_rank_random_states=None,
        )
        for rank in range(2):
            result = gathered.get_for_rank(rank, n_ranks=2)
            assert result.corrector_state is not None
            expected = torch.arange(rank * 2, (rank + 1) * 2).reshape(2, 1, 1).float()
            torch.testing.assert_close(
                result.corrector_state.global_dry_air_mass, expected
            )
            assert result.random_state is None

    def test_get_for_rank_selects_random_state(self):
        rs0 = _make_random_state(10)
        rs1 = _make_random_state(20)
        gathered = GatheredStepperState(
            corrector_state=None,
            per_rank_random_states=[rs0, rs1],
        )
        assert gathered.get_for_rank(0, n_ranks=2).random_state is rs0
        assert gathered.get_for_rank(1, n_ranks=2).random_state is rs1

    def test_scatter_random_state_keeps_full_corrector(self):
        corrector = CorrectorState(
            global_dry_air_mass=torch.arange(4).reshape(4, 1, 1).float()
        )
        rs0 = _make_random_state(10)
        rs1 = _make_random_state(20)
        gathered = GatheredStepperState(
            corrector_state=corrector,
            per_rank_random_states=[rs0, rs1],
        )
        result = gathered.scatter_random_state(rank=0)
        assert result.corrector_state is corrector
        assert result.random_state is rs0

        result1 = gathered.scatter_random_state(rank=1)
        assert result1.corrector_state is corrector
        assert result1.random_state is rs1

    def test_from_per_rank_states(self):
        s0 = _make_stepper_state(n_samples=2, seed=10)
        s1 = _make_stepper_state(n_samples=2, seed=20)
        gathered = GatheredStepperState.from_per_rank_states([s0, s1])
        assert gathered.n_ranks == 2
        result0 = gathered.get_for_rank(0, n_ranks=2)
        result1 = gathered.get_for_rank(1, n_ranks=2)
        assert result0.random_state is s0.random_state
        assert result1.random_state is s1.random_state
        assert result0.corrector_state is not None
        torch.testing.assert_close(
            result0.corrector_state.global_dry_air_mass,
            s0.corrector_state.global_dry_air_mass,
        )

    def test_round_trip_state_dict(self):
        rs0 = _make_random_state(10)
        rs1 = _make_random_state(20)
        torch.randn(5, generator=rs0.generator)

        gathered = GatheredStepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[1.0]], [[2.0]]])
            ),
            per_rank_random_states=[rs0, rs1],
        )

        state_dict = gathered.to_state_dict()
        restored = GatheredStepperState.from_state_dict(state_dict)

        assert restored.n_ranks == 2
        for i in range(2):
            original = gathered.get_for_rank(i, n_ranks=2)
            restored_rank = restored.get_for_rank(i, n_ranks=2)
            assert original.random_state is not None
            assert restored_rank.random_state is not None
            assert torch.equal(
                original.random_state.generator.get_state(),
                restored_rank.random_state.generator.get_state(),
            )

    def test_from_state_dict_raises_on_ungathered(self):
        stepper = _make_stepper_state(n_samples=2, seed=42)
        state_dict = stepper.to_state_dict()
        with pytest.raises(UngatheredStateDictError):
            GatheredStepperState.from_state_dict(state_dict)

    def test_n_ranks_marker_in_state_dict(self):
        gathered = GatheredStepperState(
            corrector_state=None,
            per_rank_random_states=[_make_random_state(0), _make_random_state(1)],
        )
        state_dict = gathered.to_state_dict()
        assert "random_state.n_ranks" in state_dict
        assert state_dict["random_state.n_ranks"].item() == 2

    def test_per_sample_state_keys_excludes_random_state(self):
        keys = GatheredStepperState.per_sample_state_keys()
        assert all("corrector" in k for k in keys)
        assert not any("random" in k for k in keys)

    def test_get_for_rank_validates_n_ranks(self):
        gathered = GatheredStepperState(
            corrector_state=None,
            per_rank_random_states=[
                _make_random_state(0),
                _make_random_state(1),
            ],
        )
        with pytest.raises(ValueError, match="does not match"):
            gathered.get_for_rank(0, n_ranks=4)

    def test_from_state_dict_corrector_only(self):
        """A corrector-only state dict (no random state) round-trips."""
        stepper = StepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[3.0]]])
            ),
            random_state=None,
        )
        state_dict = stepper.to_state_dict()
        gathered = GatheredStepperState.from_state_dict(state_dict)
        assert gathered._corrector_state is not None
        torch.testing.assert_close(
            gathered._corrector_state.global_dry_air_mass,
            torch.tensor([[[3.0]]]),
        )
        assert gathered._per_rank_random_states is None

    def test_from_per_rank_states_no_random(self):
        s0 = _make_stepper_state(n_samples=2, seed=None)
        s1 = _make_stepper_state(n_samples=2, seed=None)
        gathered = GatheredStepperState.from_per_rank_states([s0, s1])
        assert gathered._corrector_state is not None
        assert gathered._per_rank_random_states is None

    def test_to_cpu(self):
        gathered = GatheredStepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[1.0]]])
            ),
            per_rank_random_states=[_make_random_state(0)],
        )
        cpu_gathered = gathered.to_cpu()
        result = cpu_gathered.get_for_rank(0, n_ranks=1)
        assert result.corrector_state is not None
        assert result.corrector_state.global_dry_air_mass.device.type == "cpu"
