import torch

from fme.core.corrector.state import CorrectorState
from fme.core.random_state import RandomState
from fme.core.stepper_state import GatheredStepperState, StepperState


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

    def test_round_trip_state_dict(self):
        rs0 = _make_random_state(10)
        rs1 = _make_random_state(20)
        # Advance rs0 so the two states differ.
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
        assert restored.corrector_state is not None
        torch.testing.assert_close(
            restored.corrector_state.global_dry_air_mass,
            gathered.corrector_state.global_dry_air_mass,
        )
        for i in range(2):
            assert restored.per_rank_random_states is not None
            original_state = gathered.per_rank_random_states[i].generator.get_state()
            restored_state = restored.per_rank_random_states[i].generator.get_state()
            assert torch.equal(original_state, restored_state)

    def test_from_state_dict_legacy_single_generator(self):
        """A StepperState state dict (single generator) round-trips through
        GatheredStepperState as a 1-rank list."""
        stepper = _make_stepper_state(n_samples=2, seed=42)
        state_dict = stepper.to_state_dict()

        gathered = GatheredStepperState.from_state_dict(state_dict)
        assert gathered.n_ranks == 1
        assert gathered.per_rank_random_states is not None

        original_gen_state = stepper.random_state.generator.get_state()
        restored_gen_state = gathered.per_rank_random_states[0].generator.get_state()
        assert torch.equal(original_gen_state, restored_gen_state)

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

    def test_to_cpu(self):
        gathered = GatheredStepperState(
            corrector_state=CorrectorState(
                global_dry_air_mass=torch.tensor([[[1.0]]])
            ),
            per_rank_random_states=[_make_random_state(0)],
        )
        cpu_gathered = gathered.to_cpu()
        assert cpu_gathered.corrector_state is not None
        assert cpu_gathered.corrector_state.global_dry_air_mass.device.type == "cpu"
