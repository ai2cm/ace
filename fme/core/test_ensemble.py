import dataclasses

import pytest
import torch

from fme.core.ensemble import get_crps, get_energy_score


@dataclasses.dataclass
class CRPSExperiment:
    name: str
    truth_amount: float
    random_amount: float


@pytest.mark.parametrize("n_ensemble", [2, 5])
@pytest.mark.parametrize("alpha", [1.0, 0.95])
def test_crps(n_ensemble: int, alpha: float):
    """
    Test that get_crps is a proper scoring rule.

    Scoring rules that are proper are proven to have the lowest
    expected score if the predicted distribution equals the
    underlying distribution of the target variable. Note that
    the assumptions in this test are only valid for values of
    alpha near 1.
    """
    torch.manual_seed(0)
    nx = 1
    ny = 1
    n_batch = 10000
    n_sample = n_ensemble
    truth_amount = 0.8
    random_amount = 0.5
    experiments = [
        CRPSExperiment("perfect", truth_amount, random_amount),
        CRPSExperiment("extra_variance", truth_amount, random_amount * 1.1),
        CRPSExperiment("less_variance", truth_amount, random_amount * 0.9),
        CRPSExperiment("deterministic", truth_amount, random_amount * 1e-5),
    ]
    x_predictable = torch.rand(n_batch, 1, nx, ny)
    x = truth_amount * x_predictable + random_amount * torch.rand(n_batch, 1, nx, ny)
    crps_values = {}
    for experiment in experiments:
        x_sample = (
            experiment.truth_amount * x_predictable
            + experiment.random_amount * torch.rand(n_batch, n_sample, nx, ny)
        )
        crps_values[experiment.name] = get_crps(
            gen=x_sample, target=x, alpha=alpha
        ).mean()
    assert crps_values["perfect"] < crps_values["extra_variance"]
    assert crps_values["perfect"] < crps_values["less_variance"]
    assert crps_values["extra_variance"] < crps_values["deterministic"]
    assert crps_values["less_variance"] < crps_values["deterministic"]


def test_energy_score_two_members_matches_explicit():
    """The pairwise implementation reproduces the closed form for 2 members."""
    torch.manual_seed(0)
    gen = torch.randn(8, 2, 3, 5, 5, dtype=torch.cfloat)
    target = torch.randn(8, 1, 3, 5, 5, dtype=torch.cfloat)
    explicit = torch.abs(gen - target).mean(dim=1) - 0.5 * torch.abs(
        gen[:, 0, ...] - gen[:, 1, ...]
    )
    torch.testing.assert_close(get_energy_score(gen, target), explicit)


def test_energy_score_n_members_matches_manual_pairs():
    """For n members the internal term is the mean |X_i - X_j| over pairs i<j."""
    torch.manual_seed(0)
    n_ens = 4
    gen = torch.randn(8, n_ens, 3, 5, 5, dtype=torch.cfloat)
    target = torch.randn(8, 1, 3, 5, 5, dtype=torch.cfloat)
    target_term = torch.abs(gen - target).mean(dim=1)
    pair_terms = [
        torch.abs(gen[:, i, ...] - gen[:, j, ...])
        for i in range(n_ens)
        for j in range(i + 1, n_ens)
    ]
    internal_term = -0.5 * torch.stack(pair_terms, dim=1).mean(dim=1)
    torch.testing.assert_close(
        get_energy_score(gen, target), target_term + internal_term
    )


def test_energy_score_single_member_raises():
    gen = torch.randn(8, 1, 3, 5, 5, dtype=torch.cfloat)
    target = torch.randn(8, 1, 3, 5, 5, dtype=torch.cfloat)
    with pytest.raises(NotImplementedError):
        get_energy_score(gen, target)
