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


def _complex_rand(*shape: int) -> torch.Tensor:
    return torch.complex(torch.rand(*shape), torch.rand(*shape))


@pytest.mark.parametrize("n_ensemble", [2, 3, 5])
def test_energy_score_matches_pairwise_reference(n_ensemble: int):
    """
    get_energy_score equals E|X - y| - 1/2 * mean_{i<j} |X_i - X_j| computed
    explicitly over every unique pair of ensemble members.
    """
    torch.manual_seed(0)
    gen = _complex_rand(4, n_ensemble, 3, 5)
    target = _complex_rand(4, 1, 3, 5)
    target_term = torch.abs(gen - target).mean(dim=1)
    pairs = [
        torch.abs(gen[:, i] - gen[:, j])
        for i in range(n_ensemble)
        for j in range(i + 1, n_ensemble)
    ]
    expected = target_term - 0.5 * torch.stack(pairs, dim=1).mean(dim=1)
    torch.testing.assert_close(get_energy_score(gen, target), expected)


@pytest.mark.parametrize("n_ensemble", [2, 3])
def test_energy_score_is_proper(n_ensemble: int):
    """
    Test that get_energy_score is a proper scoring rule: the perfectly
    calibrated ensemble scores best, as in test_crps.
    """
    torch.manual_seed(0)
    nx = 1
    ny = 1
    n_batch = 10000
    truth_amount = 0.8
    random_amount = 0.5
    experiments = [
        CRPSExperiment("perfect", truth_amount, random_amount),
        CRPSExperiment("extra_variance", truth_amount, random_amount * 1.1),
        CRPSExperiment("less_variance", truth_amount, random_amount * 0.9),
        CRPSExperiment("deterministic", truth_amount, random_amount * 1e-5),
    ]
    x_predictable = _complex_rand(n_batch, 1, nx, ny)
    x = truth_amount * x_predictable + random_amount * _complex_rand(n_batch, 1, nx, ny)
    scores = {}
    for experiment in experiments:
        x_sample = (
            experiment.truth_amount * x_predictable
            + experiment.random_amount * _complex_rand(n_batch, n_ensemble, nx, ny)
        )
        scores[experiment.name] = get_energy_score(gen=x_sample, target=x).mean()
    assert scores["perfect"] < scores["extra_variance"]
    assert scores["perfect"] < scores["less_variance"]
    assert scores["extra_variance"] < scores["deterministic"]
    assert scores["less_variance"] < scores["deterministic"]
