import dataclasses

import pytest
import torch

from fme.core.ensemble import get_crps, get_patch_energy_score


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


def _patch_vectors(field: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Explicitly build the patch vector at each point of ``[..., H, W]``:
    periodic in the last (lon) dim, zero-padded in the second-to-last (lat)
    dim. Returns ``[..., H, W, patch_size**2]``."""
    halo = patch_size // 2
    n_lat = field.shape[-2]
    padded = torch.nn.functional.pad(field, (0, 0, halo, halo))
    entries = []
    for i in range(patch_size):
        for dj in range(-halo, halo + 1):
            rolled = torch.roll(padded, shifts=-dj, dims=-1)
            entries.append(rolled[..., i : i + n_lat, :])
    return torch.stack(entries, dim=-1)


def _brute_force_patch_energy_score(
    gen: torch.Tensor, target: torch.Tensor, patch_size: int
) -> torch.Tensor:
    px0 = _patch_vectors(gen[:, 0], patch_size)
    px1 = _patch_vectors(gen[:, 1], patch_size)
    py = _patch_vectors(target[:, 0], patch_size)
    d0 = torch.linalg.norm(px0 - py, dim=-1)
    d1 = torch.linalg.norm(px1 - py, dim=-1)
    d01 = torch.linalg.norm(px0 - px1, dim=-1)
    return (0.5 * (d0 + d1) - 0.5 * d01) / patch_size


@pytest.mark.parametrize("patch_size", [1, 3, 5])
def test_patch_energy_score_matches_brute_force(patch_size: int):
    torch.manual_seed(0)
    gen = torch.randn(2, 2, 3, 8, 16)
    target = torch.randn(2, 1, 3, 8, 16)
    result = get_patch_energy_score(gen, target, patch_size=patch_size)
    expected = _brute_force_patch_energy_score(gen, target, patch_size)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)


def test_patch_energy_score_is_proper():
    """Like test_crps: the perfectly-calibrated ensemble scores best."""
    torch.manual_seed(0)
    n_batch = 5000
    nx, ny = 4, 8
    truth_amount = 0.8
    random_amount = 0.5
    x_predictable = torch.rand(n_batch, 1, nx, ny)
    x = truth_amount * x_predictable + random_amount * torch.rand(n_batch, 1, nx, ny)
    scores = {}
    for name, amount in [
        ("perfect", random_amount),
        ("extra_variance", random_amount * 1.2),
        ("less_variance", random_amount * 0.8),
        ("deterministic", random_amount * 1e-5),
    ]:
        x_sample = truth_amount * x_predictable + amount * torch.rand(
            n_batch, 2, nx, ny
        )
        scores[name] = get_patch_energy_score(x_sample, x, patch_size=3).mean()
    assert scores["perfect"] < scores["extra_variance"]
    assert scores["perfect"] < scores["less_variance"]
    assert scores["extra_variance"] < scores["deterministic"]
    assert scores["less_variance"] < scores["deterministic"]


def test_patch_energy_score_periodic_in_lon():
    torch.manual_seed(0)
    gen = torch.randn(2, 2, 3, 8, 16)
    target = torch.randn(2, 1, 3, 8, 16)
    score = get_patch_energy_score(gen, target, patch_size=3)
    rolled_score = get_patch_energy_score(
        torch.roll(gen, shifts=5, dims=-1),
        torch.roll(target, shifts=5, dims=-1),
        patch_size=3,
    )
    torch.testing.assert_close(rolled_score, torch.roll(score, shifts=5, dims=-1))


def test_patch_energy_score_size_one_is_fair_crps():
    torch.manual_seed(0)
    gen = torch.randn(2, 2, 3, 8, 16)
    target = torch.randn(2, 1, 3, 8, 16)
    patch = get_patch_energy_score(gen, target, patch_size=1)
    crps = get_crps(gen, target, alpha=1.0)
    torch.testing.assert_close(patch, crps, rtol=1e-5, atol=1e-5)


def test_patch_energy_score_finite_gradient_at_zero_spread():
    member = torch.randn(2, 1, 3, 8, 16)
    gen = member.repeat(1, 2, 1, 1, 1).requires_grad_(True)
    target = member.clone()
    score = get_patch_energy_score(gen, target, patch_size=3)
    score.sum().backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()


def test_patch_energy_score_rejects_bad_inputs():
    gen = torch.randn(2, 3, 3, 8, 16)
    target = torch.randn(2, 1, 3, 8, 16)
    with pytest.raises(NotImplementedError):
        get_patch_energy_score(gen, target)
    with pytest.raises(ValueError):
        get_patch_energy_score(torch.randn(2, 2, 3, 8, 16), target, patch_size=2)
