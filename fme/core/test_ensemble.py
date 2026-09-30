import dataclasses
import math

import pytest
import torch

from fme.core.ensemble import get_crps, get_variogram_edge_offsets, get_variogram_score


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


def test_variogram_edge_offsets_window_3():
    assert set(get_variogram_edge_offsets(3)) == {(0, 1), (1, 1), (1, 0), (1, -1)}


def test_variogram_edge_offsets_window_5():
    offsets = get_variogram_edge_offsets(5)
    assert len(offsets) == 12
    assert len(set(offsets)) == 12
    for di, dj in offsets:
        assert (di, dj) != (0, 0)
        assert (-di, -dj) not in offsets
        assert abs(di) <= 2 and abs(dj) <= 2
    assert set(get_variogram_edge_offsets(3)) <= set(offsets)


@pytest.mark.parametrize("window_size", [0, 1, 2, 4])
def test_variogram_edge_offsets_invalid_window(window_size: int):
    with pytest.raises(ValueError):
        get_variogram_edge_offsets(window_size)


def _variogram_score_oracle(
    gen: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    window_size: int,
    p: float,
    eps: float,
) -> torch.Tensor:
    """Variogram score by explicit enumeration of every unordered pair of
    distinct grid points within the window (periodic in longitude, no pole
    crossing), for gen [B, 2, C, n_lat, n_lon] and scale [n_edges, C] ordered
    as get_variogram_edge_offsets(window_size).
    """
    n_batch, _, n_channel, n_lat, n_lon = gen.shape
    halo = window_size // 2
    kind_index = {o: k for k, o in enumerate(get_variogram_edge_offsets(window_size))}
    pairs: dict[frozenset, int] = {}
    for i1 in range(n_lat):
        for j1 in range(n_lon):
            for i2 in range(n_lat):
                for j2 in range(n_lon):
                    if (i1, j1) == (i2, j2) or abs(i2 - i1) > halo:
                        continue
                    # signed periodic longitude step in [-n_lon/2, n_lon/2)
                    dj = (j2 - j1 + n_lon // 2) % n_lon - n_lon // 2
                    if abs(dj) > halo:
                        continue
                    di = i2 - i1
                    if (di, dj) not in kind_index:
                        continue  # the mirror orientation of the same pair
                    pairs[frozenset([(i1, j1), (i2, j2)])] = kind_index[(di, dj)]
    result = torch.zeros(n_batch, n_channel, dtype=torch.float64)
    for pair, k in pairs.items():
        (i1, j1), (i2, j2) = sorted(pair)
        # orient the increment along the kind's offset
        di, dj = get_variogram_edge_offsets(window_size)[k]
        if (i2 - i1, (j2 - j1 + n_lon // 2) % n_lon - n_lon // 2) != (di, dj):
            (i1, j1), (i2, j2) = (i2, j2), (i1, j1)

        def d(field: torch.Tensor) -> torch.Tensor:
            inc = (field[..., i2, j2] - field[..., i1, j1]) / scale[k]
            return (inc**2 + eps) ** (p / 2)

        d_y = d(target[:, 0].double())
        d_1 = d(gen[:, 0].double())
        d_2 = d(gen[:, 1].double())
        result += (d_y - d_1) * (d_y - d_2)
    n_expected = sum(
        (n_lat - di) * n_lon for di, _ in get_variogram_edge_offsets(window_size)
    )
    assert len(pairs) == n_expected
    return result / len(pairs)


@pytest.mark.parametrize("window_size", [3, 5])
@pytest.mark.parametrize("p", [0.5, 1.0])
def test_variogram_score_matches_pairwise_oracle(window_size: int, p: float):
    torch.manual_seed(0)
    n_batch, n_channel, n_lat, n_lon = 2, 3, 4, 7
    gen = torch.randn(n_batch, 2, n_channel, n_lat, n_lon, dtype=torch.float64)
    target = torch.randn(n_batch, 1, n_channel, n_lat, n_lon, dtype=torch.float64)
    offsets = get_variogram_edge_offsets(window_size)
    scale = torch.rand(len(offsets), n_channel, dtype=torch.float64) + 0.5
    result = get_variogram_score(gen, target, scale, offsets, p=p, eps=1e-6)
    expected = _variogram_score_oracle(
        gen, target, scale, window_size=window_size, p=p, eps=1e-6
    )
    assert result.shape == (n_batch, n_channel)
    torch.testing.assert_close(result, expected)


def test_variogram_score_does_not_cross_poles():
    """Members are zero and the target is 0 on row 0 and 1 on row 1, so every
    north-south edge from row 0 to row 1 scores 1 and every east-west edge
    scores 0. Wrapping across the poles would add north-south edges from row 1
    back to row 0 (score 0.75 instead of 0.6).
    """
    n_lon = 5
    gen = torch.zeros(1, 2, 1, 2, n_lon)
    target = torch.zeros(1, 1, 1, 2, n_lon)
    target[..., 1, :] = 1.0
    offsets = get_variogram_edge_offsets(3)
    scale = torch.ones(len(offsets), 1)
    result = get_variogram_score(gen, target, scale, offsets, p=1.0, eps=0.0)
    # 3 north-south kinds x 1 valid row x n_lon edges score 1, out of
    # (2 rows x n_lon east-west) + (3 x 1 x n_lon north-south) edges
    torch.testing.assert_close(result, torch.tensor([[0.6]]))


@pytest.mark.parametrize("window_size", [3, 5])
def test_variogram_score_longitude_roll_invariant(window_size: int):
    torch.manual_seed(0)
    gen = torch.randn(2, 2, 3, 6, 8)
    target = torch.randn(2, 1, 3, 6, 8)
    offsets = get_variogram_edge_offsets(window_size)
    scale = torch.rand(len(offsets), 3) + 0.5
    result = get_variogram_score(gen, target, scale, offsets)
    rolled = get_variogram_score(
        torch.roll(gen, 3, dims=-1), torch.roll(target, 3, dims=-1), scale, offsets
    )
    torch.testing.assert_close(result, rolled)


def test_variogram_score_fair_estimator_expectation():
    """Averaged over many independent member pairs, the fair two-member score
    is (D_y - E[D])^2 per edge. With members iid standard normal at every
    point, each increment is N(0, 2) and E|Δ|^p = 2^p Γ((p + 1) / 2) / sqrt(π).
    """
    torch.manual_seed(0)
    p = 0.5
    n_samples, n_lat, n_lon = 20000, 2, 4
    target_field = torch.randn(1, 1, 1, n_lat, n_lon, dtype=torch.float64)
    target = target_field.expand(n_samples, 1, 1, n_lat, n_lon)
    gen = torch.randn(n_samples, 2, 1, n_lat, n_lon, dtype=torch.float64)
    offsets = get_variogram_edge_offsets(3)
    scale = torch.ones(len(offsets), 1, dtype=torch.float64)
    score = get_variogram_score(gen, target, scale, offsets, p=p, eps=0.0).mean()
    expected_d = 2**p * math.gamma((p + 1) / 2) / math.sqrt(math.pi)
    squared_errors = []
    for di, dj in offsets:
        lower = target_field[..., : n_lat - di, :]
        upper = torch.roll(target_field[..., di:, :], shifts=-dj, dims=-1)
        d_y = (upper - lower).abs() ** p
        squared_errors.append(((d_y - expected_d) ** 2).flatten())
    expected = torch.cat(squared_errors).mean()
    torch.testing.assert_close(score, expected, rtol=0.03, atol=0.0)


@pytest.mark.parametrize("identical_members", [True, False])
def test_variogram_score_finite_gradients_at_zero_increments(
    identical_members: bool,
):
    """Spatially constant fields have zero increments, where |Δ|^0.5 has an
    unbounded derivative; the smoothed power must keep gradients finite.
    """
    gen = torch.zeros(1, 2, 2, 4, 6)
    if not identical_members:
        gen[:, 1] = 1.0  # constant, different from member 0
    gen.requires_grad_(True)
    target = torch.zeros(1, 1, 2, 4, 6)
    offsets = get_variogram_edge_offsets(5)
    scale = torch.ones(len(offsets), 2)
    score = get_variogram_score(gen, target, scale, offsets, p=0.5)
    score.sum().backward()
    assert gen.grad is not None
    assert torch.isfinite(gen.grad).all()


def test_variogram_score_requires_two_members():
    offsets = get_variogram_edge_offsets(3)
    with pytest.raises(NotImplementedError):
        get_variogram_score(
            torch.zeros(1, 3, 1, 4, 6),
            torch.zeros(1, 1, 1, 4, 6),
            torch.ones(len(offsets), 1),
            offsets,
        )
