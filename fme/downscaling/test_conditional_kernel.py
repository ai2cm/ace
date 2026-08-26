import pytest
import torch

from fme.downscaling.conditional_kernel import (
    ConditionEncoder,
    _project_to_simplex,
    build_kernel_basis,
    condition_features,
    empirical_covariance,
    project_onto_kernel_hull,
)
from fme.downscaling.noise import uniform_frame_times


@pytest.mark.parametrize("pin_endpoints", [True, False])
def test_build_kernel_basis_is_psd_and_trace_normalized(pin_endpoints):
    tau = uniform_frame_times(9)
    specs = (
        [("independent", None), ("brownian_bridge", None), ("ou", 0.5)]
        if pin_endpoints
        else [("independent", None), ("ou", 0.3), ("ou", 0.6), ("rbf", 0.4)]
    )
    basis = build_kernel_basis(tau, specs, pin_endpoints)
    assert basis.shape == (len(specs), 9, 9)

    expected_trace = 7.0 if pin_endpoints else 9.0
    for k in basis:
        eigvals = torch.linalg.eigvalsh(k)
        assert (eigvals >= -1e-5).all(), "basis kernel must be PSD"
        assert torch.allclose(k.trace(), torch.tensor(expected_trace), atol=1e-4)
        assert torch.allclose(k, k.T, atol=1e-6), "basis kernel must be symmetric"


def test_build_kernel_basis_rejects_bridge_without_pinning():
    tau = uniform_frame_times(9)
    with pytest.raises(ValueError, match="pin_endpoints"):
        build_kernel_basis(tau, [("brownian_bridge", None)], pin_endpoints=False)


def test_build_kernel_basis_rejects_unknown_kernel():
    tau = uniform_frame_times(9)
    with pytest.raises(ValueError, match="Unknown kernel"):
        build_kernel_basis(tau, [("bogus", None)], pin_endpoints=False)


@pytest.mark.parametrize(
    "v,expected",
    [
        (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])),
        (
            torch.tensor([0.4, 0.3, 0.3]),
            torch.tensor([0.4, 0.3, 0.3]),
        ),  # already on simplex -> unchanged
        (torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5])),
    ],
)
def test_project_to_simplex_known_cases(v, expected):
    projected = _project_to_simplex(v)
    assert torch.allclose(projected, expected, atol=1e-5)
    assert torch.allclose(projected.sum(), torch.tensor(1.0), atol=1e-5)
    assert (projected >= -1e-6).all()


def test_project_onto_kernel_hull_recovers_known_weights_in_population():
    """Prop 4: if Sigma_hat is EXACTLY a convex combination of the basis,
    the projection recovers those weights (to high precision, given enough
    projected-gradient iterations on this tiny QP)."""
    tau = uniform_frame_times(9)
    specs = [("independent", None), ("ou", 0.3), ("ou", 0.6), ("rbf", 0.4)]
    basis = build_kernel_basis(tau, specs, pin_endpoints=False)

    w_true = torch.tensor([0.1, 0.2, 0.6, 0.1])
    sigma_hat = torch.einsum("b,bij->ij", w_true, basis)

    w_hat = project_onto_kernel_hull(basis, sigma_hat)
    # The basis Gram matrix is fairly ill-conditioned (OU at nearby length
    # scales/RBF look similar over a 9-frame window), so convergence to
    # exact recovery is slow relative to the fit-residual convergence --
    # see project_onto_kernel_hull's n_iters docstring. 1e-2 comfortably
    # separates "converged toward the true weights" from "stuck near the
    # uniform initialization" or a bug in the projection direction.
    assert torch.allclose(w_hat, w_true, atol=1e-2)
    assert torch.allclose(w_hat.sum(), torch.tensor(1.0), atol=1e-4)
    assert (w_hat >= -1e-6).all()
    k_hat = torch.einsum("b,bij->ij", w_hat, basis)
    assert torch.norm(k_hat - sigma_hat) < 1e-2


def test_project_onto_kernel_hull_out_of_hull_projects_to_closest_point():
    """When Sigma_hat is NOT in the hull, the projection should still land
    on the simplex and strictly reduce the Frobenius distance to Sigma_hat
    relative to an arbitrary interior point (e.g. the uniform mixture)."""
    tau = uniform_frame_times(9)
    specs = [("independent", None), ("ou", 0.3), ("ou", 0.6)]
    basis = build_kernel_basis(tau, specs, pin_endpoints=False)
    torch.manual_seed(0)
    sigma_hat = torch.randn(9, 9)
    sigma_hat = sigma_hat @ sigma_hat.T  # arbitrary PSD, generically outside the hull

    w_hat = project_onto_kernel_hull(basis, sigma_hat)
    assert torch.allclose(w_hat.sum(), torch.tensor(1.0), atol=1e-4)
    assert (w_hat >= -1e-6).all()

    k_hat = torch.einsum("b,bij->ij", w_hat, basis)
    uniform = torch.full((len(specs),), 1.0 / len(specs))
    k_uniform = torch.einsum("b,bij->ij", uniform, basis)
    assert torch.norm(k_hat - sigma_hat) <= torch.norm(k_uniform - sigma_hat) + 1e-4


def test_condition_encoder_outputs_simplex_weights():
    torch.manual_seed(1)
    encoder = ConditionEncoder(n_channels=5, n_basis=4, hidden=8)
    features = torch.randn(3, 5)
    weights = encoder(features)
    assert weights.shape == (3, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert (weights >= 0).all()


def test_condition_features_zero_when_endpoints_equal():
    clip = torch.zeros(2, 3, 5, 4, 4)
    clip[:, :, 0] = 1.0
    clip[:, :, -1] = 1.0  # identical endpoints -> zero difference
    features = condition_features(clip)
    assert torch.allclose(features, torch.zeros(2, 3))


def test_condition_features_matches_manual_computation():
    torch.manual_seed(2)
    clip = torch.randn(2, 3, 5, 4, 4)
    features = condition_features(clip)
    expected = (clip[:, :, -1] - clip[:, :, 0]).abs().mean(dim=(-2, -1))
    assert torch.allclose(features, expected)


def test_empirical_covariance_recovers_known_correlation():
    """Mix white noise through a known (T, T) matrix M; the empirical
    covariance of the mixed signal, pooled over many independent
    (batch/channel/spatial) draws, should converge to M @ M.T."""
    torch.manual_seed(3)
    tau = uniform_frame_times(9)
    basis = build_kernel_basis(tau, [("ou", 0.4)], pin_endpoints=False)
    true_cov = basis[0]

    mixing = torch.linalg.cholesky(true_cov + 1e-6 * torch.eye(9))
    n_batch, n_channels, height, width = 8, 5, 16, 16
    z = torch.randn(n_batch, n_channels, 9, height, width)
    mixed = torch.einsum("ij,bcjhw->bcihw", mixing, z)

    sigma_hat = empirical_covariance(mixed)
    assert sigma_hat.shape == (9, 9)
    assert torch.allclose(sigma_hat, true_cov, atol=0.05)
