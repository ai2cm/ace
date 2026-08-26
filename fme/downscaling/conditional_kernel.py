"""Conditional (state-dependent) kernel mixtures for two-block noise.

See ``idea/conditional_kernel_theory.md`` (repo root, sibling of ``ace/``)
for the full derivation. Summary: instead of one FIXED per-block temporal
kernel (``fme.downscaling.twoblock``'s ``r_kernel``/``d_kernel``), fix a
small basis of ``B`` trace-normalized PSD kernels and learn a conditioning
network ``g_phi(c) -> simplex weights w(c)``, giving a state-dependent
covariance ``K_{w(c)} = sum_b w_b(c) K^(b)``.

Critically (the doc's central caveat, "plain DSM does NOT learn w*"):
``g_phi`` must NOT be trained through the denoising (DSM) loss -- that
selects the likelihood-optimal noise, which the doc shows numerically is
misaligned with the consistency-optimal (moment-matched) noise. Instead
``g_phi`` is trained by a separate, convex moment-matching objective
(Prop. 4): project the batch's empirical residual covariance onto the
basis hull via a simplex-constrained least squares (equivalent to NNLS
projection), and regress ``g_phi``'s output toward that target. Callers
(``video_models.py``) are responsible for keeping the two training signals
separate -- detach ``w(c)`` before using it to build the noise actually fed
to the DSM loss, and only let ``weight_fit_loss`` (built from this
module's ``project_onto_kernel_hull``) backprop into ``g_phi``.
"""

import torch

from fme.downscaling.noise import (
    brownian_bridge_mixing_matrix,
    ou_mixing_matrix,
    rbf_mixing_matrix,
)

KernelSpec = tuple[str, float | None]  # (kernel name, length_scale or None)


def _identity_mixing(
    n_timesteps: int, pin_endpoints: bool, device: torch.device
) -> torch.Tensor:
    if not pin_endpoints:
        return torch.eye(n_timesteps, device=device)
    n_interior = n_timesteps - 2
    mixing = torch.zeros(n_timesteps, n_timesteps, device=device)
    mixing[1 : 1 + n_interior, 1 : 1 + n_interior] = torch.eye(
        n_interior, device=device
    )
    return mixing


def _mixing_for_spec(
    tau: torch.Tensor, spec: KernelSpec, pin_endpoints: bool
) -> torch.Tensor:
    kernel, length_scale = spec
    if kernel == "independent":
        return _identity_mixing(tau.shape[0], pin_endpoints, tau.device)
    if kernel == "brownian_bridge":
        if not pin_endpoints:
            raise ValueError("brownian_bridge requires pin_endpoints=True.")
        return brownian_bridge_mixing_matrix(tau)
    if kernel == "ou":
        assert length_scale is not None
        return ou_mixing_matrix(tau, length_scale, pin_endpoints)
    if kernel == "rbf":
        assert length_scale is not None
        return rbf_mixing_matrix(tau, length_scale, pin_endpoints)
    raise ValueError(f"Unknown kernel {kernel!r}")


def build_kernel_basis(
    tau: torch.Tensor, specs: list[KernelSpec], pin_endpoints: bool
) -> torch.Tensor:
    """``(B, T, T)`` stack of trace-normalized PSD covariance matrices, one
    per entry of ``specs``.

    Each ``K^(b) = M_b @ M_b^T`` where ``M_b`` is a unit-diagonal
    (on its support) correlation-style mixing matrix from ``noise.py``'s
    kernel builders, so every basis element already has the same trace
    (``n_timesteps`` if ``pin_endpoints=False``, ``n_timesteps - 2`` if
    ``True``) by construction -- no separate normalization step needed
    (see ``conditional_kernel_theory.md``'s Setup: "each trace-normalized").

    Args:
        tau: Normalized frame times, ``(T,)``, as in ``noise.uniform_frame_times``.
        specs: Basis kernel specs, e.g. ``[("independent", None),
            ("brownian_bridge", None), ("ou", 0.5)]``. Should be linearly
            independent (as ``(T, T)`` vectors) for the weights to be
            identifiable (Prop. 4) -- picking meaningfully different
            kernels/length-scales is sufficient in practice.
        pin_endpoints: Whether every basis kernel is endpoint-pinned (the
            ``r``-block case) or stationary/full-grid (the ``d``-block
            case). Must match how the basis will be used downstream.
    """
    mats = [_mixing_for_spec(tau, spec, pin_endpoints) for spec in specs]
    mixing = torch.stack(mats, dim=0)  # (B, T, T)
    return mixing @ mixing.transpose(-1, -2)


class ConditionEncoder(torch.nn.Module):
    """Tiny MLP ``g_phi``: cheap per-channel condition statistics -> simplex
    weights over a fixed kernel basis.

    See ``conditional_kernel_theory.md``'s "Practical parameterization
    notes": "tiny MLP/CNN on cheap statistics of (x_0^A, x_0^B)... per-
    channel weights are consistent with the per-channel matched-kernel
    evidence" -- this v1 keeps ONE shared weight vector per block (not per
    physical channel, matching two_block's existing "fixed kernel, not
    per-channel" scope), fed by per-channel condition FEATURES (richer
    input, coarser output). Per-channel weights are a natural follow-up
    once per-channel moment estimates are reliable enough.
    """

    def __init__(self, n_channels: int, n_basis: int, hidden: int = 16):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_channels, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, n_basis),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features``: ``(B, n_channels)`` -> ``weights``: ``(B, n_basis)``,
        each row on the simplex.
        """
        return torch.softmax(self.net(features), dim=-1)


def condition_features(coarse_clip: torch.Tensor) -> torch.Tensor:
    """Cheap per-channel condition statistics from the two coarse endpoints:
    mean absolute endpoint difference over space, per channel -- the
    simplest instance of ``conditional_kernel_theory.md``'s
    ``||x_0^B - x_0^A||`` per-channel suggestion.

    Args:
        coarse_clip: ``(B, C, T, Hc, Wc)``, endpoints at index 0 and -1.

    Returns:
        ``(B, C)``.
    """
    diff = coarse_clip[:, :, -1] - coarse_clip[:, :, 0]  # (B, C, Hc, Wc)
    return diff.abs().mean(dim=(-2, -1))


def empirical_covariance(residual: torch.Tensor) -> torch.Tensor:
    """``(T, T)`` empirical second-moment matrix of ``residual`` ``(B, C, T,
    H, W)``, pooling batch/channel/spatial dims as independent observations
    of the ``T``-length residual vector -- a minibatch-level moment
    estimate (``conditional_kernel_theory.md``'s DSM-caveat resolution,
    option 2: "Sigma_hat from the current minibatch's residuals", rather
    than a persistent cross-dataset binning scheme). Assumes mean-zero
    residuals, true for the ``r``/``d`` targets by construction (see
    ``twoblock.py``).
    """
    flat = residual.permute(0, 1, 3, 4, 2).reshape(-1, residual.shape[2])  # (N, T)
    return (flat.T @ flat) / flat.shape[0]


def _project_to_simplex(v: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of a vector onto the probability simplex
    (Held/Wolfe/Crowder 1974; Duchi et al. 2008's ``O(n log n)`` algorithm).
    """
    n = v.shape[0]
    sorted_v, _ = torch.sort(v, descending=True)
    cssv = torch.cumsum(sorted_v, dim=0) - 1.0
    idx = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = sorted_v - cssv / idx > 0
    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = cssv[rho] / (rho + 1.0)
    return torch.clamp(v - theta, min=0.0)


def project_onto_kernel_hull(
    basis: torch.Tensor, sigma_hat: torch.Tensor, n_iters: int = 2000
) -> torch.Tensor:
    """``w_dagger = argmin_{w in simplex} ||sum_b w_b K^(b) - Sigma_hat||_F^2``
    (Prop. 4 of ``conditional_kernel_theory.md``): a small
    (``B``-dimensional, ``B <~ 8``) strongly convex QP, solved by projected
    gradient descent with an exact Euclidean simplex projection at each
    step -- ``B`` is small enough that a dedicated NNLS/QP solver isn't
    necessary.

    Args:
        basis: ``(n_basis, T, T)``, e.g. from ``build_kernel_basis``.
        sigma_hat: ``(T, T)``, e.g. from ``empirical_covariance``.
        n_iters: Projected-gradient steps. Each step is a ``(B, B)``
            mat-vec (``B`` tiny, ~4-8) plus an ``O(B log B)`` simplex
            projection, so even thousands of iterations cost microseconds
            -- convergence speed is governed by the basis Gram matrix's
            condition number (similar-shaped kernels, e.g. OU at nearby
            length scales, are closer to collinear and converge slower),
            not by ``B`` itself, so the default is set generously rather
            than tuned to a specific basis.

    Returns:
        A DETACHED ``(n_basis,)`` simplex vector -- always used as a fixed
        regression target, never differentiated through (see this module's
        docstring on why weight fitting must stay separate from the DSM
        graph).
    """
    n_basis = basis.shape[0]
    gram = torch.einsum("bij,cij->bc", basis, basis)  # (n_basis, n_basis)
    target = torch.einsum("bij,ij->b", basis, sigma_hat)  # (n_basis,)
    lipschitz = torch.linalg.eigvalsh(gram)[-1].clamp(min=1e-8)
    step = 1.0 / lipschitz
    w = torch.full((n_basis,), 1.0 / n_basis, device=basis.device, dtype=basis.dtype)
    for _ in range(n_iters):
        grad = gram @ w - target
        w = _project_to_simplex(w - step * grad)
    return w.detach()
