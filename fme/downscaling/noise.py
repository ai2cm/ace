import abc
import dataclasses
from collections.abc import Callable

import torch

from fme.core.rand import log_normal_sample, log_uniform_sample, randn_like


@dataclasses.dataclass
class ConditionedTarget:
    """
    A class to hold the conditioned targets and the loss weighting.

    Attributes:
        latents: The normalized targets with noise added.
        sigma: The noise level.
        weight: The loss weighting.
    """

    latents: torch.Tensor
    sigma: torch.Tensor
    weight: torch.Tensor


class NoiseDistribution(abc.ABC):
    @abc.abstractmethod
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        pass


@dataclasses.dataclass
class LogNormalNoiseDistribution(NoiseDistribution):
    p_mean: float
    p_std: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return log_normal_sample(
            p_mean=self.p_mean,
            p_std=self.p_std,
            shape=(batch_size, 1, 1, 1),
            dtype=torch.float32,
        ).to(device)


@dataclasses.dataclass
class LogUniformNoiseDistribution(NoiseDistribution):
    p_min: float
    p_max: float

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return log_uniform_sample(
            p_min=self.p_min,
            p_max=self.p_max,
            shape=(batch_size, 1, 1, 1),
            dtype=torch.float32,
        ).to(device)


def uniform_frame_times(n_timesteps: int) -> torch.Tensor:
    """Normalized times of ``n_timesteps`` uniformly spaced frames on ``[0, 1]``.

    Endpoints land exactly on 0 and 1; interior frames on ``k / (T - 1)``.
    """
    return torch.linspace(0.0, 1.0, n_timesteps, dtype=torch.float64)


def brownian_bridge_mixing_matrix(tau: torch.Tensor) -> torch.Tensor:
    """Temporal mixing matrix for endpoint-pinned, time-correlated noise.

    Given the normalized frame times ``tau`` (1-D, length ``T``, with
    ``tau[0] == 0`` and ``tau[-1] == 1`` marking the pinned endpoints and the
    interior strictly inside ``(0, 1)``), returns a ``(T, T)`` matrix ``M`` such
    that white noise ``Z ~ N(0, I)`` mixed along the time axis as ``E = M @ Z``
    has a Brownian-bridge *correlation* structure across the interior frames and
    is exactly zero on the two endpoint frames.

    The Brownian-bridge covariance ``k_BB(s, t) = min(s, t) - s * t`` (bridge
    length 1) vanishes at both endpoints, matching the endpoint-pinned residual
    the video model diffuses. We normalize it to a *correlation* matrix (unit
    diagonal on the interior) so that, when scaled by the EDM noise level
    ``sigma``, each frame's marginal noise std stays ``sigma`` -- only the
    cross-time correlation is introduced. ``M`` is the Cholesky factor of that
    correlation matrix placed into the interior block; the endpoint rows/cols are
    all zero.

    Because every entry depends only on its own pair ``(tau_i, tau_j)``, the
    matrix built for any *subset* of frame times equals the full-grid matrix
    restricted to those frames. Generating a subset of frames therefore draws
    from the exact *marginal* of the full-window bridge -- the noise obeys the
    same process whether or not the other frames are requested.

    Args:
        tau: Normalized frame times, shape ``(T,)`` with ``T >= 3``, sorted with
            endpoints at 0 and 1 and interior values in ``(0, 1)``.

    Returns:
        A ``(T, T)`` float32 tensor on ``tau``'s device.
    """
    if tau.ndim != 1 or tau.shape[0] < 3:
        raise ValueError(
            "Brownian-bridge noise needs at least 3 frames (2 endpoints + 1 "
            f"interior); got tau with shape {tuple(tau.shape)}."
        )
    n_timesteps = tau.shape[0]
    n_interior = n_timesteps - 2
    tau_i = tau[1:-1].to(torch.float64)
    s = tau_i.reshape(-1, 1)
    t = tau_i.reshape(1, -1)
    cov = torch.minimum(s, t) - s * t  # interior Brownian-bridge covariance
    std = cov.diagonal().sqrt()
    corr = cov / torch.outer(std, std)  # normalize to unit diagonal
    chol = torch.linalg.cholesky(corr)  # lower-triangular, corr == chol @ chol.T
    mixing = torch.zeros(
        n_timesteps, n_timesteps, dtype=torch.float64, device=tau.device
    )
    mixing[1 : 1 + n_interior, 1 : 1 + n_interior] = chol
    return mixing.to(torch.float32)


def _interior_cholesky_mixing_matrix(
    tau: torch.Tensor, corr_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> torch.Tensor:
    """Shared scaffold for endpoint-pinned mixing matrices: builds the
    unit-diagonal interior correlation matrix via ``corr_fn(tau_i, tau_j)``,
    Cholesky-factors it, and embeds it in a zero-padded ``(T, T)`` matrix
    (endpoint rows/cols zero). Used by ``brownian_bridge_mixing_matrix``,
    ``ou_mixing_matrix``, and ``rbf_mixing_matrix`` -- they differ only in
    ``corr_fn``.
    """
    if tau.ndim != 1 or tau.shape[0] < 3:
        raise ValueError(
            "Endpoint-pinned noise needs at least 3 frames (2 endpoints + 1 "
            f"interior); got tau with shape {tuple(tau.shape)}."
        )
    n_timesteps = tau.shape[0]
    n_interior = n_timesteps - 2
    tau_i = tau[1:-1].to(torch.float64)
    s = tau_i.reshape(-1, 1)
    t = tau_i.reshape(1, -1)
    corr = corr_fn(s, t)
    chol = torch.linalg.cholesky(corr)  # lower-triangular, corr == chol @ chol.T
    mixing = torch.zeros(
        n_timesteps, n_timesteps, dtype=torch.float64, device=tau.device
    )
    mixing[1 : 1 + n_interior, 1 : 1 + n_interior] = chol
    return mixing.to(torch.float32)


def ou_mixing_matrix(tau: torch.Tensor, length_scale: float) -> torch.Tensor:
    """Temporal mixing matrix for endpoint-pinned, Ornstein-Uhlenbeck-
    correlated noise: ``corr(s, t) = exp(-|s - t| / length_scale)``.

    Same endpoint-pinned-interior-Cholesky structure as
    ``brownian_bridge_mixing_matrix`` (see its docstring for the subset-
    marginal property, which holds here too since OU correlation is also a
    pure function of each frame-time pair), but with an OU kernel instead of
    the Brownian-bridge kernel. Empirically (see
    ``toy/process_residual_bridge_report.md``), real interior-frame
    correlations for wind/precipitation channels decay more like OU than
    like a bridge.

    Args:
        tau: Normalized frame times, shape ``(T,)``, as in
            ``brownian_bridge_mixing_matrix``.
        length_scale: OU decorrelation length, in the same units as ``tau``
            (hours, for this codebase's frame times).
    """
    return _interior_cholesky_mixing_matrix(
        tau, lambda s, t: torch.exp(-torch.abs(s - t) / length_scale)
    )


def rbf_mixing_matrix(tau: torch.Tensor, length_scale: float) -> torch.Tensor:
    """Temporal mixing matrix for endpoint-pinned, RBF/squared-exponential-
    correlated noise: ``corr(s, t) = exp(-(s - t)^2 / (2 * length_scale^2))``.

    See ``ou_mixing_matrix`` for the shared structure. **Caution:** RBF
    kernels become numerically near-singular fast as ``length_scale`` grows
    relative to the frame spacing -- verify ``det`` isn't degenerate before
    using this in training (see ``toy/process_residual_bridge_report.md``:
    RBF was empirically the best-fitting kernel for PRMSL and T2m, but its
    determinant there is ~1e-16 / ~1e-11, i.e. effectively rank-deficient --
    OU was used for those channels instead for exactly this reason).

    Args:
        tau: Normalized frame times, shape ``(T,)``, as in
            ``brownian_bridge_mixing_matrix``.
        length_scale: RBF length scale, in the same units as ``tau``.
    """
    return _interior_cholesky_mixing_matrix(
        tau, lambda s, t: torch.exp(-((s - t) ** 2) / (2 * length_scale**2))
    )


def condition_with_noise_for_training(
    targets_norm: torch.Tensor,
    noise_distribution: NoiseDistribution,
    sigma_data: float,
    loss_weight_exponent: float = 1.0,
) -> ConditionedTarget:
    """
    Condition the targets with noise for training.

    Args:
        targets_norm: The normalized targets.
        noise_distribution: The noise distribution to use for conditioning.
        sigma_data: The standard deviation of the data,
            used to determine loss weighting.
        loss_weight_exponent: Exponent applied to the base EDM loss weight
            ``(sigma^2 + sigma_data^2) / (sigma * sigma_data)^2``. The default
            of 1.0 gives the standard EDM weighting (~1/sigma^2 for small
            sigma). Use 0.5 for ~1/sigma weighting (square root of EDM weight).

    Returns:
        The conditioned targets and the loss weighting.
    """
    sigma = noise_distribution.sample(targets_norm.shape[0], targets_norm.device)
    weight = (
        (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2
    ) ** loss_weight_exponent
    noise = randn_like(targets_norm) * sigma
    latents = targets_norm + noise
    return ConditionedTarget(latents=latents, sigma=sigma, weight=weight)
