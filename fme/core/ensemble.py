import torch
import torch.nn.functional as F


def get_crps(
    gen: torch.Tensor, target: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """
    Compute the CRPS loss for a single variable at a single timestep.

    Supports almost-fair modification to CRPS from
    https://arxiv.org/html/2412.15832v1, which claims to be helpful in
    avoiding numerical issues with fair CRPS.

    Args:
        gen: The generated ensemble members, of shape [n_batch, n_ensemble, ...].
        target: The target, of shape [n_batch, 1, ...].
        alpha: The alpha value for the CRPS loss. Corresponds to the alpha value
            for "almost fair" CRPS from https://arxiv.org/html/2412.15832v1. Default
            behavior uses fair CRPS (alpha=1.0).

    Returns:
        The CRPS loss.
    """
    n_ens = gen.shape[1]
    epsilon = (1.0 - alpha) / 2.0

    # Term 1: E|X - y|
    target_term = torch.mean(torch.abs(gen - target), dim=1)

    if n_ens == 1:
        internal_term = torch.zeros_like(target_term)
    else:
        # Indices for unique pairs i < j
        idx = torch.triu_indices(n_ens, n_ens, offset=1, device=gen.device)
        i, j = idx[0], idx[1]  # [n_pairs]

        # Only materialize the needed pairs: [B, n_pairs, ...]
        pairwise = (gen[:, i, ...] - gen[:, j, ...]).abs()

        # Mean over pairs
        internal_term = -0.5 * pairwise.mean(dim=1)

    crps = target_term + (1.0 - epsilon) * internal_term
    return crps


def get_energy_score(
    gen: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the energy score for a single complex-valued variable at a single
    timestep.

    The energy score is defined as

    .. math::

        E[||X - y||^{beta}] - 1/2 E[||X - X'||^{beta}]

    where :math:`X` is the ensemble, :math:`y` is the target, and :math:`||.||`
    is the complex modulus. It is a proper scoring rule for beta in (0, 2). Here
    we use beta=1. See Gneiting and Raftery (2007) [1]_ Section 4.3 for more details.

    Args:
        target: The target tensor without a sample dimension
        prediction: The prediction tensor with a sample dimension
        sample_dim: The dimension of `prediction` corresponding to sample.

    .. [1] https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf

    Args:
        gen: The complex-valued generated ensemble members, of shape
            [n_batch, n_ensemble, ...].
        target: The complex-valued target, of shape [n_batch, 1, ...].

    Returns:
        The energy score.
    """
    if gen.shape[1] != 2:
        raise NotImplementedError(
            "Energy score is written here specifically for 2 ensemble members, "
            f"got {gen.shape[1]} ensemble members. "
            "Update this function (and its tests) to support more."
        )
    # CRPS is `E[|X - y|] - 1/2 E[|X - X'|]`
    # below we compute the first term as the average of two ensemble members
    # meaning the 0.5 factor can be pulled out
    target_term = torch.abs(gen - target).mean(axis=1)
    internal_term = -0.5 * torch.abs(gen[:, 0, ...] - gen[:, 1, ...])
    return target_term + internal_term


def _patch_l2_distance(diff: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Compute the per-point Euclidean distance between the patches of two
    fields, given their pointwise difference.

    ||patch(u) - patch(v)||^2 at each point equals the patch_size x patch_size
    box-sum of (u - v)^2, because patch extraction is linear and the windows
    of the two fields are aligned; no patch tensor is ever materialized.

    Args:
        diff: The pointwise difference of two fields, of shape
            [..., n_lat, n_lon].
        patch_size: Odd width of the square patch in grid points.

    Returns:
        The patch distance, of the same shape as diff.
    """
    halo = patch_size // 2
    shape = diff.shape
    squared = (diff * diff).reshape(-1, 1, shape[-2], shape[-1])
    squared = F.pad(squared, (halo, halo, 0, 0), mode="circular")
    squared = F.pad(squared, (0, 0, halo, halo), mode="constant", value=0.0)
    box_sum = F.avg_pool2d(
        squared, kernel_size=patch_size, stride=1, divisor_override=1
    )
    # sqrt has an unbounded gradient at zero (identical members, constant
    # patches); clamping gives a zero subgradient there, like abs() in CRPS.
    return torch.sqrt(box_sum.clamp_min(1e-12)).reshape(shape)


def get_patch_energy_score(
    gen: torch.Tensor,
    target: torch.Tensor,
    patch_size: int = 3,
) -> torch.Tensor:
    """
    Compute a per-channel patch energy score.

    Each grid point is scored with the energy score of the
    patch_size x patch_size patch of a single channel centered on it, using
    the Euclidean norm over patch values:

    .. math::

        E[||X - y||] - 1/2 E[||X - X'||]

    The east-west boundary is periodic and the north-south boundary is
    zero-padded; since generated and target fields receive identical padding,
    out-of-domain entries contribute zero to every patch distance, so polar
    patches are effectively truncated to their valid entries.

    The result is divided by patch_size (the square root of the patch
    dimensionality) so its magnitude is comparable to CRPS; for
    patch_size=1 it equals fair CRPS exactly.

    Args:
        gen: The generated ensemble members, of shape
            [n_batch, n_ensemble, ..., n_lat, n_lon].
        target: The target, of shape [n_batch, 1, ..., n_lat, n_lon].
        patch_size: Odd width of the square patch in grid points.

    Returns:
        The patch energy score, of shape [n_batch, ..., n_lat, n_lon].
    """
    if gen.shape[1] != 2:
        raise NotImplementedError(
            "Patch energy score is written here specifically for 2 ensemble "
            f"members, got {gen.shape[1]} ensemble members. "
            "Update this function (and its tests) to support more."
        )
    if patch_size < 1 or patch_size % 2 == 0:
        raise ValueError(f"patch_size must be a positive odd integer, got {patch_size}")
    n_batch = gen.shape[0]
    # One pad + pool pass over all three difference fields, stacked on batch.
    diffs = torch.cat(
        [
            gen[:, 0] - target[:, 0],
            gen[:, 1] - target[:, 0],
            gen[:, 0] - gen[:, 1],
        ],
        dim=0,
    )
    distance = _patch_l2_distance(diffs, patch_size)
    target_term = 0.5 * (distance[:n_batch] + distance[n_batch : 2 * n_batch])
    internal_term = -0.5 * distance[2 * n_batch :]
    return (target_term + internal_term) / patch_size
