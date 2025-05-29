import torch


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
    if gen.shape[1] != 2:
        raise NotImplementedError(
            "CRPS loss is written here specifically for 2 ensemble members, "
            f"got {gen.shape[1]} ensemble members. "
            "Update this function (and its tests) to support more."
        )
    # CRPS is `E[|X - y|] - 1/2 E[|X - X'|]`
    # below we compute the first term as the average of two ensemble members
    # meaning the 0.5 factor can be pulled out
    alpha = 0.95
    epsilon = (1 - alpha) / 2
    target_term = torch.abs(gen - target).mean(axis=1)
    internal_term = -0.5 * torch.abs(gen[:, 0, ...] - gen[:, 1, ...])
    return target_term + (1 - epsilon) * internal_term


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
