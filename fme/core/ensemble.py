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
