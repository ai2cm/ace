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


def get_variogram_edge_offsets(window_size: int) -> list[tuple[int, int]]:
    """
    Return the edge kinds of the variogram score for a square window.

    An edge kind is an index offset ``(di, dj)`` from a grid point to a
    neighbor within the ``window_size x window_size`` window centered on it,
    where ``di`` steps the latitude index and ``dj`` the longitude index. Only
    one of each mirror pair ``(di, dj)`` / ``(-di, -dj)`` is kept (the
    half-plane ``di > 0``, or ``di == 0`` and ``dj > 0``), since both describe
    the same set of point pairs, and ``(0, 0)`` is excluded. A window of size
    ``w`` has ``(w**2 - 1) / 2`` edge kinds, e.g. 4 for ``w = 3`` (E, SW, S,
    SE) and 12 for ``w = 5``; a smaller window's kinds are a subset of a
    larger one's.

    Args:
        window_size: Odd width of the window in grid points, at least 3.

    Returns:
        The ``(di, dj)`` offsets, ordered by ``di`` then ``dj``.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError(
            f"window_size must be an odd integer of at least 3, got {window_size}"
        )
    halo = window_size // 2
    offsets = [(0, dj) for dj in range(1, halo + 1)]
    offsets += [(di, dj) for di in range(1, halo + 1) for dj in range(-halo, halo + 1)]
    return offsets


def get_variogram_score(
    gen: torch.Tensor,
    target: torch.Tensor,
    scale: torch.Tensor,
    offsets: list[tuple[int, int]],
    p: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute a per-channel, grid-local variogram score of order p with the fair
    two-member estimator.

    The variogram score of Scheuerer and Hamill (2015) [1]_ sums, over pairs
    of points, the squared difference between the observed and the expected
    ``|x_i - x_j|^p``. Here the pairs are all grid-point pairs of one channel
    separated by one of the ``offsets`` (see
    :func:`get_variogram_edge_offsets`). With ``D = |Δ / scale|^p`` for the
    increment ``Δ = x[i + di, j + dj] - x[i, j]``, each edge contributes

    .. math::

        (D_y - D_1)(D_y - D_2),

    where ``D_1``, ``D_2`` are the two ensemble members' values and ``D_y``
    the target's. For independent members this is unbiased for the
    population term ``(D_y - E_F[D])^2`` (it can be negative for one sample).

    Longitude (the last dimension) is periodic. No edge crosses a pole: an
    edge kind with latitude step ``di`` has ``n_lat - di`` valid rows. The
    result is the unweighted mean over all valid edges of all edge kinds
    (each kind contributes its valid-edge count; no area or distance
    weighting).

    ``|Δ / scale|^p`` has an unbounded derivative at zero for ``p < 1``, so it
    is evaluated as ``((Δ / scale)^2 + eps)^(p / 2)``, which keeps gradients
    finite for zero increments and identical members.

    .. [1] https://doi.org/10.1175/MWR-D-14-00269.1

    Args:
        gen: The generated ensemble members, of shape
            [n_batch, 2, ..., n_lat, n_lon].
        target: The target, of shape [n_batch, 1, ..., n_lat, n_lon].
        scale: The increment scale of each edge kind, of shape
            [n_edges, ...], where the trailing dims broadcast against the
            non-horizontal dims of gen after the ensemble dim (e.g.
            [n_edges, n_channel] for gen of shape
            [n_batch, 2, n_channel, n_lat, n_lon]).
        offsets: The ``(di, dj)`` edge kinds, one per entry of scale's first
            dim; ``di`` must be in [0, n_lat).
        p: Order of the variogram score.
        eps: Smoothing added to the squared scaled increment before the power.

    Returns:
        The variogram score, of shape [n_batch, ...] (horizontal dims reduced).
    """
    if gen.shape[1] != 2:
        raise NotImplementedError(
            "Variogram score is written here specifically for 2 ensemble "
            f"members, got {gen.shape[1]} ensemble members. "
            "Update this function (and its tests) to support more."
        )
    if scale.shape[0] != len(offsets):
        raise ValueError(
            f"scale has {scale.shape[0]} edge kinds but {len(offsets)} offsets "
            "were given"
        )
    n_lat = gen.shape[-2]
    n_lon = gen.shape[-1]
    # index 0 along dim 1 is the target, 1 and 2 are the members
    x = torch.cat([target, gen], dim=1)
    total: torch.Tensor | None = None
    n_edges = 0
    for k, (di, dj) in enumerate(offsets):
        if di < 0 or di >= n_lat:
            raise ValueError(f"latitude offset must be in [0, n_lat), got {di}")
        lower = x[..., : n_lat - di, :]
        upper = x[..., di:, :]
        if dj != 0:
            # rolled[..., j] = upper[..., j + dj], periodic in longitude
            upper = torch.roll(upper, shifts=-dj, dims=-1)
        scale_k = scale[k].reshape(*scale.shape[1:], 1, 1)
        scaled = (upper - lower) / scale_k
        d = (scaled * scaled + eps) ** (p / 2)
        score = (d[:, 0] - d[:, 1]) * (d[:, 0] - d[:, 2])
        edge_sum = score.sum(dim=(-2, -1))
        total = edge_sum if total is None else total + edge_sum
        n_edges += (n_lat - di) * n_lon
    assert total is not None
    return total / n_edges
