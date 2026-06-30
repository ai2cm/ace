import dataclasses

import torch


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Synthetic training-time input variable masking.

    Combines two masking mechanisms into a single mask that is broadcast across
    the whole batch (shape ``[1, n_channels]``), so every sample and every
    ensemble member receives the same mask:

    1. Uniform masking: a count ``k`` is drawn uniformly from
       ``[0, min(max_masked_vars, n_channels)]`` and ``k`` random channels are
       dropped. The minimum dropped count is hard-coded to 0, so a draw may
       drop nothing.
    2. Per-variable Bernoulli masking: for each variable named in
       ``variable_masking_rates`` a ``Bernoulli(rate)`` is drawn; channels that
       fire are guaranteed to be dropped (subject to the edge cases below).

    The two mechanisms do not OR/AND. Instead each fired Bernoulli channel that
    is not already in the uniform drop set evicts a randomly-chosen channel that
    is in the drop set and is not itself a fired Bernoulli channel, then takes
    its place. The total dropped count therefore stays equal to the uniform
    count ``k``.

    Edge cases:
        - If ``k == 0`` there are no slots to evict, so no Bernoulli var can be
          dropped (replacement is a no-op).
        - If more Bernoulli vars fire than there are evictable (non-fired)
          uniform slots, only ``k``-worth are dropped; which fired vars win is
          random.

    Parameters:
        max_masked_vars: Maximum number of uniformly-masked variables. The
            count is sampled uniformly from ``[0, max_masked_vars]`` (capped at
            the number of input channels), so a draw may mask no variables.
        variable_masking_rates: Optional mapping of variable name to Bernoulli
            masking rate in ``[0, 1]``. Names not present among the input
            channels are ignored.
    """

    max_masked_vars: int = 0
    variable_masking_rates: dict[str, float] | None = None

    def __post_init__(self):
        if not isinstance(self.max_masked_vars, int) or self.max_masked_vars < 0:
            raise ValueError(
                "max_masked_vars must be a non-negative int, got "
                f"{self.max_masked_vars!r}"
            )
        if self.variable_masking_rates is not None:
            for name, rate in self.variable_masking_rates.items():
                if not 0.0 <= rate <= 1.0:
                    raise ValueError(
                        f"variable_masking_rates[{name!r}] must be in [0, 1], "
                        f"got {rate}"
                    )

    def sample_mask(self, names: list[str], device: torch.device) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[1, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        The leading dimension is 1 so the mask broadcasts over the batch (and
        hence over ensemble members) when applied.
        """
        n = len(names)
        max_n = min(self.max_masked_vars, n)
        # Uniform count of dropped channels in [0, max_n].
        k = int(torch.randint(0, max_n + 1, (1,), device=device).item())
        # Uniform drop set: the first k of a random permutation.
        perm = torch.randperm(n, device=device)
        drop = set(perm[:k].tolist())

        # Bernoulli-fired channels.
        fired: set[int] = set()
        rates = self.variable_masking_rates or {}
        for i, name in enumerate(names):
            rate = rates.get(name)
            if rate is None:
                continue
            if bool(torch.rand((), device=device) < rate):
                fired.add(i)

        # Combine by replacement: each fired channel not already dropped evicts
        # a non-fired channel from the drop set and takes its place.
        fired_to_add = sorted(f for f in fired if f not in drop)
        evictable = sorted(d for d in drop if d not in fired)
        if fired_to_add and evictable:
            n_replace = min(len(fired_to_add), len(evictable))
            add_idx = torch.randperm(len(fired_to_add), device=device)[:n_replace]
            evict_idx = torch.randperm(len(evictable), device=device)[:n_replace]
            for a, e in zip(add_idx.tolist(), evict_idx.tolist()):
                drop.discard(evictable[e])
                drop.add(fired_to_add[a])

        present = torch.ones(1, n, dtype=torch.bool, device=device)
        if drop:
            idx = torch.tensor(sorted(drop), device=device, dtype=torch.long)
            present[0, idx] = False
        return present
