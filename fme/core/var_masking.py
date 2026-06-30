import dataclasses

import torch


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Synthetic training-time input variable masking.

    Combines two masking mechanisms into a single mask that is broadcast across
    the whole batch (shape ``[1, n_channels]``), so every sample and every
    ensemble member receives the same mask. The two mechanisms govern disjoint
    sets of channels (Bernoulli takes precedence over uniform):

    1. Per-variable Bernoulli masking: for each variable named in
       ``variable_masking_rates`` a ``Bernoulli(rate)`` is drawn; the channel is
       dropped iff its draw fires. These channels are governed *solely* by their
       rate, so the marginal probability that such a channel is dropped equals
       its configured ``rate`` exactly.
    2. Uniform masking over the remaining (unnamed) channels: a count ``k`` is
       drawn uniformly from ``[0, min(max_masked_vars, n_uniform)]`` where
       ``n_uniform`` is the number of channels with no configured rate, and
       ``k`` of those channels are dropped. The minimum dropped count is
       hard-coded to 0, so a draw may drop nothing.

    Because the two channel sets are disjoint, the mechanisms are fully
    decoupled: a configured ``rate`` is the true per-variable drop probability,
    independent of ``max_masked_vars``.

    Parameters:
        max_masked_vars: Maximum number of uniformly-masked variables. The
            count is sampled uniformly from ``[0, max_masked_vars]`` (capped at
            the number of uniformly-maskable channels), so a draw may mask no
            variables.
        variable_masking_rates: Optional mapping of variable name to Bernoulli
            masking rate in ``[0, 1]``. Names not present among the input
            channels are ignored. Named channels are excluded from the uniform
            pool.
    """

    max_masked_vars: int = 0
    variable_masking_rates: dict[str, float] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.max_masked_vars, int)
            or isinstance(self.max_masked_vars, bool)
            or self.max_masked_vars < 0
        ):
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
        present = torch.ones(1, n, dtype=torch.bool, device=device)
        rates = self.variable_masking_rates or {}

        # Bernoulli channels use their rate and are excluded from uniform masking.
        uniform_pool = torch.tensor(
            [i for i, name in enumerate(names) if name not in rates],
            device=device,
            dtype=torch.long,
        )
        n_uniform = uniform_pool.numel()
        max_n = min(self.max_masked_vars, n_uniform)
        # Uniform count of dropped channels in [0, max_n], over the pool only.
        k = int(torch.randint(0, max_n + 1, (1,), device=device).item())
        perm = torch.randperm(n_uniform, device=device)
        present[0, uniform_pool[perm[:k]]] = False

        # Per-variable Bernoulli masking over the named channels.
        for i, name in enumerate(names):
            rate = rates.get(name)
            if rate is None:
                continue
            if bool(torch.rand((), device=device) < rate):
                present[0, i] = False
        return present
