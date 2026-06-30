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
       fire are dropped.

    The two mechanisms combine by logical OR: a channel is dropped if the
    uniform draw selected it or its Bernoulli fired. The total dropped count is
    therefore at least ``k`` and may exceed it (up to ``n_channels``).

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
        present = torch.ones(1, n, dtype=torch.bool, device=device)
        # Uniform drop set: the first k of a random permutation.
        perm = torch.randperm(n, device=device)
        present[0, perm[:k]] = False

        # Per-variable Bernoulli masking, OR-combined with the uniform drop set.
        rates = self.variable_masking_rates or {}
        for i, name in enumerate(names):
            rate = rates.get(name)
            if rate is None:
                continue
            if bool(torch.rand((), device=device) < rate):
                present[0, i] = False
        return present
