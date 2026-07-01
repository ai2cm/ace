import dataclasses

import torch


@dataclasses.dataclass
class MaskingGroupConfig:
    """
    A group of input variables that share a single Bernoulli masking draw.

    One ``Bernoulli(rate)`` is drawn for the whole group; when it fires every
    variable in ``variables`` is dropped together. A group of a single variable
    reproduces per-variable masking.

    Parameters:
        variables: Names of the input channels in this group. Names not present
            among the input channels are ignored.
        rate: Bernoulli masking rate in ``[0, 1]`` shared by the whole group.
    """

    variables: list[str]
    rate: float


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Synthetic training-time input variable masking.

    Combines two masking mechanisms into a single mask that is broadcast across
    the whole batch (shape ``[1, n_channels]``), so every sample and every
    ensemble member receives the same mask. The two mechanisms govern disjoint
    sets of channels (grouped Bernoulli takes precedence over uniform):

    1. Per-group Bernoulli masking: for each group in ``variable_masking_groups``
       a single ``Bernoulli(rate)`` is drawn; when it fires every variable in the
       group is dropped together. These channels are governed *solely* by their
       group's rate, so the marginal probability that a group is dropped equals
       its configured ``rate`` exactly.
    2. Uniform masking over the remaining (ungrouped) channels: a count ``k`` is
       drawn uniformly from ``[0, min(max_masked_vars, n_uniform)]`` where
       ``n_uniform`` is the number of channels in no group, and ``k`` of those
       channels are dropped. The minimum dropped count is hard-coded to 0, so a
       draw may drop nothing.

    Because the two channel sets are disjoint, the mechanisms are fully
    decoupled: a configured group ``rate`` is the true per-group drop
    probability, independent of ``max_masked_vars``.

    Parameters:
        max_masked_vars: Maximum number of uniformly-masked variables. The
            count is sampled uniformly from ``[0, max_masked_vars]`` (capped at
            the number of uniformly-maskable channels), so a draw may mask no
            variables.
        variable_masking_groups: Optional list of variable groups, each with a
            shared Bernoulli masking rate. Grouped channels are excluded from
            the uniform pool. A variable may appear in at most one group.
    """

    max_masked_vars: int = 0
    variable_masking_groups: list[MaskingGroupConfig] | None = None

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
        if self.variable_masking_groups is not None:
            seen: set[str] = set()
            for group in self.variable_masking_groups:
                if not 0.0 <= group.rate <= 1.0:
                    raise ValueError(
                        f"masking group rate must be in [0, 1], got {group.rate}"
                    )
                if len(group.variables) == 0:
                    raise ValueError("masking group variables must be non-empty")
                for name in group.variables:
                    if name in seen:
                        raise ValueError(
                            f"variable {name!r} appears in more than one masking group"
                        )
                    seen.add(name)

    def sample_mask(self, names: list[str], device: torch.device) -> torch.Tensor:
        """
        Sample a boolean presence mask of shape ``[1, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        The leading dimension is 1 so the mask broadcasts over the batch (and
        hence over ensemble members) when applied.
        """
        n = len(names)
        present = torch.ones(1, n, dtype=torch.bool, device=device)
        groups = self.variable_masking_groups or []

        # Map each channel to its group index (-1 if ungrouped); no RNG sync.
        name_to_idx = {name: i for i, name in enumerate(names)}
        group_id = torch.full((n,), -1, dtype=torch.long, device=device)
        for g, group in enumerate(groups):
            for name in group.variables:
                if name in name_to_idx:
                    group_id[name_to_idx[name]] = g

        # Grouped channels are excluded from uniform masking; pool is ungrouped.
        uniform_pool = (group_id == -1).nonzero(as_tuple=True)[0]
        n_uniform = uniform_pool.numel()
        max_n = min(self.max_masked_vars, n_uniform)
        # Uniform count of dropped channels in [0, max_n], over the pool only.
        k = int(torch.randint(0, max_n + 1, (1,), device=device).item())
        perm = torch.randperm(n_uniform, device=device)
        present[0, uniform_pool[perm[:k]]] = False

        # Draw one Bernoulli per group; drop every member of a fired group.
        if len(groups) > 0:
            rates = torch.tensor(
                [group.rate for group in groups],
                device=device,
                dtype=torch.float,
            )
            fired = torch.rand(len(groups), device=device) < rates
            grouped = group_id >= 0
            # clamp -1 -> 0 to keep the gather in-range; grouped masks the bogus lookup
            present[0] &= ~(grouped & fired[group_id.clamp(min=0)])
        return present
