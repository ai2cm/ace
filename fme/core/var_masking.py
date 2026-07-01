import abc
import dataclasses

import torch


class MaskingGenerator(abc.ABC):
    """Owns a disjoint slice of channels and answers which it drops per step.

    Each generator is built for a fixed list of channel ``names`` (its private
    pool) and draws independently every step. The final presence mask assembled
    by :class:`VariableMasking` is the union of every generator's dropped names.
    """

    @abc.abstractmethod
    def sample(self, device: torch.device) -> list[str]:
        """Return the names this generator drops on this step."""


class BernoulliMaskingGenerator(MaskingGenerator):
    """Drops all owned channels together with probability ``rate``.

    One ``Bernoulli(rate)`` is drawn for the whole pool; when it fires every
    name is dropped, otherwise none are. The marginal drop probability of the
    pool equals ``rate`` exactly.
    """

    def __init__(self, names: list[str], rate: float):
        self._names = list(names)
        self._rate = rate

    def sample(self, device: torch.device) -> list[str]:
        fired = bool((torch.rand(1, device=device) < self._rate).item())
        return list(self._names) if fired else []


class UniformMaskingGenerator(MaskingGenerator):
    """Drops ``k`` random owned channels, ``k`` uniform in ``[0, max]``.

    ``k`` is drawn uniformly from ``[0, min(max_masked_vars, n)]`` where ``n``
    is the pool size, so a draw may drop nothing.
    """

    def __init__(self, names: list[str], max_masked_vars: int):
        self._names = list(names)
        self._max_masked_vars = max_masked_vars

    def sample(self, device: torch.device) -> list[str]:
        n = len(self._names)
        max_n = min(self._max_masked_vars, n)
        k = int(torch.randint(0, max_n + 1, (1,), device=device).item())
        if k == 0:
            return []
        perm = torch.randperm(n, device=device)[:k]
        return [self._names[i] for i in perm.tolist()]


@dataclasses.dataclass
class BernoulliMaskingConfig:
    """All-or-nothing Bernoulli masking of a channel pool.

    Parameters:
        rate: Bernoulli masking rate in ``[0, 1]``; the marginal probability
            the pool is dropped on any step.
    """

    rate: float

    def __post_init__(self):
        if not 0.0 <= self.rate <= 1.0:
            raise ValueError(f"masking rate must be in [0, 1], got {self.rate}")

    def build(self, names: list[str]) -> BernoulliMaskingGenerator:
        return BernoulliMaskingGenerator(names, self.rate)


@dataclasses.dataclass
class UniformMaskingConfig:
    """Uniform masking of a random count of channels from a pool.

    Parameters:
        max_masked_vars: Maximum number of masked channels. The count is
            sampled uniformly from ``[0, min(max_masked_vars, n)]`` where ``n``
            is the pool size, so a draw may mask no variables.
    """

    max_masked_vars: int

    def __post_init__(self):
        # bool is a subclass of int, so reject it explicitly to avoid a
        # silently-coerced True/False being treated as a masked-var count.
        if (
            not isinstance(self.max_masked_vars, int)
            or isinstance(self.max_masked_vars, bool)
            or self.max_masked_vars < 0
        ):
            raise ValueError(
                "max_masked_vars must be a non-negative int, got "
                f"{self.max_masked_vars!r}"
            )

    def build(self, names: list[str]) -> UniformMaskingGenerator:
        return UniformMaskingGenerator(names, self.max_masked_vars)


# Disjoint required fields (`rate` vs `max_masked_vars`) let dacite discriminate
# this union under strict=True, matching the SchedulerConfig precedent.
MaskingConfig = BernoulliMaskingConfig | UniformMaskingConfig


@dataclasses.dataclass
class MaskingGroupConfig:
    """A named group of variables masked together by one generator.

    Parameters:
        variables: Names of the input channels in this group. Every name must
            match a packed input channel (validated by
            ``VariableMaskingConfig.build``); a typo raises rather than silently
            masking nothing.
        masking: The masking scheme applied to this group's channels.
    """

    variables: list[str]
    masking: MaskingConfig

    def __post_init__(self):
        if len(self.variables) == 0:
            raise ValueError("masking group variables must be non-empty")


@dataclasses.dataclass
class VariableMaskingConfig:
    """Synthetic training-time input variable masking.

    Each configured masking scheme owns a disjoint slice of channels and answers
    which of its channels to drop this step; the final mask is the union of all
    outputs, broadcast across the batch (shape ``[1, n_channels]``) so every
    sample and ensemble member is masked identically.

    Channels named in an ``override_groups`` entry are governed solely by that
    group's masking scheme; all remaining (ungrouped) channels are governed by
    ``default``. Because the pools are disjoint, the schemes are fully decoupled:
    a group ``rate`` is the true per-group drop probability regardless of
    ``default``.

    Parameters:
        default: Masking scheme for ungrouped channels. Defaults to
            ``UniformMaskingConfig(0)``, which masks no ungrouped channels.
        override_groups: Optional list of variable groups, each with its own
            masking scheme. A variable may appear in at most one group.
    """

    default: MaskingConfig = dataclasses.field(
        default_factory=lambda: UniformMaskingConfig(0)
    )
    override_groups: list[MaskingGroupConfig] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        seen: set[str] = set()
        for group in self.override_groups:
            for name in group.variables:
                if name in seen:
                    raise ValueError(
                        f"variable {name!r} appears in more than one masking group"
                    )
                seen.add(name)

    def build(self, names: list[str]) -> "VariableMasking":
        """Build the runtime masking object for the packed channel ``names``.

        Raises ``ValueError`` naming any grouped variable absent from ``names``
        so a typo fails loudly at build time rather than silently masking
        nothing. ``names`` is the authoritative packed channel set (input
        channels plus any GMR extra sentinels).
        """
        valid = set(names)
        unknown = [
            name
            for group in self.override_groups
            for name in group.variables
            if name not in valid
        ]
        if unknown:
            raise ValueError(
                f"masking group variable(s) {unknown} not in packed input "
                f"channels {names}"
            )
        grouped = {name for group in self.override_groups for name in group.variables}
        ungrouped = [name for name in names if name not in grouped]
        generators: list[MaskingGenerator] = [self.default.build(ungrouped)]
        generators += [
            group.masking.build(group.variables) for group in self.override_groups
        ]
        return VariableMasking(names, generators)


class VariableMasking:
    """Runtime union of masking generators over a fixed channel list."""

    def __init__(self, names: list[str], generators: list[MaskingGenerator]):
        self._names = list(names)
        self._generators = list(generators)

    def sample_mask(self, device: torch.device) -> torch.Tensor:
        """Sample a boolean presence mask of shape ``[1, n_channels]``.

        ``True`` means the channel is present; ``False`` means it is dropped.
        The leading dimension is 1 so the mask broadcasts over the batch (and
        hence over ensemble members) when applied.
        """
        dropped: set[str] = set()
        for generator in self._generators:
            dropped.update(generator.sample(device))
        present = torch.tensor(
            [[name not in dropped for name in self._names]],
            dtype=torch.bool,
            device=device,
        )
        return present
