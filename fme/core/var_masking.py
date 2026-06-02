import dataclasses
import re
from typing import ClassVar, Literal

import torch


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Dropout configuration for randomly masking input channels per sample
    during training.

    Keys in ``rates`` ending with ``_`` are prefix keys: they match all
    variables whose name is ``{prefix}{digits}`` (e.g. ``air_temperature_``
    matches ``air_temperature_0`` through ``air_temperature_7``). Variables
    matched by the same prefix key share one random draw per sample — either
    all levels are masked or none.

    Keys without a trailing ``_`` are exact variable names and receive
    independent draws. Variables not matched by any key in ``rates`` use
    ``default_rate`` with independent draws.

    Parameters:
        default_rate: Mask probability for variables not listed in ``rates``.
            0.0 means never mask (default).
        rates: Per-variable or per-prefix mask probabilities in [0, 1].
    """

    _LEVELED_PATTERN: ClassVar[re.Pattern] = re.compile(r"^(.+_)(\d+)$")

    default_rate: float = 0.0
    rates: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.default_rate <= 1.0:
            raise ValueError(f"default_rate must be in [0, 1], got {self.default_rate}")
        for key, rate in self.rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Mask rate for '{key}' must be in [0, 1], got {rate}")

    def _rate_for(self, name: str) -> float:
        if name in self.rates:
            return self.rates[name]
        m = self._LEVELED_PATTERN.match(name)
        if m and m.group(1) in self.rates:
            return self.rates[m.group(1)]
        return self.default_rate

    def _group_key(self, name: str) -> str:
        """Return the masking group key for a variable (prefix or exact name)."""
        if name in self.rates:
            return name
        m = self._LEVELED_PATTERN.match(name)
        if m and m.group(1) in self.rates:
            return m.group(1)
        return name

    def sample_masks(
        self,
        variable_names: list[str],
        batch_size: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample presence masks (True = present, False = masked).

        Args:
            variable_names: Names of variables to consider for masking.
            batch_size: Number of samples in the batch.
            device: Device for the output tensors.

        Returns:
            Mapping from variable name to [batch_size] bool tensor where
            True means present and False means masked. Only variables with
            non-zero mask rate are included.
        """
        groups: dict[str, list[str]] = {}
        for name in variable_names:
            if self._rate_for(name) > 0.0:
                groups.setdefault(self._group_key(name), []).append(name)
        result: dict[str, torch.Tensor] = {}
        for group_vars in groups.values():
            rate = self._rate_for(group_vars[0])
            masked = torch.rand(batch_size) < rate
            if device is not None:
                masked = masked.to(device)
            present = ~masked
            for name in group_vars:
                result[name] = present
        return result


@dataclasses.dataclass
class UniformVariableMaskingConfig:
    """
    Dropout configuration that masks a uniformly-sampled count of input channels
    per sample during training.

    A random integer ``n`` is drawn from ``[min_vars, max_vars]`` (inclusive) for
    each sample, then ``n`` variables are chosen uniformly at random from those not
    in ``ignore_vars`` and masked.

    Parameters:
        min_vars: Minimum number of variables to mask per sample. Use ``"min"`` to
            default to 0 at sample time.
        max_vars: Maximum number of variables to mask per sample. Use ``"max"`` to
            default to the number of eligible variables at sample time.
        ignore_vars: Variables that are never eligible for masking.
    """

    min_vars: int | Literal["min"] = "min"
    max_vars: int | Literal["max"] = "max"
    ignore_vars: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if self.min_vars != "min" and (
            not isinstance(self.min_vars, int) or self.min_vars < 0
        ):
            raise ValueError(
                f"min_vars must be 'min' or a non-negative int, got {self.min_vars!r}"
            )
        if self.max_vars != "max" and (
            not isinstance(self.max_vars, int) or self.max_vars < 0
        ):
            raise ValueError(
                f"max_vars must be 'max' or a non-negative int, got {self.max_vars!r}"
            )
        if isinstance(self.min_vars, int) and isinstance(self.max_vars, int):
            if self.min_vars > self.max_vars:
                raise ValueError(
                    f"min_vars ({self.min_vars}) must be <= max_vars ({self.max_vars})"
                )

    def sample_masks(
        self,
        variable_names: list[str],
        batch_size: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample presence masks (True = present, False = masked).

        Args:
            variable_names: Names of variables to consider for masking.
            batch_size: Number of samples in the batch.
            device: Device for the output tensors.

        Returns:
            Mapping from variable name to [batch_size] bool tensor where
            True means present and False means masked. Only variables masked
            in at least one sample are included.
        """
        eligible = [v for v in variable_names if v not in self.ignore_vars]
        n_eligible = len(eligible)
        lo = 0 if self.min_vars == "min" else self.min_vars
        hi = n_eligible if self.max_vars == "max" else self.max_vars
        if isinstance(self.min_vars, int) and lo > n_eligible:
            raise ValueError(
                f"min_vars ({lo}) exceeds number of eligible variables ({n_eligible})"
            )
        if isinstance(self.max_vars, int) and hi > n_eligible:
            raise ValueError(
                f"max_vars ({hi}) exceeds number of eligible variables ({n_eligible})"
            )
        if lo > hi:
            raise ValueError(
                f"min_vars ({lo}) must be <= max_vars ({hi}) after resolving strings"
            )
        if hi == 0:
            return {}
        # [batch_size, n_eligible] True = present, False = masked
        present = torch.ones(batch_size, n_eligible, dtype=torch.bool)
        for i in range(batch_size):
            n = int(torch.randint(lo, hi + 1, (1,)).item())
            if n > 0:
                indices = torch.randperm(n_eligible)[:n]
                present[i, indices] = False
        result: dict[str, torch.Tensor] = {}
        for j, name in enumerate(eligible):
            col = present[:, j]
            if not col.all():
                if device is not None:
                    col = col.to(device)
                result[name] = col
        return result
