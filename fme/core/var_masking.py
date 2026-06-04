import dataclasses
import re
from typing import ClassVar, Literal

import torch


@dataclasses.dataclass
class UniformMaskingConfig:
    """
    Configuration for masking a uniformly sampled count of variables.

    A random integer ``n`` is drawn from ``[min_vars, max_vars]`` (inclusive) for
    each sample, then ``n`` variables are chosen uniformly at random from those not
    in ``ignore_vars`` and masked.
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

    def _eligible_and_bounds(
        self, variable_names: list[str]
    ) -> tuple[list[str], int, int]:
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
        return eligible, lo, hi

    def sample_masks(
        self,
        variable_names: list[str],
        batch_size: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample presence masks (True = present, False = masked)."""
        eligible, lo, hi = self._eligible_and_bounds(variable_names)
        n_eligible = len(eligible)
        if hi == 0:
            return {}
        if device is None:
            present = torch.ones(batch_size, n_eligible, dtype=torch.bool)
        else:
            present = torch.ones(
                batch_size, n_eligible, dtype=torch.bool, device=device
            )
        for i in range(batch_size):
            n = int(torch.randint(lo, hi + 1, (1,)).item())
            if n > 0:
                if device is None:
                    indices = torch.randperm(n_eligible)[:n]
                else:
                    indices = torch.randperm(n_eligible, device=device)[:n]
                present[i, indices] = False
        result: dict[str, torch.Tensor] = {}
        for j, name in enumerate(eligible):
            col = present[:, j]
            if not bool(col.all().item()):
                result[name] = col
        return result


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Dropout configuration for randomly masking input channels per sample
    during training.

    Keys in ``per_variable`` ending with ``_`` are prefix keys: they match all
    variables whose name is ``{prefix}{digits}`` (e.g. ``air_temperature_``
    matches ``air_temperature_0`` through ``air_temperature_7``). Variables
    matched by the same prefix key share one random draw per sample, so either
    all levels are masked or none.

    ``default_rate`` is a reserved key in ``per_variable``. Variables not
    matched by an exact or prefix key use that rate with independent draws.
    Uniform masking is applied only to variables not explicitly handled by an
    exact or prefix key, and default-rate variables are excluded from uniform
    masking when ``default_rate`` is greater than zero.
    """

    _DEFAULT_RATE_KEY: ClassVar[str] = "default_rate"
    _LEVELED_PATTERN: ClassVar[re.Pattern] = re.compile(r"^(.+_)(\d+)$")

    uniform: UniformMaskingConfig | None = None
    per_variable: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.uniform, dict):
            self.uniform = UniformMaskingConfig(**self.uniform)
        for key, rate in self.per_variable.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Mask rate for '{key}' must be in [0, 1], got {rate}")

    @property
    def _default_rate(self) -> float:
        return self.per_variable.get(self._DEFAULT_RATE_KEY, 0.0)

    def _explicit_group_key(self, name: str) -> str | None:
        """Return the exact or prefix group key handling ``name``, if any."""
        if name != self._DEFAULT_RATE_KEY and name in self.per_variable:
            return name
        m = self._LEVELED_PATTERN.match(name)
        if m:
            prefix = m.group(1)
            if prefix in self.per_variable:
                return prefix
        return None

    def _explicit_groups(self, variable_names: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for name in variable_names:
            key = self._explicit_group_key(name)
            if key is not None:
                groups.setdefault(key, []).append(name)
        return groups

    def _explicit_handled_names(self, variable_names: list[str]) -> set[str]:
        return {
            name
            for group_names in self._explicit_groups(variable_names).values()
            for name in group_names
        }

    def _uniform_candidate_names(self, variable_names: list[str]) -> list[str]:
        explicit_handled = self._explicit_handled_names(variable_names)
        default_handled = (
            set(variable_names) - explicit_handled
            if self._default_rate > 0.0
            else set()
        )
        excluded = explicit_handled | default_handled
        return [name for name in variable_names if name not in excluded]

    def _uniform_eligible_and_bounds(
        self, variable_names: list[str]
    ) -> tuple[list[str], int, int]:
        if self.uniform is None:
            return [], 0, 0
        return self.uniform._eligible_and_bounds(
            self._uniform_candidate_names(variable_names)
        )

    def _sample_rate_mask(
        self,
        rate: float,
        batch_size: int,
        device: torch.device | None,
    ) -> torch.Tensor:
        if device is None:
            masked = torch.rand(batch_size) < rate
        else:
            masked = torch.rand(batch_size, device=device) < rate
        return ~masked

    def _merge_mask(
        self,
        result: dict[str, torch.Tensor],
        name: str,
        mask: torch.Tensor,
    ) -> None:
        mask = mask.to(dtype=torch.bool)
        if name in result:
            result[name] = result[name].to(device=mask.device, dtype=torch.bool) & mask
        else:
            result[name] = mask

    def sample_masks(
        self,
        variable_names: list[str],
        batch_size: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample presence masks (True = present, False = masked)."""
        result: dict[str, torch.Tensor] = {}

        explicit_groups = self._explicit_groups(variable_names)
        for key, group_vars in explicit_groups.items():
            rate = self.per_variable[key]
            if rate > 0.0:
                present = self._sample_rate_mask(rate, batch_size, device)
                for name in group_vars:
                    self._merge_mask(result, name, present)

        default_rate = self._default_rate
        explicit_handled = {
            name for group_vars in explicit_groups.values() for name in group_vars
        }
        if default_rate > 0.0:
            for name in variable_names:
                if name not in explicit_handled:
                    present = self._sample_rate_mask(default_rate, batch_size, device)
                    self._merge_mask(result, name, present)

        if self.uniform is not None:
            uniform_masks = self.uniform.sample_masks(
                self._uniform_candidate_names(variable_names), batch_size, device
            )
            for name, mask in uniform_masks.items():
                self._merge_mask(result, name, mask)
        return result

    def validate_variable_names(self, variable_names: list[str]) -> None:
        if self.uniform is not None:
            self._uniform_eligible_and_bounds(variable_names)

    def can_mask(self, name: str, variable_names: list[str]) -> bool:
        if name not in variable_names:
            return False
        explicit_key = self._explicit_group_key(name)
        if explicit_key is not None:
            return self.per_variable[explicit_key] > 0.0
        if self._default_rate > 0.0:
            return True
        if self.uniform is None:
            return False
        eligible, _, hi = self._uniform_eligible_and_bounds(variable_names)
        return name in eligible and hi > 0
