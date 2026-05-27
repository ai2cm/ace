import abc
import dataclasses
import re

import torch

from fme.ace.data_loading.batch_data import BatchData


@dataclasses.dataclass
class VariableMaskingConfig:
    """
    Randomly masks variables per sample to train robustness to missing inputs.

    Keys in ``rates`` ending with ``_`` are prefix keys: they match all
    variables whose name is ``{prefix}{digits}`` (e.g. ``air_temperature_``
    matches ``air_temperature_0`` through ``air_temperature_7``). Variables
    matched by the same prefix key share one random draw per sample — either
    all levels are masked or none.

    Keys without a trailing ``_`` are exact variable names and receive
    independent draws. Variables not matched by any key in ``rates`` use
    ``default_rate`` with independent draws.

    Masked samples have their IC tensor values replaced with NaN so the model
    learns to handle missing inputs.  Target timesteps are never touched,
    keeping the loss signal intact.  ``channel_mask`` indicators in the step
    tell the model which IC channels are absent.

    Parameters:
        default_rate: Mask probability for variables not listed in ``rates``.
            0.0 means never mask (default).
        rates: Per-variable or per-prefix mask probabilities in [0, 1].
    """

    default_rate: float = 0.0
    rates: dict[str, float] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.default_rate <= 1.0:
            raise ValueError(f"default_rate must be in [0, 1], got {self.default_rate}")
        for key, rate in self.rates.items():
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Mask rate for '{key}' must be in [0, 1], got {rate}")

    def build_modifier(self, n_ic_timesteps: int = 1) -> "VariableMaskingModifier":
        return VariableMaskingModifier(self, n_ic_timesteps=n_ic_timesteps)


@dataclasses.dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation.

    Attributes:
        rotate_probability: The probability of rotating the sphere by 180 degrees,
            as a value between 0.0 and 1.0.
        additional_directional_names: Names of variables whose sign is flipped when
            the poles are reversed. By default this includes known directional
            names as stored in RotateModifier.FLIP_NAMES.
    """

    rotate_probability: float = 0.0
    additional_directional_names: list[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.rotate_probability <= 1.0:
            raise ValueError(
                "rotate_probability must be between 0.0 and 1.0, "
                f"got {self.rotate_probability}"
            )

    def build_modifier(self) -> "BatchModifierABC":
        if self.rotate_probability > 0.0:
            return RotateModifier(
                self.rotate_probability, self.additional_directional_names
            )
        return NullModifier()


class BatchModifierABC(abc.ABC):
    @abc.abstractmethod
    def __call__(self, batch: BatchData) -> BatchData: ...


class RotateModifier(BatchModifierABC):
    """
    Modifier that rotates the sphere by 180 degrees so that the poles swap
    places. This is the same as flipping both zonal and meridional axes.

    Also flips the sign of horizontal directional variables such as horizontal
    winds in specific directions, so their new values reflect the rotated axes.
    The names of such variables are stored in the `FLIP_NAMES` class variable.
    Variables not included in this list are not flipped.

    Specifically, the regex pattern r'{name}(_?[0-9]+m?)?$' is used to match the
    names of variables whose sign is flipped when the poles are reversed, for
    each name in `FLIP_NAMES`. This will match both names that end with something
    like "_0", "_1", etc. or something like "10m" or "2m".

    Note that seasons are handled by the fact that solar insolation is a data
    variable, but time is not modified. This means monthly or seasonal averages
    using this data will be affected by the rotation.
    """

    # names of variables whose sign is flipped when the poles are reversed
    FLIP_NAMES = [
        "eastward_wind",
        "northward_wind",
        "UGRD",
        "VGRD",
        "U",
        "V",
    ]

    def __init__(
        self,
        rotate_probability: float,
        additional_directional_names: list[str],
    ):
        self.rotate_probability = rotate_probability
        self.additional_directional_names = additional_directional_names
        self._pattern = re.compile(
            r"({})(_?[0-9]+m?)?$".format(
                "|".join(self.FLIP_NAMES + self.additional_directional_names)
            )
        )

    def __call__(self, batch: BatchData) -> BatchData:
        if batch.horizontal_dims != ["lat", "lon"]:
            raise NotImplementedError(
                "Horizontal dimensions must be lat and lon to rotate the sphere, got "
                f"{batch.horizontal_dims}"
            )
        example_value = next(iter(batch.data.values()))
        apply = (
            torch.rand(example_value.shape[0]).to(example_value.device)
            < self.rotate_probability
        )
        while len(apply.shape) < len(example_value.shape):
            apply = apply.unsqueeze(-1)
        new_data = {}
        for name, value in batch.data.items():
            new_value = torch.flip(value, dims=[-2, -1])
            if self._pattern.match(name):
                new_value = -1 * new_value
            new_data[name] = torch.where(apply, new_value, value)
        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            labels=batch.labels,
            data_mask=batch.data_mask,
        )


class NullModifier(BatchModifierABC):
    def __call__(self, batch: BatchData) -> BatchData:
        return batch


class VariableMaskingModifier(BatchModifierABC):
    """
    Randomly masks IC timesteps of variables by filling with NaN.
    See VariableMaskingConfig for the masking contract.
    """

    # Matches names like air_temperature_0, specific_total_water_7
    _LEVELED_PATTERN = re.compile(r"^(.+_)(\d+)$")

    def __init__(self, config: VariableMaskingConfig, n_ic_timesteps: int = 1):
        if n_ic_timesteps < 1:
            raise ValueError(f"n_ic_timesteps must be >= 1, got {n_ic_timesteps}")
        self._config = config
        self._n_ic_timesteps = n_ic_timesteps

    def _rate_for(self, name: str) -> float:
        if name in self._config.rates:
            return self._config.rates[name]
        m = self._LEVELED_PATTERN.match(name)
        if m and m.group(1) in self._config.rates:
            return self._config.rates[m.group(1)]
        return self._config.default_rate

    def _group_key(self, name: str) -> str:
        """Return the masking group key for a variable (prefix or exact name)."""
        if name in self._config.rates:
            return name
        m = self._LEVELED_PATTERN.match(name)
        if m and m.group(1) in self._config.rates:
            return m.group(1)
        return name

    def sample_masks(
        self,
        variable_names: list[str],
        batch_size: int,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        """Sample per-sample presence masks (True = present, False = masked)."""
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

    def __call__(self, batch: BatchData) -> BatchData:
        variable_names = list(batch.data.keys())
        example = next(iter(batch.data.values()))
        batch_size = example.shape[0]

        # Group variables by their masking key; skip those with rate == 0
        groups: dict[str, list[str]] = {}
        for name in variable_names:
            if self._rate_for(name) > 0.0:
                groups.setdefault(self._group_key(name), []).append(name)

        if not groups:
            return batch

        new_data = dict(batch.data)
        any_masked = False
        n_ic = self._n_ic_timesteps

        for group_vars in groups.values():
            rate = self._rate_for(group_vars[0])
            # One random draw per sample for the entire group
            masked = torch.rand(batch_size) < rate  # True → mask this sample
            if not masked.any():
                continue
            any_masked = True
            for name in group_vars:
                tensor = new_data[name]  # [batch, time, ...]
                # Mask only IC timesteps; leave target timesteps intact so the
                # loss can still supervise predictions for masked-input samples.
                ic_slice = tensor[:, :n_ic]
                broadcast = masked.view(batch_size, *([1] * (ic_slice.dim() - 1)))
                masked_ic = torch.where(
                    broadcast, torch.full_like(ic_slice, float("nan")), ic_slice
                )
                new_data[name] = torch.cat([masked_ic, tensor[:, n_ic:]], dim=1)

        if not any_masked:
            return batch

        return BatchData(
            data=new_data,
            time=batch.time,
            horizontal_dims=batch.horizontal_dims,
            epoch=batch.epoch,
            labels=batch.labels,
            n_ensemble=batch.n_ensemble,
            data_mask=batch.data_mask,
        )
